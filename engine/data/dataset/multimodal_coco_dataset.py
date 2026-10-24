"""
Multimodal COCO dataset for RGB + NPY inputs.

The RAM cache stores the deterministic, aspect-ratio-preserving resize only.
Random augmentation and the final training resize still run for every sample.
"""

import math
import os
import random
import shutil
import threading
import weakref
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import tqdm
from PIL import Image

from ...core import register
from .coco_dataset import LOCAL_RANK, NUM_THREADS, CocoDetection


@dataclass(frozen=True)
class _MultimodalCacheRecord:
    rgb_offset: int
    npy_offset: int
    height: int
    width: int
    annotations: List[dict]
    npy_tensor: Optional[torch.Tensor] = None


@dataclass
class _MultimodalRamCache:
    rgb_buffer: torch.Tensor
    npy_buffer: Optional[torch.Tensor]
    records: Dict[int, _MultimodalCacheRecord]
    size_bytes: int
    shared_memory: bool


_MULTIMODAL_CACHE_REGISTRY = weakref.WeakValueDictionary()
_MULTIMODAL_CACHE_LOCK = threading.RLock()


def _clear_multimodal_cache_registry():
    """Clear process-local cache references. Intended for isolated tests."""
    with _MULTIMODAL_CACHE_LOCK:
        _MULTIMODAL_CACHE_REGISTRY.clear()


@register()
class MultimodalCocoDetection(CocoDetection):
    __inject__ = ["transforms"]
    __share__ = [
        "remap_mscoco_category",
        "cache_imgsz",
        "ram_cache",
        "num_classes",
        "multimodal_cache_share_memory",
    ]

    def __init__(
        self,
        img_folder,
        npy_folder,
        ann_file,
        transforms,
        return_masks=False,
        remap_mscoco_category=False,
        cache_imgsz=None,
        ram_cache=False,
        num_classes=None,
        npy_size_mismatch="error",
        npy_dtype="float32",
        multimodal_cache_share_memory=True,
    ):
        self.npy_folder = str(npy_folder)
        self.npy_size_mismatch = npy_size_mismatch
        self.npy_dtype = npy_dtype
        self.multimodal_cache_share_memory = bool(multimodal_cache_share_memory)
        self._multimodal_ram_cache: Optional[_MultimodalRamCache] = None

        if self.npy_size_mismatch not in {"error", "resize"}:
            raise ValueError("npy_size_mismatch must be one of {'error', 'resize'}")

        # Multimodal caching must be deferred until the NPY settings are known,
        # otherwise the parent class eagerly creates a second RGB-only cache.
        super().__init__(
            img_folder=img_folder,
            ann_file=ann_file,
            transforms=transforms,
            return_masks=return_masks,
            remap_mscoco_category=remap_mscoco_category,
            cache_imgsz=cache_imgsz,
            ram_cache=ram_cache,
            num_classes=num_classes,
            defer_ram_cache=True,
        )

        if self.ram_cache:
            self._initialize_multimodal_ram_cache()

    def __getitem__(self, idx):
        sample, target = self.load_item(idx)
        if self._transforms is not None:
            sample, target, _ = self._transforms(sample, target, self)
        return sample, target

    def load_item(self, idx):
        cache = self._multimodal_ram_cache
        if cache is None:
            rgb, target = super().load_item(idx)
            image_id = self.ids[idx]
            npy = self._read_npy_tensor(image_id, rgb)
        else:
            image_id = self.ids[idx]
            record = cache.records[image_id]
            rgb_numel = record.height * record.width * 3
            rgb_array = (
                cache.rgb_buffer[record.rgb_offset : record.rgb_offset + rgb_numel]
                .view(record.height, record.width, 3)
                .numpy()
            )
            # Detach the per-sample PIL image from the shared backing buffer so
            # an in-place transform cannot corrupt later samples.
            rgb = Image.fromarray(rgb_array).copy()
            rgb, target = self._prepare_loaded_item(idx, rgb, record.annotations)

            if cache.npy_buffer is not None:
                npy_numel = record.height * record.width
                npy = (
                    cache.npy_buffer[record.npy_offset : record.npy_offset + npy_numel]
                    .view(1, record.height, record.width)
                    .clone()
                )
            else:
                npy = record.npy_tensor.clone()

        sample = {"rgb": rgb, "npy": npy}
        return sample, target

    def _initialize_multimodal_ram_cache(self):
        key = self._multimodal_cache_key()
        with _MULTIMODAL_CACHE_LOCK:
            cache = _MULTIMODAL_CACHE_REGISTRY.get(key)
            if cache is None:
                if not self.check_multimodal_cache_ram():
                    return
                cache = self._populate_multimodal_ram_cache()
                _MULTIMODAL_CACHE_REGISTRY[key] = cache
            self._attach_multimodal_ram_cache(cache)

    def _multimodal_cache_key(self):
        ann_path = Path(self.ann_file).expanduser().resolve()
        ann_stat = ann_path.stat()

        def _normal_path(path):
            return os.path.normcase(str(Path(path).expanduser().resolve()))

        return (
            _normal_path(self.img_folder),
            _normal_path(self.npy_folder),
            _normal_path(ann_path),
            ann_stat.st_size,
            ann_stat.st_mtime_ns,
            self.cache_imgsz,
            self.npy_size_mismatch,
            self.npy_dtype,
            self.multimodal_cache_share_memory,
        )

    def _target_cache_size(self, image_id):
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root, image_info["file_name"])
        with Image.open(image_path) as image:
            width, height = image.size

        ratio = self._cache_resize_ratio(height, width)
        if ratio == 1:
            return height, width

        new_width = min(math.ceil(width * ratio), self.cache_imgsz)
        new_height = min(math.ceil(height * ratio), self.cache_imgsz)
        return new_height, new_width

    def _build_cache_layout(self):
        layout = {}
        rgb_offset = 0
        npy_offset = 0
        for image_id in self.ids:
            height, width = self._target_cache_size(image_id)
            layout[image_id] = (rgb_offset, npy_offset, height, width)
            rgb_offset += height * width * 3
            npy_offset += height * width
        return layout, rgb_offset, npy_offset

    def _allocate_cache_buffers(self, rgb_numel, npy_numel):
        rgb_buffer = torch.empty(rgb_numel, dtype=torch.uint8)
        npy_buffer = None
        if self.npy_dtype == "float32":
            npy_buffer = torch.empty(npy_numel, dtype=torch.float32)

        use_shared_memory = self.multimodal_cache_share_memory
        requested_bytes = rgb_buffer.numel() * rgb_buffer.element_size()
        if npy_buffer is not None:
            requested_bytes += npy_buffer.numel() * npy_buffer.element_size()

        shared_memory_path = Path("/dev/shm")
        if use_shared_memory and os.name == "posix" and shared_memory_path.exists():
            shared_memory_free = shutil.disk_usage(shared_memory_path).free
            if requested_bytes > shared_memory_free * 0.9:
                use_shared_memory = False
                print(
                    f"{requested_bytes / (1 << 30):.1f}GB shared memory required for the "
                    f"multimodal cache but only {shared_memory_free / (1 << 30):.1f}GB "
                    "is available in /dev/shm; using process-local packed storage instead."
                )

        if use_shared_memory:
            try:
                rgb_buffer.share_memory_()
                if npy_buffer is not None:
                    npy_buffer.share_memory_()
            except (OSError, RuntimeError) as exc:
                print(
                    "Unable to place the complete multimodal cache in shared memory; "
                    f"continuing with process-local packed storage. Reason: {exc}"
                )

        shared_memory = (
            rgb_buffer.is_shared()
            and npy_buffer is not None
            and npy_buffer.is_shared()
        )
        return rgb_buffer, npy_buffer, shared_memory

    def _populate_multimodal_ram_cache(self):
        layout, rgb_numel, npy_numel = self._build_cache_layout()
        rgb_buffer, npy_buffer, shared_memory = self._allocate_cache_buffers(
            rgb_numel,
            npy_numel,
        )
        records = {}
        cached_bytes = 0
        gb = 1 << 30

        def _load_pair(image_id):
            _, rgb, annotations = self.load_image_and_label_from_coco(image_id)
            rgb_offset, npy_offset, height, width = layout[image_id]
            if rgb.size != (width, height):
                raise RuntimeError(
                    f"Unexpected cached RGB size for image_id={image_id}: "
                    f"got={rgb.size}, expected=({width}, {height})"
                )

            rgb_tensor = torch.from_numpy(np.array(rgb, dtype=np.uint8, copy=True)).contiguous()
            rgb_buffer[rgb_offset : rgb_offset + rgb_tensor.numel()].copy_(rgb_tensor.view(-1))

            npy_tensor = self._read_npy_tensor(image_id, rgb).contiguous()
            if tuple(npy_tensor.shape) != (1, height, width):
                raise RuntimeError(
                    f"Unexpected cached NPY size for image_id={image_id}: "
                    f"got={tuple(npy_tensor.shape)}, expected={(1, height, width)}"
                )

            stored_npy = None
            if npy_buffer is not None:
                if npy_tensor.dtype != npy_buffer.dtype:
                    raise TypeError(
                        f"Cached NPY dtype mismatch for image_id={image_id}: "
                        f"got={npy_tensor.dtype}, expected={npy_buffer.dtype}"
                    )
                npy_buffer[npy_offset : npy_offset + npy_tensor.numel()].copy_(npy_tensor.view(-1))
            else:
                stored_npy = npy_tensor

            # Release the temporary parent-class cache entry immediately. The
            # packed buffers are the sole pixel cache after this point.
            self.ims[image_id] = None
            self.labs[image_id] = None

            record = _MultimodalCacheRecord(
                rgb_offset=rgb_offset,
                npy_offset=npy_offset,
                height=height,
                width=width,
                annotations=annotations,
                npy_tensor=stored_npy,
            )
            pair_bytes = rgb_tensor.numel() * rgb_tensor.element_size()
            pair_bytes += npy_tensor.numel() * npy_tensor.element_size()
            return image_id, record, pair_bytes

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(_load_pair, self.ids)
            pbar = tqdm.tqdm(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for _, (image_id, record, pair_bytes) in pbar:
                records[image_id] = record
                cached_bytes += pair_bytes
                pbar.desc = f"Caching RGB+NPY pairs ({cached_bytes / gb:.1f}GB RAM)"
            pbar.close()

        # Keep these empty: MultimodalCocoDetection reads only from the packed
        # pair cache, while the single-modal CocoDetection cache is untouched.
        self.ims = {}
        self.labs = {}
        return _MultimodalRamCache(
            rgb_buffer=rgb_buffer,
            npy_buffer=npy_buffer,
            records=records,
            size_bytes=cached_bytes,
            shared_memory=shared_memory,
        )

    def _attach_multimodal_ram_cache(self, cache):
        self._multimodal_ram_cache = cache
        self.ims = {}
        self.labs = {}
        self.cache_flag = False
        for image_id, record in cache.records.items():
            self.coco.imgs[image_id]["width"] = record.width
            self.coco.imgs[image_id]["height"] = record.height
            self.coco.imgToAnns[image_id] = record.annotations
            for annotation in record.annotations:
                self.coco.anns[annotation["id"]] = annotation

    def _read_npy_tensor(self, image_id, rgb_image):
        npy_path = self._resolve_npy_path(image_id)
        raw = np.load(npy_path, allow_pickle=True)
        if isinstance(raw, np.lib.npyio.NpzFile):
            try:
                keys = list(raw.files)
                if "arr_0" in keys:
                    npy = raw["arr_0"]
                elif len(keys) == 1:
                    npy = raw[keys[0]]
                else:
                    raise ValueError(
                        "NPZ file with multiple arrays must contain key 'arr_0', "
                        f"got keys={keys} for {npy_path}"
                    )
            except KeyError as exc:
                raise ValueError(
                    "NPZ file must contain a single array or key 'arr_0', "
                    f"got keys={list(raw.files)} for {npy_path}"
                ) from exc
            finally:
                raw.close()
        else:
            npy = raw

        npy = np.squeeze(npy)
        if npy.ndim != 2:
            raise ValueError(
                f"NPY modality must be 2D (H, W), got shape {tuple(npy.shape)} for {npy_path}"
            )

        if self.npy_dtype == "float32":
            npy = npy.astype(np.float32, copy=False)
        npy_tensor = torch.from_numpy(npy).unsqueeze(0)

        rgb_width, rgb_height = rgb_image.size
        npy_height, npy_width = npy_tensor.shape[-2:]
        if (npy_height, npy_width) != (rgb_height, rgb_width):
            if self.npy_size_mismatch == "error":
                raise ValueError(
                    f"NPY/RGB size mismatch for image_id={image_id}: "
                    f"npy=({npy_height}, {npy_width}), rgb=({rgb_height}, {rgb_width})"
                )

            npy_tensor = F.interpolate(
                npy_tensor.unsqueeze(0),
                size=(rgb_height, rgb_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return npy_tensor

    def _estimate_multimodal_cache_bytes(self, sample_ids):
        # The normal path stores RGB uint8 (3 bytes/pixel) and NPY float32
        # (4 bytes/pixel). Use 8 bytes for an unspecified source NPY dtype.
        npy_element_size = 4 if self.npy_dtype == "float32" else 8
        total = 0
        for image_id in sample_ids:
            height, width = self._target_cache_size(image_id)
            total += height * width * (3 + npy_element_size)
        return total

    def check_multimodal_cache_ram(self, safety_margin=0.5):
        """Check the combined resized RGB + NPY cache against available RAM."""
        if self.ni == 0:
            return True

        gb = 1 << 30
        sample_count = min(self.ni, 30)
        sample_ids = random.sample(self.ids, sample_count)
        sampled_bytes = self._estimate_multimodal_cache_bytes(sample_ids)
        required_bytes = sampled_bytes * self.ni / sample_count * (1 + safety_margin)
        memory = psutil.virtual_memory()
        can_cache = required_bytes < memory.available
        if not can_cache:
            print(
                f"{required_bytes / gb:.1f}GB RAM required to cache resized RGB+NPY pairs "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{memory.available / gb:.1f}/{memory.total / gb:.1f}GB available, "
                "not caching multimodal data ⚠️"
            )
        return can_cache

    def _resolve_npy_path(self, image_id):
        image_info = self.coco.loadImgs(image_id)[0]
        stem = Path(image_info["file_name"]).stem
        npy_path = os.path.join(self.npy_folder, f"{stem}.npy")
        npz_path = os.path.join(self.npy_folder, f"{stem}.npz")

        if os.path.exists(npy_path):
            return npy_path
        if os.path.exists(npz_path):
            return npz_path

        raise FileNotFoundError(
            f"Missing modality file for image_id={image_id}. Tried: {npy_path} and {npz_path}"
        )

    def extra_repr(self) -> str:
        s = super().extra_repr()
        s += f"\n npy_folder: {self.npy_folder}"
        s += f"\n npy_size_mismatch: {self.npy_size_mismatch}"
        s += f"\n multimodal_cache_share_memory: {self.multimodal_cache_share_memory}"
        return s
