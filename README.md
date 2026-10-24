# D²Q-MoE

This repository contains the submission-time public code snapshot for **D²Q-MoE**, a pseudo-depth-assisted object detection model built on DEIM/D-FINE.

D²Q-MoE introduces three components in the detection pipeline:

- **GTDMoE**: a depth feature adaptation module.
- **RADF**: a cross-modal feature fusion module.
- **SRQM**: a query classification modulation head.

The public snapshot retains the general framework, training entry, module interfaces, and an example RGB-D configuration chain. The three components use simplified public implementations. The full research designs and paper-specific training settings are reserved for a later release after paper acceptance.

## Project Layout

```text
D2Q-MoE/
├── train.py
├── configs/
│   ├── base/
│   ├── dataset/
│   │   └── rgbd_detection.yml
│   ├── dfine/
│   └── d2q_moe/
│       └── d2q_m_public.yml
├── d2q_moe/
│   ├── modules.py
│   └── model.py
└── engine/
    ├── backbone/
    ├── core/
    ├── data/
    ├── deim/
    ├── misc/
    ├── optim/
    └── solver/
```

## Environment

```bash
conda create -n d2q_moe python=3.10 -y
conda activate d2q_moe
pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA environment. Pretrained backbone loading is disabled in the example configuration.

## Dataset

The public configuration expects COCO-style detection annotations, RGB images, and spatially aligned single-channel pseudo-depth maps saved as `.npy` files. Each depth file should have the same relative stem as its RGB image, for example `sample.jpg` and `sample.npy`.

Edit the paths and number of classes in:

```text
configs/dataset/rgbd_detection.yml
```

For custom annotations, use consecutive category IDs from `0` to `num_classes - 1` with `remap_mscoco_category: false`.

The private substation dataset, annotations, and pseudo-depth files used in the manuscript are not included.

## Training

Run the simplified public model from the repository root:

```bash
python train.py -c configs/d2q_moe/d2q_m_public.yml --device cuda:0 --use-amp
```

For CPU execution, use `--device cpu` and omit `--use-amp`.

The separate RGB baseline remains available through `configs/dfine/dfine_hgnetv2_m_custom.yml`.

## Evaluation

Evaluate a checkpoint trained with the public configuration:

```bash
python train.py -c configs/d2q_moe/d2q_m_public.yml --device cuda:0 --test-only --resume path/to/public_checkpoint.pth
```
