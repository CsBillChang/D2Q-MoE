"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""  

import copy, torch, thop, time, os
   
from typing import Tuple     
from ..logger_module import get_logger

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m" 
logger = get_logger(__name__)  
   
     




def _build_profile_inputs(input_shape, device, model=None):
    rgb = torch.randn(size=input_shape, device=device)
    if getattr(model, 'expects_multimodal_input', False):
        depth = torch.randn((input_shape[0], 1, *input_shape[-2:]), device=device)
        return ({'rgb': rgb, 'npy': depth},)
    return (rgb,)
 
def stats(   
    cfg,    
    input_shape: Tuple=(1, 3, 640, 640), module=None) -> Tuple[int, dict]:

    # base_size = cfg.train_dataloader.collate_fn.base_size    
    input_shape = (1, 3, *cfg.yaml_cfg['eval_spatial_size'])     
  
    if module:
        model_for_info = copy.deepcopy(module)
    else:     
        model_for_info = copy.deepcopy(cfg.model).deploy()   
   
    if cfg.device:     
        device = torch.device(cfg.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_for_info.to(device)   
    profile_inputs = _build_profile_inputs(input_shape, device, model_for_info)
    flops, macs = None, None     

    try:
        from calflops import calculate_flops
        flops, macs, _ = calculate_flops(
            model=model_for_info, 
            input_shape=input_shape,
            output_as_string=True,   
            output_precision=4,
            print_detailed=False
        )
    except Exception:     
        logger.warning(RED + "calculate_flops failed.. using thop instead.." + RESET)

    if flops is None or macs is None: 
        try:    
            macs_val = thop.profile(model_for_info, inputs=profile_inputs, verbose=False)[0]    
            macs, flops = thop.clever_format([macs_val, macs_val * 2], format="%.3f")
        except Exception:  
            logger.warning(RED + "thop profile failed.. set FLOPs/MACs to N/A." + RESET)
            flops, macs = "N/A", "N/A"   
    params = sum(p.numel() for p in model_for_info.parameters())
    del model_for_info  
   
    return params, {"Model FLOPs:%s   MACs:%s   Params:%s" %(flops, macs, params)}
 
def get_weight_size(module):
    timestamp = f'{int(time.time() * 1000)}.pth' # 毫秒级时间戳
    torch.save(module.state_dict(), timestamp)   
    stats = os.stat(timestamp)
    logger.info(ORANGE + f'only model size: {stats.st_size / (1024 ** 2):.1f} MB' + RESET) 
    os.remove(timestamp)
