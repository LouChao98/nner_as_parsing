import logging

import fastNLP
import pytorch_lightning
import wandb
from hydra._internal.utils import is_under_debugger
from hydra.utils import HydraConfig
from omegaconf import ListConfig, OmegaConf

pl_logger = logging.getLogger('lightning')
pl_logger.propagate = False

wandb_logger = logging.getLogger('wandb')
# wandb_logger.propagate = False

# OmegaConf.register_new_resolver('in', lambda x, y: x in y)
OmegaConf.register_new_resolver('lang', lambda x: x.split('_')[0])
OmegaConf.register_new_resolver('last', lambda x: x.split('/')[-1])
# OmegaConf.register_new_resolver('cat', lambda x, y: x + y)
OmegaConf.register_new_resolver('in_debugger', lambda x, default=None: x if is_under_debugger() else default)
OmegaConf.register_new_resolver('path_guard', lambda x: x.replace('/', '-')[:240])

def name_guard(fallback):
    try:
        return HydraConfig.get().job.override_dirname
    except ValueError as v:
        if 'HydraConfig was not set' in str(v):
            return fallback
        raise v

OmegaConf.register_new_resolver('name_guard', name_guard)

def choose_accelerator(gpus):
    if isinstance(gpus, int):
        return 'ddp' if gpus > 1 else None
    elif isinstance(gpus, str):
        return 'ddp' if len(gpus.split(',')) > 1 else None
    elif isinstance(gpus, (list, ListConfig)):
        return 'ddp' if len(gpus) > 1 else None
    elif gpus is None:
        return None
    raise ValueError(f'Unrecognized {gpus=} ({type(gpus)})')


OmegaConf.register_new_resolver('accelerator', choose_accelerator)


def process_ckpt_name(path, watch_field, test_when_val):
    if test_when_val:
        return path + '-{%s:.2f}' % watch_field.replace('val', 'test')
    else:
        return path


OmegaConf.register_new_resolver('ckpt_name', process_ckpt_name)


g_cfg = None
