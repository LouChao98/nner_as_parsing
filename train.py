from __future__ import annotations

import logging
import os
import random
import string

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import HydraConfig, instantiate
from omegaconf import DictConfig

import src
from src.data.datamodule import DataModule
from src.runners.basic import BasicRunner
from src.utils.callback import BestWatcherCallback
from src.utils.fn import instantiate_no_recursive

log = logging.getLogger(__name__)


# torch.set_anomaly_enabled(True)

@hydra.main('conf', 'conf')
def train(cfg: DictConfig):
    src.g_cfg = cfg
    log.info(f'Working directory: {os.getcwd()}')
    if cfg.name == '@@@AUTO@@@':
        # In the case we can not set name={hydra:job.override_dirname} in config.yaml
        cfg.name = HydraConfig.get().job.override_dirname

    # init multirun
    if (num := HydraConfig.get().job.get('num')) is not None and num > 1:
        # set group in wandb, if use joblib, this will be set from joblib.
        if 'MULTIRUN_ID' not in os.environ:
            os.environ['MULTIRUN_ID'] = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(4))

        if 'logger' in cfg.trainer and 'tags' in cfg.trainer.logger:
            cfg.trainer.logger.tags.append(os.environ['MULTIRUN_ID'])

    if (seed := cfg.seed) is not None:
        pl.seed_everything(seed)
        # torch.use_deterministic_algorithms(True)
        # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    assert not (cfg.runner.load_from_checkpoint is not None and cfg.trainer.resume_from_checkpoint is not None), \
        'You should not use load_from_checkpoint and resume_from_checkpoint at the same time.'
    assert cfg.watch_field.startswith('val/')

    trainer: pl.Trainer = instantiate(cfg.trainer)
    datamodule: DataModule = instantiate_no_recursive(cfg.datamodule)
    runner: BasicRunner = instantiate_no_recursive(cfg.runner, dm=datamodule)
    trainer.fit(runner, datamodule)

    log.info(f'Working directory: {os.getcwd()}')

    # Return metric score for hyperparameter optimization
    if 'optimized_metric' not in cfg: return

    callbacks = trainer.callbacks
    for c in callbacks:
        if isinstance(c, BestWatcherCallback):
            return c.best_model_metric[cfg.optimized_metric]


if __name__ == '__main__':
    train()
