import logging

import hydra
import pytorch_lightning as pl
import src
from hydra import compose
from hydra.utils import HydraConfig, instantiate
from omegaconf import OmegaConf
from src.my_typing import *
from src.utils.fn import instantiate_no_recursive

log = logging.getLogger(__name__)


@hydra.main('conf', 'conf_test')
def test(cfg: DictConfig):

    if (seed := cfg.seed) is not None:
        pl.seed_everything(seed)

    if cfg.runner.load_from_checkpoint is None:
        log.warning('Testing a random-initialized model.')

    if (p := cfg.runner.load_from_checkpoint) is not None:
        p = Path(p)
        if p.parts[-2] == 'checkpoint':
            config_folder = p.parents[1] / 'config'
        else:
            config_folder = p.parent / 'config'
        if config_folder.exists():
            # Load saved config.
            # Note that this only load overrides. Inconsistency happens if you change sub-config's file.
            # From Hydra's author:
            # https://stackoverflow.com/questions/67170653/how-to-load-hydra-parameters-from-previous-jobs-without-having-to-use-argparse/67172466?noredirect=1
            log.info('Loading saved overrides')
            original_overrides = OmegaConf.load(config_folder / 'overrides.yaml')
            current_overrides = HydraConfig.get().overrides.task
            # hydra_config = OmegaConf.load(config_folder / 'hydra.yaml')
            config_name = 'conf_test'  # hydra_config.hydra.job.config_name
            overrides = original_overrides + current_overrides
            cfg = compose(config_name, overrides=overrides)
            log.info(OmegaConf.to_yaml(cfg))

    src.g_cfg = cfg

    trainer: pl.Trainer = instantiate(cfg.trainer)
    datamodule: DataModule = instantiate_no_recursive(cfg.datamodule)
    runner: BasicRunner = instantiate_no_recursive(cfg.runner, dm=datamodule)
    trainer.test(runner, datamodule=datamodule)

    runner.write_prediction('predict_on_test', 'test')


if __name__ == '__main__':
    test()
