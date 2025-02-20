import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

@hydra.main(config_path='configs', config_name='td3_v2', version_base=None)
def main(cfg: DictConfig):
    # instantiate the trainer, env is defined in the config
    trainer = instantiate(cfg.trainer, cfg)
    trainer.train()

if __name__ == '__main__':
    main()
