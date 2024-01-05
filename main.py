import hydra
from omegaconf import OmegaConf
import logging
log = logging.getLogger(__name__)


from mlops_exercises.train_model import train


@hydra.main(config_path="configs", config_name="default_config.yaml")
def main(config):
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    config = config.experiments
    
    if config.mode == "train":
        train(**config.params)
        
        
if __name__ == "__main__":
    main()