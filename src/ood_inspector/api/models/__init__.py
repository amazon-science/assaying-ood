import hydra.core.config_store as hydra_config_store

from ood_inspector.api.models import mock_config, timm_config, torchvision_config

config_store = hydra_config_store.ConfigStore.instance()
timm_config.register_configs()
torchvision_config.register_configs()
config_store.store(group="model", name="mock", node=mock_config.MockModelConfig)
