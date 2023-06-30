import dataclasses
from typing import Any, Callable, List

import hydra
import hydra.core.config_store as hydra_config_store
import pytest


@dataclasses.dataclass
class TestConfig:
    datasets: Any
    model: Any
    transform: Any


@pytest.fixture
def inspector_config() -> Callable:
    def config(overrides: List[str]):
        """Create a configuration for testing."""
        config_store = hydra_config_store.ConfigStore.instance()
        config_store.store(name="test", node=TestConfig)
        with hydra.initialize_config_module(version_base="1.1", config_module="ood_inspector/api"):
            return hydra.compose(config_name="test", overrides=overrides)

    return config
