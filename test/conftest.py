import pytest
import torch


class TestDataloader:
    def __call__(
        self, inputs: torch.Tensor, targets: torch.Tensor, batch_size: int = 1
    ) -> torch.utils.data.DataLoader:
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


@pytest.fixture
def generate_dataloader():
    return TestDataloader()
