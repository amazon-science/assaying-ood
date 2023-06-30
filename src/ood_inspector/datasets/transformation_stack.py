from torchvision.transforms import transforms as tv_transforms


def decompose_transform(transform: tv_transforms.Compose):
    pre_normalization = []
    normalization = None
    post_normalization = []

    iterator_pre_norm = True
    for subtransform in transform.transforms:
        if isinstance(subtransform, tv_transforms.Compose):
            pre_norm, norm, post_norm = decompose_transform(subtransform)
            pre_normalization += pre_norm
            if normalization is None:
                normalization = norm
            else:
                raise ValueError("Two normalizations are not supported.")
            post_normalization += post_norm
        if isinstance(subtransform, tv_transforms.Normalize):
            if not iterator_pre_norm:
                raise ValueError("Two normalizations are not supported.")
            iterator_pre_norm = False
            normalization = subtransform
        else:
            if iterator_pre_norm:
                pre_normalization.append(subtransform)
            else:
                post_normalization.append(subtransform)
    return pre_normalization, normalization, post_normalization


class TransformationStack:
    def __init__(self, transformation, augmenter) -> None:
        self._pre_normalization, self.normalization, self._post_normalization = decompose_transform(
            transformation
        )

        self.pre_normalization = tv_transforms.Compose(self._pre_normalization)
        self._post_normalization = tv_transforms.Compose(self._post_normalization)

        if augmenter:
            self.pre_normalization = augmenter.compose_with_transform(self.pre_normalization)

    def __call__(self, sample):
        sample["image"] = self.pre_normalization(sample["image"])
        sample["image"] = self.normalization(sample["image"])
        sample["image"] = self.post_normalization(sample["image"])
        return sample

    @property
    def pre_normalization(self):
        """All transformation before the normalization.
        Last transform in the stack is always `toTensor` transform."""
        return self._pre_normalization

    def check_returns_tensor(self, transforms: tv_transforms.Compose):
        returns_tensor = False
        for subtransform in transforms.transforms:
            if isinstance(subtransform, tv_transforms.Compose):
                returns_tensor = self.check_returns_tensor(subtransform)
            if isinstance(subtransform, tv_transforms.ToTensor):
                returns_tensor = True
            if isinstance(subtransform, tv_transforms.ToPILImage):
                returns_tensor = False
        return returns_tensor

    @pre_normalization.setter
    def pre_normalization(self, pre_normalization_transforms):
        if not self.check_returns_tensor(pre_normalization_transforms):
            raise ValueError(
                "Pre normalization does not call ToTensor or is overwritten by ToPILImage"
            )
        self._pre_normalization = pre_normalization_transforms

    @property
    def post_normalization(self):
        """All transformation after the normalization. They must act on torch.Tensor() objects"""
        return self._post_normalization
