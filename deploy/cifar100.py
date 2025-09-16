"""Code for dealing with the CIFAR100 dataset.
Used by FAR models.

https://www.cs.toronto.edu/~kriz/cifar.html
"""

import os
import pathlib
import pickle
import typing
from collections.abc import Sequence

import numpy as np
import onnxruntime
import onnxruntime.quantization
from PIL import Image
from torchvision import transforms

_CIFAR_IMAGE_SIZE = 32
_CIFAR_MEAN = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32).reshape(3, 1, 1)
_CIFAR_STD = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32).reshape(3, 1, 1)


def _resize_images(
    images: np.ndarray,
    output_size: int,
    normalize: bool = True,
    input_mean: np.ndarray = _CIFAR_MEAN,
    input_std: np.ndarray = _CIFAR_STD,
) -> np.ndarray:
    """Resize all images from (3, 32, 32) to (3, output_size, output_size)
    using bicubic interpolation.
    """

    size = int(output_size / 0.875) if output_size > 32 else output_size
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(output_size),
            transforms.ToTensor(),
        ]
    )
    resized = []
    for img in images:
        img = np.transpose(img, (1, 2, 0))  # [C, H, W] -> [H, W, C]
        img_resized = transform(img).numpy()
        if normalize:
            img_resized = (img_resized - input_mean) / input_std
        else:
            img_resized = (img_resized * 255).astype(np.uint8)
        resized.append(img_resized)
    return np.stack(resized, axis=0)


class CIFAR100Dataset(Sequence):
    """Reads in the CIFAR100 dataset, including images and labels."""

    def __init__(
        self,
        cifar_dir: os.PathLike,
        split: str = "train",
        output_size: int = 224,
        normalize: bool = False,
        input_mean: typing.Optional[np.ndarray] = None,
        input_std: typing.Optional[np.ndarray] = None,
    ):
        self.split = split
        self._input_mean = _CIFAR_MEAN if input_mean is None else input_mean
        self._input_std = _CIFAR_STD if input_std is None else input_std
        self.cifar_dir = pathlib.Path(cifar_dir)
        if split not in {"train", "test"}:
            raise ValueError(f"Unexpected split: {split}. Use 'train' or 'test'.")
        self.data_file = self.cifar_dir / split
        if not self.data_file.exists():
            raise FileNotFoundError(
                f"Missing CIFAR-100 {split} file at {self.data_file}"
            )

        with open(self.data_file, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
        data = entry.get("data")
        labels = entry.get("fine_labels")
        if data is None or labels is None:
            raise ValueError(
                "Missing required keys 'data' or 'fine_labels' in CIFAR-100 pickle file"
            )
        self.images = data.reshape(-1, 3, _CIFAR_IMAGE_SIZE, _CIFAR_IMAGE_SIZE).astype(
            np.uint8
        )
        if self.images.shape[1:] != (3, _CIFAR_IMAGE_SIZE, _CIFAR_IMAGE_SIZE):
            raise ValueError(
                f"Unexpected image shape: {self.images.shape[1:]}, expected (3, 32, 32)"
            )
        self.images = _resize_images(
            images=self.images,
            output_size=output_size,
            normalize=normalize,
            input_mean=self._input_mean,
            input_std=self._input_std,
        )
        if self.images.shape[1:] != (3, output_size, output_size):
            raise ValueError(
                f"Unexpected image shape: {self.images.shape[1:]}, \
                  expected (3, {output_size}, {output_size})"
            )
        self.labels = np.array(labels, dtype=np.int64)
        if len(data) != len(labels):
            raise ValueError(f"Mismatch: {len(data)} images and {len(labels)} labels")

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        if isinstance(index, slice):
            # Return multiple items
            images = self.images[index]  # shape: (N, 3, 32, 32)
            labels = self.labels[index]  # shape: (N,)
            return images, labels
        else:
            # Return a single item
            image = self.images[index]  # shape: (3, 32, 32)
            label = self.labels[index]  # int
            return image, label

    def all_images(self) -> np.ndarray:
        """Reads all images into a single numpy array."""
        return self.images

    def all_images_and_labels(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Reads all images and labels into a pair of numpy arrays."""
        return self.images, self.labels


class CIFAR100CalibrationData(onnxruntime.quantization.calibrate.CalibrationDataReader):
    """Provides the CIFAR100 dataset as calibration data for ONNXRuntime.

    This provides an interface to read the dataset that can be used by the
    ONNXRuntime static quantization.
    """

    def __init__(
        self,
        cifar100_data_dir: os.PathLike,
        input_value_name: str,
        input_size: int = 224,
        max_examples: typing.Optional[int] = None,
        input_mean: typing.Optional[np.ndarray] = None,
        input_std: typing.Optional[np.ndarray] = None,
    ):
        super().__init__()
        self._input_value_name = input_value_name
        self._input_mean = _CIFAR_MEAN if input_mean is None else input_mean
        self._input_std = _CIFAR_STD if input_std is None else input_std
        assert self._input_mean.shape == (3, 1, 1)
        assert self._input_std.shape == (3, 1, 1)
        self._dataset = CIFAR100Dataset(
            cifar100_data_dir,
            split="train",
            output_size=input_size,
            normalize=True,
            input_mean=self._input_mean,
            input_std=self._input_std,
        )
        self._indices = np.arange(len(self._dataset))
        np.random.shuffle(self._indices)
        if max_examples is not None and max_examples < len(self._dataset):
            self._indices = self._indices[:max_examples]

        self._current_index = 0

    def __len__(self):
        return self._indices.size

    def get_next(self):
        if self._current_index >= len(self):
            return None

        image, _ = self._dataset[self._indices[self._current_index]]
        # Move image to NCHW, following ONNX tensor convention.
        image = np.expand_dims(image, axis=0)

        result = {self._input_value_name: image}
        self._current_index += 1
        return result

    def rewind(self):
        self._current_index = 0
