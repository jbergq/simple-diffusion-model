from typing import Optional, Union

import torch
import numpy as np
from torch import Tensor

U = Union[Tensor, np.ndarray]


def to_uint8(array: U):
    """Converts array to uint8.

    Args:
        array (Union[Tensor, np.ndarray]): Input array of type Tensor or np.ndarray.

    Returns:
        Union[Tensor, np.ndarray]: Converted uint8 array.
    """
    if isinstance(array, Tensor):
        array = array.type(torch.uint8)
    elif isinstance(array, np.ndarray):
        array = array.astype(np.uint8)

    return array


def repeat_channels(array: U, repeats: int, axis: int):
    """Repeats array along axis.

    Args:
        array (Union[Tensor, np.ndarray]): Input array of type Tensor or np.ndarray.
        repeats (int): The number of repetitions along axis.
        axis (int): The axis along which to repeat values.

    Returns:
        Union[Tensor, np.ndarray]: Repeated array which has the same shape as input, except along the given axis.
    """
    if isinstance(array, Tensor):
        array = array.repeat_interleave(repeats, axis)
    elif isinstance(array, np.ndarray):
        array = array.repeat(repeats, axis)

    return array


def to_image(array: U, min: Optional[Union[float, int]] = 0.0, max: Optional[Union[float, int]] = 1.0):
    """Converts array to uint8 image with values in range [0, 255].

    Args:
        array (Union[Tensor, np.ndarray]): Array to convert.
        min (Union[float, int], optional): Minimum value of range for input image. Defaults to 0.0.
        max (Union[float, int], optional): Maximum value of range for input image. Defaults to 1.0.

    Returns:
        Union[Tensor, np.ndarray]: Converted array.
    """

    n, c, h, w = array.shape

    assert isinstance(array, (Tensor, np.ndarray)), "`array` must be either torch Tensor or numpy ndarray"
    assert c == 1 or c == 3, "Argument `array` must have 1 or 3 channels."

    # Scale to [0, 255].
    array = (array - min) / (max - min) * 255

    # Convert to uint8.
    array = to_uint8(array)

    # Ensure returned image has 3 channels by repeating single channel.
    if c == 1:
        array = repeat_channels(array, 3, 1)

    return array
