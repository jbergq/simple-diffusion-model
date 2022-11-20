from typing import Optional, Tuple, Union

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
        array = array.clone().type(torch.uint8)
    elif isinstance(array, np.ndarray):
        array = array.astype(np.uint8)

    return array


def to_float32(array: U, value_range: Optional[Tuple] = (0.0, 1.0)):
    """Converts uint8 array to float32 while also rescaling array to specified range. Assumes input array has values in range [0, 255].

    Args:
        array (Union[Tensor, np.ndarray]): Input array of type Tensor or np.ndarray.
        value_range (Optional[Tuple], optional): Range to scale array to. Defaults to (0.0, 1.0).

    Returns:
        Union[Tensor, np.ndarray]: Converted and rescaled float32 array.
    """
    if isinstance(array, Tensor):
        array = array.type(torch.float32)
    elif isinstance(array, np.ndarray):
        array = array.astype(np.float32)

    # Scale to new range.
    min_r, max_r = value_range
    array = array / 255 * (max_r - min_r) + min_r

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
        array = array.clone().repeat_interleave(repeats, axis)
    elif isinstance(array, np.ndarray):
        array = array.repeat(repeats, axis)

    return array


def to_image(array: U, value_range: Optional[Tuple] = (0.0, 1.0)):
    """Converts array to uint8 image with values in range [0, 255].

    Args:
        array (Union[Tensor, np.ndarray]): Array to convert.
        value_range (Optional[Tuple], optional): Range of input array. Defaults to (0.0, 1.0).

    Returns:
        Union[Tensor, np.ndarray]: Converted array.
    """

    n, c, h, w = array.shape

    assert isinstance(array, (Tensor, np.ndarray)), "`array` must be either torch Tensor or numpy ndarray."
    assert c == 1 or c == 3, "Argument `array` must have 1 or 3 channels."
    assert (
        len(value_range) == 2 and value_range[0] < value_range[1]
    ), "`value_range` must be tuple of length 2, with `value_range[0] < value_range[1]`."

    # Scale to [0, 255].
    min_r, max_r = value_range
    array = (array - min_r) / (max_r - min_r) * 255

    # Convert to uint8.
    array = to_uint8(array)

    # Ensure returned image has 3 channels by repeating single channel.
    if c == 1:
        array = repeat_channels(array, 3, 1)

    return array
