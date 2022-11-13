from typing import Optional, Union

import numpy as np
from torch import Tensor

U = Union[Tensor, np.ndarray]


def to_uint8(image: U, min: Optional[Union[float, int]] = 0.0, max: Optional[Union[float, int]] = 1.0):
    """Converts an image to uint8. Will also scale values to range [0, 255].

    Args:
        image (U): Image to convert.
        min (Union[float, int], optional): Minimum value of range for input image. Defaults to 0.0.
        max (Union[float, int], optional): Maximum value of range for input image. Defaults to 1.0.
    """
    image = (image - min) / (max - min) * 255
    image = int(image)

    return image
