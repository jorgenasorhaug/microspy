"""Convenience functions for creating HyperSpy signals to use as 
navigators with :meth: '~hyperspy.signals.Signal2D.plot'.
"""
from hyperspy.signals import Signal2D
import numpy as np
from skimage.exposure import rescale_intensity
from matplotlib.colors import ListedColormap

DEFAULT_COLORS = (
    'red',
    'blue',
    'yellow',
    'magenta',
    'green',
    'indigo',
    'darkorange',
    'cyan',
    'pink',
    'yellowgreen',
)

def get_rgb_navigator(image: np.ndarray, 
                      dtype: str | np.dtype | type = 'uint16') -> Signal2D:
    """Create an RGB navigator signal which is suitable to pass to
    :meth:`~hyperspy._signals.signal2d.Signal2D.plot` as the
    ``navigator`` parameter.

    Parameters
    ----------
    image
        RGB color image of shape ``(n rows, n columns, 3)``.
    dtype
        Which data type to cast the signal data to, either ``"uint16"``
        (default) or ``"uint8"``. Must be a valid :class:`numpy.dtype`
        identifier.

    Returns
    -------
    s
        Signal with an (n columns, n rows) signal shape and no
        navigation shape, of data type either ``rgb8`` or ``rgb16``.
    """
    dtype = np.dtype(dtype)
    image_rescaled = rescale_intensity(image, out_range = dtype.type).astype(dtype)
    s = Signal2D(image_rescaled)
    s = s.transpose(signal_axes = 1)
    s.change_dtype({"uint8" : "rgb8",
                    "uint16" : "rgb16"}[dtype.name])
    return s

def get_discrete_colour_map(colours : list | None,
                            bg_colour = 'w') -> ListedColormap:
    """Create a listed colourmap. The colours will be tiled 500 times.
    """
    from copy import deepcopy

    if colours == None: colors = DEFAULT_COLORS
    else: colors = deepcopy(colours)

    colors = np.array(colors)

    colors = colors[~(colors == bg_colour)]

    colors = np.tile(colors, 500)

    colors = np.insert(colors, 0, bg_colour)
    
    return ListedColormap(colors)