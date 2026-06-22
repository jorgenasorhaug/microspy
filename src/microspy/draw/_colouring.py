#
# Copyright 2026 the microspy developer(s)
#
# This file is part of microspy.
#
# microspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# microspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with microspy. If not, see <http://www.gnu.org/licenses/>.
#
# This code is inspired by exspy : ~exspy._misc.elements.py

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

def generate_unique_rgb_colors(
    n : int
) -> list:
    """
    Generate n unique RGB colors.
    
    Parameters
    ----------
    n 
        Number of unique colors to generate
        
    Returns
    -------
    list of tuples 
        Each tuple is (R, G, B)
    """
    if n > 256**3:
        raise ValueError("Cannot generate more than 16,777,216 unique RGB colors.")
    
    colors : list = []
    
    while len(colors) < n:
        colors.append(np.random.rand(3))
    
    return list(colors)






#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%% DELETE THESE: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

def get_discrete_colour_map(
    colours : list | None,
    n : int,
    bg_colour : str = (1.,1.,1.),
    **kwargs
) -> ListedColormap:
    """Create a custom colourmap. 
    """
    from copy import deepcopy

    bkgr_colour = kwargs.get("background_colour")
    if bkgr_colour is None:
        bkgr_colour = bg_colour

    if colours == None: colors = DEFAULT_COLORS
    else: colors = deepcopy(colours)
    colors = np.array(colors)
    colors = colors[~(colors == bkgr_colour)]
    colors = np.insert(arr=colors, obj=0, values=bkgr_colour)
    return ListedColormap(colors)