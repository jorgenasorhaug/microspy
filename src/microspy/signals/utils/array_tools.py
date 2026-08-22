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

import numpy as np
from microspy.misc import exceptions

allowed_hor_dirs = ("l2r", "r2l")
allowed_vert_dirs = ("t2b", "b2t")

def depad_arrays(
    arr : np.ndarray,
    **kwargs
) -> list[np.ndarray, ...]:
    """Depad 3 dimensional arrays and return a list of depadded arrays.

    Parameters
    ----------
    arr
        Array to depad.
    kwargs
        keyword argument passed on to :func:'depad_array'.

    Returns
    -------
    depadded
        List of depadded arrays.
    """

    ndim = np.ndim(arr)
    
    if ndim < 2 or ndim > 3:
        raise AttributeError(f"Array of dimension {ndim} "
                "is not supported.")
        
    depadded = []

    for a in arr:
        depadded.append(
            depad_array(a, **kwargs)
        )

    return depadded   

def depad_array(
    arr : np.ndarray,
    **kwargs
) -> np.ndarray:
    """Depad 2-dimensional array.

    Parameters
    ----------
    arr
        Array to depad.
    kwargs
        empty edges properties; The function looks for 'empty_value' keyword
        to pass on to :func:'_crop_away_empty_edges'. 

    Returns
    -------
    depadded
        Depadded array.
    """
    ndim = np.ndim(arr)
    
    if ndim != 2:
        raise AttributeError(f"Array of dimension {ndim} "
                             "is not supported.")
    
    empty_val = kwargs.get("empty_value")
    if empty_val is None:
        empty_val = 0.0
    
    return _crop_away_empty_edges(arr = arr, empty_value = empty_val)

def _crop_away_empty_edges(
    arr : np.ndarray,
    empty_value : int | float = 0.0
) -> np.ndarray:
    """Crop away empty edges from an array.
    
    Parameters
    ----------
    arr
        Array to remove empty edges from.
    empty_value 
        Value considered empty and will be removed from the returned array.
    
    Returns
    -------
    arr
        Array with empty edges removed    
    """
    # Identify the not-padded area
    y0, x0 = np.where(arr > empty_value)
    
    if len(y0) > 0: 
        y00,y01 = y0.min(), y0.max()+1
        arr = arr[y00:y01,:]
        
    if len(x0) > 0: 
        x00,x01 = x0.min(), x0.max()+1
        arr = arr[:,x00:x01]
    
    return arr

def _3Darray_2_4Darray(
    arr : np.ndarray, 
    to_shape : list | tuple, 
    flip_axis : int | tuple | list | None = (1,) 
) -> np.ndarray:
    """Reshape a stack of arrays (3 dimensions) into a 4-dimensional grid.

    Parameters
    ---------
    arr
        Stack of images.
    to_shape
        Navigation shape.
    flip_axis
        Flip array axis/axes.

        Example usage:
        If the images are to be read from right to left,
        flip_axis = 1 else 0

        if flip_axis is a list, flipping will be done 
        chronologically.

    Returns
    -------
    grid
        Reshaped array into 4D grid

    Examples
    --------
    >>> num = 6
    >>> arr = np.arange(num*16).reshape(num, 4,4)
    >>> _3Darray_2_4Darray(arr, to_shape = (2,3)).shape
    (2,3,4,4)
    """
    
    if len(to_shape) != 2:

        raise AttributeError("Argument 'to_shape' cannot have "
                            "more than 2 navigation axes.")

    N, H, W = arr.shape

    assert N == np.prod(to_shape)

    grid = np.reshape(arr, to_shape + (H, W))

    if flip_axis is not None: 

        if isinstance(flip_axis, int | np.integer):
    
            flip_axes = (flip_axis,)

        else: flip_axes = flip_axis

        for flip in flip_axes:
            
            if flip not in (0,1):
                
                if flip is None:
                    _exception = f"Flip argument {flip} is not supported. "
                    _exception += "Returning unflipped array."
                    
                    exceptions.formatted_warning(
                        message = _exception
                    )
                    
                    return grid
    
                else:
                    
                    raise AttributeError("Axis flipping along axis "
                                     f"{flip} is not supported.")
            
            grid = np.flip(grid, flip)

    return grid


def _4Darray_2_3Darray(
    arr : np.ndarray, 
    flip_axis : int | tuple | list | None = (1,)
) -> np.ndarray:
    """Reshape a stack of images from a 4-dimensional to a 3D grid.

    Parameters
    ---------
    arr
        Stack of images.
    to_shape
        Navigation shape.
    flip_axis
        Axis to flip.

    Returns
    -------
    degrid
        Reshaped array into 3D grid.
    """
    #print("4->3: ", flip_axis)
    Y,X,H,W = arr.shape
    new_shape = (Y * X,) + (H, W)
    degrid = arr

    if flip_axis is not None:

        if isinstance(flip_axis, int | np.integer):
            flip_axis = (flip_axis,)

        for flip in flip_axis:
            if flip not in (0,1):
                if flip is None:
                    _exception = f"Flip argument {flip} is not supported. "
                    _exception += "Returning unflipped array."
                    
                    exceptions.formatted_warning(
                        message = _exception
                    )
                    
                    return degrid
    
                else:
                    raise AttributeError("Axis flipping along axis "
                                     f"{flip} is not supported.")

            degrid = np.flip(degrid, flip)

    return np.reshape(degrid, new_shape)

