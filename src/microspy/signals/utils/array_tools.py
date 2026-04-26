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
from src.microspy._misc import exceptions

allowed_hor_dirs = ("l2r", "r2l")
allowed_vert_dirs = ("t2b", "b2t")

def stitch_array(
    image_array : np.ndarray,
    navigation_shape : tuple | list | None = None, 
    horisontal_order_direction : str = "r2l", 
    vertical_order_direction : str = "t2b", 
    scale : int | float | None | np.integer | np.float = None
) -> np.ndarray:
    """Stitch the individual images according to argument
    directions.

    Note!
    The horisontal_orde_direction has not yet been set.

    Parameters
    ----------
    navigation_shape 
        Shape in which the images will be stitched into. 
        If set to None, the images will be stitched as
        they are.
        
        Note! 
        The convention is identical to matplotlib's row/
        column convention. Example: 4 images in horisontal 
        direction and 3 images in vertical direciton will 
        be correctly stitched by using navigation_shape = 
        (3,4).

        Note! 
        If a row/column of images is to be stitched, define 
        navigation_shape as (ROWS,1) / (1,COLS)
    horisontal_order_direction, vertical_order_direction
        The images' horisontal/vertical order in which to be stitched.
        
        "r2l"/"l2r" : right ->/<- left.
        "t2b"/"b2t" : top ->/<- bottom.
        
    scale
        rescale the stitched image's navigation shape.
        
    Returns
    -------
    stitched_array
        Stitched images
    """
    
    if horisontal_order_direction not in allowed_hor_dirs:

        raise AttributeError(f"{horisontal_order_direction} is not"
                            "a recognised image order direction.\n"
                            f"Allowed ones are: {allowed_hor_dirs}.")
    
    if vertical_order_direction not in allowed_vert_dirs:

        raise AttributeError(f"{vertical_order_direction} is not"
                            "a recognised image order direction.\n"
                            f"Allowed ones are: {allowed_vert_dirs}.")

    
    from copy import deepcopy
    
    stitched = deepcopy(image_array)

    # 3D or 4D array
    arr_shape = np.shape(stitched)
    nav_shape = arr_shape[:-2]
    H, W = arr_shape[-2:]

    # Array shape is 3D else 4D
    if len(nav_shape) == 1:

        if navigation_shape is None:
    
            raise exceptions.ShapeError("A navigation shape must "
                                       "be provided to stitch the "
                                       "the array correctly.")

        rows, cols = navigation_shape

    else: rows, cols = nav_shape
    
    # Identify axes to flip
    flip_axis = None
    
    if horisontal_order_direction == "r2l":
        
        flip_axis = 1

    if vertical_order_direction == "b2t":

        if flip_axis is not None:

            flip_axis = (flip_axis, 0)

        else: flip_axis = 0
    
    stitched = _3Darray_2_4Darray(
        arr = stitched,
        to_shape = (rows, cols), 
        flip_axis = flip_axis 
    )
    
    if scale is not None:

        from skimage.transform import rescale
        
        stitched = rescale(
            image = stitched, 
            scale = scale, 
            anti_aliasing = False
        )
    
    # Rearrange and merge axes
    stitched = stitched.transpose(0, 2, 1, 3)
    stitched = stitched.reshape(rows * H, cols * W)
    return stitched

def _3Darray_2_4Darray(
    arr : np.ndarray, 
    to_shape : list | tuple, 
    flip_axis : int | None | list = 1 
) -> np.ndarray:
    """Reshape a stack of images into a 4D grid.

    Note! 
    The argument arr is changed, and not copied.

    Parameters
    ---------
    arr
        Stack of images
    to_shape
        Navigation shape
    flip_axis
        Flip array axis

        Example usage:
        If the images are to be read from right to left,
        flip_axis = 1 else 0

        if flip_axis is a list, flipping will be done 
        chronologically.

    Returns
    -------
    grid
        Reshaped array into 4D grid

    Example
    -------
    
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
    
                raise AttributeError("Axis flipping along axis "
                                     f"{flip} is not supported.")
            
            grid = np.flip(grid, flip)

    return grid


def _4Darray_2_3Darray(
    arr : np.ndarray, 
    flip_axis : int | None | list = 1
) -> np.ndarray:
    """Reshape a stack of images into a 3D grid.

    Note! 
    The argument arr is changed, and not copied.

    Parameters
    ---------
    arr
        Stack of images
    to_shape
        Navigation shape
    flip_axis
        Axis to flip

    Returns
    -------
    degrid
        Reshaped array into 4D grid
    """

    Y, X, H, W = arr.shape
        
    new_shape = (Y*X,) + (H,W)

    degrid = arr

    if flip_axis is not None:

        if isinstance(flip_axis, int | np.integer):
    
            flip_axes = (flip_axis,)

        for flip in flip_axes:
            
            if flip not in (0,1):
                
                raise AttributeError("Axis flipping along axis "
                                     f"{flip} is not supported.")

            degrid = np.flip(degrid, flip)

    return np.reshape(degrid, new_shape)

