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
from copy import deepcopy

def get_stitched_Parent_signal(
    array : np.ndarray,
    grid_shape : tuple | list = None,
    horisontal_direction : str | None = None,
    vertical_direction : str | None = None,
):
    """Stitch a 4D grid array into an image

    Parameters
    ----------
    array
        array grid
    nav_shape
        2D grid shape. If None, the grid is stitched as is.
    horisontal/vertical_direction
        String stating which direction to stitch the array if 
        nav_shape is not None.

        Allowed arguments:
        "l2r" (left to right) and "r2l" (right to left)
        "t2b" (top to bottom) and "b2t" (bottom to top)

    Returns
    -------
    stitch
        stitched image
    """

    hd = ["l2r", "r2l"]
    vd = ["t2b", "b2t"]

    shape = np.shape(array)
    ndim = np.ndim(array)

    if horisontal_direction is not None:
        if horisontal_direction not in hd:
            raise ValueError(f"Argument {horisontal_direction} is not "
                             f"recognised. Allowed arguments are {hd}")
    else:
        if ndim == 3:
            print("Note! Image acquisition order is assumed to be. "
                  "left -> right. See function"
                  "particle_analysis.misc.vendorImageAquisitionOrder())")
            
    if vertical_direction is not None:
        if vertical_direction not in vd:
            raise ValueError(f"Argument {vertical_direction} is not "
                             f"recognised. Allowed arguments are {vd}")
    else:
        if ndim == 3:
            print("Note! Image acquisition order is assumed to be. "
                  "top -> bottom. See function"
                  "particle_analysis.misc.vendorImageAquisitionOrder())")
    
    if grid_shape is not None:

        # Mismatching dimensions:
        if len(grid_shape) == ndim == 2:

            raise ValueError("Unable to stitch 2D images")

    else:

        if ndim == 3:

            raise ValueError("Unable to stitch 2D images without "
                             "a fourth dimension. For N rows, set "
                             "nav_shape = (N,1). For N columns, set "
                             "nav_shape = (1,N).")

    # transpose the array | fix array order unless it's 4D
    if ndim < 4:
        rows, cols = grid_shape
        N = array.shape[0]
    
        if N != rows * cols:
            raise ValueError("grid_shape does not match number of images")
    
        # Reshape into grid
        grid = array.reshape(rows, cols, *array.shape[1:])
    
        # Step 2: flip horizontally if needed
        if horisontal_direction == 'r2l':

            grid = grid[:, ::-1]
    
        # Flip vertically if needed
        if vertical_direction == 'b2t':
            
            grid = grid[::-1, :]

        # Transpose back:
        grid = grid.reshape(N, *array.shape[1:])

    elif ndim == 4:

        N = np.prod(array.shape[:2])
        grid = array.reshape(N, *array.shape[2:])
        grid_shape = shape[:2]

    else:

        raise ValueError(f"Cannot stitch array of dimension {ndim}.")

    return _stitch_nd_to_2d(
        arr = grid,
        grid_shape = grid_shape
        )


def _stitch_nd_to_2d(
    arr : np.ndarray, 
    grid_shape : tuple | list | None = None, 
):
    """Stitch an ND array (>=3D) to a 2D image.

    Parameters
    ----------
    arr 
        Input array of shape (n_slices, h, w, ...) or (n_slices, h, w)
    grid_shape 
        Layout of tiles. If None, a near-square grid is used.

    Returns
    -------
    stitched : np.ndarray
        2D (or 3D if channels exist) stitched image
    """

    if arr.ndim < 3:
        
        raise ValueError("Input array must have at least 3 dimensions")

    n_slices = arr.shape[0]
    tile_shape = arr.shape[1:]

    # Determine grid
    if grid_shape is None:
        cols = int(np.ceil(np.sqrt(n_slices)))
        rows = int(np.ceil(n_slices / cols))
    else:
        rows, cols = grid_shape
    
    # Output shape
    h, w = tile_shape[:2]
    
    out_shape = (
        rows * h,
        cols * w,
    ) 

    # Stitched image
    stitched = np.zeros(out_shape, dtype=arr.dtype)

    # Place tiles
    for idx in range(n_slices):
        
        r = idx // cols # row
        c = idx % cols # col

        # intervals
        y = r * h 
        x = c * w

        stitched[y:y+h, x:x+w, ...] = arr[idx]

    return stitched
