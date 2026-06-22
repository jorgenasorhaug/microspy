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
from src.microspy._misc import exceptions
import warnings

def get_stitched_grid_signal(
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

def get_phase_maps_from_label_map(
    classes : list | tuple,
    label_map : np.ndarray,
    background_label : int | None = -1,
    **kwargs
) -> np.ndarray:
    """Creates a map of phase labels according to unique classes

    Parameters
    ----------
    classes
        list of classes
    label_map
        Map of labels with as many unique labels as classes
    background_label
        Index of background label in label_map

    Returns
    -------
    phase_map
        Map of class labels

    Note
    ----
    kwargs takes image_indices as argument
        list, tuple or ndarray of where the labels are
    """
   
    from tqdm import tqdm_notebook
    print("Not (yet) supporting where indices are expected to "
          "be found in array of images.")
    
    num_classes = len(classes)
    
    unique_classes = np.unique(classes)
    unique_labels = np.unique(label_map)

    if background_label is None:
        background_label = -1
    
    if background_label not in unique_labels:
        warnings.warn(f"Background label {background_label} not "
                     "found in label_map.")
    else:
        unique_labels = np.delete(
            arr = unique_labels,
            obj = unique_labels == background_label
        )
    
    if num_classes != len(unique_labels):
        
        raise exceptions.ShapeError(
            f"Number of classes ({num_classes}) does "
            "not match the number of unique labels "
            f"({len(unique_labels)})"
        )

    else:
        
        phase_maps = dict()
        
        for cl in tqdm_notebook(
            unique_classes,
            total = len(unique_classes),
            desc = "Mapping class positions",
            position = 0
        ):
            
            labels = unique_labels[classes == cl]
            phase_maps[cl] = np.zeros_like(
                label_map,
                dtype = bool
            )
            
            for label in tqdm_notebook(
                labels,
                total = len(labels),
                desc = cl,
                position = 0
            ):

                phase_maps[cl][label_map==label] = True

    return phase_maps