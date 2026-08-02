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
from microspy.misc import exceptions
import warnings
from tqdm import tqdm_notebook

def stitch_grid_signal(
    array : np.ndarray,
    grid_shape : tuple | list = None,
    horisontal_direction : str | None = None,
    vertical_direction : str | None = None,
) -> np.ndarray:
    """Stitch an array into a 2-dimensional image.

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
        stitched image from grid signal
        
    Example
    -------
    >>> import microspy as ms
    >>> s = ms.load("results.csv")
    >>> s
    <Particle analysis, title:, dimensions: (15)>
    
    >>> s.load_images()
    >>> s.Images.ParentSig
    <MicroSpySignal2D_Parent, title: Acquisition, dimensions: (280|1024, 768)>
    >>> s.Images.ParentSig.data.shape
    (280, 768, 1024)
    
    >>> stitched_image = ms.draw.stitch_grid_signal(
           array = s.Images.ParentSig.data,
           grid_shape = (70,4),
           horisontal_direction = "r2l" # Jeol's acquisition order
       )
    >>> stitched_image.shape
    (3072, 71680)
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
            print("Note! Image acquisition order is assumed to be "
                  "left -> right.")
            
    if vertical_direction is not None:
        if vertical_direction not in vd:
            raise ValueError(f"Argument {vertical_direction} is not "
                             f"recognised. Allowed arguments are {vd}")
    else:
        if ndim == 3:
            print("Note! Image acquisition order is assumed to be "
                  "top -> bottom.")
    
    if grid_shape is not None:
        # Mismatching dimensions:
        if len(grid_shape) == ndim == 2:
            raise ValueError("Unable to stitch 2D images")

    else:
        if ndim == 3:
            raise ValueError(
                    "Unable to stitch 2D images without a fourth dimension. "
                    "For N rows, set 'grid_shape = (N,1)'. For N columns, "
                    "set 'grid_shape = (1,N)'."
                )

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

def label2rgb(
    label_map : np.ndarray,
    colours : list | tuple | dict | None = None,
    underlay_image : np.ndarray | None = None,
    bkgr_label : int | None = None,
    bkgr_colour : str | tuple = "whitesmoke",
    **kwargs
) -> tuple(list(tuple, ...), np.ndarray):
    """Create a coloured label map using :func:'skimage.color.label2rgb'.
    
    Parameters
    ----------
    label_map
        Integer array of labels (with the same shape as overlay).
    colours 
        List (or dictionary) of colours. 
        If it's a dictionary, the key should represent the labels.
        The values are then colours (str or rgb's)
    underlay_image
        Image to overlay be overlayed by the coloured label map.
    bkgr_label
        Background label.
    bkgr_colour
        Colour of the background.
    kwargs
        keywords passed on to :func:'skimage.color.label2rgb'.
    
    Returns
    -------
    coloured_map
        Coloured label map according to colours.
    colours_rgb
        rgb colours used for colouring.
        
    Examples
    --------
    >>> import microspy as ms
    >>> s = ms.load("filename.hdf5")
    
    >>> s.update_phase_maps()
    >>> classes, pm = s.Images.get_phase_map()
    >>> pm.shape
    (4, 70, 768, 1024)
    
    >>> coloured_phase_map = ms.draw.label2rgb(
            label_map = pm, 
            colours = ["red", "blue", "green"]
        )
    >>> coloured_phase_map.shape
    (4, 70, 768, 1024, 3) # rgb values
    """
    
    from copy import deepcopy
    from matplotlib.colors import to_rgb
    from skimage.color import label2rgb
    from tabulate import tabulate
    from ._colouring import Closest_colorname
    
    unique_labels = np.unique(label_map)
    
    if bkgr_label is None:
        bkgr_label = np.min(unique_labels)
        exceptions.formatted_warning(
           "No background label is provided. The minimum label value in the "
          f"label map ('{bkgr_label}') will be interpreted as the background "
          "label."
        )
        
    # Ignore the background label
    unique_labels = np.delete(
        arr = unique_labels,
        obj = np.where(unique_labels == bkgr_label)
    )
    num_labels = len(unique_labels)
    
    if isinstance(colours, list | tuple):
        num_colours = len(colours)
        if num_colours < num_labels:
            raise exceptions.InputError(
                    f"The number of provided colours ('{num_colours}')do not match "
                    f"the number of unique labels ('{num_labels}')."
                )
        _colours = deepcopy(colours)
        
    elif isinstance(colours, dict):
        # Check if all keys are integers:
        allKeys_int = True if any(
            x.is_integer() for x in colours.keys()
        ) else False
        
        if not allKeys_int:
            raise exceptions.InputError(
                    "All colour keys must be integers representing the labels."
                )
        
        num_colours = len(colours)
        if num_colours < num_labels:
            raise exceptions.InputError(
                    f"The number of provided colours ('{num_colours}') do not "
                    f"match the number of unique labels ('{num_labels}')."
                )
        
        values = np.asarray(list(colours.keys())).astype(int)
        missing_labels = set(unique_labels) - set(values)
        if len(missing_labels) > 0:
            raise exceptions.InputError(
                    f"'{missing_labels}' is/are missing from the unique "
                    f"labels ('{unique_labels}')"
                )
        
        missing_values = set(values) - set(unique_labels)
        if len(missing_values) > 0:
            raise exceptions.InputError(
                    f"{missing_values} is/are missing from the colours "
                    "dictionary."
                )
                
        colours = dict(
            sorted(
                colours.items(), key=lambda x: x[0]
            )
        )
        _colours = list(colours.values())
    
    elif colours is None:
        from ._colouring import DEFAULT_COLOURS
        colours = DEFAULT_COLOURS
        
        if len(colours) < num_labels:
            exceptions.InputError(
                "The colours must be set to return a coloured label map."
            )
            
        print("Setting matplotlib's default colours.")
    else:
        raise exceptions.InputError(
                f"Colour argument of type {type(colours)} is not supported."
            )
    if underlay_image is not None:
        if underlay_image.shape != label_map.shape:
            raise exceptions.InputError(
                    "overlay image and label map have different shapes "
                   f"({underlay_image.shape}\u2260{label_map.shape})."
                )
    
    col_rgb = [to_rgb(col) for col in _colours]
    
    coloured_map = label2rgb(
        label = label_map, 
        image = underlay_image, 
        colors = col_rgb, 
        bg_label = bkgr_label,
        bg_color = bkgr_colour,
        **kwargs
    )
    
    result = (col_rgb, coloured_map)
    
    # Print label-colour overview
    tabular_data = list(
        zip(
            unique_labels, 
            [Closest_colorname(
                rgb_values = 255*np.asarray(rgb)
            ).closest_color_name()[0] for rgb in col_rgb]
        )
    )
    
    bkgr_col_name = Closest_colorname(
        rgb_values = 255*np.asarray(to_rgb(bkgr_colour))
    ).closest_color_name()[0]
    
    tabular_data.insert(0, (bkgr_label, bkgr_col_name))
    
    print(
        tabulate(
            tabular_data = tabular_data,
            headers = ["Label", "Colour"]
        )
    )
    
    return result
    
def get_grid_mask(
    grid_shape : tuple,
    edge_width : int = 1,
) -> np.ndarray:
    """Create a "grid mask" representing for a visual representation of the 
    image grid.
    
    Parameters
    ----------
    grid_shape
        Shape of image grid.
    edge_width
        Width of the edge per image.
    
    Returns
    -------
    grid_mask
        ndarray representing a grid mask.
    """
    
    if len(grid_shape) != 4:
        raise exceptions.InputError(
                f"The grid shape must be 4D, and not {len(grid_shape)}."
            )
    
    grid_mask = np.zeros(
        shape = grid_shape,
        dtype = bool
    )
    
    grid_mask[:,:,edge_width:-edge_width, edge_width:-edge_width] = True
    
    return stitch_grid_signal(array = grid_mask)
    
def _stitch_nd_to_2d(
    arr : np.ndarray, 
    grid_shape : tuple | list | None = None, 
):
    """Stitch an ND array (>=3D) to a 2D image.
    
    Not meant to be used directly.

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

def _get_phase_maps_from_label_map(
    classes : list | tuple,
    label_map : np.ndarray,
    background_label : int | None = -1,
    **kwargs
) -> np.ndarray:
    """Create phase maps per class. The function is used whenever the phase
    maps are updated.
    
    Not meant to be used directly.

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
    Length of classes must be identical to the number of unique labels.
    
    kwargs takes image_indices as argument.
        list, tuple or ndarray of where the labels are expected to be found.
        
    Example
    -------
    >> import microspy as ms
    >> from microspy.draw import _get_phase_maps_from_label_map
    >> s = ms.load("filename.csv")
    >> s.load_images()
    >> s.map_particles()
    ...
    >> _get_phase_maps_from_label_map(
           classes = s.particle_classes,
           label_map = s.Images.ParticleMap.data,
           # **{"acquisition_order" : s.metadata.Sample.acquisition_order}
       )
    {'Type B': array([[[[False, False, False, ..., False, False, False],
          [False, False, False, ..., False, False, False],
          [False, False, False, ..., False, False, False],
          ...,
          [False, False, False, ..., False, False, False],
          [False, False, False, ..., False, False, False],
          [False, False, False, ..., False, False, False]],
 
         [[False, False, False, ..., False, False, False],
          [False, False, False, ..., False, False, False],
          [False, False, False, ..., False, False, False],
          ...,
          [False, False, False, ..., False, False, False],
          [False, False, False, ..., False, False, False],
          [False, False, False, ..., False, False, False]],
          ...
          [False, False, False, ..., False, False, False]]]],
       shape=(4, 70, 768, 1024))}
    """
    
    num_classes = len(classes)
    unique_labels = np.unique(label_map)

    if background_label is None:
        exceptions.formatted_warning(
            message = "Background label set as -1."
        )
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
            f"Total number of classes ({num_classes}) does "
            "not match the number of unique labels "
            f"({len(unique_labels)})"
        )

    else:
        
        # Create phase maps
        phase_maps = _map_classes_labels(
            particle_classes = classes,
            label_map = label_map,
            background_label = background_label,
            **kwargs
        )
    return phase_maps
    
def _map_classes_labels(
    particle_classes : np.ndarray,
    label_map : np.ndarray,
    background_label : int = -1,
    **kwargs
) -> np.ndarray:
    """Creates a label map per class. Helping function to 
    :func:'_get_phase_maps_from_label_map'.
    
    Parameters
    ----------
    particle_classes
        Class names 
    label_map
        Label map where each class instance exist
    kwargs
        keyword argument. See :func: _get_phase_maps_from_label_map
    """
    
    phase_maps = dict()
    
    # Get Child-Parent acquisition order (if any).
    acq_order = kwargs.get("acquisition_order")
    
    unique_classes = np.unique(particle_classes)
    unique_labels = np.unique(label_map)
    
    if background_label not in unique_labels:
        warnings.warn(f"Background label {background_label} not "
                     "found in label_map.")
    else:
        unique_labels = np.delete(
            arr = unique_labels,
            obj = unique_labels == background_label
        )
   
    # Unique classes:
    for cl in tqdm_notebook(
            unique_classes,
            total = len(unique_classes),
            desc = "Mapping class positions:",
            position = 0
        ):
            
            phase_maps[cl] = np.zeros_like(
                    label_map,
                    dtype = bool
                )
            
            # Check if acquisition order is accessible for faster mapping
            if isinstance(acq_order, list | np.ndarray):
                
                ndim = np.ndim(label_map)
                
                if ndim > 3:
                    raise exceptions.ShapeError(
                        "Cannot map particles from un-degridified "
                        "array.")
                        
                # Labels of current interest:
                class_indices = np.where(particle_classes == cl)[0]
                
                # Iterate through each class and image instance
                for p_idx, img_idx in tqdm_notebook(
                    zip(
                        class_indices, 
                        acq_order[class_indices]
                    ), 
                    desc=cl, 
                    total = len(class_indices)
                ): 
                    
                    phase_maps[cl][img_idx][
                        np.where(
                            label_map[img_idx] == unique_labels[p_idx]
                            )
                        ] = True
                        
            else:
                
                # Check the entire ndarray
                labels = unique_labels[particle_classes == cl]
                
                # Iterate labels of current interest:
                for label in tqdm_notebook(
                    labels,
                    total = len(labels),
                    desc = cl,
                    position = 0
                ):

                    phase_maps[cl][label_map==label] = True
                
    return phase_maps