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
from tqdm import tqdm_notebook
from scipy.ndimage import center_of_mass

from .array_tools import depad_arrays
from microspy.misc import exceptions

def _find_crop_rectangles(
    original_image : np.ndarray,
    cropped_images : list | np.ndarray,
    labels : list | tuple | None = None,
    progressbar : bool = True
) -> np.ndarray:
    """
    Finds bounding rectangles where cropped images originate from.

    Parameters
    ----------
    original_image
        Parent image
    cropped_images
        List of child images (cropped from parent)
    labels
        List of child image labels. If None, the first label
        is set equal to 1.
    progressbar
        Whether to display progressbar
        
    Returns
    -------
    results
        List of dictionaries with keys:
            "label" (image label)
            "top_left" (child image upper left coord.) 
            "bottom_right" (child image bottom right coord.)
            "shape" (child image shape)
    """
    from skimage.feature import match_template
    
    results = []

    if labels is None: 
        labels = np.arange(1, 1 + len(cropped_images))

    elif len(labels) != len(cropped_images):
        raise ValueError(f"The length of cropped images ({len(cropped_images)}) "
                         "is different from the length of child image labels "
                        f"({len(labels)}).")
    
    for idx, crop in tqdm_notebook(
        zip(labels, cropped_images),
        total = len(labels),
        disable = not progressbar
    ): 

        result = match_template(original_image, crop)
        
        ij = np.unravel_index(np.argmax(result), result.shape)
        y, x = ij
        h, w = crop.shape

        results.append({
            "label": np.int32(idx),
            "top_left": (np.uint32(y), np.uint32(x)),
            "bottom_right": (np.uint32(y + h), np.uint32(x + w)),
            "shape": (np.uint32(h), np.uint32(w)),
        })

    return results
    
def _get_crop_rectangles(
    original_image : np.ndarray,
    cropped_images : list(np.ndarray),
    use_labels : list | tuple | None = None,
    progressbar : bool = True
) -> list:
    """Identify bounding rectangles where cropped images originate from. 

    Parameters
    ----------
    original_image
        Parent image
    cropped_images
        List of child images (cropped from parent). Their positions are 
        unknown.
    key_words 
        List of key_words defining the order of the use_labels.
        Specify the labels to use.
    progressbar
        Whether to display progressbar

    Returns
    -------
    results
        list of dictionaries with keys representing the individual labels'
        properties like label, image positions and shape. 
    """

    if use_labels is not None:
        if len(use_labels) != len(cropped_images):
            raise ValueError(f"The number of labels {len(use_labels)}"
                            "is not matching the number of cropped "
                            f"images ({len(cropped_images)}).")
    
    results = _find_crop_rectangles(
        original_image, 
        cropped_images,
        labels = use_labels,
        progressbar = progressbar
    )
    
    return results

def _create_label_map_com(
    label_image_shape : tuple,
    crop_rectangles : list[list],
    background_label : int = -1
) -> np.ndarray:
    """Creates a label map of the cropped images. Overlapping label areas 
    are resolved using center-of-mass distance. 
    
    Not meant to be used directly.
    
    See :func:'_create_label_map'.
    
    Parameters
    ----------
    label_image_shape
        Shape of the label image
    crop_rectangles
        List of lists containing information about the cropped images,
        s.a. label, top_left and bottom right position, and shape.
    key_words 
        List of key_words defining the order of the crop_rectangles 
        information.
    background_label 
        Label of the background
    
    Returns
    -------
    label_map
        dictionary with keyword argument representing the labels,
        and the corresponding values representing the single label 
        map.
    """
    
    # Label map where overlapping regions are adjusted
    label_map = np.full(
        shape = label_image_shape,
        fill_value = background_label,
        dtype=np.int16
    )
    
    # Covers/keep track of all the label regions (boolean).
    master_mask = np.zeros_like(
        label_map, 
        dtype=bool
    )
    
    centers = {} # CoM
    masks = {} 

    # Create rectangular masks and save centers
    for rect in crop_rectangles:

        label = rect.get("label")

        y0, x0 = rect.get("top_left")
        y1, x1 = rect.get("bottom_right")
        
        mask = np.zeros(label_image_shape, dtype=bool)
        mask[y0:y1, x0:x1] = True

        # Key: label ID
        masks[label] = mask
        master_mask[mask] = True

        # Key: label
        centers[label] = center_of_mass(mask)

    # Identified regions and assign labels
    iters = np.where(master_mask)
    
    for y,x in zip(iters[0], iters[1]):
        
        # Identify candidates (single/overlaps)
        candidates = [
            label for label, mask in masks.items()
            if mask[y, x]
        ]
        
        # Set label
        if len(candidates) == 1:
            label_map[y, x] = candidates[0]
        else:
            # Resolve overlap by nearest center
            distances = [
                np.hypot(y - centers[l][0], x - centers[l][1])
                for l in candidates
            ]
            
            label_map[y, x] = candidates[
                int(
                    np.argmin(distances)
                )
            ]
                
    return label_map
    
def _create_label_map_by_minimum_area(
    label_image_shape : tuple,
    crop_rectangles : list[list],
    background_label : int = -1
) -> np.ndarray:
    """Creates a label map of the cropped images. Overlapping label areas 
    are resolved using center-of-mass distance. 
    
    Not meant to be used directly.
    
    See :func:'_create_label_map'.
    
    Parameters
    ----------
    label_image_shape
        Shape of the label image
    crop_rectangles
        List of lists containing information about the cropped images,
        s.a. label, top_left and bottom right position, and shape.
    key_words 
        List of key_words defining the order of the crop_rectangles 
        information.
    background_label 
        Label of the background
    
    Returns
    -------
    label_map
        dictionary with keyword argument representing the labels,
        and the corresponding values representing the single label 
        map.
    """
    
    # Label map where overlapping regions are adjusted
    label_map = np.full(
        shape = label_image_shape,
        fill_value = background_label,
        dtype=np.int16
    )
    
    # Covers/keep track of all the label regions (boolean).
    master_mask = np.zeros_like(
        label_map, 
        dtype=bool
    )
    
    centers = {} # CoM
    masks = {} 

    # Create rectangular masks and save centers
    for rect in crop_rectangles:

        label = rect.get("label")

        y0, x0 = rect.get("top_left")
        y1, x1 = rect.get("bottom_right")
        
        mask = np.zeros(label_image_shape, dtype=bool)
        mask[y0:y1, x0:x1] = True

        # Key: label ID
        masks[label] = mask
        master_mask[mask] = True

        # Key: label
        centers[label] = center_of_mass(mask)

    # Identified regions and assign labels
    iters = np.where(master_mask)
    
    for y,x in zip(iters[0], iters[1]):
        
        # Identify candidates (single/overlaps)
        candidates = [
            label for label, mask in masks.items()
            if mask[y, x]
        ]
        
        # Favour the label region with the smallest area:
        area = [
            np.sum(masks[l]) for l in candidates
        ]
        
        # Set label regions
        label_map[y, x] = candidates[
            int(
                np.argmin(area)
            )
        ]
        
    return label_map
    
def _create_label_map(
    image_shape : np.ndarray, 
    crop_rectangles : list[list],
    background_label : int = -1
) -> np.ndarray:
    """
    Creates a label map of the cropped images. 

    Parameters
    ----------
    original_image
        Parent image
    crop_rectangles
        List of lists containing information about the cropped images,
        s.a. label, top_left and bottom right position, and shape.
    key_words 
        List of key_words defining the order of the crop_rectangles order.
    background_label 
        Label of the background
        
    Returns
    -------
    label_map
        Map of labels
    """
    
    # Create label map
    label_map = _create_label_map_com(
        label_image_shape = image_shape,
        crop_rectangles = crop_rectangles,
        background_label = background_label
    )
    
    # --- Check if labels are missing due to CoM approach ---
    current_labels = np.unique(label_map)
    current_labels = np.delete(
        arr = current_labels, 
        obj = np.where(current_labels == background_label)
    )
    expected_labels = np.asarray(
        [x["label"] for x in crop_rectangles]
    )

    # ----- Check if labels have been removed: -------------
    missing_labels = np.setdiff1d(
        ar1 = expected_labels, 
        ar2 = current_labels
    )
    
    if len(missing_labels) != 0:
        # ------------ Recover missing labels --------------
        label_map = _create_label_map_by_minimum_area(
            label_image_shape = image_shape,
            crop_rectangles = crop_rectangles,
            background_label = background_label
        )

    return label_map        

def _map_particle_regions(
    childArray : np.ndarray,
    parentArray : np.ndarray,
    parentOrder : np.ndarray | None = None,
    background_label : int = -1,
    nested_progressbar : bool = True
) -> np.ndarray:
    """Map the child arrays onto the parent arrays.

    Parameters
    ----------
    childArray
        Array of child arrays
    parentArray
        Array of parent arrays
    parentOrder
        Order of parentArray, i.e. multiple child arrays may be found in 
        the same parent array, so instead of checking the entire parent 
        array, only individual parent arrays will be investigated.
    background_label 
        Label of the background. Default: -1.
    nested_progressbar
        Whether to use a nested progress bar or not. Default: True

    Returns
    -------
    label_maps
        Array of mapped and labelled child arrays.
    """
    
    cndim = np.ndim(childArray)
    pndim = np.ndim(parentArray)
    shape = parentArray.shape[-2:]

    if not 1 < cndim <= 3:
        raise ValueError(f"Child array dimension {cndim} is "
                        "not supported.")
    if not 1 < cndim <= 3:
        raise ValueError(f"Parent array dimension {pndim} is "
                        "not supported.")

    # Future: determine dtype by childArray length
    if np.iinfo(np.int16).min > len(childArray) > np.iinfo(np.int16).max:
        exceptions.formatted_warning(
            "Future fix: determine dtype from chilArray length."
        )
        dtype = np.int64
    else:
        dtype = np.int16
        
    label_maps = np.full_like(
        a = parentArray, 
        fill_value = background_label,
        dtype = dtype
    )
    
    unique_parentOrder = np.unique(
        parentOrder
    )
    
    # Previous parentOrder index:
    prev_index = unique_parentOrder[0]
    incrementor = - background_label

    # Iterate through the (unique) parent array:
    for index in tqdm_notebook(
        unique_parentOrder,
        desc = "Progress"
    ):
 
        # Update parent array:
        parr = parentArray[index]

        # Depad child images corr. to the parent image:
        depaddedChildArr = depad_arrays(
            childArray[np.where(parentOrder == index)]
        )

        # Labels for Child arrays
        child_labels = np.arange(
            start = background_label + 1,
            stop = len(depaddedChildArr) + background_label + 1,
            step = 1
        )

        # Get child array labels and locations:
        rectangles = _get_crop_rectangles(
            original_image = parr,
            cropped_images = depaddedChildArr,
            use_labels = child_labels,
            progressbar = nested_progressbar
        )
        
        # Label child array locations in the 'label_maps'
        label_maps[index] = _create_label_map(
            image_shape = shape,
            crop_rectangles = rectangles,
            background_label = background_label
        )
            
        # increment labels wrt. existing labels:
        if index != prev_index:
            label_maps[index][
                label_maps[index] > background_label
                ] += (incrementor + np.max(label_maps[prev_index]))
            prev_index = index    
        
    return label_maps