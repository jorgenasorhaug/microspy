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

from microspy.misc import exceptions

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

def _dict2list(
    dictionary : dict,
    key_words : list = ["label",
                        "top_left",
                        "bottom_right",
                        "shape"]
) -> list:
    """Get dictionary values by argument keywords. Helping function to 
    :func:'_listOfDicts2listOfLists'.

    Parameters
    ----------
    dictionary
        Dictionary with values to get
    key_words
        List of keywords to get the values from the argument
        dictionary

    Returns
    -------
    array
        array of dictionary values according to the order of the 
        keywords in key_words
    """

    #Check if the symmetric difference is empty:
    diff = set(key_words) ^ set(dictionary.keys())
                                
    if not diff:

        array = []

        for key in key_words:

            val = dictionary.get(key)

            if isinstance(val, int | np.integer):
                array.append(val)
            elif isinstance(val, list | tuple):
                array.extend(val)
            else:
                raise TypeError(f"dictionary value type ({type(val)}) "
                                "is not supported.")
        return array
    else:
        raise ValueError("The dictionary keywords and argument key_words "
                         f"are symmetrically different by {diff}")

def _listOfDicts2listOfLists(
    list_of_dicts : list(dict),
    key_words : list = ["label",
                        "top_left",
                        "bottom_right",
                        "shape"]
) -> list[list, ...]:
    """Return a list of lists with values from the dictionaries in a list
    in the same order as the keywords in key_words.
    
    A helping function to :func:'_get_crop_rectangles'.

    Parameters
    ----------
    list_of_dicts
        List of dictionaries with expected keys as in key_words
    key_words
        keywords to get values from the dictionaries in list_of_dicts

    Returns
    -------
    results
        list of lists containing the values in the order defined by key_words
    """
    
    results = []
    
    for d in list_of_dicts:
        results.append(
            _dict2list(d, key_words)
        )
        
    return results

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
    key_words : list = ["label",
                        "top_left",
                        "bottom_right",
                        "shape"],
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
        List of lists with values

        For example:
        key_words = ["label",
                     "top_left",
                     "bottom_right",
                     "shape"]
                     
         provides the following:
     
        results[0] : image label
        results[1], results[2] : child image upper left coord. 
        results[3], results[4] : child image bottom right coord.
        results[5], results[6] : "shape" (child image shape)
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
    
    return _listOfDicts2listOfLists(results, key_words)

def _create_label_map(
    image_shape : np.ndarray, 
    crop_rectangles : list(list),
    key_words : list = ["label",
                        "top_left",
                        "bottom_right",
                        "shape"],
    background_label : int = -1
) -> np.ndarray:
    """
    Creates a label map of the cropped images. Overlapping label areas 
    are resolved using center-of-mass distance.

    Parameters
    ----------
    original_image
        Parent image
    cropped_images
        List of child images (cropped from parent)
    key_words 
        List of key_words defining the order of the
    background_label 
        Label of the background
        
    Returns
    -------
    label_map
        Map of labels
    """
    from scipy.ndimage import center_of_mass

    # Set the value indices
    kw_idxs = []
    counter = 0
    for kw in key_words:
        if kw == "label": 
            kw_idxs.append(counter)
            counter += 1
        else: 
            kw_idxs.append([counter, counter+1])
            counter += 2
            
    H, W = image_shape
    label_map = np.full(
        shape = (H, W),
        fill_value = background_label,
        dtype=np.int16
    )

    # A master mask to restrict the second iteration 
    # underneath
    master_mask = np.zeros_like(
        label_map, 
        dtype=bool
    )

    centers = {} # CoM
    masks = {} 

    # Create rectangular masks and centers
    for rect in crop_rectangles:

        label = rect[kw_idxs[key_words.index("label")]]
        
        # Top-left coords.
        y0 = rect[kw_idxs[key_words.index("top_left")][0]]
        x0 = rect[kw_idxs[key_words.index("top_left")][1]]

        # Bottom-right coords.
        y1 = rect[kw_idxs[key_words.index("bottom_right")][0]]
        x1 = rect[kw_idxs[key_words.index("bottom_right")][1]]

        mask = np.zeros((H, W), dtype=bool)
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
                int(np.argmin(distances))
                ]

    return label_map

def _map_particle_regions(
    childArray : np.ndarray,
    parentArray : np.ndarray,
    parentOrder : np.ndarray | None = None,
    background_label : int = -1,
    nested_progressbar : bool = True
):
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
    label_maps = np.full_like(
        a = parentArray, 
        fill_value = background_label,
        dtype = np.int16
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