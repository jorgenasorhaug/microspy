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

def depad_arrays(
    arr : np.ndarray
) -> list:
    """Depad arrays and return a list of depadded arrays

    Parameters
    ----------
    arr
        Array of images to depad

    Returns
    -------
    depadded
        Array of depadded images
    """

    ndim = np.ndim(arr)
    
    if ndim < 2 or ndim > 3:
        raise AttributeError(f"Array of dimension {ndim} "
                "is not supported.")
        
    depadded = []

    for a in arr:
        depadded.append(
            depad_array(a)
        )

    return depadded   

def depad_array(
    arr : np.ndarray
) -> np.ndarray:
    """Depad array

    Parameters
    ----------
    arr
        Array to depad

    Returns
    -------
    depadded
        Depadded array
    """
    ndim = np.ndim(arr)
    
    if ndim != 2:
        raise AttributeError(f"Array of dimension {ndim} "
                             "is not supported.")

    return _crop_away_empty_edges(arr)

def _crop_away_empty_edges(
    arr : np.ndarray
) -> np.ndarray:
    """Crop away empty edges from array.
    
    Parameters
    ----------
    arr
        Array to remove empty edges from
    
    Returns
    -------
    arr
        Array with empty edges removed    
    """
    # Identify the not-padded area
    y0, x0 = np.where(arr > 0)
    
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
    """Get dictionary values by argument keywords

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
) -> list:
    """Return a list of lists with values from the dictionaries in a list
    in the same order as the keywords in key_words.

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
    Finds bounding rectangles where cropped images originate 
    from.

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
        disable = ~progressbar
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
    """Identify bounding rectangles where cropped images 
    originate from.

    Parameters
    ----------
    original_image
        Parent image
    cropped_images
        List of child images (cropped from parent)
    key_words 
        List of key_words defining the order of the 
    use_labels
        Specify the labels to use
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
    parentOrder : np.ndarray,
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
        Order of parentArray, i.e. multiple child arrays
        may be found in the same parent array, so instead
        of checking the entire parent array, only individual 
        parent arrays will be investigated.
    background_label 
        Label of the background. Default: -1
    nested_progressbar
        Whether to use a nested progress bar or not.
        Default: True

    Returns
    -------
    label_maps
        Array of mapped (labelled) child arrays
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

        # Depad child images 
        depaddedChildArr = depad_arrays(
            childArray[np.where(parentOrder == index)]
        )

        # Label for Child arrays
        child_labels = np.arange(
            start = background_label + 1,
            stop = len(depaddedChildArr) + background_label + 1,
            step = 1
        )

        # Child array labels and locations:
        rectangles = _get_crop_rectangles(
            original_image = parr,
            cropped_images = depaddedChildArr,
            use_labels = child_labels,
            progressbar = nested_progressbar
        )

        # Label child array locations
        label_maps[index] = _create_label_map(
            image_shape = shape,
            crop_rectangles = rectangles,
            background_label = background_label
        )
        
        #print("initial:", np.unique(label_maps[index]))
        #print("Previous:", np.unique(label_maps[prev_index]))
        # increment labels
        if index != prev_index:
            label_maps[index][
                label_maps[index] > background_label
                ] += (incrementor + np.max(label_maps[prev_index]))
            prev_index = index    
        #print("Final:", np.unique(label_maps[index]))
    return label_maps


    









#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%% Image class utility functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
from src.microspy.signals import _images_DELETEME
from src.microspy._misc import exceptions as _errors

def _gridify_4D_array_to_3D(arr, flip_axis = 1):
    """Make a 3D ndarray from a 4D ndarray 
    See _gridify_3D_array_to_4D
    """

    if np.shape(arr) == (3,): raise ValueError("Input array is already a 3D array")
        
    new_shape = (np.prod(arr.shape[:2]),) + np.shape(arr)[2:]
    
    return np.reshape(np.flip(arr, flip_axis), new_shape)

def _gridify_ND_array_to_nD(arr, flip_axis = 1):
    """Make a 3D ndarray from a 4D ndarray 
    See _gridify_3D_array_to_4D
    """
        
    new_shape = (np.prod(arr.shape[:2]),) + np.shape(arr)[2:]
    
    return np.reshape(np.flip(arr, flip_axis), new_shape)

"""def _colour_region_in_image(im_to_colour, location, rgb_vals):
    Colour an image according to array location with rgb values

    Example
    ------
    >>> im_to_colour[location] = rgb_vals
    >>> im_to_colour
    np.array([ ...
               ...
               ])
    
    im_to_colour[location] = rgb_vals
    
    return im_to_colour"""

#def _crop_away_empty_edges(arr):
#    """Crop away empty edges"""
#    # Identify the not-padded area
#    y0,x0 = np.where(arr > 0)
#    
#    y00,y01 = y0.min(), y0.max()+1
#    x00,x01 = x0.min(), x0.max()+1
#    
#    return arr[y00:y01, x00:x01]

def _put_array_content_into_larger_array(arr, new_shape):
    """Copy (and return) the content from arr and paste it into the new array with shape defined by
    new_shape
    """
    shape = arr.shape
    to_arr = np.zeros(new_shape, dtype = arr.dtype)
    
    arr_shape_diff = np.array(new_shape) - shape

    if arr_shape_diff[0] != 0:
        if arr_shape_diff[1] != 0:
            to_arr[arr_shape_diff[0]//2:-arr_shape_diff[0]//2, arr_shape_diff[1]//2:-arr_shape_diff[1]//2] = arr
        else: 
            to_arr[arr_shape_diff[0]//2:-arr_shape_diff[0]//2, :] = arr
    else:
        if arr_shape_diff[1] != 0:
            to_arr[:, arr_shape_diff[1]//2:-arr_shape_diff[1]//2] = arr
        else: to_arr = arr
    return to_arr

def _put_array_content_into_smaller_array(arr, new_shape):
    """Copy (and return) the content from arr and paste it into the new array with shape defined by
    new_shape
    """
    shape = arr.shape
    to_arr = np.zeros(new_shape, dtype = arr.dtype)
    
    arr_shape_diff = shape - np.array(new_shape)

    if arr_shape_diff[0] != 0:
        if arr_shape_diff[1] != 0:
            to_arr = arr[arr_shape_diff[0]//2:-arr_shape_diff[0]//2, arr_shape_diff[1]//2:-arr_shape_diff[1]//2]
        else: 
            to_arr = arr[arr_shape_diff[0]//2:-arr_shape_diff[0]//2, :]
    else:
        if arr_shape_diff[1] != 0:
            to_arr = arr[:, arr_shape_diff[1]//2:-arr_shape_diff[1]//2]
        else: to_arr = arr
    return to_arr

def _copy_images_into_common_array_shape(arr1 : np.ndarray, 
                                         arr2 : np.ndarray):
    """The idea with the function is to create a new arra with dimensions (np.max([arr1.shape[0], arr2.shape[0]]), 
    np.max([arr1.shape[1], arr2.shape[1]])) and copy the data in arr1 and arr2 into this new array shape.

    If axis is provided, the new shape will only be changed along this axis. The other axis is set equal to the first 
    array.

    """
    if arr1.dtype == arr2.dtype:
        
        shape1, shape2 = arr1.shape, arr2.shape

        new_shape = (np.max([shape1[0], shape2[0]]), np.max([shape1[1], shape2[1]]))

        arr1_new = _put_array_content_into_larger_array(arr1, new_shape)
        arr2_new = _put_array_content_into_larger_array(arr2, new_shape)

        return arr1_new, arr2_new
        
    else: raise ValueError(f"Array data types are not compatible ({arr1.dtype} vs {arr2.dtype})")



"""def _make_large_particle_images_shape_compatible(arr1, arr2):
    Look for zero-valued areas in both images and crop these away. arr1 and arr2 will get updated.
    
    
    arr1_new = _crop_away_empty_edges(arr1)
    arr2_new = _crop_away_empty_edges(arr2)
        
    arr1_shape = arr1_new.shape
    arr2_shape = arr2_new.shape

    new_shape = (np.min([arr1_shape[0], arr2_shape[0]]), np.min([arr1_shape[1], arr1_shape[1]]))
    
    if new_shape != arr1_shape or new_shape != arr2_shape:
        arr1_new = _put_array_content_into_smaller_array(arr1_new, new_shape)
        arr2_new = _put_array_content_into_smaller_array(arr2_new, new_shape)
    
    return arr1_new, arr2_new
"""

def _get_rectangle_mask_after_image_shift(image, threshold = 1.0, top_hat_filter = False):
    """Assuming a padded image, create a rectangular mask for the signal"""
    from skimage.morphology import white_tophat
    if top_hat_filter: mask = (image > threshold) * ~white_tophat(image > threshold)
    else: mask = image > threshold
    return mask

def _assess_successful_stitching(shift, shift_threshold = 1.0):
    """Returning True if the shift distance (Eucledian) is a success, i.e. the shift distance is 
    greater than the set threshold. (1.0 by default).
    """
    if np.sqrt(np.sum(np.square(shift))) > shift_threshold: return True
    else: return False

def _remove_empty_edges_from_image(image):
    """Look for empty edges in an image, and return a new image with these removed
    """
    args = np.where(image)
    sp_y0, sp_y1, sp_x0, sp_x1 = args[0].min(), args[0].max()+1, args[1].min(), args[1].max()+1
    return image[sp_y0:sp_y1, sp_x0:sp_x1]

def _rectangulerize_mask(mask_image):
    """Create a rectangular mask from argument where min and max define the boundaries"""
    y,x = np.where(mask_image)
    rect_mask = np.zeros_like(mask_image, bool)
    if len(y) > 0 and len(x) > 0: rect_mask[y.min():y.max()+1, x.min():x.max()+1] = True
    return rect_mask

def _remove_padding_along_axis(arr, axis):
    """Remove padding along a specified axis"""
    if axis not in [0,1]: raise ValueError(f"axis {axis} is not supported yet")

    else:

        y,x = np.where(arr > 0)

        if axis == 0: return arr[y.min():y.max(),:]

        else: return arr[:, x.min():x.max()]

def _get_particle_mask(pimage : np.ndarray,
                       conditions : dict,
                       #remove_edge_artefacts : bool = False,
                       background_label : int = 0,
                       #current_mask : None | np.ndarray = None
                      ):
    """Create a particle image mask based on the conditions provided in the conditions dictionary. 
    The watershed image segmentation is used to remove artefacts from edges, which are typical from image stitching.

    Obs! Note that the particle image will first be threhsolded, then segmentet using watershed segmentation to remove edge artefacts, before the remaining conditions will be applied.

    Parameters
    ---------
    pimage
        image of a particle
    conditions
        Dictionary of conditions describing intensity thresholds and minimum and maximum particles sizes.

    Example
    -------
    pimage_mask = _get_particle_mask(
        pimage,
        conditions = {'>' : 80, # Particle should have intensity > 80
                      '<=' : 160, # article should have intensity <160 
                      'min_size' : 20, # Minimum particle size: 20, i.e. particles with 20 or less pixels will get removed.
                      'binary_erosion' : 2 # The mask will be eroded, iterating twice
                      'fill_holes' : None | 0 | np.inf | 'test', # The content is arbitrary and doesn't do anything.
                      }
    )                
    """
    
    first_set_of_conditions = ['<','<=','>','>=']
    allowed_condition_arguments = ['<','<=','>','>=',
                                   'min_size','max_size', 'min_diameter', 'max_diameter',
                                   'binary_erosion','binary_dilation', 'fill_holes']

    if len(pimage.shape) > 2:
        raise _errors.ShapeError(f"The provided particle image shape ({pimage.shape}) is not supported.")
        
    if type(conditions) != dict:
        raise TypeError(f"The provided conditions dictionary is not a dictionary.")
    
    cond_keys = list(conditions.keys())

    if not all(key in allowed_condition_arguments for key in cond_keys):
        not_allowed = []
        for k in cond_keys:
            if k not in allowed_condition_arguments:
                not_allowed.append(k)
        raise KeyError(f"The provided conditions contains key arguments that are not supported. ({k})")

    first_condition_round = dict()
    second_condition_round = dict()
    
    for cond in conditions.keys():
        if cond in first_set_of_conditions:
            first_condition_round[cond] = conditions[cond]
        else:
            second_condition_round[cond] = conditions[cond]

    #print(first_condition_round, second_condition_round)

    # Define a thresholded image(s)
    if len(first_condition_round) == 0:
        if type(current_mask) == type(None):
            raise TypeError(f"To create a mask, a threshold must first be set.")
        elif np.sum(current_mask) == 0:
            raise TypeError(f"To create a mask, a threshold must first be set.")
    #    else:
    #        current_mask = _create_mask_from_conditions(
    #            image = pimage.copy(),
    #            current_mask = current_mask,
    #            conditions = first_condition_round
    #        )
    #else:
    #if type(current_mask) == type(None):
    #    current_mask = 
    #else:
    #    if np.sum(current_mask) == 0:
    #        warnings.warn('The provided mask is empty.')
    
    mask = _create_mask_from_conditions(
        image = pimage.copy(),
        current_mask = np.zeros_like(pimage),
        conditions = first_condition_round
    )
    
    # Remove obvious edge artefacts like rectangles
    #tmp = _identify_and_remove_rectangles_from_mask(current_mask, background_label = background_label).astype(int)

    """if remove_edge_artefacts:

        from skimage import segmentation, filters, morphology
        from skimage.feature import peak_local_max
        from scipy import ndimage as ndi
    
        # Perform watershed segmentation
        distance = ndi.distance_transform_edt(tmp)
        coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=tmp)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = segmentation.watershed(-distance, markers, mask=tmp)
    
        # Define an edge mask to remove edge artefacts
        #edge_mask = np.ones_like(labels, bool)
        #edge_mask[1:-1,1:-1] = False
    
        # Remove edge artefacts:
        labels = _identify_and_remove_rectangles_from_mask(
            labels,
            background_label = background_label,
            condition_scale = 0.95)
            
        # Relabel:
        labels = morphology.label(labels > 0)

        # Define the particle centre and identify the correct particle label
        centre = (tmp.shape[0]//2, tmp.shape[1]//2)
    
        particle_label = labels[centre[0], centre[1]]
        for l in np.unique(labels):
            if l != particle_label:
                labels[labels == l] = 0

        # Check if the particle is still intact. If not, reset and don't remove edge artefacts.
        if len(np.unique(labels)) == 1:
            labels = tmp

    else: labels = tmp """
        
    if len(second_condition_round) > 0:
        
        # Perform the remaining conditions
        return _create_mask_from_conditions(
            image = pimage.copy(),
            current_mask = mask > 0,
            conditions = second_condition_round
        )  
        
        # Remove obvious artefacts one last time:
        #return _identify_and_remove_rectangles_from_mask(
        #    labels > 0,
        #    background_label = background_label,
        #    condition_scale = 1.0) 
        
                              
    
    else: return mask > 0

def _identify_and_remove_rectangles_from_mask(mask,
                                              background_label : int = 0,
                                              condition_scale : float = 1.0):
    """As the name suggests... The function returns a updated mask with 
    all obvious rectangles removed"""
    from skimage.measure import label, regionprops_table
    
    tmp = label(mask > 0, connectivity = 1)
    
    for l in np.unique(tmp):
        if l != background_label:
            y, x = np.where(tmp == l)
            Y = y.max() + 1 - y.min()
            X = x.max() + 1 - x.min()
            area = X*Y
            props = regionprops_table(tmp, 
                                      properties = ('label', 'area'))
            if float(props['area'][np.where(props['label'] == l)]) == condition_scale * float(area):
                tmp[tmp == l] = 0

    return tmp > 0
        
def _create_mask_from_conditions(image,
                                 current_mask,
                                 conditions):
    """A support function for _get_particle_mask().
    Provide an empty ndarray as current_mask. The threshold conditions will then be summed
    into this mask.
    """
    mask = current_mask.copy() 
    
    for cond in conditions:
        if cond == '<': mask += image < conditions[cond]
        if cond == '<=': mask += image <= conditions[cond]
        if cond == '>': mask += image > conditions[cond]
        if cond == '>=': mask += image >= conditions[cond]
        if cond == 'min_size':
            from skimage.morphology import remove_small_objects
            mask = remove_small_objects(
                mask, min_size=conditions[cond])
        if cond == 'max_size':
            from skimage.morphology import remove_small_objects
            mask_ = remove_small_objects(
                mask, min_size=conditions[cond])
            mask = bool(mask.astype(int) - mask_.astype(int))
        if cond == 'binary_erosion':
            from scipy.ndimage import binary_erosion
            mask = binary_erosion(mask, iterations = conditions[cond])
        if cond == 'binary_dilation':
            from scipy.ndimage import binary_dilation
            mask = binary_dilation(mask, iterations = conditions[cond])
        if cond == 'fill_holes':
            from scipy.ndimage import binary_fill_holes
            mask = binary_fill_holes(mask)
        if 'diameter' in cond:
            from skimage.morphology import label 
            from skimage.measure import regionprops_table
            mask = label(mask)
            if cond == 'min_diameter': 
                prop = regionprops_table(mask, 
                                         properties = ('label', 'axis_minor_length'))
                remove = prop['axis_minor_length'] < conditions[cond]
                for l in prop['label'][remove]:
                    mask[mask == l] = 0
                mask = mask > 0
            elif cond == 'max_diameter':
                prop = regionprops_table(mask, 
                                         properties = ('label', 'axis_major_length'))
                remove = prop['axis_major_length'] > conditions[cond]
                for l in prop['label'][remove]:
                    mask[mask == l] = 0
                mask = mask > 0
    return mask

def _segment_clustered_particles(particle_images : list,
                                 particle_image_masks : list, 
                                stitched_image : np.ndarray,
                                segmented_stitched_image : np.ndarray,
                                conditions : dict):
    """By setting a set of pre-defined conditions as done in the particle analysis software,
    search for single particle images that overlap with regions in the stitched image, and 
    segment these regions out to measure new geometric properties. 

    Parameters
    ----------
    particle images
        List of single particle images
    particle_image_masks
        list of particle image masks where particle region of interest
        is located.
    stitched_image
        Stitched image containing all the particle images
    conditions
        Conditions to segment the particle regions of interest

    Returns
    -------
    clustered_particles
        List of clustered particle labels
    ...
    """
    from skimage.morphology import label

    # Relabel 
    masked_stitch = label(segmented_stitched_image)

    image_labels = np.zeros((len(particle_images),), int)

    images = np.zeros((len(particle_images),) + stitched_image.shape)

    for pim_id in range(len(image_labels)):

        # Identify the particle region within the stitched image
        p_region = _get_single_particle_map(
            SEM_image = stitched_image,
            particle_image = particle_images[pim_id]
        )

        # Map the masked particle onto the stitched region
        p_thr = particle_image_masks[pim_id].copy()
        """p_thr = s.Images._create_particle_mask(
            particle_images[pim_id],
            conditions = conditions, 
            return_single_particle_only=False
        )"""

        # Paste the particle mask onto the mapped region
        tmp = p_region.copy()
        p_region[p_region] = p_thr.reshape(np.prod(p_thr.shape))

        images[pim_id] = p_region.copy()
        
        # Identify the unique labels from the labelled masked stitched 
        # image where the particle image is overlapping
        unique_labels = np.unique(p_region * masked_stitch)
        unique_labels = np.delete(unique_labels, np.where(unique_labels == 0))
        num_pixels = np.zeros_like(unique_labels)
        for i, l in zip(range(len(num_pixels)), unique_labels):
            num_pixels[i] = np.sum((p_region * masked_stitch) == l) 

        # Identify the segmented particle with the most pixels within this
        # region
        image_labels[pim_id] = unique_labels[np.argmax(num_pixels)]
        #unique_labels = np.unique(image_labels)

    return image_labels, images
        
def _concatenate_two_edge_images(img0 : np.ndarray, 
                                 img1 : np.ndarray, 
                                 loc0 : tuple | list, 
                                 loc1 : tuple | list):
    """Concatenate particle images img0 and img1 based on their edge positions in an overview image.
    Example: If img0 is at an image top and img1 is at an image bottom, img1 will be concatenated 
    above img0.

    Parameters
    ----------
    img0, img1
        numpy.ndarray of similar shapes
    loc0, loc1
        2D tuple describing the images' positions wrt. the overview SEM image.
        These values can be obtained from the function "_identify_a_particles_position_wrt_SEM_image_edges()"
        See also the function _translate_location_position()

    Returns
    -------
    img0 + img1 
        concatenated along the axis described in loc0 and loc1.
    """
    
    s0, s1 = img0.shape, img1.shape

    if len(s0) > 2 or len(s1) > 2: 
        raise _errors.ShapeError(f"Array 1 of shape {s0} can not be concatenated with array 2 of shape {s1}")
    
    if s0[0] != s1[0] and s0[1] != s1[1]: 
        raise _errors.ShapeError(f"Array 1 of shape {s0} can not be concatenated with array 2 of shape {s1}")
    
    # Concatenate the equally sized images if they are not at an image edge
    if loc0[0] == 0: # Bottom
        if loc1[0] == 1: # Bottom-Top
            if s0[1] != s1[1]: raise _errors.ShapeError(f"Array 1 of shape {s0} can not be concatenated with array 2 of shape {s1}")
            return np.concatenate([img1, img0], axis = 0)
        elif loc1[0] == 0: # Bottom-Bottom
            if loc0[1] == 0 and loc1[1] == 1: # Left-Right
                if s0[0] != s1[0]: raise _errors.ShapeError(f"Array 1 of shape {s0} can not be concatenated with array 2 of shape {s1}")
                return np.concatenate([img1, img0], axis = 1)
            elif loc0[1] == 1 and loc1[1] == 0: # Right-left
                if s0[0] != s1[0]: raise _errors.ShapeError(f"Array 1 of shape {s0} can not be concatenated with array 2 of shape {s1}")
                return np.concatenate([img0, img1], axis = 1)
            else: return np.array([])
        else: return np.array([])
    elif loc0[0] == 1: # Top
        if loc1[0] == 0: # Top-bottom
            if s0[1] != s1[1]: raise _errors.ShapeError(f"Array 1 of shape {s0} can not be concatenated with array 2 of shape {s1}")
            return np.concatenate([img0, img1], axis = 0)
        elif loc1[0] == 1: # Top-top
            if loc0[1] == 0 and loc1[1] == 1: # Left-Right
                if s0[0] != s1[0]: raise _errors.ShapeError(f"Array 1 of shape {s0} can not be concatenated with array 2 of shape {s1}")
                return np.concatenate([img1, img0], axis = 1)
            elif loc0[1] == 1 and loc1[1] == 0: # Right-left
                if s0[0] != s1[0]: raise _errors.ShapeError(f"Array 1 of shape {s0} can not be concatenated with array 2 of shape {s1}")
                return np.concatenate([img0, img1], axis = 1)
            else: return np.array([])
        else: return np.array([])
    elif loc0[1] == 0: #Left side
        if loc1[1] == 1: # right side
            if s0[0] != s1[0]: raise _errors.ShapeError(f"Array 1 of shape {s0} can not be concatenated with array 2 of shape {s1}")
            return np.concatenate([img1, img0], axis = 1)
        else: return np.arary([])
    elif loc0[1] == 1: # Right side
        if loc1[1] == 0: # Right-left side
            if s0[0] != s1[0]: raise _errors.ShapeError(f"Array 1 of shape {s0} can not be concatenated with array 2 of shape {s1}")
            return np.concatenate([img0, img1], axis = 1)
        else: return np.array([])
    else: return np.array([])


def _get_single_particle_map(SEM_image, particle_image):
    """Identify a particle' probable upper left corner position in an image (SEM_image) by running template matching.
    (See skimage.feature's function match_template)

    Parameters
    ----------
    SEM_image
        SEM image array to look for the particle. 
    particle_image
        Particle image array. 
    
    Returns
    -------
    particle_map
        Map with the identified particle.
    """
    from skimage.feature import match_template
    
    particle = np.zeros_like(SEM_image, dtype = bool)

    if SEM_image.shape == particle_image.shape:

        return np.ones_like(particle_image)

    tm = match_template(SEM_image, particle_image)

    x, y = np.unravel_index(np.argmax(tm), tm.shape)[::-1] # upper left corner

    start = (x, y) 
    
    extent = (x + particle_image.shape[1], y + particle_image.shape[0])
    
    particle[start[1]:extent[1], start[0]:extent[0]] = True

    return particle

def _get_edge_particles_gridMap_from_labelled_particles_gridMap(particles_region_map, edge_width = 1):
    """The function looks for separated/cut particles at image edges that might be the same as they are cut in half by
    the image acquisition. 

    Parameters
    ----------
    particles_region_map
        3D or 4D grid with SEM images
    edge_width
        Width of edge to look for potential split particles, i.e. particles within the edge width
        are considered as potential cut particles.

    Returns
    -------
    same_particles_map
        array containing particles that have been cut during particle analysis.
    """
    from tqdm import tqdm

    ndim = len(particles_region_map.shape)

    shape = particles_region_map.shape
    
    if ndim == 3: 
        
        img_shape = particles_region_map[0].shape

        nav_shape = particles_region_map.shape[0]
    
    elif ndim == 4: 
        
        img_shape = particles_region_map[0,0].shape

        nav_shape = particles_region_map.shape[:2]
    
    else: 
        
        print(f"Argument shape {particles_region_map.shape} is invalid.")
    
        return particles_region_map

    edges = np.ones(img_shape, dtype = bool)

    # Setting centre region to zero
    edges[edge_width:-edge_width, edge_width:-edge_width] = False
    
    SEM_images_grid = particles_region_map[:,:] * edges
    
    # Map of the cropped particles
    sum_im = np.zeros_like(SEM_images_grid, dtype = SEM_images_grid.dtype)
    
    # Unique labels in overview image: 
    uniques = np.unique(SEM_images_grid)
    
    uniques = np.delete(uniques, uniques == 0)

    empty_image = np.zeros(img_shape, bool)

    # Reshape arrays to speed up things
    print('FUTURE FIX: TO SPEED THINGS UP A BIT, INSPECT ONLY THE IMAGE EDGES.')
    #SEM_images_grid = np.reshape(SEM_images_grid, np.prod(shape))
    
    for l in tqdm(uniques, desc="Identifying edge particles"):
    
        if ndim == 4:
    
            row, col, imrow, imcol = np.where(SEM_images_grid == l)
            
            r, c = row[0], col[0]
            
            # Identify the overlapping regions
            if r != 0: # Current image * Image above

                edge_mask = empty_image.copy()

                edge_mask[:edge_width,:] = True
                
                sum_im[r,c] += (SEM_images_grid[r,c] * np.flip(SEM_images_grid[r-1,c], axis = 0) * edge_mask)
    
            if r != nav_shape[0] - 1: # Current image * Image below

                edge_mask = empty_image.copy()

                edge_mask[-edge_width:,:] = True
            
                sum_im[r,c] += (SEM_images_grid[r,c] * np.flip(SEM_images_grid[r+1,c], axis = 0) * edge_mask)
    
            if c != 0: # Current image * Image to the left

                edge_mask = empty_image.copy()

                edge_mask[:,:edge_width] = True
            
                sum_im[r,c] += (SEM_images_grid[r,c] * np.flip(SEM_images_grid[r,c-1], axis = 1) * edge_mask)
            
            if c != nav_shape[1] - 1: # Current image * Image to the right

                edge_mask = empty_image.copy()

                edge_mask[:,-edge_width:] = True
                
                sum_im[r,c] += (SEM_images_grid[r,c] * np.flip(SEM_images_grid[r,c+1], axis = 1) * edge_mask)
    
        elif ndim == 3:

            print('WARNING! THE FUNCTION IS ASSUMING THE IMAGES ARE TAKEN FROM RIGHT TO LEFT')
    
            row, imrow, imcol = np.where(SEM_images_grid == l)
            
            r = row[0]
        
            # Identify the overlapping regions
            if r != 0: # Current image * Image to the right

                edge_mask = empty_image.copy()

                edge_mask[:,-edge_width:] = True 
                
                sum_im[r] += SEM_images_grid[r] * np.flip(SEM_images_grid[r-1], axis = 1) * edge_mask
    
            if r != nav_shape - 1: # Current image * Image to the left

                edge_mask = empty_image.copy()

                edge_mask[:,:edge_width] = True
                
                sum_im[r] +=  SEM_images_grid[r] * np.flip(SEM_images_grid[r+1], axis = 1) * edge_mask
    
        else: 
            
            raise _errors.ShapeError(f'The provided array shape ({particles_region_map.shape}) is not expected.')
        
    sum_im = sum_im > 0

    clipped_particle_map = np.zeros_like(SEM_images_grid)

    cropped_particles = np.unique(sum_im * particles_region_map)

    cropped_particles = np.delete(cropped_particles, np.where(cropped_particles == 0))

    for p in tqdm(cropped_particles, desc = "Mapping cropped particles"):
    
           clipped_particle_map[np.where(particles_region_map == p)] = p

    return clipped_particle_map

"""def _check_if_idx_is_in_list_with_arrays(list_with_arrays, idx):
    Check if idx is in list_with_arrays.

    Example
    -------
    >>> list_with_arrays = [np.array([1,2]), np.array([3,4])]
    >>> idx = 2
    >>> check_if_idx_is_in_list_with_arrays(list_with_arrays, idx)
    True
    >>> check_if_idx_is_in_list_with_arrays(list_with_arrays, 5)
    False
    
    for i in range(len(list_with_arrays)):

        if idx in list_with_arrays[i]: return True

    return False"""

def _identify_particle_edge_clusters(cropped_particles_map, image_edges_grid):#edge_width = 1):
    """Identify labelled regions (particles) at (SEM) image edges which might be the same label (particle). The 
    function assumes *._get_edge_particles_gridMap_from_labelled_particles_gridMap() has been run.

    Parameters
    ----------
    cropped_particles_map
        (n,m) shaped image of labelled particles 

    image_edges_grid
        (n,m) shaped mask with One/True-valued rows and columns defining which particles to consider.

    Returns
    -------
    particle_clusters
        pairs clustered together at image edges defined by image_edges_grid 

    Example
    -------
    >>> import particle_analysis as pa
    >>> s = pa.load(filename)
    >>> s.load_images(image_path)
    >>> grid_shape = (4,4)
    >>> s.gridify_SEM_images(grid_shape)
    >>> s.identify_particle_regions()
    >>> mask = s.get_navigation_grid(edge_width = 1) # default value
    >>> mask
    array([[False, False, False, ..., False, False, False],
        [False,  True,  True, ...,  True,  True, False],
        [False,  True,  True, ...,  True,  True, False],
        ...,
        [False,  True,  True, ...,  True,  True, False],
        [False,  True,  True, ...,  True,  True, False],
        [False, False, False, ..., False, False, False]],
        shape=(n, m))
    >>> np.unique(~image_edges_grid * pa.utils.stitch_images(s.Images.cropped_particles_map.data, grid_shape))
    array([   0,   78,  105,  174,  301,  303,  304,  326,  355,  391,  411,
        412,  413,  414,  415,  507,  508,  509,  617,  618,  619,  620,
        809,  810,  866,  873,  900,  901,  902,  985,  986,  987, 1104,
       1151, 1152, 1153, 1155, 1229, 1230, 1232, 1233, 1305, 1317, 1318])
    """

    particle_clusters = []
    
    if np.shape(cropped_particles_map) != np.shape(image_edges_grid):

        raise _errors.ShapeError(f"The shape of array1 ({np.shape(cropped_particles_map)}) is not identical to the shape of array 2 ({np.shape(image_edges_grid)}).")

    else:

        from skimage.morphology import label
        from tqdm import tqdm

        # Identify connected edge particles
        edge_particle_map = cropped_particles_map * image_edges_grid

        # Relabel connected edge particles:
        relabelled_edge_particle_map = label(edge_particle_map > 0)

        unique_particle_labels = np.unique(relabelled_edge_particle_map)

        # Get the unique new particle labels
        unique_particle_labels = np.delete(unique_particle_labels, np.where(unique_particle_labels == 0))

        # Identify the clustered particles
        for p in tqdm(unique_particle_labels, desc = 'Grouping edge particles'):

            particle_clusters.append(np.unique(edge_particle_map[(edge_particle_map * (relabelled_edge_particle_map == p)).astype(bool)]))     
        
        return particle_clusters



def _stitch_images(data_array, shape, horisontal_direction = 'r2l', vertical_direction = 't2b'):
    """Stitch an array of images into a 2D image. 

    Parameters
    ----------
    data_array
        Array with images to stitch. Expected shape: 4D (2 navigation, 2 image) dimensions
    shape 
        list or tuple of the concatenated image shape. 
        Note that the shape should reflect the numpy convention (i.e. 20 images -> 4x5 => shape = (5,4))
    vertical_direction
        Vertical direction at which the images are to be stitched.
        Allowed arguments: l2r and r2l (default: r2l: right to left)
    horisonal_direction
        Horisontal direction at which the images are to be stitched 
        Allowed arguments: t2b and b2t (default: t2b: top to bottom)

    Returns
    -------
    stitched_image 
        Concatenated images as a numpy array
    """
    from tqdm import tqdm
    
    data_arr = data_array.copy()

    #if len(data_array.shape) == 4: 
    
    #    if len(data_array.shape[-1]) != 3: data_arr = _gridify_4D_array_to_3D(data_array)

    stitched_image = []

    ndim = len(shape)

    if ndim == 1: shape = (1, shape[0])
    
    horisontals = np.arange(0, shape[1])
    
    if horisontal_direction == 'r2l': horisontals = np.flip(horisontals)
    
    for col in range(shape[0]):
        
        #if horisontal_direction == 'l2r' : stitched_image.append(np.concatenate(data_arr[hors * shape[0] : (hors + 1)*shape[0]], axis = 0))
        #else:
        
       stitched_image.append(np.concatenate([data_arr[hors] for hors in horisontals], axis = -1))
    
       horisontals += shape[1]
    
    if vertical_direction == 't2b': stitched_image = np.concatenate(np.array(stitched_image))
    
    else: stitched_image = np.concatenate(np.array([stitched_image[i] for i in np.flip(np.arange(0,shape[0]))]))

    return stitched_image

def _gridify_3D_array_to_4D(arr, to_shape, flip_axis = 1):
    """Make a 4D array of the SEM images

    Parameters
    ---------
    SEM_images
        3D array with SEM images
    to_shape
        4D grid shape (end shape)
    flip_axis
        Flip the navigation order as the images are read from right to left in the particle analysis software.
        Default: axis = 1

    Returns
    -------
    Gridified arr (numpy ndarray)
    """
    
    if arr.shape != to_shape:

        if len(arr.shape) == 3 and _errors._check_for_numpy_ndarray(arr): 

            print(f'Reshaping images into a 4D signal of shape ({to_shape[0]}, {to_shape[1]}|{to_shape[2]}, {to_shape[3]})')
            
            return np.flip(np.reshape(arr, to_shape), flip_axis)

        else:
            
            print(f'Provided image array of shape {arr.shape} is not valid.')

            return arr

    else: 
        
        print(f'The shape {to_shape} is identical to the input array shape ({arr.shape})')

        return arr

def _gridify_nD_array_to_ND(arr, to_shape, flip_axis = 1):
    """Make a ND array for N = n+1 of the SEM images

    Parameters
    ---------
    SEM_images
        3D array with SEM images
    to_shape
        4D grid shape ("end" shape)
    flip_axis
        Flip the navigation order as the images are read from right to left in the particle analysis software.
        Default: axis = 1

    Returns
    -------
    Gridified arr (numpy ndarray)

    Example
    -------
    >>> arr = np.zeros((3,4,10,15))
    >>> arr.shape
    (3,4,10,15)
    
    >>> _gridify_nD_array_to_ND(arr, to_shape = (3,4,10,3,5)).shape
    (3,4,10,3,5)
    """
    
    if arr.shape != to_shape:

        if  _errors._check_for_numpy_ndarray(arr): 

            print(f'Reshaping images into a ND signal of shape <({to_shape[:2]} | {to_shape[2:]})>')
            
            return np.flip(np.reshape(arr, to_shape), flip_axis)

        else:
            
            print(f'Provided image array of shape {arr.shape} is not valid.')

            return arr

    else: 
        
        print(f'The shape {to_shape} is identical to the input array shape ({arr.shape})')

        return arr

"""
def _get_map_of_particle_regions(particle_images, SEM_images, SEM_image_IDs, label_particles = True):
    Identify the particles' position in each individual SEM image by running template matching.
    (See skimage.feature's function match_template)

    Parameters
    ----------
    particle_images
        Array of particle images. Note that the first index is 0
    SEM_image_IDs
        Array of SEM image IDs where the particles were found. 
        Note that it is assumed that the particles' image indices corr. to their label names.
        I.e. SEM image 1 corr. to index 0. 
    SEM_image_shape 
        list or tuple of the SEM image shape (individual viewing images)

    Note that overlapping regions are split up and divided ~equally 

    Example
    -------
        E.g. numpy.array([0,0,2,5,9,9]) -> two particles in SEM image 0, onw in 2 and 5, and 
        two in SEM image nr. 9 

    Returns
    -------
    particle_maps
        Map of identified particles.
        The background is set to zero, whilst the particles are labelled as one if label_particles
        is set to False. If not, they are labelled according to their particle IDs.
    
    overlapping_pixels
        Map of overlapping pixels    
    
    from tqdm import tqdm
    from skimage.measure import regionprops_table
    from skimage.segmentation import expand_labels
    
    if len(particle_images) == len(SEM_image_IDs):

        if label_particles: 
            
            print('\nParticle labelling: partly overlapping particle regions will be shared equally.\n')
        
            dtype = int

        else: dtype = bool

        # Map of particle "positions"
        particle_map = np.zeros(SEM_images.shape, dtype = dtype)

        # Keeping track of which pixels are overlaping, as the particle images are stored as rectangles
        overlapping_pixels = np.zeros(SEM_images.shape, dtype = bool)

        # Check if we can speed things up a bit by not re-reading SEM images
        increasing_image_idx = all(SEM_image_IDs[i] <= SEM_image_IDs[i+1] for i in range(len(SEM_image_IDs)-1))

        if increasing_image_idx: 
            
            curr_im_idx = 0

            sem_image = SEM_images[curr_im_idx].copy()

        else: print('THIS HAS NOT BEEN TESTED YET')

        print('w,h in the loop here has been changed and not tested. The function _get_single_particle_map has been changed accordingly.')
        
        for p, im_id in tqdm(zip(np.arange(len(particle_images)), SEM_image_IDs - 1), total = len(particle_images), desc = 'Mapping particles'):
            
            # Get where the particle image is within the frame
            #indices = np.where(particle_images[p] > 0)
            
            # Width, height of particle frame
            #w, h = 1 + indices[1].max() - indices[1].min(), 1 + indices[0].max() - indices[0].min()
            
            # Get image
            p_im = particle_images[p]#[indices[0].min():indices[0].max(), indices[1].min():indices[1].max()].reshape((h-1, w-1))

            # Update SEM image
            if increasing_image_idx:

                if curr_im_idx < im_id: sem_image = SEM_images[im_id].copy()

            else: sem_image = SEM_images[im_id].copy()

            # Identify the particle within the SEM image:
            pmap = _get_single_particle_map(sem_image, p_im)

            # Label the particle according to the order of the data acquisition
            if label_particles: 

                label_indices = np.where(pmap)
                
                pmap = pmap.astype(int)
                
                pmap[label_indices] = p + 1 # Particle index + 1 => label
            
            # Remove overlapping pixels: sum of stored image and new particle region image
            overlap_im = (particle_map[im_id] > 0).astype(int) + (pmap > 0).astype(int)

            # Identify the overlapping region
            overlap_im = overlap_im > 1

            if label_particles:
                
                # If the number of overlaping pixels are identical to the smallest particle in this region, then only 
                # keep the small particle reigon and ignore the larger one.
                overlapping_indices = np.where(overlap_im)
    
                particle_indices = np.where(pmap > 0)

                older_particle_label = np.unique(overlap_im * particle_map[im_id])

                if len(older_particle_label) == 2: older_particle_label = older_particle_label[-1]

                elif len(older_particle_label) > 2:
                    
                    min_p_size = np.inf
                    _temp_im = overlap_im.copy() * particle_map[im_id].copy()
                    # Identify the smallest particle among the labels
                    for pID in older_particle_label:

                        temp_size = np.sum(_temp_im == pID)

                        if temp_size < min_p_size: 
                            
                            min_p_size = temp_size
                            
                            older_particle_label = pID

                    # For later: a map of the smallest particle within the overlapping region:
                    small_within_overlap_im = (overlap_im.copy() * particle_map[im_id].copy()) == older_particle_label 
                    
                else: small_within_overlap_im = np.zeros_like(overlap_im)

                older_particle_indices = np.where(particle_map[im_id] == older_particle_label)

                small_within_overlap = False
                
                # Prepare the conditions for when there is a previously mapped particle that is within the new overlapping region
                if np.shape(np.where(small_within_overlap_im)) == np.shape(np.where(particle_map[im_id] == older_particle_label)):
                    # Have to check the shape first...
                    condition0 = np.where(small_within_overlap_im)[0] == np.where(particle_map[im_id] == older_particle_label)[0]
                    condition1 = np.where(small_within_overlap_im)[1] == np.where(particle_map[im_id] == older_particle_label)[1]
                    if condition0.all() and condition1.all(): small_within_overlap = True
                
                if np.shape(overlapping_indices) == np.shape(particle_indices):
                    # If the new particle is overlapping 100% with another particle <=> the previously mapped particle is larger 
                    # than the new particle => keep only the new particle region as it is the smallest.
                    particle_map[im_id][overlapping_indices] = pmap[particle_indices]
                    
                elif np.shape(overlapping_indices) == np.shape(older_particle_indices):
                    # If an old particle is overlapping 100% with the new particle <=> the old particle is smaller than
                    # the new particle => only keep the old particle region as it is the smallers.
                    particle_map[im_id] += pmap

                    particle_map[im_id][overlapping_indices] = older_particle_label

                elif small_within_overlap:
                    # If the overlapping region are covering 100% of an older and smaller particle:
                    particle_map[im_id][np.where(overlap_im)] = 0
                    
                    particle_map[im_id] += pmap

                    particle_map[im_id][np.where(small_within_overlap_im)] = older_particle_label

                else:

                    # Remove the overlapping region from the particle map
                    particle_map[im_id] += pmap
        
                    particle_map[im_id] *= np.invert(overlap_im)
        
                    overlapping_pixels[im_id] += overlap_im
                
                    # Expand the labels in the overlapping region
                    if np.sum(overlapping_pixels[im_id]) > 0:
        
                        initial_map = (particle_map[im_id] > 0) + overlapping_pixels[im_id]
        
                        props = regionprops_table(particle_map[im_id], properties=('label', 'axis_major_length'))
                        
                        particle_map[im_id] = expand_labels(particle_map[im_id], 
                                                            distance = np.max(props['axis_major_length']))
                        particle_map[im_id] *= initial_map
    
                if increasing_image_idx: curr_im_idx = im_id.copy()

            else: # No particle labelling

                particle_map[im_id] += pmap

                overlapping_pixels[im_id] += overlap_im
            
        return particle_map, overlapping_pixels
            
    else: print(f"The number of particle images ({len(particle_images)}) does not match the number of SEM images ({len(SEM_images)}) provided...")
"""

def _translate_location_position(loc : tuple):#v, h):
    """Translate the particle position values (-1,0,1) into a string. These values can be determined by
    the function _identify_a_particles_position_wrt_SEM_image_edges().

    Example
    -------
    >>> loc
    (-1, 0)
    
    >>> _translate_location_position(loc)
    Particle's position in the vertical direction:
    	Bottom.
    Particle's position in the horisontal direction:
    	Not near the horisontal edge.
    """
    vstring = {-1 : 'Not at the top or the bottom', 
                0 : 'Top edge',
                1 : 'Bottom edge'}
    hstring = {-1 : 'Not to the left or the right', 
                0 : 'Left edge',
                1 : 'Right edge'}
    print(f"Particle's position in the vertical direction:\n\t{vstring[loc[0]]}.")
    print(f"Particle's position in the horisontal direction:\n\t{hstring[loc[1]]}.")
    
def _identify_a_particles_position_wrt_SEM_image_edges(particle_map_label : int, 
                                                       particle_map : np.ndarray,
                                                       print_location : bool = False):
    """Locate a particle's position (label given by particle_map_label) wrt. the particle_map's 
    edges (ndarray). Note that the indices can be translated by the _translate_location_position 
    function.
    
    Parameters
    ----------
    particle_map_label
        Label representing the particle's position in the particle_map
    particle_map
        numpy ndarray of labelled particle regions.
    Returns
    -------
    loc
        (vertical, horisontal) location positions represented by -1, 0, or 1:
            -1: Not near the vert./hor. edge
             0: Top/left edge
             1: Bottom/right edge
         
    
    Example
    -------
    >>> loc = __identify_a_particles_position_wrt_SEM_image_edges(1, particle_map)
    >>> loc
    (1, -1)

    >>> _translate_location_position(loc)
    Particle's position in the vertical direction:
    	Bottom.
    Particle's position in the horisontal direction:
    	Not near the horisontal edge.
    """
    ndim = len(particle_map.shape)
    
    if ndim > 2:
        if ndim >= 4:
            sig_shape = particle_map.shape[-2:]
            argwhere = np.where(particle_map == particle_map_label)
            Y0,X0 = argwhere[0][0], argwhere[1][0]
            miny0, maxy0 = argwhere[2].min(), argwhere[2].max()
            minx0, maxx0 = argwhere[3].min(), argwhere[3].max()
        else:
            sig_shape = particle_map.shape[-2:]
            argwhere = np.where(particle_map == particle_map_label)
            Y0 = argwhere[0][0]
            miny0, maxy0 = argwhere[1].min(), argwhere[1].max()
            minx0, maxx0 = argwhere[2].min(), argwhere[2].max()
    else:
        sig_shape = particle_map.shape
        argwhere = np.where(particle_map == particle_map_label)
        miny0, maxy0 = argwhere[0].min(), argwhere[0].max()
        minx0, maxx0 = argwhere[1].min(), argwhere[1].max()
    
    # Particle location wrt. acquired navigation image (SEM image)
    # Vertical position
    vloc0 = -1 # [-1, 0, 1] <=> [not_relevant, top, bottom]
    # horisontal position
    hloc0 = -1 # [-1, 0, 1] <=> [not_relevant, left, right]
    if miny0 == 0: vloc0 = 0
    elif maxy0 == sig_shape[0]-1: vloc0 = 1
    if minx0 == 0: hloc0 = 0
    elif maxx0 == sig_shape[1] - 1: hloc0 = 1
    
    loc = (vloc0, hloc0)
    
    if print_location: _translate_location_position(loc)

    if ndim > 3: return loc, (Y0, X0) 
    if ndim == 3: return loc, (Y0,) 
    else: return loc

def _crop_out_a_random_size_larger_particle_area_from_SEM_image(sem_img : np.ndarray,
                                                                region_img : np.ndarray,
                                                                particle_label : int,
                                                                particle_shape : tuple,
                                                                particle_position : tuple,
                                                                extra_pixels : float = 0.2,
                                                                print_status = False):
    """Crop out a larger area from the sem image where the particle label is located in the region_img.
    Note that if a particle is located at an image edge, the extra pixels will be allocated to both axes.
    
    Parameters
    ----------
    sem_img
        2D image from where a larger piece than the area defined iun region_img is to be cropped out. 
        The extra pixels is defined by extra_pixels (vertical, horisontal)
    region_img
        image keeping track of the particle regions
    particle_label
        particle identify (label) of interest
    particle_shape
        Shape of the image which may restrict one of the dimensions for the new and larger images
    extra_pixels
        a percentage of extra pixels defined by np.max(sem_img.shape). Note that no extra pixels will be 
        added to the new image in the vertical direction if the particle is located at the top/bottom of
        the image. The same applies for the horisontal edges.
        Default: 20%
    """
    ndim = len(sem_img.shape)
    if ndim == 2 and len(region_img.shape) == ndim:
        
        sig_shape = sem_img.shape
        im_arg = np.where(region_img == particle_label)
        im_y0, im_y1, im_x0, im_x1 = im_arg[0].min(), im_arg[0].max(), im_arg[1].min(), im_arg[1].max()
    
        # Ambigous name, but it virtually tells how much more image to add
        extra_pixels *= sig_shape[1]
        extra_pixels = int(extra_pixels)
        if print_status: print("Extra pixels:", extra_pixels)
        
        if particle_position[0] == 0: # top
            if particle_position[1] in [0,1]: #top edge
                i0, i1 = 0, particle_shape[0] + extra_pixels 
            else: i0, i1 = 0, particle_shape[0]
        elif particle_position[0] == 1: # bottom
            if particle_position[1] in [0,1]: # bottom edge
                i0, i1 = -particle_shape[0]-1-extra_pixels, -1
            else: i0, i1 = -particle_shape[0]-1, -1
        else:
            diff = particle_shape[0] - (im_y1 - im_y0)
            i0, i1 = im_y0 - extra_pixels//2, extra_pixels//2 + im_y1 + diff
        
        if particle_position[1] == 0: #left
            if particle_position[0] in [0,1]: # top edge
                j0, j1 = 0, particle_shape[1] + extra_pixels
            else: j0, j1 = 0, particle_shape[1]
        elif particle_position[1] == 1: # right
            if particle_position[0] in [0,1]: # bottom-edge
                j0, j1 = -particle_shape[1]-1-extra_pixels, -1    
            else: j0, j1 = -particle_shape[1]-1, -1
        else:
            diff = particle_shape[1] - (im_x1 - im_x0)
            j0, j1 = im_x0 - extra_pixels//2, extra_pixels//2 + im_x1 + diff
            
        temp_im = sem_img[i0:i1, j0:j1]

        if 0 in temp_im.shape:
            
            # Correct for negative numbers???
            if i0 < 0: 
                diffi = abs(i0)
                i0 = 0
                i1 += diffi
                
            if i1 > sig_shape[0]: i1 = sig_shape[0]
    
            if j0 < 0: 
                diffj = abs(j0)
                j0 = 0
                j1 += diffj
                
            if j1 > sig_shape[1]: j1 = sig_shape[1]
            
            temp_im = sem_img[i0:i1, j0:j1]
        
        return temp_im, extra_pixels
    
    else: raise ValueError(f"The function only takes in 2D images. The provided ndarray shapes are sem_img: {sem_img.shape}, and region_img: {region_img.shape}.")

def _crop_out_a_specified_particle_area_from_SEM_image(sem_img : np.ndarray,
                                                       region_img : np.ndarray,
                                                       get_shape : tuple,
                                                       background_id = 0,
                                                       print_status = False):
    """Crop out a larger area from the sem image where the particle label is located in the region_img.
    
    Parameters
    ----------
    sem_img
        2D image from where a larger piece than the area defined iun region_img is to be cropped out. 
        The extra pixels is defined by extra_pixels (vertical, horisontal)
    region_img
        image keeping track of the particle regions
    get_shape
        Shape of the new desired image
    background_id 
        label id of the background. Default: 0

    Returns
    -------
    cropped_image
        A cropped image of shape get_shape from sem_img, including the labelled area in region_img. 
    """
    if len(np.unique(region_img)) > 2: raise ValueError(f"The labels in the provided region_img's number of unique labels ({len(np.unique(region_img))}) should not exceed 2.")

    sig_shape = sem_img.shape

    if print_status: print('Get shape:', get_shape)
    
    if len(sig_shape) == 2 and region_img.shape == sig_shape:

        im_arg = np.where(region_img > background_id)
        im_y0, im_y1, im_x0, im_x1 = im_arg[0].min(), im_arg[0].max()+1, im_arg[1].min(), im_arg[1].max()+1
        
        diffy, diffx = get_shape[0] - (im_y1 - im_y0), get_shape[1] - (im_x1 - im_x0)
        if np.mod(diffy,2) == 0: diffy05 = (diffy//2, diffy//2)
        else: diffy05 = (diffy//2, diffy//2+1)
        
        if np.mod(diffx,2) == 0: diffx05 = (diffx//2, diffx//2)
        else: diffx05 = (diffx//2, diffx//2+1)
            
        
        # Adjust y-range:
        if im_y0 == 0: # Image top 
            im_y1 = im_y1 + diffy
        
        elif im_y1 == sig_shape[0]: # Image btm.
            im_y0 = im_y0 - diffy
        
        else: # Not at vertical edges, but somewhere between
            im_y0 -= diffy05[0]
            im_y1 += diffy05[1]
            
            if im_y0 < 0: # Near top edge
                im_y0 = 0
                im_y1 = get_shape[0]
            if im_y1 > sig_shape[0]: # Near btm. edge
                im_y0 = sig_shape[0] - get_shape[0]
                im_y1 = sig_shape[0]

        # Adjust x-range:
        if im_x0 == 0: # Image left 
            im_x1 = im_x1 + diffx
        
        elif im_x1 == sig_shape[1]: # Image right
            im_x0 = im_x0 - diffx
        
        else: # Not at vertical edges, but somewhere between
            im_x0 -= diffx05[0]
            im_x1 += diffx05[1]
            
            if im_x0 < 0: # Near top edge
                im_x0 = 0
                im_x1 = get_shape[1]
            if im_x1 > sig_shape[1]: # Near btm. edge
                im_x0 = sig_shape[1] - get_shape[1]
                im_x1 = sig_shape[1]
        
        if print_status: print('Returning shape:', sem_img[im_y0:im_y1, im_x0:im_x1].shape)
        
        return sem_img[im_y0:im_y1, im_x0:im_x1]
    else:
        raise ValueError(f"The function only takes in 2D images. The provided ndarray shapes are sem_img: {sem_img.shape}, and region_img: {region_img.shape}.")

def _concatenate_a_stitched_image_with_a_non_stitched_image(particle_id : int, 
                                                            stitched_particle_ids, 
                                                            stitched_particle_image,
                                                            particle_map, 
                                                            navigation_map, 
                                                            to_shape : tuple,
                                                            ncc_score_threshold = 0.5):
    """The idea behind this hidden function is to concatenate a single particle image with a particle image that e.g. has 
    already been stitched. This will be done by mapping the stitched particle's area onto the non-stitched particle and 
    concatenate the two together. 

    Obs! There are no guarantees that the stitching suceeds.

    Parameters
    ----------
    particle_id
        Particle label (integer) as it is stored in the particle_map
    particle_map
        numpy ndimage of particle labels
    stitched_particle_image
        numpy ndarray of the stitched particle image
    navigation_map
        numpy ndimage of navigation signal
    to_shape
        To what shape the new image is to have (2D tuple)

    Returns
    -------
    concatenated_image
        The finally concatenated image
    """
    from skimage.feature import match_template
    # Whether to concatenate or not
    concatenate = True
    
    _loc_, _IDXs_ = _identify_a_particles_position_wrt_SEM_image_edges(particle_id, particle_map)

    # Get image indices and identify 
    list_of_indices = []
    for sid in stitched_particle_ids: 
        loc, idx = _identify_a_particles_position_wrt_SEM_image_edges(sid, particle_map)
        list_of_indices.append(idx)

    # Check if the stitched images are all in the same navigation image:
    for idxs in list_of_indices[1:]: 
        if list_of_indices[0] != idxs: concatenate = False

    if concatenate:
        
        artificial_particle_map = particle_map[idx[0], idx[1]].copy()
        
        for pid in stitched_particle_ids: artificial_particle_map[np.where(artificial_particle_map == pid)] = particle_id

        sem_img = navigation_map[_IDXs_[0], _IDXs_[1]].copy()
        nav_img = particle_map[_IDXs_[0], _IDXs_[1]].copy()
        p_img = _remove_empty_edges_from_image(nav_img == particle_id)
        pid_shape = p_img.shape
        
        # Flip the image to define the area where cropping is to be done
        # Define axis based on _locX and Change to_shape to make it compatible 
        # with the image argument along the edge side:
        if loc[0] in (0,1): 
            artificial_particle_map = np.flip(artificial_particle_map, axis = 0)
            to_shape = (pid_shape[0], to_shape[1])
        elif loc[1] in (0,1): 
            artificial_particle_map = np.flip(artificial_particle_map, axis = 1)
            to_shape = (to_shape[0], pid_shape[1])
        
        new_im = _crop_out_a_specified_particle_area_from_SEM_image(navigation_map[_IDXs_[0], _IDXs_[1]],
                                                                    artificial_particle_map == particle_id,
                                                                    get_shape = to_shape)
        
        # Check if the cropped image is the same as the wanted image:
        pmap = match_template(sem_img * (particle_map[_IDXs_[0], _IDXs_[1]] == particle_id), new_im)

        if pmap.max() >= ncc_score_threshold:

            if _loc_[0] == 0: return np.concatenate([stitched_particle_image, new_im])
            elif _loc_[0] == 1: return np.concatenate([new_im, stitched_particle_image])
            elif _loc_[1] == 0: return np.concatenate([stitched_particle_image, new_im], axis = 1) 
            elif _loc_[1] == 1: return np.concatenate([new_im, stitched_particle_image], axis = 1)
            else:
                print('Could not concatenate the images')
                return np.array([])
        else: return np.array([])
    else: return np.array([])

def _stitch_two_particle_images(img0, img1,
                                particle_label1, particle_label2,
                                particle_map, 
                                navigation_signal,
                                shift_threshold : float = 1.0,
                                remove_non_matched_region : bool = False,
                                success_threshold : float = 0.25):
    """stitch two particle images. The particles are identified by their corresponding labels.

        Parameters
        ----------
        particle_label1, particle_label2
            particle labels (integers)
        shift_threshold
            float describing whether a image stitching is sucessful or not. Default: 1.0 (Eucledian distance)
        remove_non_matched_region 
            Whether to remove the non-matched region from the stitched image. The non-matched region originates
            from the larger area extracted from the particle's corresponding SEM image.
            Default: True
        success_threshold
            A normalised cross-correlation score dictating whether a stitching succeeded or not. This threshold
            is compared with the product of the NCC score between the overlapping region and a larger area with 
            the stitched image. Default: 0.25

        Returns
        -------
        stitched_image
            depadded numpy.ndarray of the stitched image

        success
            Whether the stitching was successful or not based on image shifting (bool) and ncc
            score between the individual images and the stitched image in the stitched region.
        """
    from skimage.exposure import rescale_intensity

    minI = min([np.min(img0), np.min(img1)]) 
    maxI = max([np.max(img0), np.max(img1)])
    
    # Identify the particles' position wrt. the SEM image edges:
    loc0, IM_IDX0 = _identify_a_particles_position_wrt_SEM_image_edges(particle_label1, particle_map.data)
    loc1, IM_IDX1 = _identify_a_particles_position_wrt_SEM_image_edges(particle_label2, particle_map.data)

    # Pad the images to make them have identical shapes:
    pim0, pim1 = _copy_images_into_common_array_shape(img0, img1)

    # Identify the SEM and particle images
    sem_img0 = navigation_signal.inav[IM_IDX0[1],IM_IDX0[0]].data
    region_img0 = particle_map.inav[IM_IDX0[1],IM_IDX0[0]].data
    sem_img1 = navigation_signal.inav[IM_IDX1[1],IM_IDX1[0]].data
    region_img1 = particle_map.inav[IM_IDX1[1],IM_IDX1[0]].data

    # Crop out a larger region from the SEM image. Have this region to be a scalar if it's not too lage
    if pim0.shape[0] / sem_img0.shape[0] < 0.1 and pim0.shape[1] / sem_img0.shape[1] < 0.1: 
        im0, extra_pixels = _crop_out_a_random_size_larger_particle_area_from_SEM_image(
            sem_img0, 
            region_img0, 
            particle_label1, 
            pim0.shape, 
            loc0
        )
    elif pim0.shape[0] / sem_img0.shape[0] < 0.1:
        get_shape = [pim0.shape[0], int(0.1 * sem_img0.shape[0])]
        if get_shape[1] < pim0.shape[1]: get_shape[1] = pim0.shape[1]
        im0 = _crop_out_a_specified_particle_area_from_SEM_image(
            sem_img0, 
            region_img0 == particle_label1, 
            get_shape = tuple(get_shape)
        )
    elif pim0.shape[1] / sem_img0.shape[1] < 0.1:
        get_shape = [int(0.1 * sem_img0.shape[1]), pim0.shape[1]]
        if get_shape[0] < pim0.shape[0]: get_shape[0] = pim0.shape[0]
        im0 = _crop_out_a_specified_particle_area_from_SEM_image(
            sem_img0, 
            region_img0 == particle_label1, 
            get_shape = tuple(get_shape)
        )
    else:
        # The image is either large or long - crop out a similarly shaped image
        im0 = _crop_out_a_specified_particle_area_from_SEM_image(
            sem_img0, 
            region_img0 == particle_label1, 
            get_shape = pim0.shape
        )
    
    # Get the same shape for im1 as im0:
    im1 = _crop_out_a_specified_particle_area_from_SEM_image(
        sem_img1, 
        region_img1 == particle_label2, 
        get_shape = im0.shape
    )

    # Pad the images --> avoid strange artefacts in the stitching
    pad = np.max(im0.shape)
    img0_pad = np.pad(im0.astype(float).copy(), pad_width = pad)
    img1_pad = np.pad(im1.astype(float).copy(), pad_width = pad)

    shifted_image, shift = _images.phase_cross_correlate_images(img0_pad, img1_pad)
    success = _assess_successful_stitching(shift, shift_threshold = shift_threshold)
    
    if success: # If shifting succeeded or not

        from scipy.ndimage import binary_fill_holes as bfh
        
        # Sucessful stitching
        mask_shift = _get_rectangle_mask_after_image_shift(shifted_image, top_hat_filter = True)
        mask_im = _get_rectangle_mask_after_image_shift(img1_pad)
        overlap_mask = mask_shift * mask_im
        sum_mask = (mask_shift.astype(int) + mask_im.astype(int)) > 0

        diff = (bfh(sum_mask).astype(int) - sum_mask.astype(int)).astype(bool)
        sum_mask[diff] = True
        mask_im[diff] = True

        # Create a rectangular overlap mask to assess the shifted image and the
        # statinary image's similarity - metric is normalised cross-correlation.
        rect_mask = _rectangulerize_mask(overlap_mask)

        if rect_mask.sum() > 0:
            
            overlap_score = _utils.ncc(_remove_empty_edges_from_image(img1_pad*overlap_mask),
                                       _remove_empty_edges_from_image(shifted_image*overlap_mask))
            
            rect_score = _utils.ncc(_remove_empty_edges_from_image(img1_pad*rect_mask),
                                    _remove_empty_edges_from_image(shifted_image*rect_mask))
            
            score = overlap_score * rect_score
        
        else: score = 0

        if score >= success_threshold:

            # Replace everythin but the shift region with the original image
            stitched_image = shifted_image.copy()
            stitched_image[mask_im] = img1_pad[mask_im]
            stitched_image *= sum_mask # Remove shift artefacts

            # Remove stripes and other shifting artefacts
            #stitched_image *= (stitched_image > thr(stitched_image))
            
            # Identify the original particle images in the stitched image
            p_mask = np.zeros_like(stitched_image, bool)
            for pim in [img0, img1]: p_mask += _get_single_particle_map(stitched_image, pim)

            # If only keep the original particle images in the stitched image:
            if remove_non_matched_region: 
                # Mask out the original images:
                masked_stitched_image = (p_mask * stitched_image)
            
            else:
                
                p_mask = _rectangulerize_mask(p_mask)
                
                masked_stitched_image = (p_mask * stitched_image)
                
            # Remove pads
            masked_stitched_image = _remove_empty_edges_from_image(masked_stitched_image)
            
            return rescale_intensity(masked_stitched_image, 
                                     out_range = (minI, maxI)), True

        else: return np.array([]), False 
        
    else: return np.array([]), False

def _stitch_group_of_particles(group_of_particle_images : list,
                               group_of_particle_labels : list | np.ndarray,
                               particle_map, # Signal2D
                               navigation_signal, # Signal2D
                               shift_threshold : float = 1.0,
                               remove_non_matched_region : bool = False,
                               success_threshold : float = 0.25,
                               looping_progressbar = False,
                               directly_stitch_edge_particles = False):
    """Stitch a group of particle images into one particle image. The stitched particle image's 
    intensity values will be rescaled into the range covered by all the images in 
    group_of_particle_images.

    
    """
    from tqdm import tqdm 
    from skimage.exposure import rescale_intensity
    from scipy.ndimage import binary_fill_holes as bfh

    dtype = group_of_particle_images[0].dtype

    group_of_particle_labels = np.asarray(group_of_particle_labels)

    # Keep track of temporarily successfully stitched particle images
    temp_success = []
    # Keep track of particles that have not been successfully stitched (yet)
    temporarily_not_stitched = []

    # Search for the first successful stitching to start with
    label_combinations = _utils.get_number_combinations(group_of_particle_labels)
    
    for i in tqdm(range(len(label_combinations)), position=1, leave=False, disable = ~looping_progressbar):
        
        idx0, idx1 = label_combinations[i][0], label_combinations[i][1]

        # Temporary images for intensity rescaling
        img0 = group_of_particle_images[np.where(group_of_particle_labels == idx0)[0][0]]
        img1 = group_of_particle_images[np.where(group_of_particle_labels == idx1)[0][0]]
        
        minI = min([np.min(img0), np.min(img1)]) 
        maxI = max([np.max(img0), np.max(img1)])

        img1, successful_stitching = _stitch_two_particle_images(
            img0 = img0, img1 = img1,
            particle_label1 = idx0, 
            particle_label2 = idx1,
            particle_map = particle_map,
            navigation_signal = navigation_signal,
            shift_threshold = shift_threshold,
            remove_non_matched_region = remove_non_matched_region,
            success_threshold = success_threshold)

        if successful_stitching: # Reset the indices
            idx0, idx1 = label_combinations[i][0], label_combinations[i][1]
            break
    
    if successful_stitching: 
        # Two images are currently successfully stitched. Loop thorugh the rest

        if idx0 not in temp_success: temp_success.append(idx0)
        if idx1 not in temp_success: temp_success.append(idx1)

        loop_through = list(group_of_particle_labels.copy())
        for pidx in temp_success: loop_through.remove(pidx)
            
        continue_stitching = True
        terminate = False 
        # Loop through all non-stitched particles TWICE: 
        # (i) end of first loop: terminate = True
        # (ii) end of second loop: terminate = True -> continue_stitching = False
        while continue_stitching: 
        
            for pidx in tqdm(loop_through, position = 1, leave = False, disable = ~looping_progressbar):
                
                # Get the new particle image 
                #      group_of_particle_images[np.where(group_of_particle_labels == idx0)[0][0]]
                img0 = group_of_particle_images[np.where(group_of_particle_labels == pidx)[0][0]].astype(float)

                minI = min([minI, np.min(img0)]) 
                maxI = max([maxI, np.max(img0)])

                # Identify the particle's position
                loc0, IM_IDX0 = _identify_a_particles_position_wrt_SEM_image_edges(
                    pidx,
                    particle_map.data
                )

                # Make the images' shapes compatible
                pim0, pim1 = _copy_images_into_common_array_shape(img0, img1)

                sem_img0 = navigation_signal.inav[IM_IDX0[1],IM_IDX0[0]].data
                region_img0 = particle_map.inav[IM_IDX0[1],IM_IDX0[0]].data
                
                # Get data for im1 as im0:
                im0 = _crop_out_a_specified_particle_area_from_SEM_image(
                    sem_img0, 
                    region_img0 == pidx,
                    get_shape = pim0.shape)
                
                # Pad the images
                pad = np.max(im0.shape)
                img0_part = np.pad(im0.astype(float).copy(), pad_width = pad).astype(float)
                img1_part = np.pad(pim1.astype(float).copy(), pad_width = pad).astype(float)

                # Attempt stitching:
                shifted_image, shift = _images.phase_cross_correlate_images(img0_part, img1_part)

                # If image shifting was a success:
                if _assess_successful_stitching(shift):
                    # Create masks to remove the additional image information not part of the particles
                    
                    mask_shift = _get_rectangle_mask_after_image_shift(shifted_image,
                                                                                 top_hat_filter = True)
                    mask_im = _get_rectangle_mask_after_image_shift(img1_part)
                    overlap_mask = mask_shift * mask_im
                    sum_mask = ((mask_shift.astype(int) + mask_im.astype(int)) > 0).astype(int)
                    diff = (bfh(sum_mask) - sum_mask).astype(bool)
                    sum_mask[diff] = True
                    #mask_shift[np.where(diff)] = True
                    mask_im[diff] = True
                    
                    rect_mask = _rectangulerize_mask(overlap_mask)

                    # Assess whether the shifted and the stationary images' overlapping regions match
                    if rect_mask.sum() > 0:
                        
                        overlap_score = _utils.ncc(
                            _remove_empty_edges_from_image(img1_part*overlap_mask),
                            _remove_empty_edges_from_image(shifted_image*overlap_mask))
                        
                        rect_score = _utils.ncc(
                            _remove_empty_edges_from_image(img1_part*rect_mask),
                            _remove_empty_edges_from_image(shifted_image*rect_mask))
                        
                        score = overlap_score * rect_score

                    else: score = 0

                    if score >= success_threshold:
                    
                        stitched_image = shifted_image.copy()
                        stitched_image[mask_im] = img1_part[mask_im]
                        stitched_image *= sum_mask # Remove artefacts from image shifting ...
                        #stitched_image = _image_utils._remove_empty_edges_from_image(stitched_image)

                        # Create a mask covering only the initial particles
                        p_mask = np.zeros_like(stitched_image, bool)
                        for pim in [img0, img1]: 
                            p_mask += _get_single_particle_map(stitched_image, pim)

                        # If only keep the original particle images in the stitched image:
                        if remove_non_matched_region: 
                            # Mask out the original images:
                            masked_stitched_image = (p_mask * stitched_image)
                        
                        else:
                            
                            p_mask = _rectangulerize_mask(p_mask)
                            
                            masked_stitched_image = (p_mask * stitched_image)

                        img1 = _remove_empty_edges_from_image(masked_stitched_image)
                        
                        temp_success.append(pidx) # Store the successfully stitched particles

                        # If re-stitching has succeeded, remove these from the not-stitched list
                        if pidx in temporarily_not_stitched: temporarily_not_stitched.remove(pidx) 

                    else:
                        # Store none-stitched particle images
                        if pidx not in temporarily_not_stitched: 
                            temporarily_not_stitched.append(pidx) 

                else: 
                    
                    if pidx not in temporarily_not_stitched: 
                        temporarily_not_stitched.append(pidx) 

            # Check if looping is done once or twice:
            if len(temporarily_not_stitched) > 0:

                # Don't bother looping through more than twice
                if terminate: 

                    # Unsuccessfully stitched particles in a cluster might be due to little specimen shift. 
                    # To account for this, the single particles can be concatenated with the successfuilly 
                    # stitched particle images. 
                    """if directly_stitch_edge_particles and len(temporarily_not_stitched) == 1:
                        
                        # Concatenate the images 
                        img1_ = _concatenate_a_stitched_image_with_a_non_stitched_image(
                            temporarily_not_stitched[0],
                            temp_success.copy(),
                            img1.copy(),
                            particle_map.data,
                            navigation_signal.data,
                            to_shape = img1.shape
                        )

                        if len(img1_.shape) > 1: 
                            
                            img1 = img1_
                            
                            temp_success.append(temporarily_not_stitched[0])

                        #else: 
                            
                        #    unsuccessfully_stitched.append(np.array(temporarily_not_stitched))
                    
                    elif directly_stitch_edge_particles and len(temporarily_not_stitched) == 2:
                        
                        # Try stitching the two non-stitched particles and concatenate the stitched image with
                        # the prviously stitched image
                        
                        img2_, successful_stitching2_ = _stitch_two_particle_images(
                            img0 = group_of_particle_images[np.where(group_of_particle_labels == temporarily_not_stitched[0])[0][0]], 
                            img1 = group_of_particle_images[np.where(group_of_particle_labels == temporarily_not_stitched[1])[0][0]], 
                            particle_label1 = temporarily_not_stitched[0], 
                            particle_label2 = temporarily_not_stitched[1],
                            particle_map = particle_map,
                            navigation_signal = navigation_signal,
                            shift_threshold = shift_threshold,
                            remove_non_matched_region = remove_non_matched_region,
                            success_threshold = success_threshold)

                        if successful_stitching2_:

                            loc1_, IM_IDX1_ = _identify_a_particles_position_wrt_SEM_image_edges(
                                temporarily_not_stitched[0],
                                particle_map.data)

                            loc2_, IM_IDX2_ = _identify_a_particles_position_wrt_SEM_image_edges(
                                temporarily_not_stitched[1],
                                particle_map.data)

                            # Concatenate the two images:
                            axis = None
                            if (loc1_[0] in [0,1]) and (loc2_[0] in [0,1]): axis = 1 # Remove padding top/bottom
                            elif (loc1_[1] in [0,1]) and (loc2_[1] in [0,1]): axis = 0 # Remove padding left/right
                            else: axis = -1

                            img1_loc = _identify_a_particles_position_wrt_SEM_image_edges(
                                    temp_success[0],
                                    particle_map.data)[0]
                            
                            for _pidx_ in temp_success[1:]:
                                img1_loc_ = _identify_a_particles_position_wrt_SEM_image_edges(
                                    _pidx_,
                                    particle_map.data)[0]
                                
                                if img1_loc_ != img1_loc: 
                                    # terminate image concatenation if the previously stitched image contains 
                                    # particle images from different SEM images
                                    axis = -1 

                            if axis != -1:
                                
                                pim1, pim2 = _copy_images_into_common_array_shape(img1, img2_)
                                pim2 = _remove_padding_along_axis(pim2, axis = axis)
                                
                                img1 = _concatenate_two_edge_images(img1, pim2, img1_loc_, loc1_)
                                
                                temp_success.append(temporarily_not_stitched[0])
                                temp_success.append(temporarily_not_stitched[1])
                                temporarily_not_stitched = [] # empty the list"""

                            #else: unsuccessfully_stitched.append(np.array(temporarily_not_stitched))
                                
                    #else: unsuccessfully_stitched.append(np.array(temporarily_not_stitched))
                    
                    continue_stitching = False

                # Set the terminating argument to loop through a second time
                terminate = True 

                loop_through = temporarily_not_stitched # Re-define the looping list

            else: continue_stitching = False # Terminate if all were successfully stitched
        
        img1 = rescale_intensity(img1, out_range = (minI, maxI))

        return img1, temp_success, temporarily_not_stitched
    
    else: 
        
        # If no particles in the cluster were successfully stitched:
        return np.array([]), [], group_of_particle_labels
    
def _check_two_stitched_particle_images_concatenation_compatibilities(
    list_of_labels0 : list | np.ndarray, 
    list_of_labels1 : list | np.ndarray, 
    particle_map : np.ndarray):
    """Iterate through loc0 and loc1 to see if two individually stitched particle images
    can be concatenated or not

    Parameters
    list_of_labels0, list_of_labels1
        List of particle labels found in the particle_map array
    particle_map
        Map of particle labels

    Returns
    -------
    compatible
        Whether or not the list of labels are compatible for concatenation
    axis
        Suggested concatenation axis
    """

    success = False
    axis = -1
    
    list_of_image_locations0, list_of_image_locations1 = [], []
    
    for p0 in list_of_labels0:
        list_of_image_locations0.append(np.array(_identify_a_particles_position_wrt_SEM_image_edges(
            p0, particle_map)[0]))
        
    for p1 in list_of_labels1:
        list_of_image_locations1.append(np.array(_identify_a_particles_position_wrt_SEM_image_edges(
            p1, particle_map)[0]))

    list_of_image_locations0 = np.asarray(list_of_image_locations0)
    list_of_image_locations1 = np.asarray(list_of_image_locations1)

    # if one of the "columns" are all unequal to -1 (== not at vertical or horisontal edge)
    if (list_of_image_locations0[:,0] != -1).all() or (list_of_image_locations0[:,1] != -1).all():
        if (list_of_image_locations0[:,0] == 0).all() or (list_of_image_locations0[:,0] == 1).all():
            success = True
            #axis = 0
            loc0 = list_of_image_locations0[0]
        elif (list_of_image_locations0[:,1] == 0).all() or (list_of_image_locations0[:,1] == 1).all():
            success = True
            #axis = 1
            loc0 = list_of_image_locations0[0]
        else: success = False
    else: return False, ((-1,-1), (-1,-1)), -1
    
    if (list_of_image_locations1[:,0] != -1).all() or (list_of_image_locations1[:,1] != -1).all():
        if (list_of_image_locations1[:,0] == 0).all() or (list_of_image_locations1[:,0] == 1).all():
            success = True
            #axis = 0
            loc1 = list_of_image_locations1[0]
        elif (list_of_image_locations0[:,1] == 0).all() or (list_of_image_locations0[:,1] == 1).all():
            success = True
            #axis = 1
            loc1 = list_of_image_locations1[0]
        else: success = False
    else: return False, ((-1,-1), (-1,-1)), -1

    if success:

        if loc0[0] in [0,1] and loc1[0] in [0,1]: # both are top/btm
            if loc0[1] in [0,1] and loc1[1] in [0,1]: # both are left/right
                axis = 1
            else: axis = 0
        else: # left-right
            axis = 1

        return success, (tuple(loc0), tuple(loc1)), axis
        
    else: return False, ((-1,-1), (-1,-1)), -1
    
    

        






