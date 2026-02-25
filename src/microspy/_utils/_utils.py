import numpy as np
from hyperspy.api import model
from src import _errors
from exspy import material
from hyperspy.signals import Signal1D
import warnings

element_dict = material.elements.as_dictionary()

numpy_image_datatypes = [
    np.bool_, np.byte, np.ubyte, 
    np.int_, np.int8, np.int16, np.int32, np.int64,
    np.uint, np.uint8, np.uint16, np.uint32, np.uint64,
    np.float16, np.float32, np.float64
]

"""
def _check_for_numpy_ndarray(array):
    #Helping function to check if array is a numpy array
    return type(array) == np.ndarray

def _check_for_same_shape(array1, array2):
    #Helping function to check if two arrays are of the same shape
    if ~_check_for_numpy_ndarray(array1): raise TypeError(f'First argument of type {type(array1)} is not a numpy array')
    if ~_check_for_numpy_ndarray(array2): raise TypeError(f'Second argument of type {type(array2)} is not a numpy array')
    return array1.shape == array2.shape
"""

def check_array_compatibility_with_new_datatype(array, new_dtype):
    """Check if array values will not be changed if a new data type (new_dtype) set set""" 
    min_val = np.min(array)
    max_val = np.max(array)
    info_dtype = np.iinfo(new_dtype)
    if min_val < info_dtype.min: return False
    if max_val > info_dtype.max: return False
    return True  

def _get_table(data, label):
    """Structure data and labels to fit tabulate's functions
    
    Parameters
    ----------
    data
        Data to be printed
    label
        List of labels : will be printed at the left of each row 

    Returns
    -------
    table 
        List of lists that fits the tabulate functions

    Example
    -------
    >>> values_to_print = np.asarray(([1,2,3],[1,2,3],[1,2,3]))
    >>> values_to_print.shape
    (3, 3)
    >>> labels_to_print = ['row1','row2','row3']
    >>> _get_table(values_to_print, labels_to_print)
    array([['row1', 1, 2, 3],
           ['row2', 1, 2, 3],
           ['row3', 1, 2, 3]], dtype=object)
    """ 

    return np.insert(np.transpose(data.astype(object)), 0, label, axis = 1)

def _generate_random_rgb_color():
    """Generates a random RGB color as a tuple (r, g, b) with values between 0 and 1."""
    r = random.random()
    g = random.random()
    b = random.random()
    return (r, g, b)

def ncc(array, template):
    """Calculates the normalised cross-correlation (NCC) score between an image
    and a template at zero displacement.

    Parameters
    ----------
    array: np.array
        Image with the average subtracted. It is assumed that not all values are equal to zero.
    template: np.array
        Reference image with the average subtracted. It is assumed that not all values are equal to zero.

    Returns
    -------
    normalised cross-correlation value : float
        Normalised cross-correlation score between image and template.
    """
    
    cor=np.sum(array*template)
    
    nor = np.sqrt((np.sum(array**2)))*np.sqrt(np.sum(template**2))
    
    return cor / nor

def _template_match_1d(patterns, templates):#, num_particles):
    """Match patterns with templates using the normalised cross-correlation.

    Parameters
    ---------
    patterns : np.ndarray
        Patterns to match with the templates. 
        OBS! patterns.shape[0] is assumed to be the number of patterns.
    templates : np.ndarray
        Templates to match the patterns with

    Returns
    -------
    ncc : np.ndarray
        Normalised cross-correlation scores between the patterns and the tempaltes.

    Example
    -------
    >>> _template_match(patterns = np.array([[0,2,5,4,3,2], [2,5,9,8,4,6]]), 
                        templates =  np.array([[0,2,5,4,3,2], [2,5,9,8,4,6]])
    
    """
    from tqdm import tqdm

    ndim_template = len(templates.shape)

    ndim_patterns = len(patterns.shape)
    
    if ndim_template == 1: num_templates = 1
        
    else: num_templates = len(templates)

    if ndim_patterns == 1: num_pats = 1
        
    else: num_pats = len(patterns)#num_particles

    matches = np.zeros((num_pats, num_templates))

    for i in tqdm(range(num_templates)):
        
        if ndim_template > 1: tmpl = np.nan_to_num(templates[i])

        else: tmpl = np.nan_to_num(templates)
        
        for j in range(num_pats): 

            if ndim_patterns > 1: pat = np.nan_to_num(patterns[j])

            else: pat = np.nan_to_num(patterns)
            
            matches[j] = ncc(pat, tmpl)
    
    return matches

def _create_dummy_eds_signal(elements, Erange = 20., steps = 0.02, weight = None, return_model = False):
    """Create a dummy EDS signal based on element peaks from elements in elements list

    Parameters
    ----------
    elements : list
        List of elements (e.g. ['Al','Cu'])
    Erange : float
        Energy range in signal
    steps : float
        Energy resolution

    Returns
    -------
    signal : exspy.signals.EDSSEMSpectrum
        Dummy EDS-SEM signal
    """
    #from tqdm.notebook import tqdm
    from tqdm import tqdm 
    
    if weight is not None: 

        weight = np.array(weight)

        if len(elements) != len(weight): raise ValueError(f"The number of elements ({len(elements)}) is not compatible with the shape of X-ray weighting ({weight.shape})")

    #Create dummy signal
    s = Signal1D(np.zeros((int(Erange/steps)), np.float32))

    s.axes_manager[-1].scale = steps

    m = s.create_model()

    lines = []

    # To correctly display a nested loop ...
    # https://stackoverflow.com/questions/56953040/resetting-tqdm-progress-bar/58657862#58657862
    # https://github.com/tqdm/tqdm/issues/1023
    it1 = tqdm(elements, leave = True)
    
    for elem in it1:

        it2 = tqdm(element_dict[elem]['Atomic_properties']['Xray_lines'].keys(), leave = True)#['Ka']['energy (keV)']
            
        for line in element_dict[elem]['Atomic_properties']['Xray_lines'].keys(): 

            lineE = element_dict[elem]['Atomic_properties']['Xray_lines'][line]['energy (keV)']

            if lineE < Erange:

                g = model.components1D.Gaussian()
                
                g.centre.value = lineE

                g.sigma.value = 0.02 * np.exp(lineE / 10)

                g.A.value = element_dict[elem]['Atomic_properties']['Xray_lines'][line]['weight']
                
                if weight is not None: 

                    g.A.value *= weight[elements.index(elem)]

                m.append(g)

                lines.append(elem + '_' + line)

            it2.update()

        it2.refresh()

    it2.close()

    m = m.as_signal()

    m.set_signal_type('EDS_SEM')

    m.axes_manager.signal_axes[0].units = 'keV'

    m.set_lines(lines)
    
    m.add_lines(lines)

    if return_model: return m

    else: m.plot(xray_lines = 'from_elements')

def _create_gaussian(x, amplitude, mean, std_dev):
    """
    Evaluates a Gaussian function at given x-values.

    Args:
        x (numpy.ndarray or float): Input value(s).
        amplitude (float): The peak value of the Gaussian.
        mean (float): The mean (center) of the Gaussian.
        std_dev (float): The standard deviation (width) of the Gaussian.

    Returns:
        numpy.ndarray or float: The corresponding y-value(s) of the Gaussian.
    """
    return amplitude * np.exp(-((x - mean)**2) / (2 * std_dev**2))
    
def _create_dummy_eds_spectra(labelled_image, 
                              elements, 
                              label_concentrations, 
                              bkgr_idx = 0,
                              Erange = 15.0,
                              steps = 0.02):
    """ Create an artificial EDS spectrum with the EDS peaks corresponding to the 
    labelled particle's maximum element concentration in the labelled image. Note 
    that the intensity in the spectrum is weighted according to the chemical 
    composition, but is not directly readable.

    Parameters
    ----------
    labelled_image 
        Labelled 2D array 
    elements
        List of elements
    label_concentration
        Array of element concentrations

    Returns
    -------
    1D hyperspy signal with a dummy spectrum
    """
    
    E_values = np.linspace(0, Erange, int(Erange / steps))
    
    if len(elements) != np.shape(label_concentrations)[0]: 
        
        raise _errors.ShapeError(f'Invalid shape between elements argument ({len(elements)},) and label_concentrations argument ({np.shape(label_concentrations)})', errors = None)
    
    signal = np.zeros(labelled_image.shape + (int(Erange/steps),), dtype = np.uint16)

    uniques = np.unique(labelled_image)
    
    labels = np.delete(uniques, np.where(uniques == bkgr_idx))
   
    lines = []
    
    for label_idx, idx in zip(labels, np.arange(len(labels))):
    
        spectrum = np.zeros_like(E_values)
        
        for elem in elements:

            for line in element_dict[elem]['Atomic_properties']['Xray_lines'].keys(): #['Ka']['energy (keV)']
    
                lineE = element_dict[elem]['Atomic_properties']['Xray_lines'][line]['energy (keV)']
    
                if lineE < Erange:

                    # "width"
                    sigma = 0.05 * np.exp(lineE / 10)

                    # "height"
                    amplitude = element_dict[elem]['Atomic_properties']['Xray_lines'][line]['weight']

                    # Scaled according to the particle concentratoin
                    amplitude *= (label_concentrations[elements.index(elem), idx] // 4)
    
                    lines.append(elem + '_' + line)

                    mean = lineE
                    
                    spectrum += _create_gaussian(E_values, amplitude, mean, sigma)

        signal[np.where(labelled_image == label_idx)] = spectrum

    return signal, lines

def _reshape_artificial_eds_map(array, nav_shape):
    """Given an array of shape (num_images, SEM_image_shape, X_ray_spectrum), the function will return a stitched signal according to nav_shape (i.e. number of images in x- and y-direction).
    """
    #from tqdm.notebook import tqdm
    from tqdm import tqdm
    
    it1 = tqdm(np.arange(nav_shape[0]), leave = True)
    it2 = tqdm(np.flip(np.arange(nav_shape[1])), leave = True)
    
    for i in it1:

        for j in np.flip(np.arange(nav_shape[1])):

            idx = i*nav_shape[1]+j

            if np.mod(j+1, nav_shape[1]) == 0: sig = array[:,:,idx,:]

            else: sig = np.concatenate([sig, array[:,:,idx,:]], axis = 1)

            it2.update()

        if i == 0: SIGNAL = sig

        else: SIGNAL = np.concatenate([SIGNAL, sig], axis = 0)

        it2.refresh()

        if i != len(np.arange(nav_shape[0]))-1: it2.reset(total=len(np.flip(np.arange(nav_shape[1]))))

    it2.close()

    return SIGNAL
                
def _check_cropped_particle_class_compatability(classes, clusters):
    """The function looks for class compatability within the cluster labels (starting at 1). 

    Parameters
    ---------
    classes
        np.array of class names (string)
    clusters
        list of np.arrays keeping track of which particle labels are clustered together

    Returns
    -------
    updated_clusters
        list of np.arrayskeeping track of which particle labels are clustered together. However,
        the list might be updated according to classes found within the cluster.

    Example
    -------
    >>> classes 
    array(['Unclassified', 'Unclassified', 'Type A', 'Type A', 'Type B', 
           'Type B'], dtype='<U12')

    >>> # The cluster labeles' value corr. to the classes index position [label value - 1]  
    >>> clusters
    [array([1, 2]), array([3, 4, 5, 6])] 
    
    >>> _check_cropped_particle_class_compatability(classes, clusters)
    [array([0, 1]),
     array([2, 3)], 
     array([4, 5])]    
    """
    
    updated_clusters = []
    
    for clust, cluster_idx in zip(clusters, np.arange(len(clusters))):
        
        clust_classes = []
        
        for idx in clust: clust_classes.append(classes[idx-1])
        
        clust_classes = np.array(clust_classes)
        
        if len(np.unique(clust_classes)) == 1: 
            
            updated_clusters.append(clusters[cluster_idx])
        
        else:
            
            for cl in np.unique(clust_classes): updated_clusters.append(clusters[cluster_idx][clust_classes == cl])

    return updated_clusters

def get_number_combinations(array_of_different_numbers):
    """Return an array of unique number combinations from the argument array
    """
    from itertools import combinations
    
    return np.array(list(combinations(array_of_different_numbers, 2)))

def norm_data(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    mean_data=np.mean(data)
    std_data=np.std(data, ddof=1)
    #return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data-mean_data)/(std_data)


def ncc(data0, data1):
    """
    normalized cross-correlation coefficient between two data sets

    Parameters
    ----------
    data0, data1 :  numpy arrays of same size
    """
    return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))


def _identify_missing_labels(current_labels, desired_labels):
    """Return True if there are incompatibilities between the two input arrays. Return also the missing numbers.
    """
    
    missing_labels = []
        
    for i in range(len(current_labels) - 1):

        diff = current_labels[i+1] - current_labels[i]
        
        if diff > 1:
            
            for j in range(1, diff): 
                
                missing_labels.append(current_labels[i]+j)

    if len(missing_labels) > 0:
        
        warnings.warn(f"\n{len(missing_labels)} labels have either been missed or removed while mapping particles. The missing labels are {missing_labels}.")

    return missing_labels

def first_nonzero_decimal_position(n):
    """
    Finds the position of the first non-zero decimal digit.
    The position starts counting from 1 after the decimal point.
    Returns None if all decimals are zero (or the number is an integer).
    """
    s = str(n)
    if '.' not in s:
        return None
    
    # Split the string by the decimal point
    decimal_part = s.split('.')[1]
    
    # Iterate through the decimal part to find the first non-zero digit
    for index, digit in enumerate(decimal_part):
        if digit != '0':
            # Position is the index + 1 (since index starts at 0)
            return index + 1
            
    return None