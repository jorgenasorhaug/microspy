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
import warnings, os
from tabulate import tabulate

from . import exceptions as _errors
from ._utils import _utils, _io

from . import material
element_dict = material.elements

VENDORS = [
    "Jeol", "jeol",
]

# https://pythonforundergradengineers.com/unicode-characters-in-python.html
GREEK_LETTERS = {
    # lowercase
    "alpha" : "\u03B1",
    "beta" : "\u03B2",
    "gamma" : "\u03B3",
    "delta" : "\u03B4",
    "epsilon" : "\u03B5",
    "zeta" : "\u03B6",
    "eta" : "\u03B7",
    "theta" : "\u03B8",
    "iota" : "\u03B9",
    "kappa" : "\u03BA",
    "lambda" : "\u03BB",
    "mu" : "\u03BC",
    "nu" : "\u03BD",
    "xi" : "\u03BE",
    "omicron" : "\u03BF",
    "pi" : "\u03C0",
    "rho" : "\u03C1",
    "zeta" : "\u03C2",
    "sigma" : "\u03C3",
    "tau" : "\u03C4",
    "upsilon" : "\u03C5",
    "phi" : "\u03C6",
    "chi" : "\u03C7",
    "psi" : "\u03C8",
    "omega" : "\u03C9",

    # uppercase
    "Alpha" : "\u0391",
    "Beta" : "\u0392",
    "Gamma" : "\u0393",
    "Delta" : "\u0394",
    "Epsilon" : "\u0395",
    "Zeta" : "\u0396",
    "Eta" : "\u0397",
    "Theta" : "\u0398",
    "Iota" : "\u0399",
    "Kappa" : "\u039A",
    "Lambda" : "\u039B",
    "Mu" : "\u039C",
    "Nu" : "\u039D",
    "Xi" : "\u039E",
    "Omicron" : "\u039F",
    "Pi" : "\u03A0",
    "Rho" : "\u03A1",
    "Sigma" : "\u03A3",
    "Tau" : "\u03A4",
    "Upsilon" : "\u03A5",
    "Phi" : "\u03A6",
    "Chi" : "\u03A7", 
    "Psi" : "\u03A8",
    "Omega" : "\u03A9",
    "Theta" : "\u03F4"
}

def _vendor2ImAquisitionOrder(vendor : str | None):
    """Return the image acquisition order according to vendor.

    Parameters
    ----------
    vendor 
        Vendor name

    Returns
    -------
    horisontal, verticals
        acquisition directions

        horisontal directions : ("r2l", "l2r")
        vertical directions : ("t2b", "b2t")
    """
    horisontal, vertical = "l2r", "t2b"
    vendor = str(vendor).lower()
    
    if isinstance(vendor, str):
        if vendor not in VENDORS:
            raise AttributeError(f"Vendor {vendor} is not recognised. "
                                 f"Allowed vendor arguments are {VENDORS}.")

        if vendor.lower() == "jeol":
            horisontal = "r2l"
    else:
        _errors.formatted_warning(
            "Assuming 'r2l' and 't2b' acquisition order."
        )
    return horisontal, vertical 
    

def _vendor2ImFlipAxes(vendor : str):
    """Return the likely image acquisition order according to 
    vendor.

    Parameters
    ----------
    vendor 
        Vendor name

    Returns
    -------
    image_orders 
        acquisition directions

        horisontal directions : ("r2l", "l2r")
        vertical directions : ("t2b", "b2t")
    """
    if isinstance(vendor, str):
        if vendor.lower() == "jeol":
            return (1,)
    else:
        return None, None
    
def tabulate_data(
    data : np.ndarray, 
    headers : list | tuple, 
    labels : str | list | tuple | np.ndarray = "", 
    decimals : int = 2,
    unit : str | None = None,
    return_table : bool = False
):
    """Nice printing function to displau particles' property like chemical 
    composition or geometry.
    
    Parameters
    ----------
    data
        Data to be printed.
    headers
        List of headers printed at the first row. The number of 
        headers should match data.shape[0].
    labels
        List of labels printed at the left column.
        The number of labels should match data.shape[1]
    decimals
        Number of decimals to print. 2 by default.
    unit
        data unit printed at the upper left cell, above the
        labels.
    return_table
        Whether to return the table, False by default.

    Returns
    -------
    table
        numpy ndarray with the table content (excluding 
        headers and labels).
        
    Examples
    --------
    >>> tabulate_data(
            data : (N,m) data array
            labels : (N,) list of labels, e.g. class names
            headers : (m,) list of headers, e.g. elements
        )
    """ 
    
    data_shape = data.shape
    ndim = np.ndim(data)
    _data = data.copy()
    
    if ndim == 2:
        
        # Set labels
        if isinstance(labels, str): 
            if labels == "": # Empty string:
                labels = np.arange(0, data_shape[0])
            else: # Single string:
                labels = np.repeat(labels, data_shape[0])
        
        if data_shape[0] == len(labels) and data_shape[1] == len(headers):

            # Insert labels in the 0th column:
            
            table = _get_table(
                data = _data, 
                decimals = decimals, 
                labels = labels
            )

            # Update headers:
            heads = headers.copy()
            heads.insert(0, "Labels\\")
            if unit is not None:
                maxChar = max([len(unit), len("Labels")])
                minChar = min([len(unit), len("Labels")])-1
                insert_string = f"{" "*minChar}\\{unit}\n"
                insert_string += f"{heads[0]}{" "*maxChar}"
                heads[0] = insert_string
                
            # If only return the table:
            if return_table: 
                return table 
            
            print(tabulate(
                tabular_data = table, 
                headers = heads, 
                tablefmt="pretty")
                 )

        else: 
            
            print(f"The data shape {data_shape} doesn't fit the "
                  f"header ({len(headers)}) and/or labels ({len(labels)}) "
                  "shape(s)")
    
    else: 
        print(f"Data must have two dimensions.")

def save_tabulate_data(
    data : np.ndarray, 
    headers : list | tuple, 
    labels : str | list | tuple | np.ndarray = "", 
    decimals : int = 2,
    unit : str | None = None,
    filename : str = 'tabulated.txt', 
):
    """Save tabulated data to a specified format stated by the filename.

    Parameters
    ----------
    data
        Data to be printed. The data is expected to fit the 
        shape (len(header), len(label))
    header
        List of headers : will be printed at the top of each column
    labels
        List of labels : will be printed at the left of each row 
    decimals
        Number of decimals to save
    unit
        Unit of the printed property.    
    filename
        Name of file. By default: txt format.
        Allowed formats: txt and csv.
    """

    ALLOWED_EXTENSIONS = [
        "txt",
        "csv"
    ]
    
    file_type = os.path.splitext(filename)[-1][1:]
    folder, filename = os.path.split(filename)

    if file_type not in ALLOWED_EXTENSIONS:
        raise AttributeError(
            f"File type {file_type} not recognised "
            "or supported yet.")

    table = tabulate_data(
        data = data,
        headers = headers, 
        labels = labels, 
        decimals = decimals,
        unit = unit,
        return_table = True,
    )

    # Update headers
    heads = headers.copy()
    heads.insert(0, "Label\\")
    if unit is not None:
        maxChar = max([len(unit), len("Label")])
        minChar = min([len(unit), len("Label")])-1
        insert_string = f"{" "*minChar}\\{unit}\n"
        insert_string += f"{heads[0]}{" "*maxChar}"
        heads[0] = insert_string

    """
    ... file_writer ...
    """
    
    if file_type == "txt":
        _io._save_tabulated_data_as_txt(
            table = table,
            headers = heads,
            path = folder,
            filename = filename
        )   
        
    elif file_type == "csv":
        _io._save_tabulated_data_as_csv(
            table = table,
            headers = heads,
            path = folder,
            filename = filename
        )

def _get_table(
    data : np.ndarray,
    labels : list | tuple,
    decimals : int = 2
):
    """Structure data and labels to fit :func:'tabulate.tabulate'.
    
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
    
    _data = np.round(
        #Force to float64 as "lower" dtypes can not be represented 
        #exactly in binary, i.e. we avoid extra digits:
        data.astype(np.float64),  
        decimals = decimals
    )
    return np.insert(_data.astype(object), obj = 0, values = labels, axis = 1)

def _get_prime_numbers(
    N : int
) -> list:
    """Factorise N and return the prime numbers.

    Parameters
    ----------
    N 
        Integer to factorise

    Returns
    -------
    prime_factors
        List of prime numbers
    """
    from sympy import factorint
    
    primes = factorint(N)
    prime_factors = []
    
    for key, val in primes.items():
        for i in range(val):
            prime_factors.append(key)
            
    return prime_factors

def _map_factor_pairs(
    prime_factors : list | tuple
) -> set[tuple, ...]:
    """Generate all unique factor pairs (a, b), (b, a) from a list of 
    prime factors.

    Parameters
    ----------
    prime_factors
        List of prime factors.
    pairs
        A set of pairs.

    Returns
    -------
    pairs
        set of unique factor pairs.
    """
    from math import prod
    from itertools import combinations
    
    N = prod(prime_factors)
    pairs = set()

    n = len(prime_factors)

    for r in range(n + 1):
        for idx in combinations(range(n), r):

            a = prod(prime_factors[i] for i in idx)
            b = N // a

            pairs.add(tuple((a, b)))
            pairs.add(tuple((b, a)))

    return pairs

def guess_ParentSig_navigation_grid_shape(
    n_images : int, 
    image_width : int | float,
    image_height : int | float,
    stitched_image_width : int | float, 
    stitched_image_height : int | float
) -> [int, int, list]:
    """The function maps potential grid shapes that fits the number of
    images, image width and height, and the 

    The function does not take overlap between images into consideration.

    Parameters
    ----------
    n_images
        Total number of images
    image_width, image_height
        Width, height of the image
    tot_area_width, tot_area_height
        Total width and height if the images were correctly stitched together.

    Returns
    -------
    rows, cols
        The best row, col match
    candidates
        All tested candidates and corr. scores determined by the ratio between tested 
        grid shape area and the actual area.
    """
    
    expected_ratio = stitched_image_width / stitched_image_height
    
    # Pairs of prime numbers:
    factor_pairs = _map_factor_pairs(
        _get_prime_numbers(
            N = n_images
        )
    )
    
    candidates = []

    for row, col in factor_pairs:
        ratio = (col * image_width) / (row * image_height)

        error = ratio / expected_ratio
        
        candidates.append((error, row, col))

    scores = np.asarray(candidates)[:,0]
    best_match = min(
        range(len(scores)), key=lambda i: abs(scores[i] - 1.0)
    )

    rows, cols = candidates[best_match][1:]
    
    return rows, cols, candidates



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%% KEEP OR DELETE FUNCTIONS? %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def plot_data_distribution(data_arr,
                           unit = '',
                           x_tick = '',
                           colors = [np.asarray([21,106,163]) / 255, # violin
                                     np.asarray([175,204,184])#191,187,152]) 
                                                     / 255, # boxplot
                                     np.asarray([206,156,168]) / 255], # scatter
                           return_fig = False):
    """Plot a box + violin + scatter plot of data_arr

    Parameters
    ----------
    data_arr
        np.ndarray of shape (N,)
    """
    
    import matplotlib.pyplot as plt

    # --- Temporary change using rc_context() ---
    with plt.rc_context({'lines.linewidth': 3, 'font.size' : 16}):
    
        fig, axs = plt.subplots(figsize=(10,10))

        boxprops = dict(linestyle='-', linewidth=2, color='k')
        
        medianprops = dict(linestyle='-', linewidth=2, color='k')
        
        bplot = axs.boxplot(data_arr, patch_artist=True, boxprops=boxprops, medianprops=medianprops)
        
        vp1 = axs.violinplot(data_arr, showmeans=False, showmedians=False, side = 'high', showextrema=False)

        for pc in vp1['bodies']:
            pc.set_facecolor(colors[0])
            pc.set_edgecolor('black')
        
        axs.set_ylabel(unit)
        
        for patch, color in zip(bplot['boxes'], [colors[1]]): 
            patch.set_alpha(0.8)
            patch.set_facecolor(color)
        
        x_arr = np.random.randint(low = 95, high = 105, size = len(data_arr)) / 100
        
        scatter = axs.scatter(x_arr, data_arr, color=colors[2], marker='o', zorder=5, alpha=.35)
        
        # Legends
        axs.legend([bplot["boxes"][0], vp1['bodies'][0], scatter], 
                   ['Box plot', 'Violin plot', 'Data pts.'], 
                   loc='upper right')

        plt.xticks([1], [x_tick])

        plt.show()

    if return_fig: return fig

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
    from hyperspy.api import model
    from hyperspy.signals import Signal1D
    
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