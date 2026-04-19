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
# This code is inspired by exspy : ~exspy._misc.elements.py


import numpy as np
from hyperspy.api import model
from src.microspy._misc import material
from hyperspy.signals import Signal1D
import warnings, os
from src.microspy._misc import exceptions as _errors

element_dict = material.elements

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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% UTILITIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
import os
import ast

def get_standalone_functions(file_path):
    """Parses a .py file and returns a list of top-level function names."""
    functions = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            node = ast.parse(f.read())
            # Only extract functions at the top level of the module (node.body)
            # This automatically skips methods inside classes
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append(item.name)
    except Exception as e:
        return [f"Error parsing: {str(e)}"]
    return functions

def generate_structure_md(root_dir, output_file="PROJECT_STRUCTURE.md"):
    """Traverses directories and builds a Markdown file of the structure."""
    md_lines = [f"# Project Structure: {os.path.basename(root_dir)}\n"]
    
    for root, dirs, files in os.walk(root_dir):
        # Calculate indentation level for the directory
        level = root.replace(root_dir, '').count(os.sep)
        indent = "  " * level
        folder_name = os.path.basename(root) or root_dir

        if folder_name not in (".ipynb_checkpoints",
                              "__pycache__"):
            md_lines.append(f"{indent}- 📁 {folder_name}/")
        
        for file in sorted(files):
            if file.endswith(".py") and file not in ("__init__.py",
                                                    "__init__.pyi") and "checkpoint" not in file:
                file_indent = "  " * (level + 1)
                md_lines.append(f"{file_indent}- 📄 {file}")
                
                # Extract and list functions under the file
                funcs = get_standalone_functions(os.path.join(root, file))
                for func in funcs:
                    func_indent = "  " * (level + 2)
                    md_lines.append(f"{func_indent}- `f` {func}()")
                    
    # Save to file
    content = "\n".join(md_lines)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Structure successfully saved to {output_file}")
    print(content)

def print_csv_metadata(csv_file):
    """Print the stored metadata in the particle analysis *.csv file

    Parameters
    ----------
    csv_file
        Either the csv filename used to load the data, or the pandas DataFrame object
    """
    
    if type(csv_file) == str: csv_file = pd.read_csv(str(csv_file))

    csv_keys = list(csv_file.keys())

    project_name = csv_keys[1]

    first_col, second_col = csv_file['Project name'], csv_file[project_name]

    # second_col[~first_col] is to avoid a vertical shift
    first_col, second_col = first_col[~first_col.isnull()], second_col[~first_col.isnull()] 

    # Replace nan == empty cell with empty string
    second_col[second_col.isnull()] = ''
    
    for first, second in zip(first_col, second_col): print(f"{first:35}{second:<20}")


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%% PRINTING PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%

def tabulate_particles_property(data, header, label = '', return_table = False):
    """Print selected particle's property like chemical composition or geometry
    
    Parameters
    ----------
    data
        Data to be printed. The data is expected to fit the shape (len(header), len(label))
    label
        List of labels : will be printed at the left of each row 
    header
        List of headers : will be printed at the top of each column

    Returns
    -------

    Example:
    >>>print_particles_property(data = (n,m) array of data, like (num elements, num particles)
                             label = (n,) list of labels, like class names
                             header = (m,) list of headers, like list of elements
    >>>
    """ 

    data_shape = data.shape

    if len(data_shape) <= 2:
        
        if type(label) == str: 
            
            if label == '': label = np.arange(0, data_shape[1])
    
        if data_shape[0] == len(header) and data_shape[1] == len(label):
        
            table = _utils._get_table(np.round(data, decimals = 2), label)

            if return_table: return table #tabulate(table, header, tablefmt="pretty") 
            
            else: print(tabulate(table, header, tablefmt="pretty"))

        else: print(f"The data shape ({data_shape}) doesn't fit the header ({len(header)}) and/or label ({len(label)}) shape(s)")
    
    else: print(f"Data of shape {data_shape} doesn't fit the table.")

def save_tabulated_particles_property(data, header, label = '', path = '', filename = 'tabulates.txt'):
    """Save the tabulated data in folder tabulates
    
    Parameters
    ----------
    data, label, header: see print_particles_property

    path
        String path to store the tabulated data. 

    filename
        Name of file being saved

    Note! The data will be stored in a filename called tabulates.txt.
    """

    folder = path

    save_results = False

    if folder[-1] != '\\' or folder[-1] != '/': folder += '\\'

    if os.path.exists(folder): save_results = True

    else: 
        
        ans = input(f"Couldn't find path {path}.\nCreate path? (y/[n])")

        if ans.upper() == 'Y' or ans == '':

            print(f"Creating path {path}")
            
            os.mkdir(path)

            save_results = True

    if save_results:

        if header == '': header = np.arange(0, np.shape(data)[1])

        table = _utils._get_table(data, label)

        filename = f"{os.path.splitext(filename)[0]}.txt"
        
        with open(folder + filename, 'w') as f:  f.write(tabulate(table, header, tablefmt="pretty"))



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% IMAGE MANIPULATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def stitch_images(images, 
                  navigation_shape, 
                  horisontal_direction = 'r2l', 
                  vertical_direction = 't2b'):
    """Stitch an array of images into a 2D image. 

    Parameters
    ----------
    images
        Array with images to stitch. Expected shape: 4D (2 navigation, 2 image) dimensions
    navigation_shape 
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
    if not _errors._check_for_numpy_ndarray(images): 
        
        raise TypeError(f'Argument data_array of type {type(images)} is not a numpy ndarray')
    
    if not len(navigation_shape) == 2: 

        if len(navigation_shape) == 1: navigation_shape = (1, navigation_shape[0])
        
        else: raise ValueError(f"Provided image shape {navigation_shape} is invalid. Provide a two dimensional shape.")

    if images.shape[:2] == navigation_shape:

        images = _image_utils._gridify_4D_array_to_3D(images)

    return _image_utils._stitch_images(images, 
                                  navigation_shape, 
                                  horisontal_direction = horisontal_direction, 
                                  vertical_direction = vertical_direction)

def stitch_rgb_phase_map(phase_map, nav_shape):
    """Stitch a 4D or 5D array with 3 rgb channels into a single image of shape 3: (x,y,channels).

    Parameters
    ----------

    Returns
    -------
    
    Example
    -------
    """
    if not _errors._check_for_numpy_ndarray(phase_map): raise TypeError('Provided phase map must be a numpy ndarray')

    if len(nav_shape) != 2: raise ValueError(f"Provided navigation shape {nav_shape} is not valid.")
    
    phase_map_s = phase_map.copy()

    # 20*20 != 20
    if np.prod(nav_shape) == np.prod(phase_map.shape[:2]) and len(phase_map) > 4: 

        phase_map_s = _image_utils._image_utils._gridify_ND_array_to_nD(phase_map_s)

    return _image_utils._stitch_images(phase_map_s, shape = nav_shape)
    
def gridify_3D_array_to_4D(arr, nav_shape):
    """Gridify the 3D array to 4D. Nav_shape defines the number of images in the different directions.

    Parameters
    ----------
    arr
        numpy.ndarray of shape (3,)
    nav_shape
        Navigation shape to shape the array into

    Example
    ------
    >>> import particle_analysis as pa
    >>> import numpy as np
    >>> img = np.asarray([[[0]*4]*4]*4)
    >>> img.shape
    (4,4,4)
    >>> img = pa.gridify_3D_array_to_4D(arr, nav_shape = (2,2))
    >>> img.shape
    (2,2,4,4)
    """
    if len(arr.shape) != 3: raise TypeError(f"Array shape {arr.shape} is not expected.")

    if len(nav_shape) != 2: raise TypeError(f"Navigation shape is not valid. Provide a 2-integer list")

    return _image_utils._gridify_3D_array_to_4D(arr, nav_shape + arr.shape[-2:])
    
def greyscale_to_rgba(grey_image, dtype_out = np.float16):
    """Return a grey-scale array as a rgb equivalent
    
    Parameters
    ----------
    grey_image
        2D grey scale image array 

    Returns
    -------
        2D image array with RGBA channels (RGB is normalised to be in the range 0,1)
    """
    fac = np.max(grey_image)
    
    if not _errors._check_for_numpy_ndarray(grey_image): raise TypeError(f"Input image is not a grey scale.")
    
    img = np.expand_dims(grey_image, axis = -1).astype(dtype_out)
    
    return np.concatenate((img / fac, img / fac, img / fac,
                           np.full_like(img, 1)), # alpha channel
                           axis = -1)

def greyscale_to_rgb(grey_image, 
                     in_range=(0, 255),
                     dtype_out = np.float16):
    """Return a grey-scale array as a normalised rgb equivalent (intensity range: 0,1)
    
    Parameters
    ----------
    grey_image
        2D grey scale image array 
    in_range
        tuple of in range intensity values that is givne to the rescale_intensity function
    dtype_out
        Datatype to return            

    Returns
    -------
        2D image array with RGBA channels (RGB is normalised to be in the range 0,1)
    """
            
    if not _errors._check_for_numpy_ndarray(grey_image): raise TypeError(f"Input image is not a grey scale.")
    
    from skimage import color
    from skimage.exposure import rescale_intensity

    grey_im = color.gray2rgb(grey_image) # Intensity values unchanged
    
    return (rescale_intensity(1.0 * grey_im, in_range = in_range)).astype(np.float32)
    
def plot_rgb_map_with_colorbar(array, 
                               colours, 
                               background_colour = 'whitesmoke',
                               return_fig = False):
    from matplotlib import colors
    import matplotlib.pyplot as plt
    
    auto_colouring  = False

    colour_type = type(colours)

    num_colors = len(colours)
    
    if colours is not None: 

        unique_classes = list(colours.keys())

        if colour_type == dict: colours = [colours[cl] for cl in unique_classes]

    else: auto_colouring = True

    # Create a unique color map
    if auto_colouring: 

        if len(colours) < 11: 
            
            print('Colouring according to tableau colors')
            
            colours = [col for col in list(colors.TABLEAU_COLORS.keys())[:len(unique_classes)]]
        
        else: 
            
            print('Generating random colours')
            # Alternatively, use: colors.CSS4_COLORS
            colours = [_utils._generate_random_rgb_color() for i in range(len(unique_classes))]

    colours.insert(0, colors.to_rgb(background_colour))

    unique_classes.insert(0, 'Matrix')
    
    phase_vals = np.arange(num_colors + 1)
    
    cmap = colors.ListedColormap(colours)
    
    norm = colors.BoundaryNorm(np.arange(-0.5, phase_vals.max() + 1.5, 1), cmap.N)

    # Plotting
    fig, ax = plt.subplots()
    cax = ax.imshow(array, cmap = cmap, norm = norm)
    # Add a colorbar with a label
    cbar = fig.colorbar(cax, ticks = phase_vals)
    
    if colour_type == dict: cbar.ax.set_yticklabels(unique_classes) 
        
    plt.axis('off')
    plt.show()

    if return_fig: return fig


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PARTICLE CHEMISTRY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

def get_label_colourmap(list_of_colours : list | None = None):
    """Create a colourmap for labels.

    Parameters
    ----------
    list_of_colours
        List of pyplot colour names
    num_colours
        Number of colours in the colourmap

    returns
    -------
    colour map
        A matplotlib.color ListedColormap
    """
    return _colouring.get_discrete_colour_map(list_of_colours)

def get_navigator_colours(image):
    """Create a navigator rgb map
    """
    return _colouring.get_rgb_navigator(image)

def get_greek_letter(letter : str):
    """Return the ¨code representing greek letters for nice printing/name setting"""
    letters = {
        "alpha" : "\u03B1",
        "Alpha" : "\u0391",
        "beta" : "\u03B2",
        "Beta" : "\u0392",
        "delta" : "\u03B4",
        "Delta" : "\u0394",
        "mu" : "\u03BC",
        "Mu" : "\u039C",
        "pi" : "\u03C0",
        "Pi" : "\u03A0",
        "sigma" : "\u03C3",
        "Sigma" : "\u03A3",
        "omega" : "\u03C9",
        "Omega" : "\u03A9"
    }

    if letter not in letters.keys():

        raise ValueError(f"The letter {letter} is unrecognised.")

    else: return letters[letter]











        

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