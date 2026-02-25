import pandas as pd
import os, warnings, re
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src._utils import check_array_compatibility_with_new_datatype

numpy_image_datatypes = [
    np.bool_, np.byte, np.ubyte, 
    np.int_, np.int8, np.int16, np.int32, np.int64,
    np.uint, np.uint8, np.uint16, np.uint32, np.uint64,
    np.float16, np.float32, np.float64
]


def get_module_function_names(module):
    """Return a list of function names from module"""
    import inspect
    function_names = []
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            function_names.append(name)
    return function_names

def _check_csv_filename(filename) -> bool:
    """Helper function to check if the filename can be read"""
    
    is_file = os.path.isfile(filename)

    read_filetype = os.path.splitext(filename)[-1]
    
    if is_file and read_filetype == '.csv' and 'Project name' in pd.read_csv(filename).keys(): return True 
    
    else: 

        warnings.warn('File not recognised')
        
        return False

def _read_number_of_particles_from_pandas_file(pandas_file) -> int:
    """Get the number of particles probed during
    particle analysis from the pandas file (csv)

    Parameters
    ----------
    pands_file
        Pandas file type

    Returns
    -------
    number_of_particles 
        
    """
    first_key = list(pandas_file.keys())[0]
    
    return len(pandas_file[first_key])

def _read_number_of_particles_from_filename(filename) -> int:
    """Get the number of particles probed during
    particle analysis from the pandas file (csv)

    Parameters
    ----------
    pands_file
        Pandas file type

    Returns
    -------
    number_of_particles 
        
    """
    if _check_csv_filename(filename):

        pandas_file = pd.read_csv(filename)
        
        first_key = list(pandas_file.keys())[0]
        
        return len(pandas_file[first_key])

def get_number_of_particles(arg):
    """Get the number of particles probed from particle analysis.

    Parameters
    ---------
    arg 
        filename string or pandas DataFrame

    
    Returns
    -------
        number_of_particles
    """
    if type(arg) == str: return _read_number_of_particles_from_filename(arg)

    elif type(arg) == pd.core.frame.DataFrame: return _read_number_of_particles_from_pandas_file(arg)

    else: warnings.warn('Argument not recognised')


def _read_elements_from_csv(filename, keyword = "[Mass%]"):
    """Read the elements stored in the particle analysis
    result file (.csv)

    Parameters
    ----------
    filename 
        Name of .csv file

    Returns
    -------
    filenames
        List of filenames
    """
    filename = str(filename)
    
    if _check_csv_filename(filename):

        s = pd.read_csv(filename)
        
        # Get identified elements from csv as a list
        return [content.replace(' ' + keyword,'') for content in list(s.keys()) if keyword in content]
        
def _read_elements_from_pandas_file(pandas_file, keyword = "[Mass%]"):
    """Alternative function to read_elements_from_csv. Here, the 
    elements are extracted from a read pandas_file
    
    Parameters
    ----------
    pandas_file 
        Pandas file with the elements to extract

    Returns
    -------
    filenames
        List of filenames
    """
    return [content.replace(' ' + keyword, '') for content in list(pandas_file.keys()) if keyword in content]

def get_elements(arg):
    """Get the elements identified during the particle analysis

    Parameters
    arg
        Either filename of pandas DataFrame
    
    Returns
    -------
    elements 
        list of identified elements
    """
    if type(arg) == str: return _read_elements_from_csv(arg)

    elif type(arg) == pd.core.frame.DataFrame: return _read_elements_from_pandas_file(arg)

    else: warnings.warn('Argument not identified')

def _read_particles_composition_from_filename(filename, keyword = "[Mass%]"):
    """Read the chemical composition of the particles from 
    particle analysis. The order of the chemical composition is
    identicla to the order of elements (see read-elements_from_csv)

        Parameters
    ----------
    filename 
        Name of .csv file

    Returns
    -------
    compositions
        Array of chemical compositions. 
    """

    filename = str(filename)

    if _check_csv_filename(filename):

        s = pd.read_csv(filename)

        return _read_particles_composition_from_pandas(s)
        
def _read_particles_composition_from_pandas(pandas_file, keyword = "[Mass%]"):
    """Alternative function to read_particles_composition. Here, the 
    concentrations are extracted from a read pandas_file

        Parameters
    ----------
    filename 
        Name of .csv file

    Returns
    -------
    compositions
        Array of chemical compositions. 
    """

    elements = get_elements(pandas_file)

    concentrations = []

    # Get the particles' concentration of element elem
    for elem in elements: 

        arr = np.asarray(pandas_file[f"{elem} {keyword}"])

        arr[np.isnan(arr)] = 0.0
        
        concentrations.append(arr)

    return np.asarray(concentrations)

def get_particles_composition(arg, unit = "[Mass%]"):
    """Get the chemical composition of probed particles from
    particle analysis

    Parameters
    ----------
    arg
        pandas DataFrame or filename string

    Returns
    -------
    compositions
        particles' chemical composition in the same order as the element list
    """
    if type(arg) == str: return _read_particles_composition_from_filename(arg, unit)
    
    elif type(arg) == pd.core.frame.DataFrame: return _get_particles_composition_from_pandas(arg, unit) 

def get_unit(arg):
    """Read the chemical composition unit stored in the csv file."""
    chemical_unit = ''
    
    if type(arg) == str: s = pd.read_csv(arg)

    elif type(arg) == pd.core.frame.DataFrame: s = arg

    key_arguments = list(s.keys())

    from exspy.material import elements as ELEMENTS

    known_elements = list(ELEMENTS.as_dictionary())[1:]

    for key in key_arguments:

        if key.split(' ')[0] in known_elements: 
            
            chemical_unit = key.split(' ')[1]

            break
    
    return chemical_unit
    
def _read_classes_from_pandas(pandas_file, keyword = 'Class name', class_dtype = '<U25'):
    """Read stored classes from a filename (csv)

    Parameters
    ----------
    filename 
        .csv filename

    Returns
    -------
    classes
        List of classes
    """

    # Get classes:
    classes = list(pandas_file[keyword])
    
    # Get unique class names
    unique_classes = pandas_file[keyword].unique()

    # Identify the non-classified classes:
    for classname in unique_classes:

        # Remove nan with a string-name
        if type(classname) == float or pd.isna(classname):
            
            for i in range(len(classes)):

                if type(classes[i]) == float: classes[i] = 'Unclassified'

    return np.asarray(classes).astype(class_dtype)
    
def _read_classes_from_csv(filename):
    """Read stored classes from a filename (csv)

    Parameters
    ----------
    filename 
        .csv filename

    Returns
    -------
    classes
        List of classes
    """
    filename = str(filename)

    if _check_csv_filename(filename):

        s = pd.read_csv(filename)

        return _read_classes_from_pandas(s)    

def get_classes(arg):
    """Get classes from a pandas DataFrame or a 
    filename referring to the csv file

    Parameters
    ---------
    arg 
        filename (str) to vsc file or pandas DataFrame

    Returns
    -------
    classes
        List of classes for each particle
    """
    if type(arg) == str: return _read_classes_from_csv(arg)

    elif type(arg) == pd.core.frame.DataFrame: return _read_classes_from_pandas(arg)

def _create_directory(path):
    """The function creates a directory in current directory

    Parameters
    ----------
    directory_name : str
        directory
    """

    # Create the directory
    try: os.mkdir(path)
    
    except FileExistsError: pass
    
    except PermissionError: print(f"Permission denied: Unable to create '{path}'.")
    
    except Exception as e: print(f"An error occurred: {e}")

def _get_particles_geometry(arg, element_keyword = "[Mass%]"):
    """Read measured geometries from particle analysis.

    Parameters
    ----------
    arg
        filename string or pandas object

    element_keyword
        Keyword not of interest to only get the geometric properties 

    Returns
    ------
    
    """
    if type(arg) == str: s = pd.read_csv(arg)

    else: s = arg
    
    key_arguments = [arg for arg in list(s.keys())[9:] if element_keyword not in arg]

    key_content = dict()

    for key in key_arguments: key_content[key] = np.array(s[key])

    return key_content

def get_subdirectories(path : str):
    """
    Returns a list of immediate subdirectory names in the given path using os.scandir().
    """
    return [entry.name for entry in os.scandir(path) if entry.is_dir()]

def get_directory_filenames(path : str):
    """
    Returns a list of immediate subdirectory names in the given path using os.scandir().
    """
    return [entry.name for entry in os.scandir(path) if not entry.is_dir()]

def _identify_subdirectories_of_interest(path : str,
                                         keyword : str):
    """Identify all subdirectories starting with keyword (argument)"""
    path = str(path)
    
    subdirs = get_subdirectories(path)

    get_dirs = []
    
    for subdir in subdirs:

        if keyword in subdir: get_dirs.append(subdir)
    
    return get_dirs

def _identify_filenames_of_interest(path : str,
                                    keyword : str):
    """Identify all subdirectories starting with keyword (argument)"""
    path = str(path)
    
    filenames = get_directory_filenames(path)

    get_names = []
    
    for name in filenames:

        if keyword in name: get_names.append(name)
    
    return get_names

def _load_stub_image(path : str,
                       image_extension : str = 'png',
                       set_dtype = None):
    """
    Load the patched image stored in path
    """
    # Get the patched image
    path = Path(path)

    try:
        
        patched_im_filename = _identify_filenames_of_interest(
            path = path,
            keyword = image_extension
        )

        if len(patched_im_filename) > 1:

            raise warnings.warn(f"{len(patched_im_filename)} images were found the directory.")

        else: patched_im_filename = patched_im_filename[0]

        print('Reading patched view images...')
        
        # 4 channels of which all contain the same information ...
        patched_im = plt.imread(path / patched_im_filename)[...,0]

    except FileNotFoundError:

        warnings.warn(f"The patched image in directory\n{path}\nCould not be found.")

        return np.asarray([], np.uint8)

    if set_dtype is not None: 
        
        if check_array_compatibility_with_new_datatype(patched_im, set_dtype): 
            
            patched_im = patched_im.astype(set_dtype)

    return patched_im

def _load_view_images(path : str,
                      image_extension = 'bmp',
                      set_dtype : bool = None):
    """
    Load the view images from particle analysis

    Expected folder structure:
    --------------------------
    Sutb_id/
    ├── View_id
    │   └── Particle_id/
    │       ├── spectrum folder
    │       ├── ParticleImage.bmp
    │       ├── Tag data
    │       └── *.xml-files
    │   ├── ViewImage.bmp 
    │   └── *.xml file
    ├── StubData.png
    └── *.xml file

    Parameters
    ----------
    path
        path to Sutb_id
    image_extension
        image extension. Default: bitmap (*.bmp)
    """
    path = Path(path)

    folders = np.sort(
        _identify_subdirectories_of_interest(
            path = path,
            keyword = 'View'
        )
    )

    first = True

    print('Loading View images...')
    for fol, idx in tqdm(zip(folders, np.arange(len(folders))), total = len(folders)):

        if first:

            image_filename = _identify_filenames_of_interest(
                path = path / fol, 
                keyword = image_extension
            )

            # If more than one image is found:
            if len(image_filename) > 1: 

                if image_extension[0] == '.': image_extension = image_extension[1:]

                image_filename = _identify_filenames_of_interest(
                    path = path / fol, 
                    keyword = f'ViewImage.{image_extension}'
                )

            else: image_filename = image_filename[0]
                
            first_im = plt.imread(path / fol / image_filename)

            im_shape = first_im.shape

            # Identify a proper image data type
            if set_dtype is not None: 
        
                if check_array_compatibility_with_new_datatype(first_im, set_dtype): 
                    
                    first_im = first_im.astype(set_dtype)

            else: set_dtype = float

            view_images = np.zeros(((len(folders),) + im_shape), dtype = set_dtype)

            view_images[idx] = first_im

            first = False

        else: view_images[idx] = plt.imread(path / fol / image_filename).astype(set_dtype)

    return view_images

def _estimate_number_of_particles_based_on_folders(path : str,
                                                   folders : list,
                                                   keyword = 'Particle'):
    """See _load_particle_images for folder structure"""

    num_particles = 0
    
    path = Path(path)

    for fol in folders:

        p_folders = _identify_subdirectories_of_interest(
            path = path / fol,
            keyword = keyword
        )

        num_particles += len(p_folders)

    return num_particles

def _load_particle_images(path : str,
                          folders = list,
                          image_extension = 'bmp',
                          set_dtype : bool = None,
                          centre_particle_images : bool = True,
                          num_images = None):
    """
    Load the particle images from particle analysis

    Expected folder structure:
    --------------------------
    Sutb_id/
    ├── View_id
    │   └── Particle_id/
    │       ├── spectrum folder
    │       ├── ParticleImage.bmp
    │       ├── Tag data
    │       └── *.xml-files
    │   ├── ViewImage.bmp 
    │   └── *.xml file
    ├── StubData.png
    └── *.xml file

    Parameters
    ----------
    path
        path to Sutb_id
    folders
        List of folders to extract particle images from
    image_extension
        image extension. Default: bitmap (*.bmp)
    set_dtype
        Set image data type if not None
    centre_particle_images
        Whether to centre the particle images or not in the new array
    num_particles 
        How many particle images to read
    """

    from scipy.ndimage import center_of_mass

    path = Path(path)

    if num_images == None:

        num_images = _estimate_number_of_particles_based_on_folders(
            path = path,
            folders = folders
        )

    # Since we don't know the common shape of the particle images, we need to start with one
    first = True

    # Image indexer
    p = 0

    print('Reading individual particle images...')
    
    for fol in tqdm(folders):

        subdirs = _identify_subdirectories_of_interest(
            path = path / fol,
            keyword = 'Particle'
        )

        for pim in subdirs:

            filename = _identify_filenames_of_interest(
                path = path / fol / pim,
                keyword = image_extension
            )

            if len(filename) > 1: 
                
                raise FileNotFoundError(f"Expected to find only one image in directory {path / fol / pim}, but {len(filename)} were found.")

            else: filename = filename[0]

            directory = path / fol / pim / filename

            # Read particle image
            tmp = plt.imread(directory)
            
            shape = np.shape(tmp)

            # Create an array matching the first particle image's shape
            if first:
                
                particle_images = np.zeros(((num_images),) + shape, dtype = tmp.dtype)
            
                particle_images[p, ...] = tmp.copy()

                first = False

            else: 

                old_shape = particle_images.shape[-2:]

                # (n_images, y, x)
                pad_width = np.asarray(
                    ((0,0),
                    (0, shape[0] - old_shape[0]), 
                    (0, shape[1] - old_shape[1]))
                )

                pad_width *= (pad_width > 0)
                
                # pad the array if the new image is larger | (n_before, n_after)
                if pad_width.sum() > 0: 
                    
                    particle_images = np.pad(particle_images, 
                                             mode = 'constant', 
                                             constant_values = 0,
                                             pad_width = pad_width)
    
                particle_images[p, :shape[0], :shape[1]] = tmp.copy()

            p += 1

    # Centre of the particle image array:
    p_im_centre = np.round(
        np.asarray([particle_images[0].shape[0]/2, 
        particle_images[0].shape[-1]/2])
    ).astype(int)

    empty_pIm = np.zeros_like(particle_images[0])

    print('Centring the particle images...')    
    for i in range(num_images):

        # Centre of mass
        cm = np.round(np.asarray(center_of_mass(particle_images[i] > 0))).astype(int)

        # Where the data is to be extracted from
        _from = np.where(particle_images[i] > 0) 
        
        # To where the data will be stored
        _to = ( _from[0] + p_im_centre[0] - cm[0] - 1, 
                _from[1] + p_im_centre[1] - cm[1] - 1) 
        
        empty_pIm[_to] = particle_images[i][_from]

        particle_images[i].fill(0)
        
        particle_images[i] = empty_pIm.copy()

        empty_pIm.fill(0) 

    if set_dtype is not None: 
        
        if check_array_compatibility_with_new_datatype(particle_images, set_dtype):
            
            particle_images = particle_images.astype(set_dtype)

    return particle_images






def _load_images(path,
                 subdir_keyword : str = 'Sutb',
                 image_extension = ['png', 'bmp','bmp'],
                 get_particle_images : bool = True,
                 centre_particle_images : bool = True,
                 set_dtype : bool = None):
    """Read the stitched overview image, the individual view images, and the identified particles
    from particle analyseis

    Parameters
    ----------
    path 
        Path to stitched overview image and the folder structure from pa
    read_order
        A list of strings with particle labels
    get_individual_particle_images 
        Whether to read the individual particle images. True by default

    Returns
    -------

    """

    if len(image_extension) != 3: 
        
        raise ShapeError(f"iamge_extension argument has shape {(len(image_extension,))}, but the expected is (3,).")
    
    folder = str(path)

    # Check if correct path is employed, i.e. to the directory with the Sutb_ folders:
    subdirs = _identify_subdirectories_of_interest(
        path = folder,
        keyword = subdir_keyword
    )
    
    if len(subdirs) > 0: 

        from pathlib import Path

        # Check data type:
        if set_dtype is not None:

            if set_dtype not in numpy_image_datatypes: 

                print(f'Data type {set_dtype} was not recognised. Reading images as default datatype.')
                
                set_dtype = None 
        
        # Iterate through the different sub-directories (e.g. Sutb1, Sutb2, etc.)
        for _subdir in subdirs:
            
            subdir = Path(os.path.join(folder, _subdir))
            
            patched_im = _load_stub_image(
                subdir,
                image_extension = image_extension[0],
                set_dtype = set_dtype
            )

            view_images = _load_view_images(
                path = subdir,
                image_extension = image_extension[1],
                set_dtype = set_dtype
            )
        
            if get_particle_images:

                # Get all folders in the directory:
                folders = np.sort(
                    _identify_subdirectories_of_interest(
                        path = subdir,
                        keyword = 'View',
                    )
                )

                particle_images = _load_particle_images(
                    path = subdir,
                    folders = folders,
                    image_extension = image_extension[2],
                    set_dtype = set_dtype,
                    centre_particle_images = True
                )
                
            else: part_im = []
            
            return patched_im, view_images, particle_images

    else: 
        
        print(f"Coudn't find directory: {folder}") 

        return [], [], []















    