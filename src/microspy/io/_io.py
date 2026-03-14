# Copyright 2026 The microspy developers
#
# This file is part of kikuchipy.
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

import pandas as pd
import os, warnings, re, glob, importlib, yaml
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.microspy._utils._utils import check_array_compatibility_with_new_datatype
from src.microspy.signals import particle_analysis 

from src.microspy.io import plugins
from src.microspy.signals import _microspy_signal

from hyperspy._signals.signal1d import Signal1D
from hyperspy._signals.signal2d import Signal2D

numpy_image_datatypes = [
    np.bool_, np.byte, np.ubyte, 
    np.int_, np.int8, np.int16, np.int32, np.int64,
    np.uint, np.uint8, np.uint16, np.uint32, np.uint64,
    np.float16, np.float32, np.float64
]

def load(filename : str):
    """Load the particle analysis resutls stored in the csv format from Jeol's
    particle analysis.

    This function is inspired by kikuchipy.

    Parameters
    ----------
    filename
        filename of the csv file

    Returns
    -------
    out
        microspy signal
    """
    
    filename = str(filename)
    
    if not os.path.isfile(filename):
        is_wildcard = False
        filenames = glob.glob(filename)
        if len(filenames) > 0:
            is_wildcard = True
        if not is_wildcard:
            raise IOError(f"No filename matches {filename!r}")


    """
    extensions ...
    readers ...
    """
    extension = os.path.splitext(filename)[-1].replace('.','')
    plugin = 'csv'
    
    if extension == plugin:

        file_reader = plugins.JEOLcsv._api.file_reader

    else:
        
        raise IOError(
            f"Could not read {filename!r}. If the file format is supported, please "
            "report this error"
        )
    
    signal_dicts = file_reader(filename)
    #return signal_dicts
    out = []
    for signal in signal_dicts:
        out.append(
            _dict2signals(signal)
        )
        directory, filename = os.path.split(os.path.abspath(filename))
        filename, extension = os.path.splitext(filename)
        #out[-1].tmp_parameters.folder = directory
        #out[-1].tmp_parameters.filename = filename
        #out[-1].tmp_parameters.extension = extension.replace(".", "")
    
    if len(out) == 1: out = out[0]
    return out

def _dict2signals(signal_dict : dict,
                 set_additional_data : bool = True):
    """Create a signal instance from a dictionary.

    Parameters
    ----------
    signal_dict
        Signal dictionary with "data", "metadata", "original_metadata"
        and axes keys. additional_data can also exist.
    set_additional_data
        Whether to set additional data that is likely not chemistry or 
        geometry. Particle classes are expected to be found here. 
    
    Returns
    -------
    signal 
        signal instance with at least "data", "metadata" and 
        "original_metadata" keys.

    Notes
    -----
    Inspired by :func:'kikuchipy.io._dict2signal'.
    """

    if not 'data' in signal_dict:
        raise AttributeError("No data identified.")
    
    md = signal_dict['metadata'] if "metadata" in signal_dict else {}
    omd = signal_dict['original_metadata'] if 'original_metadata' in signal_dict else {}
        
    if set_additional_data and 'additional_data' in signal_dict:
        add_data = signal_dict['additional_data']
    if 'axes' in signal_dict:
        axes = signal_dict['axes']

    out = []
    
    for signal_type in signal_dict['data'].keys():
        
        out.append(
            _assign_signal_subclass(
                signal_type = signal_type)(
                signal_dict['data'][signal_type]['data'],
                **{'units' : signal_dict['data'][signal_type]['units'],
                   'props' : signal_dict['data'][signal_type]['props']
                   }
                )
        )

        # Set metadata
        out[-1].metadata.add_dictionary(md)
        out[-1].metadata.set_item("original_metadata", omd)

    return out
    

def _assign_signal_subclass(
    signal_type : str = ''):
    """Return matching signal subclass given by signal_type

    Parameters
    ----------
    signal_type
        signal type

    Returns
    -------
    signal_subclass
    """
    from src.microspy.signals._microspy_signals import MicroSpySignal1D, MicroSpySignal1D_Chemistry, MicroSpySignal1D_Geometry
    
    signal_subclasses = {
        'general' : MicroSpySignal1D, 
        'chemistry' : MicroSpySignal1D_Chemistry, 
        'geometry' : MicroSpySignal1D_Geometry
    }

    if signal_type not in signal_subclasses.keys():

        raise AttributeError(f"{signal_type} is not recognised.")

    return signal_subclasses[signal_type.lower()]



    

def get_module_function_names(module):
    """Return a list of function names from module"""
    import inspect
    function_names = []
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            function_names.append(name)
    return function_names



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















    