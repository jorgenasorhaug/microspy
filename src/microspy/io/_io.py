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

import pandas as pd
import os, warnings, re, glob, importlib, yaml
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.microspy._utils._utils import check_array_compatibility_with_new_datatype
from src.microspy.signals import particle_analysis 

from src.microspy.io import plugins
from src.microspy.signals import _microspy_signals
from src.microspy.signals.particle_analysis import ParticleAnalysis

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
        out[0][-1].tmp_parameters.folder = directory
        out[0][-1].tmp_parameters.filename = filename
        out[0][-1].tmp_parameters.extension = extension.replace(".", "")
    
    if len(out) == 1: out = out[0]
    
    return ParticleAnalysis(out)

def _dict2signals(signal_dict : dict):
    """Create a signal instance from a dictionary.

    Parameters
    ----------
    signal_dict
        Signal dictionary with "data", "metadata", "original_metadata"
        and axes keys. additional_data can also exist.
    
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
        
    #if set_additional_data and 'additional_data' in signal_dict:
    #    add_data = signal_dict['additional_data']
    if 'axes' in signal_dict:
        axes = signal_dict['axes']

    out = []
    
    for signal_type in signal_dict['data'].keys():
        
        out.append(
            _assign_signal_subclass(
                signal_type = signal_type.lower())(
                signal_dict['data'][signal_type]['data'],
                **{'units' : signal_dict['data'][signal_type]['units'],
                   'props' : signal_dict['data'][signal_type]['props']
                   }
                )
        )

        # Set metadata
        out[-1].metadata.add_dictionary(md)
        out[-1].metadata.set_item("Original_metadata", omd)
        #out[-1].metadata.set_item("Additional_data", add_data)

    # Set as particle_analysis class
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

def _load_images(path : str,
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

    """
    extensions ...
    readers ...
    """
    from src.microspy.io.plugins.JEOLcsv import _identify_subdirectories_of_interest, _load_stub_image, _load_view_images, _load_particle_images

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







def get_module_function_names(module):
    """Return a list of function names from module"""
    import inspect
    function_names = []
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            function_names.append(name)
    return function_names

    