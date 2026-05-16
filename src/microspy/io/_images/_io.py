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

import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

from src.microspy.io._utils import (
    _identify_subdirectories_of_interest
)
from src.microspy.signals._microspy_signals import (
        MicroSpySignal2D,
        MicroSpySignal2D_Parent
    )

from src.microspy._misc import exceptions

def load_images(
    path : str,
    vendor : str,
    get_particle_images : bool = True,
    centre_particle_images : bool = True,
    set_dtype : bool = None
) -> list:
    """Load the stitched overview image, the individual 
    view images, and the particle images acquired during 
    particle analysis.

    "CURRENTLY ONLY SUPPORTING JEOL'S SOLUTION"

    Parameters
    ----------
    path 
        Path to stitched overview image and the folder 
        structure from pa
    vendor
        Vendor to correctly read the images.
        Currently only working for jeol
    read_order
        A list of strings with particle labels
    get_individual_particle_images 
        Whether to read the individual particle images. 
        True by default

    Returns
    -------
    out 
        list of the following images:
        patched_im
            np.ndarray representing the patched image
        view_images
            np.ndarray representing the individual 
            overview images
        particle_images
            np.ndarray representing the individual 
            particle images
    """
    from pathlib import Path
    
    folder = str(path)

    warnings.warn("Currently only supporting Jeol's version")
    vendor = 'jeol'

    """
    vendors ...
    readers ...
    """

    # Check if image data type is to be changed
    # after reading the images.
    if set_dtype is not None:

        if set_dtype not in numpy_image_datatypes: 

            print(f'Data type {set_dtype} was not recognised.' 
                  'Reading images as default datatype.')
            
            set_dtype = None 

    # Identify the vendor and import corr. image readers
    if vendor.lower() == 'jeol':
        
        from src.microspy.io._utils import (
            _identify_subdirectories_of_interest as get_subdirectories
        )
        from src.microspy.io._images.plugins.JEOL._api import (
            subdir_keyword, image_keyword, image_extension,
            _load_stub_image as stub_loader,
            _load_view_images as view_loader, 
        )

    # Check if correct path is employed, i.e. to the 
    # directory with the Sutb_ folders. E.g. ['Sutb01',
    # Sutb02'] from jeol, i.e. dir = folder / subdirs[i]
    
    folder, subdirs = get_subdirectories(
        path = folder,
        keyword = subdir_keyword
    ) # This list contains only the next subdir(s).

    out = []
    
    # Load images | iterate through diff. acquisitions
    if len(subdirs) > 0:
        
        # Iterate through the different sub-directories 
        # (e.g. Sutb1, Sutb2, etc.)
        for _subdir in subdirs:
            
            subdir = Path(os.path.join(folder, _subdir))
            
            patched_im = stub_loader(
                path = subdir,
                image_extension = image_extension[0],
                set_dtype = set_dtype
            )

            view_images = view_loader(
                path = subdir,
                image_extension = image_extension[1],
                set_dtype = set_dtype
            )
        
            if get_particle_images:
                
                from src.microspy.io._images.plugins.JEOL._api import (
                    _load_particle_images as particles_image_loader
                )

                _, folders = _identify_subdirectories_of_interest(
                    path = subdir,
                    keyword = image_keyword
                )
            
                folders = np.sort(folders)

                particle_images = particles_image_loader(
                    path = subdir,
                    folders = folders,
                    image_extension = image_extension[2],
                    set_dtype = set_dtype,
                    centre_particle_images = centre_particle_images
                )
                
            else: particle_images = []

            out.append([patched_im, 
                        view_images, 
                        particle_images]
                      )

        return out

    else: 
        
        warnings.warn(f"Coudn't find directory: {folder}") 

        return [[], [], []]
        
def save_images(
    path : str = None
):
    """
    """
    print('UNFINISHED')

def _arrays2signals(#_1dArray2list(
    array : np.ndarray | list
) -> list:
    """Assign an image subclass to the input array(s).
    
    Parameters
    ----------
    array
        list with 2D or 3D ndarrays or a single
        np.ndarray

    Returns
    -------
    out
        List of assigned subclasses
    """
    input_type = type(array)

    if isinstance(input_type, np.ndarray):

        array = [array]

    out = []

    for arr in array:

        out.append(
            _assign_image_subclass(
                array = arr
                )
            )

    return out

def _assign_image_subclass(
    array : np.ndarray
) -> MicroSpySignal2D | MicroSpySignal2D_Parent:
    """Assign image subclass to a list of 
    ndarrays

    Parameters
    ----------
    array
        List of ndarrays

    Returns
    -------
    signal
        list of image subclasses
    """

    # Return signal if it's already been set
    if isinstance(array, MicroSpySignal2D | 
                  MicroSpySignal2D_Parent
                 ):

        return array

    dim = np.ndim(array)
    
    arr = MicroSpySignal2D(array)

    if dim == 2:
        
        arr.metadata.Signal.signal_type = f"2D_Image_shape{np.shape(array)}".replace(
            ", ","x"
        ).replace('(','').replace(')','')

    elif dim > 2:

        arr.metadata.Signal.signal_type = f"ND_Image_shape{np.shape(array)}".replace(
            ", ","x"
        ).replace('(','').replace(')','')
        
    return arr

        

    