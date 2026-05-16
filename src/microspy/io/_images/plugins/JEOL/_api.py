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

import matplotlib.pyplot as plt
import numpy as np
import warnings
from pathlib import Path

from tqdm import tqdm_notebook

from src.microspy.io._utils import (
    _identify_filenames_of_interest,
    _identify_subdirectories_of_interest
)
from src.microspy._misc._misc import (
    check_array_compatibility_with_new_datatype
)

subdir_keyword = "Sutb"
image_keyword = "View"
image_extension = ["png", "bmp","bmp"]
    
def _load_stub_image(
    path : str,
    image_extension : str = 'png',
    set_dtype = None
) -> np.ndarray:
    """Load the patched image stored in path.

    Parameters
    ----------
    path 
        directory
    image_extension
        Image extension. png by default.
    set_dtype
        Set an image data type. None by default.

    Returns
    -------
    patched_im
        Patched image as stored
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
        
        # 4 channels of which all contain the same information ...
        patched_im = plt.imread(path / patched_im_filename)[...,0]

    except FileNotFoundError:
        warnings.warn(f"The patched image in directory\n{path}\nCould not be found.")
        return np.asarray([], np.uint8)

    if set_dtype is not None: 
        if check_array_compatibility_with_new_datatype(patched_im, set_dtype): 
            patched_im = patched_im.astype(set_dtype)

    return patched_im

def _load_view_images(
    path : str,
    image_extension = 'bmp',
    set_dtype : bool = None
) -> np.ndarray:
    """Load the view images from particle analysis.

    Expected folder structure:
    --------------------------
    Sutb_id/
    ├── View_id
    │   └── Particle_id/
    │       ├── spectrum folder
    │       ├── ParticleImage.bmp
    │       ├── Tag data
    │       └── *.xml-files
    │   ├── ViewImage.bmp # <-- Read | c.f. image_keyword
    │   └── *.xml file
    ├── StubData.png
    └── *.xml file

    Parameters
    ----------
    path
        path to Sutb_id
    image_extension
        Image extension. bitmap (bmp) by default.

    Returns
    -------
    view_images
        Acquired (analysed) SEM images during 
        particle analysis.
    """
    
    path, folders = _identify_subdirectories_of_interest(
            path = path,
            keyword = image_keyword
        )
    
    folders = np.sort(folders)

    path = Path(path)

    # Whether the first particle image is read or not.
    # Needed to set an initial size.
    first = True

    for fol, idx in tqdm_notebook(
        zip(
            folders,
            np.arange(len(folders))
            ), 
        total = len(folders),
        desc = "Parent Images",
        position = 0
    ):

        if first:

            image_filename = _identify_filenames_of_interest(
                path = path / fol, 
                keyword = image_extension
            )

            # If more than one image is found:
            if len(image_filename) > 1: 
                if image_extension[0] == '.': 
                    image_extension = image_extension[1:]
                    
                image_filename = _identify_filenames_of_interest(
                    path = path / fol, 
                    keyword = f'ViewImage.{image_extension}'
                )

            else: image_filename = image_filename[0]
                
            first_im = plt.imread(path / fol / image_filename)
            im_shape = first_im.shape

            # Identify a proper image data type
            if set_dtype is not None: 
                if check_array_compatibility_with_new_datatype(
                    first_im, set_dtype
                ): 
                    first_im = first_im.astype(set_dtype)
            else: set_dtype = float
                
            view_images = np.zeros(((len(folders),) + im_shape), 
                                   dtype = set_dtype)
            view_images[idx] = first_im
            first = False

        else: view_images[idx] = plt.imread(
            path / fol / image_filename
        ).astype(set_dtype)

    return view_images

def _load_particle_images(
    path : str,
    folders = list,
    image_extension = 'bmp',
    set_dtype : bool = None,
    centre_particle_images : bool = True,
    num_images = None
) -> np.ndarray:
    """Load the particle images acquired from particle analysis.
    
    Expected folder structure:
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
        List of folders to extract particle images from,
        i.e. the View_ids
    image_extension
        image extension. Default: bitmap (*.bmp)
    set_dtype
        Set image data type if not None
    centre_particle_images
        Whether to centre the particle images or not in the new array
    num_particles 
        How many particle images to read

    Returns
    -------
    particle_images
        Particle images
    """

    from scipy.ndimage import center_of_mass
    from ._utils import _estimate_number_of_particles_based_on_folders

    path = Path(path)

    if num_images == None:

        num_images = _estimate_number_of_particles_based_on_folders(
            path = path,
            folders = folders
        )
    
    # Since we don't know the common shape of the particle 
    # images, we need to start with one
    first = True

    # Image indexer
    p = 0
    
    for fol in tqdm_notebook(
        folders, 
        desc = "Child Images",
        position = 0
    ):
        fol, subdirs = _identify_subdirectories_of_interest(
            path = str(path / fol),
            keyword = 'Particle'
        )
        subdirs = np.sort(subdirs)

        for pim in subdirs:
            filename = _identify_filenames_of_interest(
                path = path / fol / pim,
                keyword = image_extension
            )

            if len(filename) > 1: 
                
                raise FileNotFoundError(f"Expected to find only one "
                                        "image in directory {path / fol /"
                                        "pim}, but {len(filename)} were "
                                        "found.")
            else: filename = filename[0]

            directory = path / fol / pim / filename

            # Read particle image
            tmp = plt.imread(directory)
            shape = np.shape(tmp)

            # Create an array matching the first particle image's shape
            if first:
                particle_images = np.zeros(
                    ((num_images),) + shape, dtype = tmp.dtype
                )
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
                    particle_images = np.pad(
                        particle_images,
                        mode = 'constant', 
                        constant_values = 0,
                        pad_width = pad_width
                    )
    
                particle_images[p, :shape[0], :shape[1]] = tmp.copy()
            p += 1

    if centre_particle_images:

        # Centre of the particle image array:
        p_im_centre = np.round(
            np.asarray([particle_images[0].shape[0]/2, 
            particle_images[0].shape[-1]/2])
        ).astype(int)
    
        empty_pIm = np.zeros_like(particle_images[0])
    
        for i in range(num_images):
    
            # Centre of mass
            cm = np.round(np.asarray(
                center_of_mass(particle_images[i] > 0)
            )).astype(int)
    
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
        if check_array_compatibility_with_new_datatype(
            particle_images, set_dtype
        ):
            particle_images = particle_images.astype(set_dtype)

    return particle_images

