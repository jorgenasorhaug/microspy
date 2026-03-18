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

"""Reader and writer of particle analysis data from Jeol's particle 
analysis."""

import pandas as pd
from pathlib import Path
import warnings, re

from matplotlib import pyplot as plt
import numpy as np

from src.microspy._utils import exceptions 

def file_reader(filename : str | Path):
    """Read particle analysis data from a .csv file using 
    pandas dataframe.

    Parameters
    ----------
    filename
        File path to dataframe file.

    Returns 
    -------
    acquisition
        [particle chemistry, particle geometry, and original metadata]
    """

    warnings.warn("\nMultiple stubs have not been tested yet ...")

    # Data dictionary
    acquisition = {}
    # metadata dictionary
    md = {}

    filename = str(filename)
    # Load pandas dataframe 
    file = pd.read_csv(filename)

    # Get the stored metadata keys and values
    # These are typically stored in the two first columns,
    # the keys in first and the metadata in the second
    mdk, mdv, pname = _get_metadata_from_jeol_csv_file(file)

    # Set keys and values (originaL metadata) to a dictionary
    omd = _lists_to_dict(mdk, mdv)

    # Get the particle data stored in the dataframe
    data, additional_data = _get_acquisition_data_from_file(file)

    # Read the total number of investigated particles
    num_particles = mdv[mdk == 'Summary']
    if isinstance(num_particles, np.ndarray):
        num_particles = num_particles[0]

    #stubs = np.where(mdk == 'Stub name')[0][1:]

    # Update metadata
    md.update(
        {
            'General' : {
                'original_filename' : filename,
                'title' : pname,
            },
            'Sample' : {
                'elements' : data['chemistry']['props'],
                'particles' : num_particles,
                'classes' : additional_data['classification'],
            },
            'Acquisition_instrument' : {
                'vendor' : 'Jeol',
                #'Stubs' : mdv[stubs]
                
            },
            'Additional_data' : additional_data
        }
    )
    del additional_data['classification']

    # Iterate through all the stubs stored in the metadata; 
    # these are separated by the keyword 'Stub name'
    karr = np.where(mdk == 'Stub name')[0]
    # The first instance is a general overview. Ignore this
    for k in karr[1:]:
        md['Acquisition_instrument'].update(
            {
                f'{mdv[k]}' : {
                    'magnification' : mdv[k+1],
                    'X_size[mm]' : mdv[k+2],
                    'Y_size[mm]' : mdv[k+3],
                    'reservation_area' : mdv[k+4],
                    'reserved_views' : mdv[k+5],
                    'analysed_area' : mdv[k+6],
                    'analysed_views' : mdv[k+7],
                    'acquisition_time' : mdv[k+8]
                }
            }
        )
    
    # Set the metadata and the data
    acquisition['metadata'] = md
    acquisition['original_metadata'] = omd
    acquisition['data'] = data
    #acquisition['additional_data'] = additional_data

    axes = {
        'chemistry' : {
            'size' : acquisition['data']['chemistry']['data'].shape,
            'name' : 'composition',
            'unit' : acquisition['data']['chemistry']['units']
        },
        'geometry' : {
            'size' : acquisition['data']['geometry']['data'].shape,
            'name' : acquisition['data']['geometry']['props'],
            'units' : acquisition['data']['geometry']['units'],
        }
    }

    acquisition['axes'] = axes
    
    return [acquisition]
    

def _get_acquisition_data_from_file(
    file : pd.core.frame.DataFrame,
    read_from_column : int = 3,
    geometric_keywords : list = [
        'um',
        'area',
        'ratio',
        'roundness',
        'orientation']
) -> tuple[dict, dict]:
    """Get acquisition data such as label name, analysis dat, class name, 
    and stub positions.

    Arguments
    ---------
    file
        pandas dataframe
    read_from_column
        column ID to start reading from
    geometric_keywords
        Geometric keywords to search for in order to identify particles' 
        geometric properties

    Returns
    -------
    data
        dictionary containing all the data from the particle
        analysis acquisition, s.a. chemistry, geometric 
        properties and classification
        
        Structure:
        ├── chemistry/
        │   ├── elements
        │   ├── data (n,m) 
        │   └── unit
        └── geometry/
            ├── prop
            ├── data (n,o) 
            └── units
            
    additional_data
        dictionary containing additional data from the particle
        analysis acquisition, e.g. particle label/number, label
        name (typically Stub[ID]-[image ID]-[particle ID], stage
        coordinates, etc.

        Structure:
        ├── keywords/
        ├──  data (n,p)
        └── classification 
    """

    from exspy.material import elements
    ELEMENTS = list(elements.as_dictionary().keys())[1:]

    # Making sure all the keywords are lower case
    geometric_keywords = [word.lower() for word in geometric_keywords]

    # Get the dataframe headers
    headers = list(file.keys())[read_from_column:]

    # We will search for all the instances of elements. 
    # The rest are particle metadata and/or geometric properties
    first_words = [s.split()[0] for s in headers]
    last_words = [s.split()[-1] for s in headers]

    # Identify elements for chemistry
    elements = []
    instance = []
    unit = ''
    for idx, fw in enumerate(first_words):
        if fw in ELEMENTS:
            instance.append(idx)
            elements.append(fw)
            if unit == '': unit = last_words[idx]
            elif unit != last_words[idx]:
                warnings.warn(f"The chemical unit of element {fw} ({last_words[idx]}) is different from the registered chemical unit {unit}!")

    # Set the chemistry as a numpy array
    chemistry = pd.concat([file[headers[i]] for i in instance], axis = 1)
    chemistry[pd.isna(chemistry)] = 0

    # Update headers
    for idx in instance[::-1]: headers.remove(headers[idx])

    # Store the chemistry
    data = {
        'chemistry' : {
            'props' : elements,
            'data' : np.asarray(chemistry),
            'units' : unit
        }
    }

    # Identify geometric keywords for geometric properties
    pattern = r'[^a-zA-Z0-9\s]' 
    instance = []
    prop = []
    units = []
    for idx, p in enumerate(headers):
        union = set(re.sub(pattern, '',p).lower().split()) & set(geometric_keywords)
        if len(union) > 0:
            instance.append(idx)
            if '[' in p.split()[-1]: 
                units.append(p.split()[-1])
            else: units.append('1')
            prop.append(p)

    # Setting geometric properties
    values = np.transpose(np.asarray([np.asarray(file[geom]) for geom in prop]))
    values[np.isnan(values)] = 0
    
    # Update headers and property keywords
    for idx in instance[::-1]: headers.remove(headers[idx])
    for p, u in zip(prop, units): 
        prop[prop.index(p)] = p.replace(f" {u}", "")
        
        
    # Store as dictionary instead?
    #geometry = _lists_to_dict(prop, [np.asarray(file[geom]) for geom in prop])
    data['geometry'] = {
        'props' : prop,
        'data' : values,
        'units' : units
    }

    # Store the classifications
    class_key = ''
    for header in headers: 
        if 'class' in header.lower(): 
            class_key = header
            break
    headers.remove(class_key)
    
    classes = file[class_key].copy().astype(object) # dataframe
    classes[pd.isna(classes)] = 'Unclassified'
    classes = np.object_(classes)

    # The remaining data is returned as additional data
    additional_data = {
        'keywords' : headers,
        'data' : np.asarray(pd.concat([file[df] for df in headers], axis = 1)),
        'classification' : classes
    }

    return data, additional_data

    
def _get_metadata_from_jeol_csv_file(file : pd.core.frame.DataFrame) -> tuple[list, list, str]:
    """Get acquisition metadata from the dataframe. 
    These are typically Acquisition date, stub name, number 
    of particles, etc.
    
    Argument
    --------
    file
        pandas dataframe

    Returns
    -------
    mdk
        metadata keys
    mdv 
        metadata values
    pname
        project name
    """

    pname = list(file.keys())[1]

    # Get stored metadata keys
    _mdk = np.asarray(file['Project name'])
    get = ~pd.isna(_mdk)
    mdk = _mdk[get] # keys

    # Get the corr. metadata values
    _mdv = np.asarray(file[pname])
    mdv = _mdv[get] # values

    # Replace string values with numerics if numeric
    numerics = np.strings.isnumeric(list(mdv))
    mdv[numerics] = mdv[numerics].astype(int)

    # Identify floats among numerics and set them accordingly
    for index, val in enumerate(mdv):
        try: 
            val = float(val)
            if val.is_integer(): val = int(val)
            mdv[index] = val
        except ValueError: pass

    # Replace nan with empty string
    mdv[pd.isna(mdv)] = ''
    
    return mdk, mdv, pname

def _lists_to_dict(
    keys : list, 
    values : list
) -> dict:
    """Set two lists of values into a dicitonary

    Parameters
    ----------
    keys, values
        numpy object arrays 

    Returns
    -------
    dictionary
        dict
    """
    
    if len(keys) != len(values):
        raise exceptions.ShapeError(f"Invalid argument shapes: {len(keys)} and {len(values)}")

    dictionary = {}
    
    for key, val in zip(keys, values): dictionary[key] = val

    return dictionary

def image_reader(
    path : str,
    read_individual_particle_images : bool = True,
    **kwargs) -> "hyperspy signals Signal2D":
    """Read the SEM images acquired during the analysis and the individual
    particle images if read_individual_particle_images is True

    Parameters
    ----------
    path
        path to image directory, typically ...
    read_individual_particle_images
        Whether to read the individual particle images. 
        These can be used to identify the particles' location
        in each individual SEM image
    """
    
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


