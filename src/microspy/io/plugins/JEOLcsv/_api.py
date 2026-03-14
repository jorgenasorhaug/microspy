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

"""Reader and writer of particle analysis data from Jeol's particle analysis. """

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
    aq
        particle chemistry, particle geometry, and original metadata
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

    num_particles = mdv[mdk == 'Summary']
    if isinstance(num_particles, np.ndarray):
        num_particles = num_particles[0]
    
    md.update(
        {
            'General' : {
                'original_filename' : filename,
                'title' : pname,
            },
            'Sample' : {
                'elements' : data['chemistry']['props'],
                'particles' : num_particles,
            },
            'Acquisition_instrument' : {}
        }
    )

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
    acquisition['additional_data'] = additional_data

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
    

def _get_acquisition_data_from_file(file : pd.core.frame.DataFrame,
                                   read_from_column : int = 3,
                                   geometric_keywords : list = [
                                       'um',
                                       'area',
                                       'ratio',
                                       'roundness',
                                       'orientation']) -> tuple[dict, dict]:
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
    classes[pd.isna(classes)] = ''
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

def _lists_to_dict(keys : list, 
                   values : list) -> dict:
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