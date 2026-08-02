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
#

import numpy as np
import json
import os

from pathlib import Path
from hyperspy.misc.utils import DictionaryTreeBrowser

EXSPY_ELEMENTS_LINES_URL = "https://raw.githubusercontent.com/hyperspy/exspy/refs/heads/main/exspy/material/xray_lines.json"
EXSPY_GENERAL_PROPS_URL = "https://raw.githubusercontent.com/hyperspy/exspy/refs/heads/main/exspy/material/elements_general_properties.json"

def _download_exspy_data(url):
    """Download exspy raw file and save it in file directory.

    Parameters
    ----------
    url 
        String url to download the data from

    Returns
    -------
    data
        Downloaded data
    """
    from requests import get
    response = get(url)
    
    if response.status_code == 200:
        print(f"Downloading material properties from {url}")
        # Convert response to a Python dictionary/list
        data = response.json()  

        if isinstance(data, list): 
            print('Fix list -> dict conversion')
    else:
        print(f"Failed to download {url}."
              f"Status code: {response.status_code}")
        data = {}

    return data

def _download_elements_general_properties():
    """Download elements_general_properties from github and save it as a 
    json file to the current directory, if it doesn't already exist.
    """

    file_path = Path(__file__).parent / "elements_general_properties.json"
    
    # Download general elements properties
    if not file_path.is_file():
        exspy_material_prop_url = EXSPY_GENERAL_PROPS_URL
        data = _download_exspy_data(exspy_material_prop_url)
        if bool(data): # not empty dictionary
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

def _download_elements_lines_properties():
    """Download xray_lines from github and save it as a json file to the 
    current directory, if it doesn't already exist.
    """
    file_path = Path(__file__).parent / "xray_lines.json"
    
    # Download general elements properties
    if not file_path.is_file():
        exspy_material_prop_url = EXSPY_ELEMENTS_LINES_URL
        data = _download_exspy_data(exspy_material_prop_url)
        if bool(data): # not empty dictionary
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        

def _load_json_data(file_path):
    """Load json data from file and extract "elements section"."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("elements", {})


def _load_elements_data():
    """Load and merge elements data from json files."""
    # Get the directory containing this file
    current_dir = Path(__file__).parent
    
    # Load data from JSON files
    general_properties_path = current_dir / "elements_general_properties.json"
    xray_lines_path = current_dir / "xray_lines.json"

    # Download properties if files do not exist
    _download_elements_general_properties()
    _download_elements_lines_properties()

    general_data = _load_json_data(general_properties_path)
    xray_data = _load_json_data(xray_lines_path)
    
    # Merge data into the expected structure
    elements = {}

    # Get all unique element symbols
    all_elements = (
        set(general_data.keys()) | set(xray_data.keys())
    )

    for element in all_elements:
        elements[element] = {}

        # Add general and physical properties
        if element in general_data:
            elem_data = general_data[element]
            if "General_properties" in elem_data:
                elements[element]["General_properties"] = elem_data[
                    "General_properties"
                ]
            if "Physical_properties" in elem_data:
                elements[element]["Physical_properties"] = elem_data[
                    "Physical_properties"
                ]

        # Add atomic properties
        atomic_props = {}

        # Add X-ray lines
        if element in xray_data:
            atomic_props["Xray_lines"] = xray_data[element]

        if atomic_props:
            elements[element]["Atomic_properties"] = atomic_props

    return elements


# Load elements data from JSON files
elements = _load_elements_data()

ELEMENTS = list(elements.keys())
ELEMENTS.sort()

elements_db = DictionaryTreeBrowser(elements)
elements_db.__doc__ = """
Database of element properties.

The following properties are included:

.. code::

    ├── Atomic_properties
    │   └── Xray_lines
    ├── General_properties
    │   ├── Z
    │   ├── atomic_weight
    │   └── name
    └── Physical_properties
        └── density_gcm-3

see : ~exspy._misc.elements
"""

def _weight_to_atomic(
    weight_percent : np.ndarray | tuple | list, 
    elements : list
) -> np.ndarray:
    """Convert weight/mass percent (wt%) to atomic percent (at.%).
    
    Note!
    This function is part of eXSpy.

    Parameters
    ----------
    weight_percent
        Weight fraction in percentage (?) 
    elements 
        A list of element abbreviations, e.g. ['Al','Fe']

    Returns
    -------
    atomic_percent 
        Composition in atomic percent.

    Examples
    --------
    Calculate the atomic percent of modern bronze given its weight percent:

    >>> exspy.material.weight_to_atomic((88, 12), ("Cu", "Sn"))
    array([ 93.19698614,   6.80301386])

    """
    atomic_weights = np.array(
        [
            elements_db[element]["General_properties"]["atomic_weight"]
            for element in elements
        ]
    )
    atomic_percent = np.array(list(map(np.divide, weight_percent, atomic_weights)))
    sum_weight = atomic_percent.sum(axis=0) / 100.0
    for i, el in enumerate(elements):
        atomic_percent[i] /= sum_weight
        atomic_percent[i] = np.where(sum_weight == 0.0, 0.0, atomic_percent[i])
    return atomic_percent


def weight_to_atomic(
    weight_percent : np.ndarray | list | tuple,
    elements : list | tuple
) -> np.ndarray:
    """Convert weight/mass percent (wt%) to atomic percent (at.%).
    
    Note!
    This function is part of eXSpy.

    Parameters
    ----------
    weight_percent
        The weight fractions.
    elements
        A list of element abbreviations, e.g. ['Al','Zn'].

    Returns
    -------
    atomic_percent
        Composition in atomic percent.

    Examples
    --------
    Calculate the atomic percent of modern bronze given its weight percent:

    >>> exspy.material.weight_to_atomic((88, 12), ("Cu", "Sn"))
    array([ 93.19698614,   6.80301386])

    See also
    --------
    exspy.material.atomic_to_weight

    """
    if len(elements) != len(weight_percent):
        raise ValueError(
            "The number of elements must match the size of the first axis"
            "of weight_percent."
        )
        
    return _weight_to_atomic(weight_percent, elements)


def _atomic_to_weight(
    atomic_percent : np.ndarray | list | tuple, 
    elements : list | tuple
) -> np.ndarray:
    """Convert atomic percent to weight percent.
    
    Note!
    This function is part of eXSpy.

    Parameters
    ----------
    atomic_percent : array
        The atomic fractions (composition) of the sample.
    elements : list of str
        A list of element abbreviations, e.g. ['Al','Zn']

    Returns
    -------
    weight_percent : numpy.ndarray of float
        Composition in weight percent.

    Examples
    --------
    Calculate the weight percent of modern bronze given its atomic percent:

    >>> exspy.material.atomic_to_weight([93.2, 6.8], ("Cu", "Sn"))
    array([ 88.00501989,  11.99498011])

    """
    
    atomic_weights = np.array(
        [
            elements_db[element]["General_properties"]["atomic_weight"]
            for element in elements
        ]
    )
    weight_percent = np.array(list(map(np.multiply, atomic_percent, atomic_weights)))
    sum_atomic = weight_percent.sum(axis=0) / 100.0
    for i, el in enumerate(elements):
        weight_percent[i] /= sum_atomic
        weight_percent[i] = np.where(sum_atomic == 0.0, 0.0, weight_percent[i])
    return weight_percent


def atomic_to_weight(
    atomic_percent : np.ndarray | list | tuple, 
    elements : list | tuple
) -> np.ndarray:
    """Convert atomic percent to weight percent.
    
    Note!
    This function is part of eXSpy.

    Parameters
    ----------
    atomic_percent : list of float or list of signals
        The atomic fractions (composition) of the sample.
    elements : list of str
        A list of element abbreviations, e.g. ['Al','Zn']. If elements is
        'auto', take the elements in en each signal metadata of the
        atomic_percent list.

    Returns
    -------
    weight_percent : numpy.ndarray of float
        Composition in weight percent.

    Examples
    --------
    Calculate the weight percent of modern bronze given its atomic percent:

    >>> exspy.material.atomic_to_weight([93.2, 6.8], ("Cu", "Sn"))
    array([ 88.00501989,  11.99498011])

    See also
    --------
    exspy.material.weight_to_atomic

    """
    if len(elements) != len(atomic_percent):
        raise ValueError(
            "The number of elements must match the size of the first axis"
            "of atomic_percent."
        )
    
    return _atomic_to_weight(atomic_percent, elements)