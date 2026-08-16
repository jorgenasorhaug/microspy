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

"""Reader and writer of microspy signals"""

import hyperspy.api as hs
import numpy as np
from pathlib import Path
import os, warnings
from h5py import File, Group

from hyperspy.signals import Signal2D

from microspy.signals.particle_analysis import ParticleAnalysis
from microspy.signals._microspy_signals import (
    Images_signals,    
    Images,
    MicroSpySignal2D,
    MicroSpySignal2D_Parent,
    MicroSpySignal1D,
    MicroSpySignal1D_Chemistry,
    MicroSpySignal1D_Geometry
)

from microspy import __version__ as microspy_version
from microspy.misc import exceptions
from microspy.signals.particle_analysis import IMAGES_SIGNAL_TYPES

from ._utils import (
    replace_none,
    _string2listWithStrings,
    _listWithStrings2string,
    sanitize
)

# Plugin descriptions
format_name = "microspy_hdf5"
manufacturer = "microspy"
file_extensions = ["hdf5"]

#----------- File Reading -----------

def file_reader(
    filename : str | Path,
    **kwargs
) -> ParticleAnalysis:
    """Read an :class:'~microspy.signals.ParticleAnalysis' signal.
    
    Not meant to be used directly; use :func:`~ParticleAnalysis.load`.

    Parameters
    ----------
    filename
        Full path of hdf5 file.
    """
    exceptions.formatted_warning("Multiple experiments not yet supported.")

    with File(filename, mode = "r", **kwargs) as f:
        file_dict = hdf5group2dict(f["/"], recursive=True)
    return dict2microspyList(file_dict)

def dict2microspyList(dictionary : dict) -> list:
    """Get a list of particle analysis data from a dictionary.

    Parameters
    ----------
    dictionary
        Dictionary with particle analysis information.
        
        Dictionary structure if saved using microspy:
        ---------------------------------------------
        dictionary
        ├──manufacturer
        ├──version
        └──signals/
            ├── MicroSpySignal1D/
            │   ├── axes_manager : dict
            │   ├── data : np.ndarray
            │   ├── name : str
            │   ├── props : str representing a list
            │   └── units : str | str representing a list
            ├── ...
            ├── MicroSpySignal2D/
            │   ├── axes_manager : dict
            │   ├── data : np.ndarray
            │   └── name : signal name typically repr. signal type 
            │              through Images_signals
            ├── ...
            ├── metadata : dict
            └── particle_classes : np.ndarray

    Returns
    -------
    -msList
        list of dictionaryies containing data, metadata, etc. compatible 
        with ParticleAnalysis class
    """
    from copy import deepcopy
    dictionary = deepcopy(dictionary)
    
    # Get manufacturer and version
    read_manufacturer = dictionary.get("manufacturer")
    read_version = dictionary.get("version")

    # Get particle classes: <Class name1> = array([...])
    pclasses = dictionary["signals"].get("particle_classes")
    particle_classes = np.empty(
        pclasses[list(pclasses.keys())[0]].size, 
        dtype=object
    )
    for cl, classified in pclasses.items():
        particle_classes[classified] = cl
    
    # Reorganise the signal's metadata
    md = dictionary.get("signals").get("metadata")
    md["General"].update(
        {
            "manufacturer" : read_manufacturer,
            "version" : read_version
        }
    )
    md["Sample"] = {
        "classes" : particle_classes,
        "acquisition_order" : dictionary.get(
            "signals"
        ).get(
            "metadata"
        ).get(
            "Acquisition_order"
            )
    }
    
    if md.get("Acquisition_order") is not None:
        del md["Acquisition_order"]
    
    # Temporarily get 1D signals (unnecessary conversion...)
    signals = dict2ListOfMicroSpySignal1D(
        dictionary.get("signals")
    )

    # Update Sample metadata
    md["Sample"]["elements"] = signals[0].metadata.get_item("Signal.props")
    
    # Set "data" key
    data = {
        "chemistry" : {
            "props" : md["Sample"].get("elements"),
            "data" : signals[0].data,
            "units" : signals[0].metadata.get_item("Signal.units")
        },
        "geometry" : {
            "props" : signals[1].metadata.get_item("Signal.props"),
            "data" : signals[1].data,
            "units" : signals[1].metadata.get_item("Signal.units")
        }
    }

    # Set images as attribute if they exists in the dictionary.
    has_images = dictionary["signals"].get("has_images") 
    if has_images:
        exceptions.formatted_warning(
            "Currently not supporting multiple experiments (Images)"
        )
        images = [
            dict2ListOfMicroSpySignal2D(dictionary["signals"])
            ]
    else: images = []
    
    # Currently only one experiment:
    msList = [
        {
            "metadata" : md,
            #"Original_metadata" : md.get("Original_metadata"),#No interest
            "data" : data,
            "images" : images
        }
    ]
    return msList

def dict2ListOfMicroSpySignal1D(
    dictionary : dict
) -> list:
    """Get a list of MicroSpySignal1D from necessary items in the dictionary.
    
    Not meant to be used directly.

    Parameters
    ----------
    dictionary
        Dictionary with MicroSpySignal1D information.
    
    Returns
    -------
    signals
        List of identified MicroSpySignal1D signals
    """
    from copy import deepcopy
    dictionary = deepcopy(dictionary)

    # Keywords to look for in the dictionary argument:
    kwds = ("Chemistry", "Geometry")
    signals = [] 
    
    for key, val in dictionary.items():
        if key in kwds:
            if key.lower() in "Chemistry".lower():
                sig = MicroSpySignal1D_Chemistry(
                    val.get("data"),
                    axes = [val.get("axes_manager")]
                )

            elif key.lower() in "Geometry".lower():
                sig = MicroSpySignal1D_Geometry(
                    val.get("data"),
                    axes = [val.get("axes_manager")]
                )

            # Set remaining metadata:
            sig.metadata.set_item("General.title", val.get("name"))
            sig.metadata.set_item("Signal.props", _string2listWithStrings(
                                        val.get("props"))
                                 )
            sig.metadata.set_item("Signal.units", _string2listWithStrings(
                                        val.get("units"))
                                 )
            signals.append(sig)

    return signals

def dict2ListOfMicroSpySignal2D(
    dictionary : dict
) -> list:
    """Get a list of MicroSpySignal2D from necessary items in a dictionary.
    
    Not meant to be used directly.

    Parameters
    ----------
    dictionary
        Dictionary with MicroSpySignal1D information.
    
    Returns
    -------
    signals
        List of identified MicroSpySignal2D signals
    """
    from copy import deepcopy
    dictionary = deepcopy(dictionary)

    # Keywords to look for:
    kwds = list(IMAGES_SIGNAL_TYPES.keys())
    signals = [] 
    
    for key, val in dictionary.items():
        if key in kwds: 
            sig = IMAGES_SIGNAL_TYPES[key](
                val.get("data"),
                axes = val.get("axes_manager").values(),
                **{"metadata" : 
                    {
                        "General" : {"title" : val.get("name")},
                        "Signal" : {"signal_type" : key}
                    }
                },
            )
            
            signals.append(sig)
            
    return signals
    
def hdf5group2dict(
    group: Group,
    dictionary: dict | None = None,
    recursive: bool = False,
    dont_read: list[str] | None = None,
) -> dict:
    """Return a dictionary with values from datasets in a group in an
    opened HDF5 file.
    
    Note meant to be used directly.

    Note!
    Copied from :orix:'_h5ebsd.hdf5group2dict'.

    Parameters
    ----------
    group
        HDF5 group object.
    dictionary
        To fill dataset values into. If not given, a new dictionary is
        created.
    recursive
        Whether to add subgroups to dictionary. Default is ``False``.
    dont_read
        List of strings of names of HDF data sets to not read.

    Returns
    -------
    dictionary
        Dataset values in group (and subgroups if ``recursive=True``).
    """
    if dictionary is None:
        dictionary = {}
    if dont_read is None:
        dont_read = []
    for key, value in group.items():
        # Check whether to extract subgroup or write value the dictionary
        if isinstance(value, Group):
            if recursive:
                dictionary[key] = {}
                hdf5group2dict(
                    group=group[key], 
                    dictionary=dictionary[key], 
                    recursive=recursive
                )
            else:
                dictionary[key] = value
        elif key not in dont_read:
            value = value[()]
            # Prepare value for entry in dictionary
            if isinstance(value, np.ndarray) and len(value) == 1:
                value = value[0]
            if isinstance(value, bytes):
                value = value.decode("latin-1")
            dictionary[key] = value
    return dictionary

#-------------- FILE WRITING ------------

def file_writer(
    signal : "ParticleAnalysis",
    filename : str | Path | None = None
) -> None:
    """Write an :class:'~microspy.signals.ParticleAnalysis' signal
    to a new hspy file.
    
    Not meant to be used directly; use :func:`ParticleAnalysis.save`.

    Parameters
    ----------
    signal
        ParticleAnalysis signal
    filename
        Full path of hspy file.
    """
    fname, fext = os.path.splitext(filename)
    if fext[1:] not in file_extensions:
        warnings.warn(f"File extension {fext[1:]} is not a supported."
                     "microspy file. Saving file with 'hdf5' extension.")
        fext = ".hdf5"
        
    try: f = File(filename, mode="w")
    except OSError:
        raise OSError(f"Cannot write to the already opened file '{filename}'.")
    
    manufacturer = signal.metadata.get_item("Acquisition_instrument.vendor")
    
    file_dict = {
        "manufacturer" : manufacturer,
        "version" : microspy_version,
        "signals" : microspySignals2dict(signal),
    }
    
    dict2hdf5group(
        file_dict,
        f["/"],
    )

    f.close()

def microspySignals2dict(
    signal : ParticleAnalysis,
    dictionary : dict | None = None,
    **kwargs
) -> dict:
    """Get a dictionary from a :class:'~particle_analysis.ParticleAnalysis'
    object with "data" and "header" keys with values.
    
    Not meant to be used directly.

    Parameters
    ----------
    signal
        Particle analysis signal
    dictionary
        Dictionary to update with microspy signal information. If not given
        (default), a new dictionary is created.
    kwargs
        Keyword arguments. See MicroSpySignal1D2dict and dict2MicroSpySignal1D

    Returns
    -------
    dictionary
        Dictionary with particle analysis information.
    """
    
    if not (isinstance(signal, ParticleAnalysis) 
        or signal is ParticleAnalysis):
        raise TypeError(f"Signal type {type(signal)} is not supported.")
    
    if dictionary is None: 
        dictionary = {}

    # Iterate through attributes and create dictionaries of them (1 and 2D):
    for attribute, value in vars(signal).items():
        if isinstance(value, MicroSpySignal1D):
            dictionary = MicroSpySignal1D2dict(
                signal = value, 
                dictionary = dictionary
            )
        elif isinstance(value, Images):
            for att, val in vars(signal.Images).items():
                if isinstance(val, MicroSpySignal2D):
                    dictionary = MicroSpySignal2D2dict(
                        signal = val,
                        dictionary = dictionary
                    )
             
    # If the signal contains Images:
    dictionary[
        "has_images"
        ] = True if hasattr(signal, "Images") else False
    
    # Append particle classes:
    dictionary.update(
        {
            "particle_classes" : {},
        }
    )

    for unique_class in np.unique(signal._particle_classes):
        dictionary["particle_classes"].update(
            {
                unique_class : signal._particle_classes == unique_class
            }
        )

    # Append metadata of interest:
    md = {}
    smd = signal._metadata.as_dictionary()

    md.update(
        {
            "General" : smd.get("General"),
            "Acquisition_instrument" : smd.get("Acquisition_instrument"),
            #"Original_metadata" : smd.get("Original_metadata"),#No interest
            #"Sample" : smd.get("Sample")#Lists with strings not supported
        }
    )
    
    # Append acquisition order:
    if dictionary["has_images"]:
        md.update(
            {
                "Acquisition_order" : smd.get("Sample").get("acquisition_order")
            }
        )
    
    dictionary["metadata"] = md

    return dictionary
    

def MicroSpySignal2D2dict(
    signal : MicroSpySignal2D,
    dictionary : dict | None = None,
) -> dict:
    """Get a dictionary of a MicroSpySignal2D signal.
    
    Not meant to be used directly.
    
    Parameters
    ----------
    signal
        MicroSpySignal2D to write to file
    dictionary
        Dictionary to update with information from a MicroSpySignal2D
        signal. If not given (default), a new dictionary is created.
    
    Returns
    -------
    dictionary
        Dictionary with information from a MicroSpySignal2D signal.
    """
    if not isinstance(signal, MicroSpySignal2D):
        raise TypeError(f"Signal type {type(signal)} is not supported.") 
        
    if dictionary is None:
        dictionary = {}
        
    signal_type = signal.metadata.get_item("Signal.signal_type")
    if not signal_type: 
        raise AttributeError("A signal type must be provided.")
    
    dictionary[signal_type] = {
            "name" : signal.metadata.get_item("General.title"),
            "data" : signal.data.copy(),
            "axes_manager" : signal.axes_manager.as_dictionary()
        }
    
    return replace_none(dictionary)

def MicroSpySignal1D2dict(
    signal : MicroSpySignal1D,
    dictionary : dict | None = None,
    **kwargs
) -> dict:
    """Get a dictionary of a MicroSpySignal1D signal.
    
    Not meant to be used directly.

    Note!
    Lists with strings are converted to single strings. To separate
    the objects, "|" is inserted in between.

    Parameters
    ----------
    signal
        MicroSpySignal1D to write to file
    dictionary
        Dictionary to update with information from a MicroSpySignal1D
        signal. If not given (default), a new dictionary is created.

    Returns
    -------
    dictionary
        Dictionary with information from a MicroSpySignal1D signal.
    """
    if not isinstance(signal, MicroSpySignal1D):
        raise TypeError(f"Signal type {type(signal)} is not supported.") 
        
    if dictionary is None:
        dictionary = {}

    signal_type = signal.metadata.get_item("Signal.signal_type")
    if not signal_type: 
        raise AttributeError("A signal type must be provided.")
        
    dictionary[signal_type] = {
        "name" : signal.metadata.get_item("General.title"),
        "data" : signal.data.copy(),
        "axes_manager" : signal.axes_manager.as_dictionary()
    }

    # Lists are not supported:
    units = signal.metadata.get_item("Signal.units")
    dictionary[signal_type]["units"] = _listWithStrings2string(
        units,
        **kwargs
    )

    props = signal.metadata.get_item("Signal.props")
    dictionary[signal_type]["props"] = _listWithStrings2string(
        props,
        **kwargs
    )
    
    return replace_none(dictionary)

def dict2hdf5group(
    dictionary : dict,
    group : Group,
    **kwargs
):
    """Write a dictionary to datasets in a new group in an opened HDF5
    file.
    
    Not meant to be used directly.

    Note!
    Copied from :orix:'orix_hdf5.dict2hdf5'.

    Parameters
    ----------
    dictionary
        Class signal types as keys with associated datasets, metadata and 
        axes_manager

    group
        HDF5 group to write dictionary to.
    **kwargs
        Keyword arguments passed to :meth:'h5py:Group.require_dataset'.
    """
    
    for key, val in dictionary.items():
        ddtype = type(val)
        dshape = (1,)
        if isinstance(val, dict):
            dict2hdf5group(val, group.create_group(key), **kwargs)
            continue  # Jump to next item in dictionary
        elif isinstance(val, str):
            # Check for special characters and replace them:
            print(val)
            val = sanitize(val)
            print(val)
            ddtype = "S" + str(len(val) + 1)
            val = val.encode()
        elif ddtype == np.dtype("O"):
            try:
                if isinstance(val, np.ndarray):
                    ddtype = val.dtype
                else:
                    ddtype = val[0].dtype
                dshape = np.shape(val)
            except TypeError:
                warnings.warn(
                    "The microspy HDF5 writer could not write the following "
                    f"information to the file: '{key} : {val}'."
                )
                break
        group.create_dataset(key, shape=dshape, dtype=ddtype, **kwargs)
        group[key][()] = val


