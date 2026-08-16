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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, re, glob, importlib, yaml

from pathlib import Path
from tqdm import tqdm_notebook as tqdm

from hyperspy._signals.signal1d import Signal1D
from hyperspy._signals.signal2d import Signal2D

PLUGINS : list = []
WRITE_EXTENSIONS : list = []
PARTSi = 5

# Look for yaml files and append file extensions and file writers:
specification_paths = list(Path(__file__).parent.rglob("specification.yaml"))
for path in specification_paths:
    if "_images" not in path.parts:
        with open(path) as file:
            # specification as dictionary
            spec = yaml.safe_load(file)
            # append directories
            spec["api"] = ".".join(path.parts[-PARTSi:-1]) 
            PLUGINS.append(spec)
            if spec["writes"]:
                for ext in spec["file_extensions"]:
                    WRITE_EXTENSIONS.append(ext)

def load(
    filename : str,
    **kwargs
):
    """Load particle analysis results such as chemical composition and 
    particles' geometric properties.

    Inspired by kikuchipy.

    Parameters
    ----------
    filename
        filename of the csv file
    **kwargs
        Keyword arguments passed on to :class:'ParticleAnalysis'.
        Example: "images" : list of MicroSpySignal2D 

    Returns
    -------
    ParticleAnalysis(out)
        particle analysis signal class. 
        Argument out is a (list of) MicroSpySignal1D

    
    Example
    -------
    >> import microspy as ms
    >> s = ms.load(filename)
    >> s
    <Particle analysis, title: sample_name, dimensions: (131)>
    """
    # Avoiding circular import
    from microspy.signals.particle_analysis import ParticleAnalysis 
    
    filename = str(filename)
    
    # Check if wildcard
    if not os.path.isfile(filename):
        is_wildcard = False
        filenames = glob.glob(filename)
        if len(filenames) > 0:
            is_wildcard = True
        if not is_wildcard:
            raise IOError(f"No filename matches {filename!r}")

    # Identify reader for file extension
    extension = os.path.splitext(filename)[-1].replace('.','')
    readers = []
    for plugin in PLUGINS:
        if extension.lower() in plugin["file_extensions"]:
            readers.append(plugin)

    reader = None
    if len(readers) == 1:
        reader = readers[0]
    else:
        raise IOError(f"Could not read {filename!r}. If the file "
                      "format is supported, please report this error.")
    
    # Import file reader
    file_reader = importlib.import_module(reader["api"]).file_reader
    
    # Read data and metadata from file per experiment
    signal_dicts = file_reader(filename) # list of experiments
    
    # Iterate through experiments:
    if isinstance(signal_dicts, list):
    
        out = []
        images = []
        for signal in signal_dicts:
            out.append(
                _dict2signals(signal)
            )
            
            images.append(signal.get("images"))
        
        if len(out) == 1: 
            out = out[0]
            images = images[0]
        
        kwargs.update({"images" : images})
        
        # Return ParticleAnalysis class:
        return ParticleAnalysis(out, **kwargs)

    else:
        raise TypeError("Make sure the api's are returning lists corr. "
                        "to experiments...")

def _dict2signals(
    signal_dict : dict
) -> list:
    """Create a signal instance from a dictionary. The dictionary is expected 
    to be structured as follows:

    signal_dict:
    ├── metadata/ 
    │   └── ...
    ├── original_metadata/
    │   └── ...
    ├── axes/ 
    ├── signal_type/ (e.g. chemistry) 
    │   ├── props # i.e. a list of quantified elements
    │   ├── data (n,m)
    │   └── unit
    └── signal_type/ (e.g. geometry)
        ├── props
        ├── data (n,o) 
        └── units

    Parameters
    ----------
    signal_dict
        Signal dictionary with "data", "metadata", "original_metadata"
        and axes keys. additional_data can also exist.
    
    Returns
    -------
    out 
        list of signal instances with at least "data", "metadata" and 
        "original_metadata".
    """

    if not 'data' in signal_dict:
        raise AttributeError("No data identified.")
    
    md = signal_dict['metadata'] if "metadata" in signal_dict else {}
    omd = signal_dict['original_metadata'] if 'original_metadata' in signal_dict else {}
        
    if 'axes' in signal_dict:
        axes = signal_dict['axes']

    out = []

    # Assign signal subclass
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
        if omd:
            out[-1].metadata.set_item("Original_metadata", omd)

    return out
    

def _assign_signal_subclass(
    signal_type : str = ''
):
    """Return matching signal subclass given by signal_type

    Parameters
    ----------
    signal_type
        signal type

    Returns
    -------
    signal_subclass
    """
    from microspy.signals._microspy_signals import (
        MicroSpySignal1D, 
        MicroSpySignal1D_Chemistry, 
        MicroSpySignal1D_Geometry
    )
    
    signal_subclasses = {
        'general' : MicroSpySignal1D, 
        'chemistry' : MicroSpySignal1D_Chemistry, 
        'geometry' : MicroSpySignal1D_Geometry
    }
    
    if signal_type not in signal_subclasses.keys():

        raise AttributeError(f"{signal_type} is not recognised.")

    return signal_subclasses[signal_type.lower()]

def _save(
    filename : str | Path,
    signal,
) -> None:
    """Write a signal to file in a supported format. The function is 
    used by the particle_analysis class object for saving. (See example.)

    Parameters
    ----------
    filename
        File path including filename
    signal
        Signal instance.
        
    Examples
    --------
    >> import microspy as ms
    >> s = ms.load(filename)
    >> s
    <Particle analysis, title: title, dimensions: (131)>
    
    >> s.save("new_filename.hdf5")
    >> s1 = s.load("new_filename.hdf5")
    >> s1
    <Particle analysis, title: title, dimensions: (131)>
    """
    filename = str(filename)
    directory = os.path.split(filename)[0]
    if not Path(directory).is_dir():
        raise OSError(f"Directory {directory!r} does not exist.")

    ext = os.path.splitext(filename)[1][1:]
    if ext == "": # Write as 
        ext = "hdf5"
        filename += f".{ext}"

    writer = None
    for plugin in PLUGINS:
        if ext.lower() in plugin["file_extensions"] and plugin["writes"]:
            writer = plugin
            break

    if writer is None:
        raise ValueError(
            f"{ext!r} does not correspond to any supported format. "
            f"Supported file extensions are: {WRITE_EXTENSIONS!r}"
        )
    else:
        # Check if file already exists
        is_file = os.path.isfile(filename)
        if is_file:
            ans = input(f"File {filename!r} already exists.\n"
                        "Overwrite file? (y/[n])")
            is_file = False if ans.lower() in ("y","yes") else True

        if not is_file:
            # Get file writer
            file_writer = importlib.import_module(writer["api"]).file_writer
                        
            file_writer(
                filename = filename, 
                signal = signal
            )