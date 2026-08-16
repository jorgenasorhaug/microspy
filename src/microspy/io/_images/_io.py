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
import matplotlib.pyplot as plt
import warnings, yaml, importlib

import os
from pathlib import Path

from microspy.io._utils import (
    _identify_subdirectories_of_interest
)
from microspy.signals._microspy_signals import (
    MicroSpySignal2D,
    MicroSpySignal2D_Parent,
)

from microspy.misc import exceptions

PLUGINS : list = []
WRITE_EXTENSIONS : list = []
PARTSi = 6

# Look for yaml files and append extensions and writers:
specification_paths = list(Path(__file__).parent.rglob("specification.yaml"))
for path in specification_paths:
    if "_images" in path.parts:
        with open(path) as file:
            # specification as dictionary
            spec = yaml.safe_load(file)
            # append directories
            spec["api"] = ".".join(path.parts[-PARTSi:-1]) 
            PLUGINS.append(spec)
            if spec["writes"]:
                for ext in spec["file_extensions"]:
                    WRITE_EXTENSIONS.append(ext)

def load_images(
    path : str | Path,
    **kwargs
) -> list:
    """Load the the stitched overview image, the individual view images, and 
    the particle images acquired during particle analysis.  

    Parameters
    ----------
    path 
        Path to stitched overview image and the folder 
        structure from pa

    Returns
    -------
    images 
        list of experiments/Images classes with the following
        ndarrays:
        
        patched_im
            np.ndarray representing the patched image
        view_images
            np.ndarray representing the individual 
            overview images
        particle_images
            np.ndarray representing the individual 
            particle images
    """
    from microspy.signals._microspy_signals import (
        Images_signals
    )

    fname = Path(path)
    if fname.is_file():
        folder, filename = os.path.splitext(path)
    elif fname.is_dir():
        # Presumably path to a folder structure
        folder = path
        filename = ""

    filename = folder + filename
    
    # Identify reader for the file extension
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

    set_dtype = kwargs.get("set_dtype")
    if set_dtype is not None:
        from microspy.io import _utils
        if not _utils._dtype_exists(set_dtype):
            print(f'Data type {set_dtype} was not recognised.' 
                  'Reading images as default datatype.')
            set_dtype = None 

    # Get file reader
    file_reader = importlib.import_module(reader["api"]).file_reader
    
    images = file_reader(
        filename,
        **kwargs
    ) 
    
    # Iterate through experiments and assign signal type
    for exp in range(len(images)):
        
        # Set images subclasses
        for (enum, im), imtype in zip(
            enumerate(images[exp]),
            Images_signals.keys()):

            sig_type = Images_signals.get(imtype)
            
            md = {
                "metadata" : 
                    {
                        "General" : {"title" : sig_type},
                        "Signal" : {"signal_type" : imtype}
                    }
            }
            
            if sig_type == list(Images_signals.values())[1]:
                im = MicroSpySignal2D_Parent(im, **md)
            else:
                # Try to set the signal. If it's empty, an exception
                # will occur.
                try: im = MicroSpySignal2D(im,**md)
                except ValueError: 
                    # Jump to the next iteration
                    continue

            images[exp][enum] = im
    
    return images
        
def _arrays2signals(
    array : np.ndarray | list
) -> list:
    """Assign an image subclass to the input array(s) if not already given.
    
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
    
    if isinstance(array, np.ndarray):
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
) -> MicroSpySignal2D | None:
    """Assign image subclass to a list of ndarrays. If subclass
    is already assigned, the argument is returned.

    Parameters
    ----------
    array
        List of ndarrays

    Returns
    -------
    signal
        image subclass or None if empty.
    """
    
    # Return signal if it's already been set
    if isinstance(array, MicroSpySignal2D):
        return array
        
    dim = np.ndim(array)

    if dim == 1:
        return None
    else:
        arr = MicroSpySignal2D(array)
    
    if dim == 2:
        arr.metadata.Signal.signal_type = f"Image_shape{np.shape(array)}".replace(
            ", ","x"
        ).replace('(','').replace(')','')

    elif dim > 2:
        arr.metadata.Signal.signal_type = f"ND_Image_shape{np.shape(array)}".replace(
            ", ","x"
        ).replace('(','').replace(')','')
        
    return arr

"""def _save(
    signal,
    filename : str | Path | None = None,
) -> None:
    Save Images class.

    Parameters
    ----------
    signal
        MicroSpy signal to save
    filename
        Complete path and filename of file.
    

    fname, ext = os.path.splitext(str(filename))
    if ext == "":
        ext = ".hdf5"

    # Identify reader for file extension
    extension = ext.replace(".","")
    writers = []
    for plugin in PLUGINS:
        if extension.lower() in plugin["file_extensions"]:
            writers.append(plugin)

    writer = None
    if len(writers) == 1:
        writer = writers[0]
    else:
        raise IOError(f"Could not read {filename!r}. If the file "
                      "format is supported, please report this error")

    # Get file writer
    try: 
        file_writer = importlib.import_module(writer["api"]).file_writer
        print("Delete this try-except in the module")
    except AttributeError:
        file_writer = importlib.import_module(writer["api"] + "._api").file_writer

    # Write file
    file_writer(
        signal = signal,
        filename = fname + ext
    )"""