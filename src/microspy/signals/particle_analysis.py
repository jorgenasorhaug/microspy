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
from tabulate import tabulate
import warnings, os
from pathlib import Path
from tqdm import tqdm_notebook
from hyperspy.misc import utils

from microspy.signals._microspy_signals import (
    MicroSpySignal2D, 
    MicroSpySignal2D_Parent,
    Images,
    MicroSpySignal1D,
    MicroSpySignal1D_Chemistry, 
    MicroSpySignal1D_Geometry,
    Images_signals
)
from microspy.io._io import _save
from microspy.io._images._utils import ALLOWED_VENDORS as ALLOWED_IMAGE_VENDORS
from microspy.misc import exceptions, ELEMENTS


IMAGES_SIGNAL_TYPES = {
    "CompositeSig" : MicroSpySignal2D, #Overview/stitched im.
    "ParentSig" : MicroSpySignal2D_Parent, # Individual images
    "ChildSig" : MicroSpySignal2D, # Cropped child images
    "ChildMap" : MicroSpySignal2D_Parent, # Map of child images
}

class ParticleAnalysis:
    """Particle analysis class.

    This class keeps track of microspy signals (extending on HyperSpy's  
    Signal1D classes for chemistry and geometry, and particle classifications. 
    The 'Images' class can also be set as an attribute to manage particle 
    images (extending on HyperSpy's Signal2D class). 

    The class has the ability to read and allocate Images
    class (MicroSpySignal2D) attribute.

    Parameters
    ----------
    signals
        List of microspy 1D signals.
        
    **kwargs
        Additional information to pass on to the signal class.

        'Unclassified_kw' : The label to use for unclassified particles.
        'images'          : list of MicroSpySignal2D signals to initialise
                            'Images' class attribute.
        'metadata_stored_in_signal_type'     
                          : signal type argument passed to 
                            :func:'_create_and_reorganise_metadata' 
                            (typically not needed to specify)
        
        Note!
        The signals in 'images' should have unique titles and signal_types
        in their metadata to be properly handled.

    Structure
    ---------
    ParticleAnalysis
    ├── MicroSpySignal1D_Chemistry/
    ├── MicroSpySignal1D_Geometry/
    └── Images/
        ├── MicroSpySignal2D
        └── MicroSpySignal2D_Parent

    Example
    -------
    >>> import microspy as ms
    
    >>> arr = np.arange(1,13).reshape(4,3)
    >>> chem = ms.signals.MicroSpySignal1D_Chemistry(
            arr, 
            **{"title" : "test"}
        )
    >>> chem.metadata.set_item("Sample.elements", ["elem0", "elem1"])
    >>> chem.metadata.get_item("General.title")
    'test'
    
    >>> geom = ms.signals.MicroSpySignal1D_Geometry(arr)
    >>> s = ms.signals.ParticleAnalysis([chem, geom])
    >>> s
    <Particle analysis, title: test, dimensions: (4)>
    
    >>> s.elements
    ['elem0', 'elem1']
    """
    def __init__(self, signals, **kwargs) -> None:
        from copy import deepcopy
        # Check if single signal is used or not
        if isinstance(signals, 
                      (MicroSpySignal1D_Geometry, 
                       MicroSpySignal1D_Chemistry)): 
            signals = [signals]
        
        in_size = signals[0].axes_manager.navigation_shape
        
        # Assign signals as attributes
        for enum, signal in enumerate(signals):
            
            signal_type = signal.metadata.get_item("Signal.signal_type")
            
            if signal_type is None:
                signal_type = f"sig{enum}"
                exceptions.formatted_warning(
                    "Detected signal with no signal type. Assigning signal "
                    f"as '{signal_type}'"
                )
            
            # Set chemistry and geometry trackers:
            setattr(
                self, 
                signal_type, 
                signal
            )
            
            curr_size = signal.axes_manager.navigation_shape
            if curr_size != in_size:
                raise exceptions.InputError(
                        "Incompatible signal navigation shapes detected!\n "
                       f"'{signal}' has a different size than '{signals[0]}'."
                    )

        # Define unclassified keyword:
        Unclassified_kw = kwargs.get(
                'Unclassified_kw'
            ) if 'Unclassified_kw' in kwargs else "Unclassified" 
            
        # Set particle classes
        classes = deepcopy(signals[0].metadata.get_item("Sample.classes"))
        if classes is None:
            classes = np.full(
                shape = in_size,
                fill_value = Unclassified_kw
            )
        
        self._set_original_classes(classes = classes)
        self._set_particle_classification(
            class_array = classes, 
            Unclassified_kw = Unclassified_kw
        )
        
        # Assign class 'Images' as attribute if provided in kwargs:
        images = kwargs.get("images")
        if images is not None:
            if len(images) > 0:
                self._set_Images(images)
        
        # Initialise metadata structure
        self._create_and_reorganise_metadata(
            signals,
            **{"metadata_stored_in_signal_type" : kwargs.get(
                    "metadata_stored_in_signal_type")
              }
        )
        
    def __repr__(self): 
        # Nice printing of information 
        
        exceptions.formatted_warning(
            "OBS! Mutliple stubs has not been tested yet!"
        )

        grid_string = ""
        cal_string = ""
        num_particles = self.num_particles

        if hasattr(self, "Images"): 
            if hasattr(self.Images, "ParentSig"):
                # 'MicroSpySignal2D_Parent, title: ..., 
                # dimensions: (X, Y|x, y | #)'
                gs_i = str(self.Images.ParentSig).index("(") + 1
                grid_string = str(self.Images.ParentSig)[gs_i:-2] + '|'
            
            if self.Images.is_calibrated: 
                
                unit = self.Images.metadata.get_item(
                    "Acquisition_instrument.Acquisition.unit"
                )
                
                # Num. pixels
                scan_area = np.prod(
                    self.Images.ParentSig.data.shape
                )
                # Scaled:
                scan_area *= np.square(
                    self.Images.metadata.get_item(
                        "Acquisition_instrument.Acquisition.scale"
                    )
                )

                # Number density
                particle_density = num_particles / scan_area
                
                cal_string = f"\nScan unit: {unit}\n"
                
                # np.format_float_position?
                cal_string += f"Particle number density: {round(
                    number = particle_density, 
                    ndigits = int(
                        str(particle_density).split('.')[1][0]
                    ) + 1 
                )} "
                
                # Area density
                area_index = self.Geometry.prop.index("Area")
                area_density = np.sum(
                    self.Geometry.data[area_index]
                ) / scan_area
                
                cal_string += f"1/{unit}\u00b2\n"
                cal_string += f"Particle area density: {round(
                    number = 100 * area_density, 
                    ndigits = int(
                        str(area_density).split('.')[1][0]
                    ) + 1
                )} %"
        
        pa_string = f"<Particle analysis, title: {self.metadata.General.title}, "
        pa_string += f"dimensions: ({grid_string}{num_particles})>{cal_string}"
        
        return pa_string

    # ---------------------------------------------------------------- #
    # ----------------------- Custom attributes ---------------------- #
    # ---------------------------------------------------------------- #

    @property 
    def is_classified(self) -> np.ndarray:
        """Returns a boolean ndarray of classified particles.
        """
        return self._is_classified
        
    @property
    def metadata(self):
        """The metadata of the signal.
        """
        return self._metadata

    @property
    def particle_classes(self) -> np.ndarray:
        """Returns an ndarray of particle classes.
        """
        return self._particle_classes
        
    @property
    def elements(self) -> np.ndarray:
        """Return the list of elements stored in the class' metadata.
        """
        return self.metadata.get_item("Sample.elements")
    
    @property
    def num_particles(self) -> int:
        """Return the total number of particles"""
        return len(self._original_classes)
        
        
    # ---------------------------------------------------------------- #
    # ------------------------ Private methods ----------------------- #
    # ---------------------------------------------------------------- #

    def _create_and_reorganise_metadata(
        self, 
        signals : list,
        metadata_stored_in_signal_type : str | None = None
    ) -> None:
        """Create the class' metadata from the metadata stored in the signal 
        with signal type as defined by argument 'signal_type'.

        Note!
        Not meant to be used directly.
        
        As the function is called, all metadata stored in the attribute 
        MicroSpySignal1D signal type 'signal_type' will be allocated to 
        this class whilst the metadata in the MicroSpySignal1D classes 
        will become updated/removed. The only metadata the MicroSpy 
        signals will keep are the 'Signal' node. 
        
        This class will point to the attribute MicroSpySignal1D classes' 
        Signal node in the metadata node Signals. 
        
        Other metadata (except the General node) in the MicroSpySignal1D 
        attribute classes are pointing to this class' metadata.
        
        Parameters
        ----------
        signals
            List of MicroSpySignal1D signals, like Chemistry and/or 
            Geometry
        metadata_stored_in_signal_type
            Which signal the metadata is to be read. This is typically not
            needed as all signals are usually allocated the same metadata
            except for the signal type.
        """
        
        from copy import deepcopy
        
        if metadata_stored_in_signal_type is None:
            signal_index = 0
        else:
            signal_index = -1
            for enum, sig in enumerate(signals):
                if sig.metadata.get_item(
                        "Signal.signal_type"
                    ) == metadata_stored_in_signal_type:
                    signal_index = enum
            if signal_index == -1:
                raise AttributeError(
                    "None of the signals have a signal type that matches "
                    f"'{metadata_stored_in_signal_type}.'"
                )

        # deepcopy metadata:
        self._metadata = deepcopy(signals[signal_index]._metadata)
        md = self.metadata

        # Define Signals node
        del md.Signal
        md.add_node('Signals')

        # Store additional data as a new node
        #if hasattr(md, 'Additional_data'):
        #    self._additional_data = md.Additional_data

        # Iterate through the signals and let all point to this class' 
        # metadata, except from the Signal node.
        for sig in signals:

            # Get signal type and delete it from the attribute
            sig_type = sig.metadata.get_item("Signal.signal_type")
            #del sig.metadata.Signal.signal_type

            # Define Signals instead of Signal
            md.set_item(
                f'Signals.{sig_type}', 
                deepcopy(sig.metadata.get_item("Signal"))
            )

            # Set the attribute signal metadata to reference the class' 
            # metadata
            smd = sig.metadata
            if hasattr(smd, 'Additional_data'):
                del smd.Additional_data
                
            smd.Acquisition_instrument = md.get_item("Acquisition_instrument")
            smd.General = md.get_item("General")
            #smd.Original_metadata = md.Original_metadata
            smd.Sample = md.get_item("Sample")

    def _set_original_classes(
        self,
        classes : np.ndarray
    ):
        """Sets an ndarray of original class names as read from 
        initialisation.
        
        Note!
        The method is not meant to be used directly.
        
        Parameters
        ----------
        classes
            ndarray of original class names.
        """
        self._original_classes = classes
    
    def _set_particle_classification(
        self, 
        class_array : np.ndarray,
        Unclassified_kw : str = 'Unclassified'
    ) -> None:
        """Set particle class names. 
        
        Note! 
        The entire 'class_array' is set as the particle classes.
        
        The method is not meant to be used directly.
        
        Parameters
        ----------
        class_array 
            Array of class names to be set.
        Unclassified_kw
            Class name of the 'unclassified' particles for 
            :func:'is_classified'.
        """
        if len(class_array) != self.num_particles:
            raise exceptions.InputError(
                    f"Input argument 'class_array' ('{class_array.shape}') "
                    "must be compatible with the total number of "
                    f"particles: ('({self.num_particles},)')"
                    
                )
            
        self._is_classified = class_array != Unclassified_kw
        self._particle_classes = class_array.copy()

        # Set phase_maps as 'not updated':
        if hasattr(self, "Images"):
            self.Images._updated_phase_maps = False
            
    def _set_Images(
        self,
        images : list(list, ...)
    ):
        """Set list of MicroSpySignal2D signals to an Images class and 
        set the class as an attribute.
        
        Note!
        The method is not meant to be used directly.
        
        If multiple experiments have been conducted, i.e. the length of 
        the images list > 1, multiple Images class will be attributed to
        this class and labelled according to the experiment (same order as 
        the images list order).

        images
            list of lists with MicroSpySignal2D classes.
        """
        if images is None:
            raise AttributeError("None argument is not allowed.")

        elif not isinstance(images, list):
            raise AttributeError(
                    f"Images argument {type(images[0])} is not supported.")

        else:
            num_experiments = len(images)
            # Iterate through multiple experiments:
            for exp in range(num_experiments):
                if num_experiments == 1: # Single experiment:
                    self.Images = Images(images[0])
                else: # Multiple experiments:
                    if exp < 10: _exp = f"0{exp}"
                    else: _exp = str(exp)
                    setattr(self, f"Images{_exp}", Images(images[exp]))
        
        
    # ---------------------------------------------------------------- #
    # -------------------------- Open methods ------------------------ #
    # ---------------------------------------------------------------- #  

    def set_min_concentration(
        self,
        threshold : int | float | list[float | int, ...] | tuple[float | int, ...] = 0.0,
        elements : list | tuple | str | dict | None = None,
        **kwargs
    ) -> None:
        """Set a minimum required concentration value for specified 
        element(s) (or a global one for all).
        
        Parameters
        ----------
        threshold
            A single threshold for all considered elements, or a list of 
            individual thresholds for individual elements.
        elements
            single element, list of elements, a dictionary of elements with 
            associated threshold values, or None (default). If None,
            the minimum abundance will be mapped onto all elements.
        kwargs
            keyword arguments passed onto 
            :func:'MicroSpySignal1D_Chemistry._update_particles_concentration'
        """
        
        if not hasattr(self, "Chemistry"):
            
            raise AttributeError(
                    "The class has no 'Chemistry' signal attribute."
                )
        else:
            
            if np.max(threshold) > 0.25 * np.max(self.Chemistry.data):
                exceptions.formatted_warning(
                    "The used threshold is > 25% of the maximum composition."
                )
            
            if elements is not None:
                if isinstance(elements, str):
                    elements = [elements]
                    
                elif isinstance(elements, dict):
                    exceptions.formatted_warning(
                        f"Input argument threshold ('{threshold}') is ignored."
                    )
                    elements, threshold = list(zip(*elements.items()))
                    elements, threshold = list(elements), list(threshold)
                else:
                    elements = list(elements)

                # Check if the provided elements exists:
                diff = set(elements) - set(ELEMENTS)
                if len(diff) > 0:
                    raise exceptions.InputError(
                            f"Element(s) '{list(diff)}' does/do not exist."
                        )
                
                # Check if the provided elements have been detected:
                particle_elements = list(self.elements)
                diff = set(elements) - set(particle_elements)
                if len(diff) > 0:
                    exceptions.formatted_warning(
                            f"Element(s) '{list(diff)}' has/have not been detected "
                            "and will be removed from the list."
                        )
                for elem in diff:
                    threshold.remove(threshold[elements.index(elem)])
                    elements.remove(elem)
            
            old_shape = self.Chemistry.data.shape
            
            self.Chemistry.threshold_data(
                threshold = threshold,
                elements = elements,
                **kwargs
            )
            
            # If elements have been removed, i.e. Chemistry's data shape 
            # is changed, update the class' metadata
            if self.Chemistry.data.shape != old_shape:
                exceptions.formatted_warning(
                    "Future fix: :func:'_create_and_reorganise_metadata'"
                )
                self.metadata.set_item(
                    "Signals.Chemistry",
                    self.Chemistry.metadata.get_item("Signal")
                )
                self.metadata.set_item(
                    "Sample.elements",
                    self.Chemistry.metadata.get_item("Sample.elements")
                )
            
    def filter_by_elements(
        self,
        elements : list | tuple | str,
        mode : str = "only",
    ) -> np.ndarray:
        """Get particles that match the provided element conditions, like 
        particles with specific elements (exclusively), or particles with 
        specific elements (and possibly others), as specified by the 'mode'
        argument.
        
        Parameters
        ----------
        elements
            List of elements
        mode
            "all" : particles containing all elements, but not exclusively. 
            "only" : particles containing exclusively the specified elements 
                    (default). 
        
        Returns
        -------
        particles fulfilling the element requirements.
        """
        
        from copy import deepcopy
        
        if not hasattr(self, "Chemistry"):
            raise AttributeError(
                    "The class has no 'Chemistry' attribute."
                )
        else:
            return self.Chemistry.get_particles_with_elements(
                elements = deepcopy(elements),
                mode = mode,
            )

    def reset_particle_classes(
        self, 
        original_classes : bool = True,
        Unclassified_kw : str = "Unclassified"
    ) -> None:
        """Reset particle classes to the original classification.
        
        Parameters
        ----------
        original_classes
            Whether to reset the particle classes to the originally set 
            classes from file reading or not. If not, all are set as
            'Unclassified_kw'.
        Unclassified_kw 
            String name of unclassified particles.
        """
        if original_classes:
            print("Setting the currently unique particle classes "
                  f"'{np.unique(self._particle_classes)}' -> "
                  f"{np.unique(self._original_classes)}")
            labels = self._original_classes
        else:
            print("Setting the currently unique particle classes "
                  f"'{np.unique(self._particle_classes)}' -> "
                  f"{Unclassified_kw}")
            labels = Unclassified_kw
            
        particles = np.full(
            shape = (self.num_particles,),
            fill_value = True,
            dtype = bool
        )
        
        self.classify_particles(
            particles = particles,
            labels = labels
        )

    def classify_particles(
        self,
        particles : np.ndarray | tuple | list,
        labels : str | tuple | list | np.ndarray,
        **kwargs
    ):
        """Classify specified particles.

        Parameters
        ----------
        particles
            (Boolean) array of particles to classify. 
            If the array have integer values, they represent particle 
            indices.
        labels
            Class label to classify the specified particles with.
        kwargs
            Keywords passed on to :func:'_set_particle_classification'.

        Examples
        --------
        >>> import microspy as ms
        >>> s = ms.load(filename)
        >>> s.is_classified
        array([False, False, True, ..., False, True, False])
        
        >>> s.particle_classes
        array(['Unclassified', 'Unclassified', 'Classified', ...,
       'Unclassified', 'Classified', 'Unclassified'], dtype=object)
        
        >>> # Classify all 'Unclassified' particles using single label string
        >>> s.classify_particles(~s.is_classified, labels = 'Classified')
        >>> s.is_classified
        array([True, True, True, ..., True, True, True])
        >>> s.particle_classes
        array(['Classified', 'Classified', 'Classified', ...,
        'Classified', 'Classified', 'Classified'], dtype=object)
        
        >>> # Reset particle classes to original (as-read) classification:
        >>> s.reset_particle_classifications() 
        >>> # Classify all 'Unclassified' particles using :func:'numpy.where'
        >>> s.classify_particles(np.where(
                ~s.is_classified), 
                labels = 'Classified'
            )
        >>> s.is_classified
        array([True, True, True, ..., True, True, True])
        
        >>> # Classify particles using lists of particle indices
        >>> s.classify_particles([0,2], "New_class")
        >>> s.particle_classes
        array(['New_class', 'Classified', 'New_class', ...,
        'Classified', 'Classified', 'Classified'], dtype=object)
        """

        tot_particles = self.num_particles
        
        pshape = np.shape(particles)
        if pshape == (0,):
            raise exceptions.InputError(
                    "Argument 'particles' can not be empty."
                )
        lshape = np.shape(labels)
        if not isinstance(labels, str):
            if lshape == (0,):
                raise exceptions.InputError(
                        "Argument 'labels' cannot be empty."
                    )
            
        # Checking input array
        if isinstance(particles, np.ndarray): 
            if len(particles) != tot_particles: # Specific indices
                if np.max(particles) >= tot_particles - 1:
                    raise ValueError(
                            f"The provided array of shape {particles.shape} ",
                            "is not compatible with the total number of "
                            f"particles ('{tot_particles}')."
                        )
                num_particles = len(particles) 
                particles = (particles,) #-> "np.where()"
            else: # boolean array 
                if not particles.dtype == np.bool_:
                    raise AttributeError(
                            "Input array 'particles' must be boolean, "
                            f"and not '{particles.dtype}'."
                        )
                num_particles = np.sum(particles)

        elif isinstance(particles, tuple | list): 
            if np.min(particles) < 0:
                raise IndexError(
                        "Unable to index negative indices."
                    )
            elif np.max(particles) > tot_particles - 1:
                raise IndexError(
                        f"Index '{np.max(particles)}' exceeds the number of "
                        f"total particles ({tot_particles})."
                    )
            
            """# List of conditions:
            if (isinstance(particles[0], list | tuple | np.ndarray) and 
                isinstance(labels, list | tuple) and 
                len(particles) == len(labels)
                ):
                for i in range(len(particles)):
                    self.classify_particles(
                        particles = particles[i]
                        labels = labels[i],
                        Unclassified_kw = Unclassified_kw
                    )
            else:
                raise ShapeError(
                        "List of particles to classify have a shape that is "
                        "uncompatible with the list of labels."
                    )"""
            
            if isinstance(particles, tuple):
                if isinstance(particles[0], np.ndarray):
                    # np.where() is assumed used:
                    if particles[0].max() >= tot_particles:
                        raise ValueError(
                                "The indices can not exceed the total number "
                                f"of particles ('{tot_particles}').")
                else:
                    # List of indices: #-> "np.where()"
                    particles = (np.asarray(particles),)
            else:
                # List of indices:
                particles = (np.asarray(particles),)
                
            num_particles = len(particles[0])
            
        # Check labels:
        if isinstance(labels, tuple | list | np.ndarray):
            # The total number is lower than the total number of particles
            if len(labels) != tot_particles:
                # The total number is unequal to the number of particles to 
                # label:
                if len(labels) != num_particles:
                    # It's a single label
                    if len(labels) == 1:
                        labels = np.asarray(labels * num_particles)
                    else:                      
                        raise ValueError(f"The number of labels ({len(labels)}) "
                                         "do not match the the number of "
                                         "particles to classify "
                                        f"({num_particles}).")
            
            # The shape of labels should match particles:
            labels = np.asarray(labels)
            
        else: 
            # Single class name string:
            labels = str(labels)
        
        # Define array of particle classes
        class_array = self.particle_classes
        
        # Set new class labels
        class_array[particles] = labels
        
        # Set particle classes
        self._set_particle_classification(
            class_array = class_array, 
            **kwargs
        )
    
    def save(
        self,
        filename : Path | str | None = None,
        extension: str | None = "hdf5"
    ) -> None:
        """Write the class to a file in the specified format. If no 
        extension is provided, the signal is written to the 'hdf5' format.

        Parameters
        ----------
        filename
            Path and filename. If None, the original filename
            is used (path + filename)
        extension
            File extension that defines the file format. The 
            options are:

            * csv: Jeol's text format
            * hdf5: 

            Each format accepts different parameters.

            If not given, the extension is determined by the
            filename.

        See also
        --------
        :func:'microspy.io.plugins.*'
        
        Notes
        -----
        This function is a modified version of :func:`kikuchipy.io._io.save`.
        """
        
        if filename is not None:
            fname, ext = os.path.splitext(str(filename))
        else:
            md = self.metadata.get_item("General")
            if md.has_item("original_filename"):
                fname = md.get_item("original_filename")
                ext = os.path.splitext(fname)[-1].replace(".","")
                if ext == "":
                    ext = "." + extension
            else:
                ValueError("filename not given.")
        
        _save(
            filename = fname + ext,
            signal = self
        )
        
    # ----------------------------------------------------------------- #
    # ------------------------ 'Images' methods ----------------------- #
                
    def load_images(
        self, 
        directory : str | Path | None = None,
        **kwargs
    ):
        """Load the acquired images and corresponding particle images from 
        the data acquisition (particle analysis).

        Parameters
        ----------
        directory
            Directory to where the images are stored. If None, the function 
            will use the directory in the original metadata and search for 
            the images.

        keyword arguments:
            read_particle_images : bool,
            centre_particle_images : bool,
            set_dtype : dtype
        """ 
        from microspy.io._images import _io

        # Check if loading images is necessary
        _load_images : bool = False
        if hasattr(self, "Images"):
            _continue = input("'Images' class has already been set. "
                              f"By reloading the images, all associated "
                              "'Images' data will become removed.\nProceed? "
                              "(y/[n])")
            if _continue.lower() in ("y", "yes"):
                _load_images = True
        else:
            _load_images = True

        if _load_images:
            # Search for a potential directory matching vendor solution:
            if directory == None:
                from microspy.io._images._utils import (
                    _image_directory_searcher as directory_searcher
                )
    
                print("Searching for images according to vendor solution.")
                vendor = self.metadata.get_item(
                    "Acquisition_instrument.vendor"
                )
                #kwargs["vendor"] = vendor
    
                if vendor == "": 
                    raise AttributeError(
                            "No vendor information is accessible. A directory "
                            "must be provided."
                        )
                    
                directory = self.metadata.get_item(
                    "General.original_filename"
                ) 
                directory = os.path.split(directory)[0]
                
                print(f"Identified directory: <{directory}>")
    
                directory_searcher = directory_searcher(vendor) 
    
                directory = Path(
                    directory_searcher(directory)
                ).parent
    
            directory = str(directory)
            
            # Loading and setting arrays as Images classes
            images = _io.load_images(
                path = directory,
                **kwargs
            )
    
            self._set_Images(images)
    
    def calibrate_navigation(
        self,
        scale : int | float,
        unit : str = "NA"
    ):
        """Calibrate the navigation signal.
        
        Parameters
        ----------
        scale
            Navigation scale (unit per pixel).
        unit 
            Spatial unit.
        """
        if not hasattr(self, "Images"):
            raise AttributeError("The signal does not keep "
                                "track of images. See "
                                "the function *.load_images().")

        self.Images.calibrate_signals(
            scale = scale,
            unit = unit
        )
    
    def gridify_ParentSig(
        self,
        navigation_shape : list | tuple | None = None,
        flip_axes : str | tuple | None = None,
    ):
        """Gridify the acquisition signals from 3 dimensions to 4 dimensions.

        Parameters
        ----------
        navigation_shape
            Shape of the navigation grid. (Obs! numpy convention)
        flip_axes
            Axes to flip
            
        Note
        ----
        Unless provided, the flip_axes argument will be attempted determined
        from information about the vendor. See 
        :func:'microspy.misc._misc._vendor2ImFlipAxes'
        """
        if not hasattr(self, "Images"):

            raise AttributeError("The signal does not keep "
                                "track of images. See "
                                "the function *.load_images().")

        if flip_axes is None:
            
            from microspy.misc._misc import _vendor2ImFlipAxes

            vendor = self.metadata.Acquisition_instrument.get_item("vendor")        
            flip_axes = _vendor2ImFlipAxes(vendor)
            
            self.Images.gridify_ParentSig(
                nav_shape = navigation_shape,
                flip_axes = flip_axes
            )

    def degridify_ParentSig(
        self,
        flip_axes : int | tuple | list | None = None,
    ):
        """Degridify the acquisition signals from 4 dimensions to 3 
        dimensions.
        
        Parameters
        ----------
        flip_axes
            axes to flip
            
        Note
        ----
        Unless provided, the flip_axes argument will be attempted determined
        from information about the vendor. See 
        :func:'microspy.misc._misc._vendor2ImFlipAxes'
        """
        if not hasattr(self, "Images"):
            raise AttributeError("The signal does not keep "
                                "track of images. See "
                                "the function *.load_images().")

        if flip_axes is None:
            
            from microspy.misc._misc import _vendor2ImFlipAxes
            
            vendor = self.metadata.Acquisition_instrument.get_item("vendor")
            flip_axes = _vendor2ImFlipAxes(vendor)
        
        self.Images.degridify_ParentSig(
            flip_axes = flip_axes
        )

    def map_particles(
        self,
        vendor : str = "",
        label_particles : bool = True,
        matrix_label : int = -1,
        **kwargs
    ):
        """Map the position of the particle images (ChildSig) onto the 
        Parent signal. The method uses template_match to identify the 
        particles in the Parent signal.

        Parameters
        ----------
        vendor 
            Vendor of the data acquisition. Used to determine the acquisition
            order.
        label_particles
            Whether to label the individual particles or not.
        matrix_label
            Label of the matrix, i.e. non-particles. All particles will have
            labels larger than the matrix label. Default: -1.
            
        Note
        ----
        The mapped particles' position will be set as a class type
        'MicroSpySignal2D_Parent', and allocated to the attribute class 
        'Images'.
        
        The minimum label value will represent the matrix.
        
        Unless the acquisition_order is provided in the kwargs, and the 
        acquisition order is accessible from stored metadata, it will be 
        stored in the metadata node "Sample.acquisition_order"
        
        Example
        -------
        >> import microspy as ms
        >> s = ms.load("particle_analysis.csv")
        >> s.load_images(True, True, np.uint8)
        >> s.Images
        <Images class, title: , dimensions: (3|)>:
        ├── CompositeSig: <title: Overview_image, dimensions: (|351, 8192)
        ├── ParentSig: <title: Acquisition, dimensions: (280,|768, 1024)
        └── ChildSig: <title: Cropped_ROIs, dimensions: (131,|78, 122)
        
        >> s.map_particles()
        >> s.Images
        <Images class, title: , dimensions: (4|)>:
        ├── CompositeSig: <title: Overview_image, dimensions: (|351, 8192)
        ├── ParentSig: <title: Acquisition, dimensions: (4, 70|768, 1024)
        ├── ChildSig: <title: Cropped_ROIs, dimensions: (131,|78, 122)
        └── ChildMap: <title: Acquisition, dimensions: (4, 70|768, 1024)
        """
        map_regions = False
        
        if not hasattr(self, "Images"):
            raise AttributeError("The signal doesn't keep track of "
                                 "any images. See :func:'load_images'")

        map_kw = list(Images_signals.keys())[3]
        
        if hasattr(
            self.Images, 
            map_kw
        ):
            ans = input("Particle regions seems to have been "
                                "mapped already.\n Proceed? (y/[n])")
            map_regions = True if ans.lower() in ("y", "yes") else False
        else: map_regions = True

        if map_regions:
            exceptions.formatted_warning(
                "Currently only supporting Jeol's solution."
            )
            if not vendor:
                vendor = self.metadata.get_item(
                    "Acquisition_instrument.vendor"
                )
            
            # The following can be moved to e.g. _misc?
            if vendor.lower() in ALLOWED_IMAGE_VENDORS:
                
                # Get acquisition order
                kwargs["acquisition_order"] = self.metadata.get_item(
                    "Sample.acquisition_order"
                )
                if kwargs["acquisition_order"] is None:
                
                    kwds = self.metadata.get_item(
                        "Additional_data.keywords"
                    )
                    
                    if kwds is not None:
                        namesId = kwds.index("Label name")
                        label_names = self.metadata.get_item(
                            "Additional_data"
                        ).data[:,namesId]
                        stringLength = len(max(label_names, key=len))
                        charSplit = np.char.split(
                            label_names.astype(f"U{stringLength}"), 
                            sep="-"
                        )
            
                        # Vendor starting index = 1
                        kwargs["acquisition_order"] = np.array(
                            [parts[1] for parts in charSplit],
                            dtype = int
                        ) - 1 
                        
                    else:
                        raise AttributeError("Currently not supporting the case where "
                                            "particle images' location in Parent Images "
                                            "are not known."
                                            )
                
                # Store the acquisition order
                self._metadata.set_item(
                    "Sample.acquisition_order",
                    kwargs.get("acquisition_order")
                )
            
            # Map the ChildSig 
            self.Images.map_ChildSig_onto_ParentSig(
                vendor = vendor,
                labelling = label_particles,
                matrix_label = matrix_label,
                **kwargs
            )
            
    def update_phase_maps(
        self,
        bkgr_label : int = -1
    ) -> None:
        """Update the phase maps. See the "Images.phase_maps" property.
        
        Parameters
        ----------
        bkgr_label 
            Label of the background (-1 by default).
        """
        if not hasattr(self, "Images"):
            raise AttributeError(
                "No 'Images' class attribute exists. "
                "See :func:'load_images'"
            )
        elif not hasattr(self.Images, "ChildMap"):
            raise AttributeError(
                "No 'ChildMap' signal attribute exists. See "
                ":func:'map_particles'"
            )
        
        if not (self.is_classified == False).all():
            # Particles have been classified:
            if not self.Images.phase_maps_updated:
                self.Images.set_phase_maps(
                    classes = self.particle_classes,
                    background_label = bkgr_label,
                    **{"acquisition_order" : self.metadata.get_item(
                        "Sample.acquisition_order"),
                        "vendor" : self.metadata.get_item(
                            "Acquisition_instrument.vendor"
                        )
                    }
                )
        else:
            exceptions.formatted_warning(
                "All particles are 'Unclassified'."
            )
            
    def plot(
        self,
        colours : list | dict | tuple | None = None,
        bkgr_colour : tuple | list | str = (1.,1.,1.),
        **kwargs
    ):
        """Plot the phase map as a hyperspy :class:'signal2d'.
        
        Parameters
        ----------
        colours
            list or dict of colours.
        bkgr_colour
            Colour of the background. Default: white.
        """

        if not hasattr(self, "Images"):
            raise AttributeError("The signal has no class 'Images'. "
                                "See :func:'load_images'.")
                                
        elif not hasattr(self.Images, f"{list(Images_signals.keys())[3]}"):
            raise AttributeError(
                    "The signal attribute 'Images' has no attribute "
                    f"{list(Images_signals.keys())[3]}. See "
                    ":func:'map_particles'.")
        
        unique_classes = np.unique(
            self.particle_classes
        )
        num_classes = len(
            unique_classes
        ) + 1 # include the matrix

        if colours is None: # Get default colours
            from microspy.draw import _colouring
            colours = _colouring.DEFAULT_COLOURS
        
        num_colours = len(colours)

        if num_colours < num_classes - 1: # Generate unique rgb colours 
            from microspy.draw._colouring import generate_unique_rgb_colors
            colours = generate_unique_rgb_colors(num_classes)
    
        # Set colours as a list of colours
        if isinstance(colours, dict):
            _colours = [colours[pclass] for pclass in colours.keys()]
        else: 
            _colours = colours
        
        # colours to rgb values
        from matplotlib.colors import to_rgb
        colours_rgb = [to_rgb(c) for c in _colours][:num_classes-1]
        
        if kwargs.get("background_label") is None:
            __attr = getattr(self.Images, list(Images_signals.keys())[3])
            bkgr_label = __attr.data.min()
            print(
                f"Reading minimum label ('{bkgr_label}') as the background "
                "label."
            )
        else:
            bkgr_label = kwargs.get("background_label")

        # Update the phase maps
        self.update_phase_maps()
        
        classes, PM = self.Images.get_phase_map(
            bkgr_label = bkgr_label,
        )
        
        # Create a ListedColormap
        from matplotlib.colors import ListedColormap, Normalize
        custom_cmap = ListedColormap(
            np.insert(# Insert background colour
                arr = colours_rgb,
                obj = 0,
                values = to_rgb(bkgr_colour), 
                axis = 0
            )
        )
        
        # Plot phase map as a hyperspy Signal2D with custom colourmap
        from hyperspy.signals import Signal2D
        _sig = Signal2D(PM)
        
        # Class names:
        classes = np.asarray(list(classes.keys()))
        
        # Text markers: classes 
        texts = np.insert(
            arr = classes,
            obj = 0,
            values = "Matrix"
        )
        
        # Can't figure out why the following fails...
        """
        from hyperspy.drawing._markers.texts import Texts
        # Label text start position
        offset = [[85,10]]
        for lower in range(len(texts)-1):   
            offset.append(
                [offset[-1][0],offset[-1][1]+5]
                 )
        
        offsets = np.stack(offset)
        m = Texts(
            offsets=offsets,
            texts=texts,
            sizes=3,
            facecolor="red"#colours_rgb,
            )
        
        _sig.add_marker(m, permanent = True)"""
        
        """norm = np.arange(len(classes)+1)-1
        norm = Normalize(
            vmin = -0.5,
            vmax = 0.5,
            clip = False
        )(norm)"""
        
        _sig.plot(
            cmap = custom_cmap, 
            vmin = float(bkgr_label)-0.5,
            vmax = float(np.max(PM))+0.5,
            #norm = norm,
            **kwargs
        )
        
        # Print the phase map information.
        from tabulate import tabulate
        print("\n",
            tabulate(
                tabular_data = zip(
                    np.arange(bkgr_label, len(texts),1), 
                    texts
                ),
                headers = ["Label", "Class name"]
            )
        )
