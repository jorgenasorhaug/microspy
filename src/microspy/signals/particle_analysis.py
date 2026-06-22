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
# tqdm(..., desc=" outer", position=0):

import numpy as np
import pandas as pd
from tabulate import tabulate
import warnings, os
from pathlib import Path
from tqdm import tqdm_notebook

# Parent dir:
from ..signals._microspy_signals import (
    MicroSpySignal2D, 
    MicroSpySignal2D_Parent,
    Images,
    MicroSpySignal1D_Chemistry, 
    MicroSpySignal1D_Geometry,
    Images_signal_type
)

from src.microspy.io._io import _save

from hyperspy.misc import utils

class ParticleAnalysis:
    """Particle analysis class

    A class keeping track of microspy signal classes like 
    particles' chemistry and geometric properties, particle
    classifications, etc.

    The class has the ability to read and allocate Images
    class (MicroSpySignal2D) attribute.

    Parameters
    ----------
    signals
        List of microspy 1D signals.
    **kwargs
        Read keywords:

        "Unclassified_kw": Define the unclassified label.

    Structure
    ---------
    ParticleAnalysis
    ├── MicroSpySignal1D_Chemistry/
    ├── MicroSpySignal1D_Geometry/
    └── Images/
        ├── Signal2D
        └── Signal2D

    Examples
    --------
    ...
    """
    def __init__(self, signals, **kwargs) -> None:

        # Check if single signal is provided or not
        if isinstance(signals, 
                      (MicroSpySignal1D_Geometry, 
                       MicroSpySignal1D_Chemistry)): 
            signals = [signals]

        # Set signals as attributes
        for signal in signals:
            # Set chemistry and geometry trackers:
            setattr(self, signal.metadata.Signal.signal_type, signal)

        # Set particle classification
        classes = signals[0].metadata.Sample.classes.copy()
        self._set_original_classes(classes = classes)
        self._set_particle_classification(
            class_array = classes, 
            Unclassified_kw = kwargs.get(
                'Unclassified_kw'
            ) if 'Unclassified_kw' in kwargs else "Unclassified" 
        )
        
        # Initialise metadata structure
        self._create_and_reorganise_metadata(signals)
        
        images = kwargs.get("images")
        if images is not None:
            if len(images) > 0:
                print("Setting 'Images' class.")
                self._set_Images(images)
        

    # Nice printing of information 
    def __repr__(self):

        print("OBS! Mutliple stubs has not been tested yet!")

        cal_string = ""
        grid_string = ""
        cal_string = ""
        read_images = True if hasattr(self, 'Images') else False 
        num_particles = len(self.particle_classes)

        if read_images:
            # 'MicroSpySignal2D_Parent, title: ..., dimensions: (X, Y|x, y | #)'
            gs_i = str(self.Images.ParentSig).index("(") + 1
            grid_string = str(self.Images.ParentSig)[gs_i:-2] + '|'
            
            if self.Images.is_calibrated: 
                
                unit = self.Images.unit
                
                # Num. pixels
                scan_area = np.prod(
                    self.Images.CompositeSig.data.shape
                )
                # Scaled:
                scan_area *= np.square(
                    self.Images.metadata.Signals.scale
                )

                # Number density
                particle_density = num_particles / scan_area

                # Area density
                area_index = self.Geometry.prop.index("Area")
                area_density = np.sum(
                    self.Geometry.data[area_index]
                ) / scan_area
                
                #decimal_position_n = _utils.first_nonzero_decimal_position(num_density)
                decimal_position_n = 2
                #decimal_position_a = _utils.first_nonzero_decimal_position(area_density)
                decimal_position_a = 2
                
                cal_string = f"\nScan unit: {unit}\n"
                cal_string += f"Particle number density: {round(
                    particle_density, 
                    decimal_position_n + 2
                )} "
                cal_string += f"1/{unit}\u00b2\n"
                cal_string += f"Particle area density: {round(
                    100 * area_density, 
                    decimal_position_a + 2
                )} %"

        else: cal_string = ""
        
        pa_string = f"<Particle analysis, title: {self.metadata.General.title}, "
        pa_string += f"dimensions: ({grid_string}{num_particles})>{cal_string}"
        
        return pa_string

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%% PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    @property 
    def is_classified(self):
        return self._is_classified
        
    @property
    def metadata(self):
        """The metadata of the signal."""
        return self._metadata

    @property
    def particle_classes(self):
        """Particle classes"""
        return self._particle_classes

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%% HIDDEN FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def _create_and_reorganise_metadata(self, signals):
        """Create class metadata. 

        Note!
        The metadata is saved in the class, whilst the metadata 
        in the attribute classes are removed.

        The only metadata the attribute classes keep track of are 
        Signal (this class will point to the attribute classes' 
        Signal through Signals). Otherwise, they point to this class'
        metadata
        """
        from copy import deepcopy

        # Copy Chemistry metadata
        self._metadata = deepcopy(signals[0]._metadata)
        md = self.metadata

        # Define Signals node
        del md.Signal
        md.add_node('Signals')

        # Store additional data as a new branch
        #if hasattr(md, 'Additional_data'):
        #    self._additional_data = md.Additional_data

        # Iterate through the signals and let all have identical 
        # metadata except from the Signal branch
        for sig in signals:

            # Get signal type and delete it from the attribute
            sig_type = sig.metadata.Signal.signal_type
            #del sig.metadata.Signal.signal_type

            # Define Signals instead of Signal
            md.set_item(
                f'Signals.{sig_type}', deepcopy(sig.metadata.Signal)
            )

            # Set the attribute signal metadata to reference the class' 
            # metadata
            smd = sig.metadata
            if hasattr(smd, 'Additional_data'):
                del smd.Additional_data
            smd.Acquisition_instrument = md.Acquisition_instrument
            smd.General = md.General
            #smd.Original_metadata = md.Original_metadata
            smd.Sample = md.Sample

    def _set_original_classes(
        self,
        classes : np.ndarray
    ):
        """Array of original classes"""
        self._original_classes = classes
    
    def _set_particle_classification(
        self, 
        class_array : np.ndarray,
        Unclassified_kw : str = 'Unclassified'
    ) -> None:
        """Set the as read particle classifications"""

        self._is_classified = class_array != Unclassified_kw
        self._particle_classes = class_array.copy()

        # Set phase_maps as 'not up-to date'
        if hasattr(self, "Images"):
            self.Images._updated_phase_maps = False
            
    def _set_Images(
        self,
        images : list
    ):
        """Set list of MicroSpySignal2D signals to Images
        class.

        images
            list of Images classes
        """
        if images is None:
            raise AttributeError("None argiment is not allowed.")

        elif not isinstance(images, list):
            raise AttributeError(f"Images argument {type(images[exp])} is not "
                                  "recognised.")

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
        
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%% OPEN FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

    def particles_logical_and(
        self,
        *arg
    ) -> None:
        """Identify particles matching provided argument conditions.

        Parameters
        ----------

        Returns
        -------
        mask
            Boolean array of met conditions
        """
        print("UNFINISHED")

    def particles_logical_or(
        self,
        *arg
    ) -> None:
        """Identify particles matching provided argument conditions.

        Parameters
        ----------

        Returns
        -------
        mask
            Boolean array of met conditions
        """
        print("UNFINISHED")

    @particle_classes.setter
    def reset_particle_classes(self):
        """Reset all particle classes to the original classification"""
        print(f"Setting the initial unique classes {np.unique(self._particle_classes)}"
              f"-> {np.unique(self._original_classes)}")
        self._particle_classes = self._original_classes
        self._is_classified = self._particle_classes != 'Unclassified'

    @particle_classes.setter
    def classify_particles(
        self,
        particles : np.ndarray | tuple,
        labels : str | tuple | list | np.ndarray
    ):
        """Classify particles

        Parameters
        ----------
        particle_array
            Array of particles to classify
        labels
            Class label

        Example
        -------
        >> s = pa.load(filename)
        >> s.is_classified
        array([False, False, True, ..., False, True, False])
        >> s.particle_classes
        array(['Unclassified', 'Unclassified', 'Classified', ...,
       'Unclassified', 'Classified', 'Unclassified'], dtype=object)
        >> s.classify(~s.is_classified, label = 'Classified')
        >> s.is_classified
        array([True, True, True, ..., True, True, True])
        >> s.particle_classes
        array(['Classified', 'Classified', 'Classified', ...,
       'Cslassified', 'Classified', 'Classified'], dtype=object)

        >> s.reset_particle_classifications()
        >> s.classify(np.where(~s.is_classified), label = 'Classified')
        >> s.is_classified
        array([True, True, True, ..., True, True, True])
        """

        tot_particles = self.metadata.Sample.particles
        
        # Checking input array
        if isinstance(particles, np.ndarray): 

            if len(particles) != tot_particles:
                if np.max(particles) >= tot_particles:
                    raise ValueError(f"The provided array of shape {particles.shape}",
                                "is not compatible with the total number of particles.")

                else: 

                    num_particles = len(particles) # "np.where()"
                    particles = (particles)
            
            else: 
                # Number of particles to classify
                num_particles = np.sum(particles)

        elif isinstance(particles, tuple): # np.where()
            if isinstance(particles[0], np.ndarray) and len(particles[0]) > 0:
                if particles[0].max() >= tot_particles:
                    raise ValueError("The indices exceeds the total "
                                     "number of particles.")
            else:
                return print("No particles to classify.")
                
            num_particles = len(particles[0])

        if num_particles == 0: 
            return print('No particles to classify.')
        
        # Checking labels
        if isinstance(labels, tuple | list | np.ndarray):

            if len(labels) != tot_particles:
                if len(labels) != num_particles:
                    if len(labels) == 1:
                        labels = np.asarray(labels * tot_particles)
                    else:                      
                        raise ValueError(f"The number of labels ({len(labels)}) "
                                         "do not match the the number of "
                                         "particles to classify "
                                        f"({num_particles}).")
            labels = np.asarray(labels)
            
        else: labels = str(labels)
        
        class_array = np.asarray(['Unclassified'] * tot_particles)
        class_array[particles] = labels
        
        # Set particle classes
        self._set_particle_classification(
            class_array = class_array, 
        )
        

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%% IMAGES %%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
    def load_images(
        self, 
        directory : str | Path | None = None,
        **kwargs
    ):
        """Load the acquired images and the corresponding 
        particle images from particle analysis.

        Parameters
        ----------
        directory
            Directory to where the images are stored.
            If None, the function will use the stored
            directory in the original metadata and search
            for the images.

        keyword arguments:
            read_particle_images : bool = True,
            centre_particle_images : bool = True,
            set_dtype : bool = None
        """ 
        from ..io._images import _io

        # Check if loading images is necessary
        _load_images : bool = False
        if hasattr(self, "Images"):
            _continue = input("'Images' class is already set. "
                              f"Stored attributes: {print(self.Images)}. "
                              "Procees? (y/[n])")
            if _continue.lower() in ("y", "yes"):
                _load_images = True
        else:
            _load_images = True

        if _load_images:
            # Search for a potential directory matching vendor solution:
            if directory == None:
                from src.microspy.io._images._utils import (
                    _image_directory_searcher as directory_searcher
                )
    
                print("Searching for images according to vendor solution.")
                vendor = self.metadata.get_item("Acquisition_instrument.vendor")
                #kwargs["vendor"] = vendor
    
                if vendor == "": 
                    raise AttributeError("No vendor information is accessible. "
                                         "A directory must be provided.")
                    
                directory = self.metadata.get_item("General.original_filename") 
                directory = os.path.split(directory)[0]
                
                print(f"Searching for particle images within \n<{directory}> "
                       "sub-directories.")
    
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
        """Calibrate the navigation signal
        
        Parameters
        ----------
        scale
            Navigation scale (unit per pixel)
        unit 
            Spatial unit
        """
        if not hasattr(self, "Images"):
            raise AttributeError("The class does not keep "
                                "track of images. See "
                                "the function *.load_images().")

        self.Images.calibrate_signals(
            scale = scale,
            unit = unit
        )
    
    def gridify_ParentSig(
        self,
        navigation_shape : list | tuple | None = None
    ):
        """Gridify the acquisition images.

        Parameters
        ----------
        navigation_shape
            Shape of the navigation grid
        """
        if not hasattr(self, "Images"):

            raise AttributeError("The class does not keep "
                                "track of images. See "
                                "the function *.load_images().")

        from src.microspy._misc._misc import _vendor2ImFlipAxes

        vendor = self.metadata.Acquisition_instrument.get_item("vendor")        
        flip_axes = _vendor2ImFlipAxes(vendor)
        
        self.Images.gridify_ParentSig(
            nav_shape = navigation_shape,
            flip_axes = flip_axes
        )

    def degridify_ParentSig(
        self
    ):
        """Degridify the acquisition images.
        """
        if not hasattr(self, "Images"):
            raise AttributeError("The class does not keep "
                                "track of images. See "
                                "the function *.load_images().")

        from src.microspy._misc._misc import _vendor2ImFlipAxes

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
        """Map particles' positions onto the Parent signal.
        The function uses template_match to identify the particles
        in the Parent signal.

        Parameters
        ----------
        """
        map_regions = False
        
        if not hasattr(self, "Images"):
            raise AttributeError("The signal doesn't keep track of "
                                 "any images. See *.load_images()")

        map_kw = list(Images_signal_type.keys())[3]
        
        if hasattr(
            self.Images, 
            map_kw
        ):
            ans = input("Particle regions seems to have been "
                                "mapped already.\n Proceed? (y/[n])")
            map_regions = True if ans.lower() in ("y", "yes") else False
        else: map_regions = True

        if map_regions:
            if not vendor:
                vendor = self.metadata.get_item(
                    "Acquisition_instrument.vendor"
                )
            
            # The following can be moved to e.g. _misc
            if vendor.lower() in ["jeol"]:
                
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
            
                        self._metadata.set_item(
                            "Sample.acquisition_order",
                            kwargs.get("acquisition_order")
                        )
                
            self.Images.map_ChildSig_onto_ParentSig(
                vendor = vendor,
                labelling = label_particles,
                matrix_label = matrix_label,
                **kwargs
            )

    def plot(
        self,
        colours : list | dict | tuple | None = None,
        bkgr_colour : tuple | list | str = (1.,1.,1.),
        **kwargs
    ):
        """Plot a phase map of the classified particles.
        
        Parameters
        ----------
        colours
            list or dict of colours.
        """

        if not hasattr(self, "Images"):
            raise AttributeError("Object has no class 'Images'. "
                                "See *.load_images().")
        elif not hasattr(self.Images, "ChildSig") and not hasattr(self.Images, "ParentSig"):
            raise AttributeError("Object class 'Images' has no attributes "
                                 "ParentSig and/or ChildSig. See *.load_images().")
        elif not hasattr(self.Images, f"{list(Images_signal_type.keys())[3]}"):
            raise AttributeError("Object class 'Images' has no attribute "
                                 f"{list(Images_signal_type.keys())[3]}. "
                                 "See *.map_particles().")
        
        unique_classes = np.unique(
            self.particle_classes
        )
        num_classes = len(
            unique_classes
        ) + 1 # incl. matrix

        if colours is None: # Get default colours
            from src.microspy.draw._colouring import DEFAULT_COLORS as colours
        
        num_colours = len(colours)

        if num_colours < num_classes - 1: # Generate unique rgb colours 
            from src.microspy.draw._colouring import generate_unique_rgb_colors
            colours = generate_unique_rgb_colors(num_classes)
    
        if isinstance(colours, dict):
            _colours = [colours[pclass] for pclass in unique_classes]
        else: _colours = colours
        
        # Set colours as rgb
        from matplotlib.colors import to_rgb
        colours_rgb = [to_rgb(c) for c in _colours][:num_classes-1]

        if kwargs.get("background_label") is None:
            __attr = getattr(self.Images, list(Images_signal_type.keys())[3])
            bkgr_label = __attr.data.min()
            print(
                f"Reading minimum label ({bkgr_label}) as background label."
            )
        else:
            bkgr_label = kwargs.get("background_label")

        # "Update phase maps"
        if not self.Images.is_phase_maps_updated:
            self.Images.set_phase_maps(
                classes = self.particle_classes,
                background_label = bkgr_label
            )
        
        PM = self.Images.get_phaseMap(
            bkgr_label = bkgr_label
        )
        
        # Create a ListedColormap
        from matplotlib.colors import ListedColormap
        custom_cmap = ListedColormap(
            np.insert(# Insert background colour
                arr = colours_rgb,
                obj = 0,
                values = to_rgb(bkgr_colour), 
                axis = 0
            )
        )
        
        from hyperspy.signals import Signal2D
        # Plot phase map as a hyperspy Signal2D with
        # custom colourmap
        _sig = Signal2D(PM)

        
        # Text markers: classes
        texts = np.insert(
            arr = unique_classes,
            obj = 0,
            values = "Matrix"
        )
        # Can't figure out why the following fails...
        """from hyperspy.drawing._markers.texts import Texts
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
        _sig.plot(
            cmap = custom_cmap, 
            vmin = float(bkgr_label)-0.5,
            vmax = float(np.max(PM))+0.5,
            **kwargs
        )
        print("Class labels:")
        for enum, cl in zip(
            np.arange(bkgr_label, len(texts),1), 
            texts
        ):
            print(f"{enum}: {cl}")

    def save(
        self,
        filename : Path | str | None = None,
        extension: str | None = "csv"
    ) -> None:
        """Write the signal to a file in the specified format.

        If no extension is provided, the signal is written to ...

        Parameters
        ----------
        filename
            Path and filename. If None, the original filename
            is used (path + filename)
        extension
            File extension that defines the file format. The 
            options are:

            * csv: Jeol's text format

            Each format accepts different parameters.

            If not given, the extension is determined by the
            filename.

        See also
        --------
        microspy.io.plugins
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

























            
#####################################################
############## OLD SOLUTIONS ########################

class _ParticleAnalysis: 
    """Create an object of the result from analysing 
    particles' chemistry and geometric properties

    Parameters
    ----------
    arg 
        filename of csv file or a pandas DataFrame

    See also
    --------

    Examples
    --------
    """
    
    # Instance properties:
    def __init__(self, *args, **kwargs):

        # Make metadata structure:
        self.metadata = _attribute_classes.metadata(arg)
        
        # Allocate Particles class
        self.Particles = _attribute_classes.Particles(arg)

        self.is_classified = self.Particles.classes != 'Unclassified'
        
        # Number of particles 
        self.number_of_particles = _io.get_number_of_particles(arg)

        self.particles_composition_shape = (len(self.Particles.elements), self.number_of_particles)

    # Print nice information about the object:
    def __repr__(self):
        
        print("OBS! Mutliple stubs has not been tested yet!")

        calibrated = False

        cal_string = ''
        
        read_images = False

        grid_string = ''

        if hasattr(self, 'Images'):

            read_images = True

            if hasattr(self.metadata, 'navigation_unit'):

                calibrated = True

        if read_images:

            grid_string = str(self.Images.navigation_shape)[1:-1] + ' | '

        if calibrated: 

            scan_area = np.prod(self.Images.navigation_shape) * np.prod(self.Images.signal_shape)

            scan_area *= np.square(self.metadata.navigation_scale)

            num_density = self.number_of_particles / scan_area

            area_density = np.sum(self.Particles.particle_geometry['Area [um²]']) / scan_area
            
            decimal_position_n = _utils.first_nonzero_decimal_position(num_density)

            decimal_position_a = _utils.first_nonzero_decimal_position(area_density)

            cal_string = f'\nScan unit: {self.metadata.navigation_unit}\nParticle number density: {round(num_density, decimal_position_n + 2)} 1/{self.metadata.navigation_unit}\u00b2\nParticle area density: {round(100 * area_density, decimal_position_a + 2)} %'
        
        pa_string = f"<Particle analysis, title: {self.metadata.General.project_name}, dimensions: ({grid_string}{self.number_of_particles})>{cal_string}"
        
        return pa_string

    ##############################################
    ############### FUNCTIONS ####################
    ##############################################
    
    def _check_array_length(self, array, num):
        """Check the length of an input array and whether it is compatible 
        with the stored number of particles in the object.
    
        Parameters
        ----------
        array
            Array whose length is to be checked
    
        Returns
            True if len(array) is identical to the number of stored particles.
        """
        if len(array) != self.number_of_particles: 
    
            print("The provided array does not match the number of particles")
            
            return False
            
        else: return True

    def _check_class_arguments(self, check_classes):
        """Check if class arguments in check_classes are classes stored in the object.
        if all classes in check_classes are found: 
            return True, []
        else: return False, [list of missing classes]
        """
        if type(check_classes) == str: check_classes = [check_classes]
        elif not type(check_classes) == list: raise TypeError(f"Provided array must be a list, and not a {type(check_classes)}")
        unique_classes = self.get_unique_particle_classes()
        missing_classes = []
        for cl in check_classes: 
            if cl not in unique_classes: missing_classes.append(cl)
        if len(missing_classes) == 0: return True, []
        else: return False, missing_classes

    def _get_max_class_label_character(self):
        """Return the maximum number of characters in the employed classes"""
        classes = self.get_unique_particle_classes(return_list=True)
        return np.max([len(x) for x in classes])
        
    def _get_max_string_length_in_list(lst):
        """Return the longest string length in list of strings"""
        return np.max([len(x) for x in lst])

    def _update_set_navigation_scale(self):
        """Update navigation_signal and particle_images' navigation unit and scale"""
        if not hasattr(self, 'Images'): 
            
            raise AttributeError('Images have not been read yet. See *.load_images()')
        
        else: 
            
            self.set_navigation_scale(self.metadata.navigation_scale,
                                        self.metadata.navigation_unit)

    def _update_particle_label_list(self, matrix_label = 0):
        """Update the unique particle labels in the Images class according to the particle_map signal.

        Parameters
        ----------
        matrix_label
            Integer label representing the matrix (i.e. not of interest)
        """
        if not hasattr(self, 'Images'): raise AttributeError("The object has no Images class attribute. See the function *.load_images()")

        if not hasattr(self.Images, 'particle_map'): raise AttributeError("The object's Images class has no attribute particle_map. See the function *.identify_particle_map().")

        unique_labels = np.unique(self.Images.particle_map)
        
        self.Images.unique_particle_labels = np.delete(unique_labels, np.where(unique_labels == matrix_label))
    
    def _stitch_cropped_particles(self,
                                  stitching_ncc_threshold = 0.25,
                                  extra_pixels = 0.2,
                                  shift_threshold = 1.0,
                                  remove_non_matched_region = False,
                                  directly_stitch_edge_particles = True,
                                  filter_stitched_images_to_correct_for_stitching_process = False,
                                  looping_progressbar = False,
                                  update_particle_properties = True):
        """The function will attempt to stitch the identified cropped particles so that measurement
        corrections can be  done. 
        
        Note that if update_particle_properties if True, the pair of stitched particles' chemical 
        composition will be averaged and their corresponding labels updated in all lists and particle
        maps by keeping the lowest label among the two.

        Parameters
        ----------
        stitching_ncc_threshold
            Float value describing the lowest allowable ncc score between two images' overlapping 
            region before stitching is considered successful.
            Default: 0.5.
        extra_pixels
            Float (percentage) defining how many extra pixels wrt. SEM image widht or height to 
            add to improve the chance of correct stitching. 
        shift_threshold 
            A number that assess whether the image stitching was successful or not by how much it was shifted 
            (Eucledian distance). Default: 1.0
        remove_non_matched_region 
            Whether to remove the region that was initially not part of the stored particle image from the 
            stitched image. By default: True
        directly_stitch_edge_particles 
            If particle stitching fails for some reason, concatenate the particle images directly. 
        filter_stitched_images_to_correct_for_stitching_process
            Whether to smooth the stitched images or not. A good reason to do this is to remove artefacts from the image shifting.
            See skimage.registration's phase_cross_correlation function.
        update_particle_properties
            Wheter to correct the cropped particle pairs' chemical composition and geometry.
        
        Returns
        -------
        Example
        -------
        >>> s = pa.load(filename)
        >>> s.load_images(image_path)
        >>> s.Images
        Particle analysis imags:
            SEM images: <(16 | 1536,2048) | Particle images: (1000 | (500,400))>

        >>> s.gridify_SEM_images(navigation_shape = (4,4))
        >>> s.Images
        Particle analysis imags:
            SEM images: <(4,4 | 1536,2048) | Particle images: (1000 | (500,400))>

        # Identify the particle regions
        >>> s.identify_particle_regions()
        >>> s.stitch_cropped_particles()
        >>> ...
        """
        stitch = True
        
        if filter_stitched_images_to_correct_for_stitching_process:
            from scipy.ndimage import median_filter
        from scipy.ndimage import binary_fill_holes as bfh
        
        if not hasattr(self, 'Images'): raise AttributeError("No images have been read yet. See the function *.load_images()")
        
        if not hasattr(self.Images, 'cropped_particles_map'): 
            
            ans = input("Cropped particles have not been mapped yet. Search for cropped particles? ([y]/n)")

            if ans.upper() == 'Y' or ans == '': self.identify_cropped_particles()

            else: stitch = False

        if not update_particle_properties:

            ans = input("Correct the stitched particles' properties? ([y]/n)")

            if ans.upper() == 'Y' or ans == '': update_particle_properties = True

        if stitch:
            
            if len(self.metadata.Particles.cropped_particle_clusters) > 0: 

                from skimage.exposure import rescale_intensity
                from tqdm import tqdm
    
                # List of arrays with particle labels
                particle_clusters = self.metadata.Particles.cropped_particle_clusters
                num_clusters = len(particle_clusters)
                stitched_particle_images = []
                successfully_stitched = []
                unsuccessfully_stitched = []
    
                for cl in tqdm(range(num_clusters), position=0, desc = "Stitching particles"):
                    
                    num_particles_in_cluster = len(particle_clusters[cl])
                    
                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    #%%%%%%%%%%%%%%%%% > 2 PARTICLE IMAGES TO BE STITCHED TOGETHER: %%%%%%%%%%%%%%%%%%%
                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    if num_particles_in_cluster > 2:
    
                        minI = 0
                        maxI = 0
    
                        for pim in particle_clusters[cl]: 
                            temp_im = self.get_depadded_particle_image(pim)
                            minI = min([minI, temp_im.min()])
                            maxI = min([maxI, temp_im.max()])
    
                        # Whether to append the second group of particles as an individual particle or not
                        append_second_stitched_image = not directly_stitch_edge_particles
    
                        img1, temp_success, temporarily_not_stitched = self.Images.stitch_group_of_particles(
                            list_of_particles = particle_clusters[cl],
                            shift_threshold = shift_threshold,
                            remove_non_matched_region = remove_non_matched_region,
                            success_threshold = stitching_ncc_threshold,
                            looping_progressbar = looping_progressbar,
                            directly_stitch_edge_particles = directly_stitch_edge_particles
                        )
    
                        # Stitch the remaining non-stitched particles (plural)
                        if len(temporarily_not_stitched) >= 2:
                                
                            if len(temporarily_not_stitched) == 2:
                                
                                img2_, successful_stitching2_ = self.Images.stitch_two_particle_images(
                                    particle_label1 = temporarily_not_stitched[0],
                                    particle_label2 = temporarily_not_stitched[1],
                                    shift_threshold = shift_threshold,
                                    remove_non_matched_region = remove_non_matched_region,
                                    success_threshold = stitching_ncc_threshold)
    
                                if successful_stitching2_:
                                    
                                    temp_success2_ = temporarily_not_stitched.copy()
    
                            else: 
                                
                                img2_, temp_success2_, temporarily_not_stitched = self.Images.stitch_group_of_particles(
                                    list_of_particles = temporarily_not_stitched,
                                    shift_threshold = shift_threshold,
                                    remove_non_matched_region = remove_non_matched_region,
                                    success_threshold = stitching_ncc_threshold,
                                    looping_progressbar = looping_progressbar,
                                    directly_stitch_edge_particles = directly_stitch_edge_particles)
    
                                successful_stitching2_ = len(temp_success2_) > 0
    
                            if successful_stitching2_ and directly_stitch_edge_particles:
    
                                concatible, (img1_loc_, loc1_), axis = _image_utils._check_two_stitched_particle_images_concatenation_compatibilities(
                                    list_of_labels0 = temp_success,
                                    list_of_labels1 = temp_success2_,
                                    particle_map = self.Images.particle_map.data)
    
                                if concatible:
    
                                    # Make the two images into the same size:
                                    pim1, pim2 = _image_utils._copy_images_into_common_array_shape(img1, img2_)
                                    
                                    # Remove padding along a specific axis:
                                    pim1 = _image_utils._remove_padding_along_axis(pim1, axis = axis)
                                    pim2 = _image_utils._remove_padding_along_axis(pim2, axis = axis)
                                                                    
                                    # Concatenate:
                                    img1 = _image_utils._concatenate_two_edge_images(pim1, pim2, img1_loc_, loc1_)
    
                                    for label in temp_success2_:
                                        temp_success.append(label)
                                    temporarily_not_stitched = []
    
                                else: append_second_stitched_image = True
    
                            if append_second_stitched_image: # img1 will be appended below
    
                                if filter_stitched_images_to_correct_for_stitching_process: 
                                    # Filter to remove artefacts from stitching
                                    img2_ = median_filter(img2_, size = 2)
                        
                                if len(temporarily_not_stitched) > 0:
                                    
                                    unsuccessfully_stitched.append(np.sort(np.array(temporarily_not_stitched)))
            
                                if len(temp_success2_) > 0:
            
                                    successfully_stitched.append(np.sort(np.array(temp_success2_)))
            
                                if np.shape(img2_) != (0,): 
                                    
                                    stitched_particle_images.append(img2_)
        
                        elif (len(temporarily_not_stitched) == 1) and directly_stitch_edge_particles: 
                            # Only one particle image hasn't been stitched with the rest yet
                               
                            img1_ = _image_utils._concatenate_a_stitched_image_with_a_non_stitched_image(
                                temporarily_not_stitched[0],
                                temp_success.copy(),
                                img1.copy(),
                                self.Images.particle_map.data,
                                self.Images.navigation_signal.data,
                                to_shape = img1.shape)
    
                            if len(img1_.shape) > 1: # If success:
                                img1 = img1_
                                temp_success.append(temporarily_not_stitched[0])
                                temporarily_not_stitched = []
                        
                        if filter_stitched_images_to_correct_for_stitching_process: 
                            # Filter to remove artefacts from stitching
                            
                            img1 = median_filter(img1, size = 2)
                
                        if len(temporarily_not_stitched) > 0:
                            
                            unsuccessfully_stitched.append(np.sort(np.array(temporarily_not_stitched)))
    
                        if len(temp_success) > 0:
    
                            successfully_stitched.append(np.sort(np.array(temp_success)))
    
                        if np.shape(img1) != (0,): 
                            
                            stitched_particle_images.append(img1)
    
    
                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    #%%%%%%%%%%%%%%%%%%%%% 2 PARTICLE IMAGES STITCHED TOGETHER: %%%%%%%%%%%%%%%%%%%%%%%
                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    else: # If only 2 particle images
    
                        idx0, idx1 = particle_clusters[cl][0], particle_clusters[cl][1]  
    
                        img0 = self.get_depadded_particle_image(idx0)
                        img1 = self.get_depadded_particle_image(idx1) 
    
                        # For intensity rescaling:
                        minI = np.min([np.min(img0), np.min(img1)])
                        maxI = np.max([np.max(img0), np.max(img1)])
    
                        img1_, successful_stitching = self.Images.stitch_two_particle_images(
                            particle_label1 = idx0,
                            particle_label2 = idx1,
                            shift_threshold = shift_threshold,
                            remove_non_matched_region = remove_non_matched_region,
                            success_threshold = stitching_ncc_threshold)
                        
                        if successful_stitching: img1 = img1_
    
                        elif not successful_stitching and directly_stitch_edge_particles:
                            # First attempt failed, try concatenate the images:
                            
                            # The following will concatenate the particle images but with padding:
                            """
                            # Make the two images into the same size:
                            pim1, pim2 = _image_utils._copy_images_into_common_array_shape(img0, img1)
                            
                            _, (loc0, loc1), axis = _image_utils._check_two_stitched_particle_images_concatenation_compatibilities(
                                list_of_labels0 = [idx0],
                                list_of_labels1 = [idx1],
                                particle_map = self.Images.particle_map.data)
                        
                            
                            # Remove padding along a specific axis:
                            pim1 = _image_utils._remove_padding_along_axis(pim1, axis = axis)
                            pim2 = _image_utils._remove_padding_along_axis(pim2, axis = axis)
                            
                            # Concatenate:
                            img1 = _image_utils._concatenate_two_edge_images(pim1, pim2, loc0, loc1)
                            """

                            img1 = self.Images.concatenate_two_cropped_particle_images(
                                particle_label1 = idx0,
                                particle_label2 = idx1)
    
                            successful_stitching = True
        
                        if successful_stitching:
    
                            if filter_stitched_images_to_correct_for_stitching_process: 
                                
                                img1 = median_filter(img1, size = 2)
    
                            stitched_particle_images.append(
                                rescale_intensity(
                                    img1.copy(), 
                                    out_range = (minI, maxI)
                                ).astype(self.Images.particle_images.data.dtype))
    
                            successfully_stitched.append(np.sort(np.array([idx0, idx1])))
    
                        else: 
                            
                            unsuccessfully_stitched.append(np.sort(np.array([idx0, idx1])))
    
                # Save the final results as hidden attributes, as pairs of particle images will become correted in the following, 
                # and their corresponding information will get updated.
                # Information about stitched images with > 2 particle images will be stored in an open attribute.
    
                unlabelled_stitched_particles_indices = []
                relabelled_stitched_particles_indices = []
                unlabelled_particle_images = []
                relabelled_particle_images = []
                
                for cl in range(len(successfully_stitched)):
                    
                    if len(successfully_stitched[cl]) > 2: 
                        #print('1', successfully_stitched[cl])
                        unlabelled_stitched_particles_indices.append(successfully_stitched[cl])
                        unlabelled_particle_images.append(stitched_particle_images[cl].copy())
                    else: 
                        #print('2', successfully_stitched[cl])
                        relabelled_stitched_particles_indices.append(successfully_stitched[cl])
                        relabelled_particle_images.append(stitched_particle_images[cl].copy())
                
                argsort = np.argsort(np.asarray(relabelled_stitched_particles_indices), axis = 0)[:,0]
                relabelled_stitched_particles_indices = np.asarray(relabelled_stitched_particles_indices)[argsort]
                _ = []
                for arg in argsort: _.append(relabelled_particle_images[arg])
                relabelled_particle_images = _
    
                # Particle images:
                self.Images._successfully_stitched_particle_images = stitched_particle_images
    
                self.Images.unrelabelled_stitched_particle_images = unlabelled_particle_images
    
                self.Images.relabelled_stitched_particle_images = relabelled_particle_images
                
                # Particle labels:
                self.metadata.Particles._successfully_stitched_particles_labels = successfully_stitched
    
                # Update the particle clusters, i.e. those remaining
                self.metadata.Particles.cropped_particle_clusters = unlabelled_stitched_particles_indices
    
                self.metadata.Particles.relabelled_stitched_particle_labels = relabelled_stitched_particles_indices
    
                self.metadata.Particles.unsuccessfully_stitched_particle_labels = unsuccessfully_stitched
    
                # Print the success rate
                number_of_cropped_particles = self.metadata.Particles.cropped_particles.shape[0]
                number_of_not_stitched_particles = 0
                for i in range(len(unsuccessfully_stitched)): 
                    number_of_not_stitched_particles += len(unsuccessfully_stitched[i])
                number_of_stithced_particles = number_of_cropped_particles - number_of_not_stitched_particles
                    
                print(f"\nSuccessful stitching: {number_of_stithced_particles}/{number_of_cropped_particles} ({np.format_float_positional(100 * number_of_stithced_particles/number_of_cropped_particles, 2)} %)")

                # Update particle information ONLY for particle pairs!
                if update_particle_properties:
                    
                    self._correct_successfully_stitched_particle_pairs()
    
            else: print('No cropped particles have been identified.')  

    def _cluster_group_of_cropped_particles(self,
                                            conditions : dict,
                                            update_particle_properties : bool = True):
        """Group particle images and labels together that are most likely corresponding to the same 
        particle, but have been separated due to particle cropping.

        

        Parameters
        ---------
        conditions
            dictionary of conditions to threshold particle images and segment them.
        update_particle_properties
            Whether to update the particles' labels and chemical composition.
            By default: True
        """
        
        if not hasattr(self.metadata, 'Particles'):
            
            raise AttributeError("The object has no Particles' class attribute. See the function *.load_images()")
        
        if not hasattr(self.metadata.Particles, 'cropped_particle_clusters'):
            
            raise AttributeError("The object has no, or has not identified cropped particles. See the function *.identify_cropped_particles().")

        # Save the info
        print('\033[1mSaved information:\033[0m')
        print('New particle images are saved in a list of dictionaries:')
        print('\t*.Images.grouped_cropped_particle_images[INDEX][KEY] \n\t~ *.metadata.Particles.cropped_particle_clusters[INDEX]')
        print('\t~ *.metadata.Particles.grouped_cropped_particle_clusters[INDEX][KEY]')
        self.Images.grouped_cropped_particle_images = []
        
        self.metadata.Particles.grouped_cropped_particle_clusters = []
        
        self.Images._grouped_cropped_particle_cluster_masks = []

        for cl in range(len(self.metadata.Particles.cropped_particle_clusters)):

            # Get particle labels grouped together
            clustered_particles = self.metadata.Particles.cropped_particle_clusters[cl].copy()

            # Get stitched image corr. to the group of particle labels
            stitched_particle_image = self.Images.unrelabelled_stitched_particle_images[cl].copy()

            # group particle images according to the stitched particle image:
            labels, masks, new_particle_images = self.Images.cluster_group_of_cropped_particles(
                list_of_cropped_particle_labels = clustered_particles,
                st_im = stitched_particle_image,
                conditions =  conditions
            )
            
            # Save the info
            self.metadata.Particles.grouped_cropped_particle_clusters.append(labels)

            self.Images._grouped_cropped_particle_cluster_masks.append(masks)

            self.Images.grouped_cropped_particle_images.append(new_particle_images)

        if update_particle_properties:
            
            self._correct_successfully_stitched_group_of_particles()

    def _correct_successfully_stitched_particle_pairs(self):
        """Following stitching of particles, only pair of particles will be relabelled, their chemistry will be updated,
        and their corresponding particle images will be updated. Stitched particles with more than two labels will 
        not be updated, as these are more challenging to automatically relabel. This should be done manually.
        """

        if not hasattr(self, 'Images'): 
            
            raise AttributeError("No images have been read yet. See the function *.load_images()")

        if not hasattr(self.Images, 'particle_map'): 
            
            raise AttributeError("No information about the particles' spatial distribution has been stored. See the function *.identify_particle_regions()")
        
        if not hasattr(self.metadata.Particles, '_successfully_stitched_particles_labels'): 
            
            raise AttributeError("No cropped particle pairs have been identified. See function *.stitch_cropped_particles()") 

        if len(self.metadata.Particles.relabelled_stitched_particle_labels) > 0:
        
            from tqdm import tqdm
    
            # Particle pairs are easy to automatically correct for. > 2 particles on the other hand...
            pairs = self.metadata.Particles.relabelled_stitched_particle_labels
    
            # Identify the largest stitched particle image shape to assess whether the signal 'particle_images' need reshaping.
            max_pair_image_shape = [0,0] # Max stitched particle image shape
            #pairs_mask = np.zeros(len(self.Images._particle_images_stitched), bool)
            for pim in self.Images.relabelled_stitched_particle_images:
                temp_shape = pim.shape 
                if temp_shape[0] > max_pair_image_shape[0]: max_pair_image_shape[0] = temp_shape[0]
                if temp_shape[1] > max_pair_image_shape[1]: max_pair_image_shape[1] = temp_shape[1]
    
            # Reshape particle images if needed:
            self.Images._reshape_particle_images(tuple(max_pair_image_shape))
    
            num_particles = self.number_of_particles
    
            # Array masks defining the particles to keep and not after stitching particles
            keep_arr = np.zeros(num_particles, bool)
            del_arr = np.zeros(num_particles, bool)
    
            pimage_shape = self.Images.particle_image_shape

            # Appending stitched images before reshaping them to become
            # compatible with the particle images' signal
            _particle_images_stitched = []
    
            # Hide old cropped particles attributes
            if not hasattr(self.Images, '_initial_cropped_particles_map'):
            
                self.Images._initial_cropped_particles_map = self.Images.cropped_particles_map.deepcopy()
            
            
            for pair, im_idx in tqdm(zip(pairs, np.arange(len(pairs))), 
                                     desc = "Correcting pair of cropped particles' composition and labels",
                                     total = len(pairs)):
    
                # Store which particles to update and which particles to remove from the class:
                keep_arr[np.where(self.Images.unique_particle_labels == pair[0])] = True # total number of particles
                del_arr[np.where(self.Images.unique_particle_labels == pair[-1])] = True # total number of particles
    
                # Update the list of cropped particles and keep the remaining labels in larger clusters:
                self.metadata.Particles.cropped_particles = np.delete(
                    self.metadata.Particles.cropped_particles, self.metadata.Particles.cropped_particles == pair[0])
                self.metadata.Particles.cropped_particles = np.delete(
                    self.metadata.Particles.cropped_particles, self.metadata.Particles.cropped_particles == pair[-1])
                
                # Update particle labels in the particle_map 
                self.Images.particle_map.data[np.where(self.Images.particle_map == pair[-1])] = pair[0]
    
                # Remove the cropped particles from the cropped_particles_map
                self.Images.cropped_particles_map.data[self.Images.cropped_particles_map == pair[0]] = 0
                self.Images.cropped_particles_map.data[self.Images.cropped_particles_map == pair[-1]] = 0
    
                # Update chemistry by averaging:
                self.Particles.composition[:, 
                    np.where(self.Images.unique_particle_labels == pair[0])] = np.mean(
                    [self.Particles.composition[:, 
                        np.where(self.Images.unique_particle_labels == pair[0])], self.Particles.composition[:,
                        np.where(self.Images.unique_particle_labels == pair[1])]], axis = 0)

                # Relabel classification if the classes are unequal
                cl0 = self.Particles.classes[np.where(self.Images.unique_particle_labels == pair[0])]
                cl1 = self.Particles.classes[np.where(self.Images.unique_particle_labels == pair[-1])]
                
                if cl0 != cl1:
                    
                    self.Particles.classes[np.where(self.Images.unique_particle_labels == pair[0])] = cl0 + cl1
    
                # Setting negative composition to 
                #self.Particles.composition[:, pair[-1]-1][:] = -1 
    
                # Reshape the stitched particle images into a shape that is compatible with the class's particle images:
                _particle_images_stitched.append(_image_utils._put_array_content_into_larger_array(
                    self.Images.relabelled_stitched_particle_images[im_idx], pimage_shape))
    
            # Update the list of unique particle labels
            self._update_particle_label_list()
            
            # Setting negative geometries to update these manufally afterwards. 
            # Geometries no longer of interest are removed.
            for geom in self.Particles.particle_geometry.keys():
    
                self.Particles.particle_geometry[geom][keep_arr] = -1
    
                self.Particles.particle_geometry[geom] = self.Particles.particle_geometry[geom][~del_arr] 
    
            # Update the total number of particles:
            self.number_of_particles -= np.sum(del_arr)

            self.Particles.classes = self.Particles.classes[~del_arr]

            self.is_classified = self.is_classified[~del_arr]
    
            # Update the array of particles' chemical composition
            self.Particles.composition = self.Particles.composition[:, ~del_arr].reshape((len(self.get_identified_elements()), 
                                                                                          self.number_of_particles))
    
            self.update_particles_composition_shape()
    
            self._update_particles_concentration()
    
            self.Images.particle_images.data[keep_arr] = np.asarray(_particle_images_stitched)
    
            self.Images.particle_images = Signal2D(self.Images.particle_images.data[~del_arr])
    
            self.metadata.Particles.label_name = list(np.asarray(self.metadata.Particles.label_name)[~del_arr])

        else: print('No cropped particles have been stitced')
    
    def _correct_successfully_stitched_group_of_particles(self):
        """Following stitching of particles, only pair of particles will be relabelled, their chemistry will be updated,
        and their corresponding particle images will be updated. Stitched particles with more than two labels will 
        not be updated, as these are more challenging to automatically relabel. This should be done manually.
        """

        if not hasattr(self, 'Images'): 
            
            raise AttributeError("No images have been read yet. See the function *.load_images()")

        if not hasattr(self.Images, 'particle_map'): 
            
            raise AttributeError("No information about the particles' spatial distribution has been stored. See the function *.identify_particle_regions()")
        
        if not hasattr(self.metadata.Particles, 'grouped_cropped_particle_clusters'): 
            
            raise AttributeError("No cropped particle pairs have been identified. See function *.stitch_cropped_particles()")

        # Iterate through all the grouped cropped particle clusters:
        if len(self.metadata.Particles.grouped_cropped_particle_clusters) > 0:
        
            from tqdm import tqdm
    
            # Particle pairs are easy to automatically correct for. > 2 particles 
            # on the other hand...
            group = self.metadata.Particles.grouped_cropped_particle_clusters
    
            # Identify the largest stitched particle image shape to assess whether the signal 
            # 'particle_images' need reshaping.
            max_pair_image_shape = [0,0] # Max stitched particle image shape
            for pim in self.Images.unrelabelled_stitched_particle_images:
                temp_shape = pim.shape 
                if temp_shape[0] > max_pair_image_shape[0]: max_pair_image_shape[0] = temp_shape[0]
                if temp_shape[1] > max_pair_image_shape[1]: max_pair_image_shape[1] = temp_shape[1]
    
            # Reshape particle images if needed:
            self.Images._reshape_particle_images(tuple(max_pair_image_shape))
    
            num_particles = self.number_of_particles
    
            # Array masks defining the particles to keep and not after stitching particles
            keep_arr = np.zeros(num_particles, bool)
            del_arr = np.zeros(num_particles, bool)

            changed_particles_labels = []
    
            pimage_shape = self.Images.particle_image_shape
    
            # Hide old cropped particles attributes
            if not hasattr(self.Images, '_initial_cropped_particles_map'):
                
                self.Images._initial_cropped_particles_map = self.Images.cropped_particles_map.deepcopy()

            for cluster, listPos in tqdm(zip(self.metadata.Particles.grouped_cropped_particle_clusters,
                                            np.arange(len(group))),
                                         desc = "Correcting group of cropped particles' composition and labels",
                                         total = len(group)):

                unique_particle_labels = np.unique(cluster)
                
                for cl in unique_particle_labels:

                    # Get the corr. particle labels
                    plabels = self.metadata.Particles.cropped_particle_clusters[listPos][cluster == cl]

                    # Save the particle labels that are kept 
                    changed_particles_labels.append(plabels[0])

                    temp_classes = []

                    keep_arr[np.where(self.Images.unique_particle_labels == plabels[0])] = True
                    
                    for plabel in plabels:
    
                        # Store which particles to update and which particles to remove from the class, 
                        # and update particle labels in the particle_map, by keeping the first label
                        if plabel != plabels[0]:
                            
                            del_arr[np.where(self.Images.unique_particle_labels == plabel)] = True 
                            
                            self.Images.particle_map.data[np.where(self.Images.particle_map == plabel)] = plabels[0]
            
                        # Remove the cropped particles from the cropped_particles_map
                        self.Images.cropped_particles_map.data[self.Images.cropped_particles_map == plabel] = 0

                        # Relabel classification if the classes are unequal
                        temp_classes.append(
                            self.Particles.classes[
                                np.where(self.Images.unique_particle_labels == plabel)
                                ]
                        )

                        # Delete the label from the cropped_particles' list:
                        self.metadata.Particles.cropped_particles = np.delete(
                            self.metadata.Particles.cropped_particles, 
                            self.metadata.Particles.cropped_particles == plabel)
                        
                    unique_classes = np.unique(np.asarray(temp_classes))

                    # Set a unique class name for the new particle
                    if len(unique_classes) > 1:

                        class_name = ''

                        for cl in unique_classes: class_name += cl

                    else: class_name = unique_classes[0]
                        
                    self.Particles.classes[
                        np.where(
                            self.Images.unique_particle_labels == plabels[0])] = class_name
                    
                    # Update chemistry by averaging:
                    self.Particles.composition[:, 
                        np.where(self.Images.unique_particle_labels == plabels[0])[0]] = np.mean(
                        [self.Particles.composition[:, 
                            np.where(self.Images.unique_particle_labels == plabel)[0]] 
                        for plabel in plabels], axis = 0)
                    
                    # Reshape the stitched particle images into a shape that is compatible with 
                    # the class's particle images:
                    self.Images.particle_images.data[
                        np.where(
                            self.Images.unique_particle_labels == plabels[0])[0]
                        ] = _image_utils._put_array_content_into_larger_array(
                            self.Images.grouped_cropped_particle_images[listPos][cl], 
                            pimage_shape)
                    
            # Delete the cropped particle clusters attribute if it's empty
            del self.metadata.Particles.cropped_particle_clusters

            self.metadata.Particles.updated_particles_from_group_of_cropped_particles = np.sort(changed_particles_labels)
            
            # Update the list of unique particle labels
            self._update_particle_label_list()
            
            # Setting negative geometries of the labels that are kept to update these manufally 
            # afterwards. Geometries no longer of interest are removed.
            for geom in self.Particles.particle_geometry.keys():
    
                self.Particles.particle_geometry[geom][keep_arr] = -1
    
                self.Particles.particle_geometry[geom] = self.Particles.particle_geometry[geom][~del_arr] 
    
            # Update the total number of particles:
            self.number_of_particles -= np.sum(del_arr)

            self.Particles.classes = self.Particles.classes[~del_arr]

            self.is_classified = self.is_classified[~del_arr]
    
            # Update the array of particles' chemical composition
            self.Particles.composition = self.Particles.composition[:, 
                ~del_arr].reshape((len(self.get_identified_elements()), 
                                   self.number_of_particles))
    
            self.update_particles_composition_shape()
    
            self._update_particles_concentration()
    
            #self.Images.particle_images.data[keep_arr] = np.asarray(_particle_images_stitched)
    
            self.Images.particle_images = Signal2D(self.Images.particle_images.data[~del_arr])
    
            self.metadata.Particles.label_name = list(np.asarray(self.metadata.Particles.label_name)[~del_arr])

        else: print('No group of cropped particles have been clustered.')
        
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%% OPEN GENERAL FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    def get_navigation_scale(self):
        """Return the stored navigation scale if it has been set"""

        if not hasattr(self.metadata, 'navigation_scale'):

            raise AttributeError("The scale hasn't been set yet. See the function *.set_navigation_scale()")

        if self.metadata.navigation_scale == 1:

            warnings.warn("The navigation scale is currently equal to 1. Make sure this is the correct one.")
        
        return self.metadata.navigation_scale
    
    def set_acceleration_voltage(self, HT):
        """Set the acceleration voltage used during particle analysis
        to the object's metadata

        Parameters
        ----------
        HT 
            High tension/acceleration voltage as integer or float
        """

        if not type(HT) in (float, int): raise TypeError(f"Provided high tension {HT} is not valid. Make sure it is a real number")
        
        self.metadata.Acquisition_settings.acc_voltage = float(HT)

    def set_navigation_scale(self, scale, unit):
        """Set the image scale

        Parameters
        ----------
        scale
            Image scale in unit/px (float)
        unit
            string
        """

        self.metadata.navigation_scale = scale

        self.metadata.navigation_unit = unit
        
        if hasattr(self, 'Images'):

            nav_shape = self.Images.navigation_shape

            sig_shape = self.Images.signal_shape

            if len(nav_shape) > 1:

                self.Images.navigation_signal.axes_manager[1].scale = scale * sig_shape[0]
                
                self.Images.navigation_signal.axes_manager[0].scale = scale * sig_shape[1]

                self.Images.navigation_signal.axes_manager[0].units = unit

                self.Images.navigation_signal.axes_manager[1].units = unit

            else:

                self.Images.navigation_signal.axes_manager[0].scale = scale * nav_shape[0] * sig_shape[0]

                self.Images.particle_images.axes_manager[0].units = unit

            self.Images.navigation_signal.axes_manager[-1].scale = scale

            self.Images.navigation_signal.axes_manager[-1].units = unit

            self.Images.navigation_signal.axes_manager[-2].scale = scale

            self.Images.navigation_signal.axes_manager[-2].units = unit

            self.Images.particle_images.axes_manager[-1].scale = scale

            self.Images.particle_images.axes_manager[-1].units = unit

            self.Images.particle_images.axes_manager[-2].scale = scale

            self.Images.particle_images.axes_manager[-2].units = unit

            if hasattr(self.Images, 'particle_map'):

                self.Images.particle_map.axes_manager[-1].scale = scale

                self.Images.particle_map.axes_manager[-1].units = unit
    
                self.Images.particle_map.axes_manager[-2].scale = scale
    
                self.Images.particle_map.axes_manager[-2].units = unit

    # Classified particles
    def classified_particles(self):
        """Returns an array of classified particles"""
        return self.Particles.classes != 'Unclassified'

    def print_max_composition(self):
        """Print the maximum chemical composition of each element"""
        max_comp = np.expand_dims(np.max(self.Particles.composition, axis=1), axis=1)
        utils.print_particles_property(max_comp,  
                                       header = self.get_identified_elements(), 
                                       label = [f'Max [{self.metadata.chemical_unit}]'])
        
    def print_min_composition(self):
        """Print the maximum chemical composition of each element"""
        min_comp = np.expand_dims(np.min(self.Particles.composition, axis=1), axis=1)
        utils.print_particles_property(min_comp,
                                       header = self.get_identified_elements(), 
                                       label = [f'Min [{self.metadata.chemical_unit}]'])

    def print_unclassified_particles_composition(self):
        """Print a table of the unclassified particles' chemistry"""
        utils.print_particles_property(self.Particles.composition[:,~self.is_classified],
                                       label = self.metadata.Particles.label_name[~self.is_classified],
                                       header = self.get_identified_elements())

    def print_classified_particles_composition(self):
        """Print a table of the unclassified particles' chemistry"""
        utils.print_particles_property(self.Particles.composition[:,self.is_classified],
                                       label = self.Particles.classes[self.is_classified],
                                       header = self.get_identified_elements())

    def print_class_composition(self, class_name):
        """Print the chemical composition of the particles classified as class_name"""
        class_name = str(class_name)
        
        if class_name not in self.get_unique_particle_classes(): raise ValueError(f"Class name {class_name} is not recognised")

        class_instances = self.Particles.classes == class_name
        
        utils.print_particles_property(self.Particles.composition[:,self.is_classified],
                                       label = self.Particles.classes[self.is_classified],
                                       header = self.get_identified_elements())

    def print_matrix_composition(self):
        """Print the matrix' composition if it has been allocated the object.
        """
        if not hasattr(self.metadata.Particles, 'matrix_elements'): raise ValueError("The object's matrix elements is not set yet. See *.set_matrix_composition()")

        if not hasattr(self.metadata.Particles, 'matrix_composition'): raise ValueError("The object's matrix composition is not set yet. See *.set_matrix_composition()")

        if len(self.metadata.Particles.matrix_composition) == 0: warnings.warn(
            'The matrix composition is empty', UserWarning)

        utils.print_particles_property(np.expand_dims(self.metadata.Particles.matrix_composition, axis = 1),
                                       header = self.metadata.Particles.matrix_elements,
                                       label = [self.metadata.chemical_unit])

    def get_removable_elements(self):
        """Based on the maximum composition stored in the object's Particles class attribute composition, 
        the function will return an array of elements to remove.
        """
        return np.asarray(self.get_identified_elements())[np.max(self.Particles.composition, axis = 1) == 0]
    
    def update_Image_shape(self):
        """Update the Image class' shape.
        """
        self.Images.shape = self.Images.navigation_shape + self.Images.signal_shape

    def get_particles_image_id(self, print_warning = True):
        """Get the image ids where the different particles are located
        
        Returns
        -------
        image_ids 
            numpy array keeping track of the image ids where specific particles were identified.

        Example
        -------
        >>> import particle_analysis as pa
        >>> s = pa.load(filename)
        >>> s.load_images(image_path)
        >>> # Image numbers (image index = image number - 1) where particle 0,1,2,3,... are located.
        >>> s.get_particle_image_id()
        array([1,1,3,3,3,6,7,7,7,...]) 
        """
        if print_warning: 
            print('Note that the image ID from this function == (index ID + 1) as in the image naming.')
            print('(See *.metadata.Particles.label_name')
        
        image_ids = []

        for i in range(self.number_of_particles): image_ids.append(int(self.metadata.Particles.label_name[i].split('-',2)[1]))

        return np.array(image_ids)

    def create_navigation_grid_mask(self, edge_width = 1):
        """Create a mask to overlay on stitched images. The mask will highlight the image edges"""
        
        return self.Images.create_stitched_image_grid_mask(edge_width = edge_width)
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%% IMAGE ANALYSIS %%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def get_particle_area_density(self):
        """Get the particle area density per image.

        Returns
        -------
        particle_area_density
            Array containing the number of particles per imaged area
        """
        image_area = np.prod(
            self.Images.signal_shape) * np.square(
            self.Images.navigation_signal.axes_manager[-1].scale)

        number_of_particles_per_image = self.Images.get_number_of_particles_per_image()

        return number_of_particles_per_image / image_area

    def get_number_of_classes_per_image(self):
        """Return the number of classified particles per class per image.

        Returns
        -------
        classes
            Classes which the numbers in number_of_classes corresponds to
        number_of_classes
            Number of particles per class per image
        """

        regridify = False
        
        if self.Images.is_gridified:

            grid_shape = self.Images.navigation_shape
            
            self.degridify_SEM_images()

            regridify = True

        classes = self.get_unique_particle_classes()
        
        number_of_classes = np.zeros((len(classes),) + self.Images.navigation_shape)

        pim_arr = self.get_particles_image_id(print_warning=False) - 1

        for im_id in range(np.max(grid_shape)):

            for cl in np.arange(len(classes)):
                
                number_of_classes[cl][im_id] = np.sum(
                    (self.Particles.classes == classes[cl])[np.where(
                        pim_arr == im_id
                    )]
                )

        number_of_classes = np.expand_dims(number_of_classes, axis = 2)
        
        if regridify:

            self.gridify_SEM_images(grid_shape)

            number_of_classes = _image_utils._gridify_nD_array_to_ND(
                number_of_classes,
                to_shape = (len(classes),) + grid_shape
            )

        return classes, number_of_classes

    def plot_bar_of_classes(self, 
                            bar_colour = 'tab:olive',
                            x_label_rotation = -25,
                            return_fig = False):
        """Plot a bar plot of the total number of classes

        Parameters
        ----------
        bar_colour 
            Colour of the bars in the bar plot
        x_label_rotation
            rotation of the x-labels
        return_fig
            Whether to return the figure or not
        """
        import matplotlib.pyplot as plt
        
        class_names = self.get_unique_particle_classes()
        
        num_particles = dict() #np.zeros((len(class_names)), int)
        
        for cname in class_names:
            
            num_particles[cname] = np.sum(self.Particles.classes == cname)
        
        x = np.arange(len(num_particles)) - 0.5
    
        fig, ax = plt.subplots(figsize = (8,8))
        
        ax.bar(x = x,
               height = num_particles.values(), 
               width = 1,
               color = bar_colour, 
               edgecolor = 'k')
        
        ax.set_xticks(
            ticks = x, 
            labels = num_particles.keys(),
            rotation = x_label_rotation,
            fontsize=14,
            ha = 'left')    
        
        plt.show()
        
        if return_fig: return fig
            
    def plot(
        self, 
        colours = None, 
        background_colour = 'whitesmoke'
    ):
        """Plot the phase maps stored in Images class
        
        Parameters
        ----------
        colours 
            list of dictionary of colours
        background_colour
            Colour of non-particles (background)
        """

        if not hasattr(self, 'Images'): 
            
            raise AttributeError("The object has no Images class attribute yet. See load_images()")

        if not hasattr(self.Images, 'phase_map'): 
            
            raise AttributeError("The class has no maps of the particles' region. See *.identify_particle_regions()")

        if len(self.Images.phase_map) == 0: 
            
            print('Phase maps need to be defined first. See function *.update_phase_maps()')

        else:
            
            if not colours is None:

                if type(colours) == type(dict()):

                    class_args =  self._check_class_arguments(list(colours.keys()))
                
                    if not class_args[0]: 
                        
                        warnings.warn(f"The provided classes {class_args[1]} in colour argument is not recognised.")

            self.Images.plot(
                unique_classes = self.get_unique_particle_classes(),
                colours = colours,
                background_colour = background_colour
            )

    def get_entire_phase_map(self):
        """Return the entire phase map as a numpy ndarray

        Returns
        -------
        phase_map
            numpy.ndarray representing the phase map. The labels corresponds to the Images' class' phase_map keys.
        phase_labels
            Phase label IDs representing the phase_map keys.

        """
        if not hasattr(self, 'Images'): raise AttributeError("Object doesn't have the Image class attribute. See *.load_images()")

        if len(self.Images.phase_map) == 0: print("The phase maps have not been set yet. See *.update_phase_maps()")

        else:

            return self.Images.get_entire_phase_map(self.get_unique_particle_classes())
                                                  
    def load_images(self, image_path, set_image_dtype = None):
        """Read the individual particle images, SEM images and the stitched image and allocate it to the object's
        Images class. Note that the image path is expected to point to the content with folders named as "View000x, 
        View0002, ..."

        Parameters
        ----------
        image_path 
            path to where the images are stored (string)
        set_image_dtype
            Whether to set the images to a data type different from the default one set by pyplot.imread
            (float64). Note that the argument data type will be checked if it will change the image data 
            or not before setting the data type. If it will change the data type, the default data type 
            is used. 

        Example
        -------
        >>> import particle_analysis as pa
        >>> s = pa.load(filename)
        
        >>> s.load_images(path, set_image_dtype = np.uint8)
        >>> s.Images.navigation_signal.data.dtype
        dtype('uint8')
        
        >>> # Plot SEM images:
        >>> s.Images.navigation_signal.plot()
        """

        try: label_names = self.metadata.Particles.label_name

        except AttributeError: raise AttributeError("Label names is not stored in the object's metadata")
        
        if not (self.number_of_particles == len(label_names)): 
            
            raise AttributeError(f"Number of particles {self.number_of_particles} is not equvalent to the number of particle label names ({len(label_names)})") 
        
        self.Images = _images.Images(path = image_path, label_names = label_names, dtype = set_image_dtype)
        
        Image_shape = self.Images.navigation_signal.data.shape

        if np.shape(Image_shape) == (3,): nav_shape, sig_shape = (Image_shape[0],), Image_shape[1:]

        elif np.shape(Image_shape) == (4,): nav_shape, sig_shape = Image_shape[:2], Image_shape[2:]

        else: raise ValueError(f"The Images' shape <({nav_shape} | {sig_shape})> are not expected.")
        
        # Store the image shape information from the particle analysis...
        self.metadata.Acquisition_settings.navigation_shape = nav_shape

        self.metadata.Acquisition_settings.signal_shape = sig_shape

        # and the current status:
        self.Images.navigation_shape = nav_shape

        self.Images.signal_shape = sig_shape

        self.update_Image_shape()
        
        print(f"Setting navigation and signal shape to ({nav_shape[0]}|{sig_shape[0]}, {sig_shape[1]})")

    def get_depadded_particle_image(self, particle_label_id):
        """As the particle images have padded (zero-valued) regions so that they have the same shape,
        the current function is intended to remove the padded edges and return only the particle image.

        Parameters
        ----------
        particle_id
            Particle label in the particle_map signal. I.e. particle_id us located at the index position == particle_id - 1

        Returns
        -------
        depadded_particle_image 
            particle image with no pad
        """

        if not hasattr(self, 'Images'): raise AttributeError("No images have been loaded. See the function *.load_images()")

        else: return self.Images.get_depadded_particle_image(particle_label_id)
        
    def get_stitched_SEM_image(self, navigation_shape = None, 
                               horisontal_direction = 'r2l', 
                               vertical_direction = 't2b',
                               scale = None):
        """Stitch the individual SEM images into one large overview image.

        Parameters
        ----------
        navigation_shape 
            Shape which the images will be stitched into. Default None, as it will try to read the object's navigation_shape
            Obs! It's the same convention as matplotlib's real space convention, i.e. 4 images in horisontal direction and 3 images 
            in vertical direciton will be correctly stitched if navigation_shape = (3,4)
        horisontal_direction
            Particle analysis acquired images from right to left, hence the default argument r2l. Alternative is to read from l2r
        vertical_direction
            Particle analysis acquired images from top to bottom, hence the default argument t2b. Alternative is to read from b2t

        Returns
        -------
        stitched_image_array
            numpy.ndarray with the stitched images
        """
        
        if not hasattr(self, 'Images'): 
            
            raise ValueError("Images class must be allocated your object, and the SEM images need to be read.")
        
        if self.Images.is_gridified and navigation_shape is None: 
            
            navigation_shape = self.Images.navigation_shape

            sem_images = _image_utils._gridify_4D_array_to_3D(self.Images.navigation_signal.data)

        elif not self.Images.is_gridified and navigation_shape is None:

            raise ValueError(f"Images are not gridified. Provide a valid navigation_shape different from '{navigation_shape}'")

        else:

            if self.Images.is_gridified: sem_images = _image_utils._gridify_4D_array_to_3D(self.Images.navigation_signal.data)

            else: sem_images = self.Images.navigation_signal.data
            
        if np.prod(navigation_shape) != np.prod(self.metadata.Acquisition_settings.navigation_shape):

            raise ValueError(f"The navigation shape {navigation_shape} is not compatible with the number of analysed images ({self.metadata.Acquisition_settings.analysed_views})")
        
        stitched_im = _image_utils._stitch_images(sem_images,
                                             navigation_shape + self.metadata.Acquisition_settings.signal_shape,
                                             horisontal_direction = horisontal_direction,
                                             vertical_direction = vertical_direction)

        if scale is not None:

            from skimage.transform import rescale
            
            stitched_im = rescale(stitched_im, scale, anti_aliasing=False)
        
        return stitched_im
    
    def gridify_SEM_images(self, navigation_shape, flip_axis = 1):
        """Reshape the object's 3D array of SEM images into a 4D array: 2 stage coordinates and 2 SEM image coordinates.

        Parameters
        ----------
        navigation_shape
            Shape of the stage navigation (list or tuple)
        flip_axis 
            The axis which to flip (as particle analysis images are acquired from right to left, top to bottom)

        Example
        >>> import particle_analysis as pa
        >>> s = pa.load("Results.csv")
        >>> s.read_images(Image_path)
        >>> s.Images.navigation_signal.data.shape
        (20, 768, 1024) 
        >>> s.gridify_SEM_images(navigation_shape = (5,4))
        >>> s.Images.navigation_signal.data.shape
        (5,4,768,1024) # (Y,X,y,x)
        """
        
        navigation_shape = tuple(navigation_shape)
        
        if not hasattr(self, 'Images'): 
            
            raise AttributeError("Images class must be allocated the object, and the SEM images need to be read.")

        if np.prod(navigation_shape) != np.prod(self.metadata.Acquisition_settings.navigation_shape):

            raise ValueError(f"The navigation shape {navigation_shape} is not compatible with the number of analysed images ({self.metadata.Acquisition_settings.analysed_views})")
        
        #if 1 not in navigation_shape:
            
        # Make the signal into the initially read 3D array
        
        if navigation_shape != self.Images.navigation_shape and len(self.Images.navigation_shape) > 1:
            
            self.degridify_SEM_images(flip_axis = flip_axis)
        
        self.Images.navigation_signal = Signal2D(
            _image_utils._gridify_3D_array_to_4D(
                self.Images.navigation_signal.data,
                to_shape = navigation_shape + self.metadata.Acquisition_settings.signal_shape,
                flip_axis = flip_axis)
        )

        # Update the image navigation shape
        self.Images.navigation_shape = navigation_shape

        self.update_Image_shape()

        # Update the other 4D signals
        self.Images._gridify_2Dsignals()

        self.Images.is_gridified = True

        self._update_set_navigation_scale()

        #else: print(f"Provided grid shape {navigation_shape} is not valid.")

    def degridify_SEM_images(self, flip_axis = 1):
        """Make the 4D image signal into a 3D image signal
        This is the opposite function of gridify_SEM_images
        """

        self.Images.navigation_shape = self.metadata.Acquisition_settings.navigation_shape

        self.update_Image_shape()

        if self.Images.navigation_signal.data.shape != self.Images.shape:
    
            print(f"Degridifying array of shape {self.Images.navigation_signal.data.shape} -> {self.Images.shape}")
            
            self.Images.navigation_signal = Signal2D(
                _image_utils._gridify_4D_array_to_3D(self.Images.navigation_signal.data,
                                                flip_axis = flip_axis)
            )
    
            self.Images.is_gridified = False

            self.Images._degrifify_2Dsignals()

            self._update_set_navigation_scale()

        else: print("Signal is already degridified.")

    def set_particle_regions(self,
                             labelled_particle_map,
                             horisontal_labelling_direction = 'l2r',
                             vertical_labelling_direction = 't2b'):
        """As an altenrative to the much slower function *.identify_particle_regions(), apply the set conditions 
        during particle analysis to re-identify the particles in the SEM images. 

        Parameters
        ----------
        conditions
            Set of conditions (dict)
        label_particles 
            Whether to label the particles and not just make a binary mask of the particles
        labelling_direction
            Direction in which the particles are labelled in the particle analysis software. Default: right to left (r2l)??
        
        """
        if hasattr(self.Images, 'particle_map'): print("The particle regions have already been defined. Run \n\t>>> del [self].Images.particle_map\nto enable redefinition of the particle regions.")

        elif labelled_particle_map.shape != self.Images.navigation_shape + self.Images.signal_shape:

            print(f'The labelled particle map shape ({labelled_particle_map.shape}) is not compatible with the signal shape {self.Images.navigation_shape + self.Images.signal_shape}')

        elif np.unique(labelled_particle_map) != np.array([0,1]) or np.unique(labelled_particle_map)[-1] != self.number_of_particles:

            print(f"The number of unique particle IDs ({np.unique(labelled_particle_map[-1])}) is not compatible with the number of particles from the particle analysis ({self.number_of_particles})\n Providing a particle map with different number of particles than the number from the data acquisition can complicate the analysis.")
                  
        else:

            print('NOT WRITTEN YET')
            print('FIX labelling direction!!!')
            print('FIX PARTICLE IMAGE ACQUISITION')
            # THE PLAN HERE IS TO MAKE THE USED PROVIDE THE USED LABELLING DIRECTION, AND THE FUNCTION WILL CORRECT FOR THIS BY RELABELLING IN ACCORDANCE WITH Jeol's LABELLING SCHEME

            self.Images.particle_map = labelled_particle_map
                                
        
    def identify_particle_regions(self, 
                                  label_particles : bool = True,
                                  return_overlapping_pixels_map : bool = False,
                                  matrix_label : int = 0):
        """Identify the particle regions in each SEM image by running template matching.
        (See skimage.feature's function match_template). The particle map is stored in the
        Images' object attribute *.particle_map

        Parameters
        ----------
        label_particles 
            Whether to get a particle map as a boolean mask or a labelled map. 
            Note that the labelling is according to the particle's labels
        
        return_overlapping_pixels_map
            Whether to return a map containing the pixels where the particle images 
            are overlapping.
            
        Note that overlapping regions are shared between the particles (~equally) 

        Returns
        -------
        omap
            Map of identified particles' overlapping pixels, given True 
            return_overlapping_pixels_map argument. 
            

        Example
        -------
        >>> import particle_analysis as pa
        >>> s = pa.load(filename)
        >>> s.load_images(path)
        
        >>> overlapping_pixels = s.identify_particle_regions(return_overlapping_pixels_map = True)
        >>> s.Images.particle_map
        <Signal2D, title: , dimensions: (4,5|1024, 768)>
        """

        if not hasattr(self, 'Images'): 
            
            raise AttributeError("The images from particle analysis must be loaded to map the particles. See the function *.load_images().")

        if not hasattr(self.Images, 'particle_images'):

            raise AttributeError("The object doesn't have images of particles. These are needed to map the particles' position in the SEM images.")

        if not hasattr(self.Images, 'navigation_signal'):

            raise AttributeError("The object doesn't have the acquired SEM images. These are needed to map the particles' position.")
        
        if len(self.Images.navigation_shape) == 2: sem_images = _image_utils._gridify_4D_array_to_3D(self.Images.navigation_signal.data)

        else: sem_images = self.Images.navigation_signal.data
        
        pmaps = list(_image_utils._get_map_of_particle_regions(
            particle_images = self.Images.get_group_of_depadded_particle_images(self.Images.unique_particle_labels),
            SEM_images = sem_images,
            SEM_image_IDs = self.get_particles_image_id(print_warning = False),
            label_particles = label_particles))
        
        if self.Images.is_gridified: 
            
            pmaps[0] = _image_utils._gridify_3D_array_to_4D(pmaps[0], to_shape = self.Images.shape)

            if return_overlapping_pixels_map: pmaps[1] = _image_utils._gridify_3D_array_to_4D(pmaps[1], to_shape = self.Images.shape)
        
        self.Images.particle_map = Signal2D(pmaps[0])

        self._update_particle_label_list(matrix_label)

        plabels = self.Images.unique_particle_labels.copy()

        # Check if some lables have been removed
        if self.number_of_particles != len(plabels):

            missing_labels = _utils._identify_missing_labels(
                current_labels = plabels, 
                desired_labels = np.arange(1, self.number_of_particles + 1)
            )
            
            if len(missing_labels) > 0:

                # Correct for the missing labels by insert single pixels:
                print("IMPROVE REINSERION OF PARTICLE REGIONS - CONSIDER 'DILATION'")
                correct = input('Correct for missing particle labels? ([y]/n)')

                if correct == '' or correct.upper() == 'Y':

                    print('Correcting for missing labels')

                    regridify = False

                    desired_label_list = np.arange(1, self.number_of_particles + 1)
                    
                    if self.Images.is_gridified:

                        regridify = True

                        grid_shape = self.Images.navigation_shape

                        self.degridify_SEM_images()

                    im_indices = self.get_particles_image_id(print_warning=False) - 1

                    for missing_label in missing_labels:

                        nav_index = im_indices[np.where(desired_label_list == missing_label)]

                        _pmap = _image_utils._get_single_particle_map(
                            SEM_image = self.Images.navigation_signal.inav[nav_index].data,
                            particle_image = self.Images.particle_images.inav[missing_label - 1].data
                        )

                        p_centre = np.round(
                            np.median(
                                np.where(_pmap), axis = 1
                            )
                        ).astype(int)

                        # Setting the centre pixel of the missing particle region as the particle label 
                        self.Images.particle_map.inav[nav_index].data[p_centre[0], p_centre[1]] = missing_label

                    if regridify:
                        
                        self.gridify_SEM_images(grid_shape)
                    
                    self._update_particle_label_list(matrix_label)

                    missing_labels = _utils._identify_missing_labels(
                        current_labels = self.Images.unique_particle_labels, 
                        desired_labels = np.arange(1, self.number_of_particles + 1)
                    )

        if return_overlapping_pixels_map: return pmaps[1]

    def identify_cropped_particles(self, edge_width = 1):
        """The function iterates through all the images according to its navigation shape
        and looks for particles that might be the same particle, but in different SEM/
        particle_map images. The identified particles' indices will be stored in the object's
        instance Images.cropped_particles_map (hyperspy signal)

        OBS! The function will separate cropped particles by their classes. I.e. it can be 
        an idea to classify the particles before mapping the cropped particles.

        Parameters
        ----------
        edge_width
            Image edge width in number of pixels to look for cropped particles. By default: 1, but a 
            higher value can be of advantage if the SEM images has a high spatial resolution. 

        Example
        -------
        >>> s = pa.load(filename)
        >>> s.load_images(image_path)
        >>> s.Images
        Particle analysis imags:
            SEM images: <(16 | 1536,2048) | Particle images: (1000 | (500,400))>

        >>> s.gridify_SEM_images(navigation_shape = (4,4))
        >>> s.Images
        Particle analysis imags:
            SEM images: <(4,4 | 1536,2048) | Particle images: (1000 | (500,400))>

        # Identify the particle regions
        >>> s.identify_particle_regions()
        >>> s.identify_cropped_particles(edge_width = 2)
        >>> s.Images.cropped_particles_map
        <Signal2D, title: , dimensions: (4,4|2048,1536)>
        """
        if not hasattr(self, 'Images'): raise AttributeError("The class doesn't have the attribute 'Images'. See the *.load_images() function.")      
            
        #cropped_particles_map = _image_utils._get_edge_particles_gridMap_from_labelled_particles_gridMap(
        #    self.Images.particle_map.data,
        #    edge_width = edge_width
        #)
        
        #self.Images.identify_cropped_particles(edge_width = edge_width)

        #print("Cropped particle map -> *.Images.cropped_particles_map.")
        #self.Images.cropped_particles_map = Signal2D(cropped_particles_map.copy())

        # Identify the cropped particles
        self.Images.map_cropped_particles(edge_width = edge_width)

        # Get the cropped particles' labels
        unique_particle_labels = np.unique(self.Images.cropped_particles_map)

        print("Cropped particle labels -> *.metadata.Particles.")
        self.metadata.Particles.cropped_particles = np.delete(unique_particle_labels, 
                                                              np.where(unique_particle_labels == 0)).astype(np.uint32)

        # Create a navigation grid mask:
        grid_mask = self.create_navigation_grid_mask(edge_width = edge_width)

        # Identify cropped particles at the grid edges:
        stitched_cropped_particles_map = utils.stitch_images(self.Images.cropped_particles_map.data, 
                                                             self.Images.navigation_shape)

        # Cluster the cropped particles
        edge_clusters = _image_utils._identify_particle_edge_clusters(stitched_cropped_particles_map,
                                                                      image_edges_grid = ~grid_mask)

        # Throw a warning if the only class is 'Unclassified'
        if len(self.get_unique_particle_classes()) == 1:

            if self.get_unique_particle_classes()[0] == 'Unclassified': 
                
                warnings.warn(f"\nThere are no classified particles. The cropped particles will not be corrected according to their classes.")
            
            else:

                print('The following has not been properly tested yet.')

                proceed = input('Proceed? ([y]/n')
                
                if proceed == '' or proceed.upper() == 'Y':
                    
                    edge_clusters = _utils._check_cropped_particle_class_compatability(classes = self.Particles.classes,
                                                                                       clusters = edge_clusters)

        print("Storing the clustered particles' corr. labels in *.metadata.Particles.cropped_particle_clusters")
        self.metadata.Particles.cropped_particle_clusters = edge_clusters

        # Hide a copy
        self.metadata.Particles._cropped_particle_clusters = edge_clusters

    def stitch_cropped_particles(self, 
                                 conditions : dict = {},
                                 stitching_ncc_threshold : float = 0.25,
                                 extra_pixels : float = 0.2,
                                 shift_threshold : float = 1.0,
                                 remove_non_matched_region : bool = False,
                                 directly_stitch_edge_particles : bool= True,
                                 filter_stitched_images_to_correct_for_stitching_process : bool = False,
                                 looping_progressbar : bool = False):
        """The function attempts to first stitch identified cropped particles, before correcting for 
        the particles composition and labels given that conditions is not empty.

        To correct the cropped particles' geometric properties, see the function *.update_particles_geometric_propertes()

        Parameters
        ----------
        stitching_ncc_threshold
            Float value describing the lowest allowable ncc score between two images' overlapping 
            region before stitching is considered successful.
            Default: 0.5.
        extra_pixels
            Float (percentage) defining how many extra pixels wrt. SEM image widht or height to 
            add to improve the chance of correct stitching. 
        shift_threshold 
            A number that assess whether the image stitching was successful or not by how much it was shifted 
            (Eucledian distance). Default: 1.0
        remove_non_matched_region 
            Whether to remove the region that was initially not part of the stored particle image from the 
            stitched image. By default: True
        directly_stitch_edge_particles 
            If particle stitching fails for some reason, concatenate the particle images directly. 
        filter_stitched_images_to_correct_for_stitching_process
            Whether to smooth the stitched images or not. A good reason to do this is to remove artefacts from the image shifting.
            See skimage.registration's phase_cross_correlation function.
        update_particle_properties
            Wheter to correct the cropped particle pairs' chemical composition and labels.
        """

        update_particle_properties = False

        if len(conditions) > 0: update_particle_properties = True

        # Stitch cropped particles and update pair of particles' labels and composition
        self._stitch_cropped_particles(
            stitching_ncc_threshold = stitching_ncc_threshold,
            extra_pixels = extra_pixels,
            shift_threshold = shift_threshold,
            remove_non_matched_region = remove_non_matched_region,
            directly_stitch_edge_particles = directly_stitch_edge_particles,
            filter_stitched_images_to_correct_for_stitching_process = filter_stitched_images_to_correct_for_stitching_process,
            looping_progressbar = looping_progressbar,
            update_particle_properties = update_particle_properties
        )

        # As some identified cropped particles might be grouped together, the following 
        # function attempts to segment these and group together labels/particles of the same origin
        self._cluster_group_of_cropped_particles(
            conditions = conditions,
            update_particle_properties = update_particle_properties
        )

    def update_particles_geometric_properties(self,
                                              conditions : dict,
                                              remeasure_properties = [
                                                  'Area [um²]',
                                                  'Maximum length [um]',
                                                  'Roundness',
                                                  'Orientation [degree]',
                                                  'Perimeter'], 
                                              labels : list | np.ndarray = [],
                                              return_single_particle_only : bool = True):
        """Update particles' geometric properties as provided by the remeasure_properties list. If the list is empty, the
        function will by default iterate through all pairs of stitched particles.
        
        The allowed properties are:
            -Area
            -Maximum length (maximum feret)
            -Roundness
            -Orientation
            -Perimiter

        See also https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
        for more properties.

        The allowed string (?) conditions are 
            < 
            <=
            >
            >=
            min_size
            max_size
            binary_erosion
            binary_dilation
            fill_holes

        Parameters
        ---------
        conditions 
            Dictionary of conditions
        remeasure_properties
            list of geometric properties to remeasure
        labels
            Which particle (labels) to remeasure geometric properties. If the list is empty,
            the stitched pair of particles will be remeasured. 
        return_single_particle_only
            Whether to only return single particles from the particle image based on the 
            largest number of segmented pixels.
        Example
        -------
        s.stitch_cropped_particles()
        s.update_stitched_particles_geometric_properties(
            conditions = {'<' : 80, # Intensity lower than 80 is of interest
                          '>=' : 160, # Intensity lower than or equal to are of interest
                          '' : ,
        })
        """

        # For regionprops_table 
        property_translator = {
            'area [um²]' : 'area',
            'area' : 'area',

            'maximum length [um]' : 'feret_diameter_max',
            'maximum length' : 'feret_diameter_max',
            'max length' : 'feret_diameter_max',

            'roundness' : 'area_convex',

            'orientation [degree]' : 'orientation',
            'orientation' : 'orientation',

            'perimeter [um]' : 'perimeter',
            'perimeter' : 'perimeter',
        }

        class_translator = {
            'area' : 'Area [um²]',
            'area [um²]' : 'Area [um²]',

            'feret_diameter_max' : 'Maximum length [um]',
            'maximum length [um]' : 'Maximum length [um]',

            'area_convex' : 'Roundness',
            'roundness' : 'Roundness',

            'orientation' : 'Orientation [degree]',
            'orientation [degree]' : 'Orientation [degree]',

            'perimeter' : 'Perimeter [um]',
            'perimeter [um]' : 'Perimeter [um]',
        }

        if not hasattr(self, 'Images'): 
            
            raise AttributeError(f"The class doesn't have any particle images. See *.load_images().")

        if len(labels) == 0:

            if not hasattr(self.metadata.Particles, '_cropped_particle_clusters'): 
            
                raise AttributeError(f"The class' Particles class doesn't have any information about cropped particle clusters. See *.identify_cropped_particles().")
            
            if not hasattr(self.Images, '_successfully_stitched_particle_images'):
                
                raise AttributeError(f"The class doesn't have any stitched particles yet. See *.stitch_cropped_particles()")
            
            print(f'Correcting the geometric properties {remeasure_properties} for pair of stitched particles.')
            
            particle_labels = self.metadata.Particles.relabelled_stitched_particle_labels[:,0]

            # Add the new grouped particle labels if they exists.
            if hasattr(self.metadata.Particles, 'updated_particles_from_group_of_cropped_particles'):

                particle_labels = np.sort(
                    np.append(
                        particle_labels, self.metadata.Particles.updated_particles_from_group_of_cropped_particles
                    )
                )
        
        else: 
            
            print('Correcting geometric properties for the specified particles.')
            
            particle_labels = labels

        # Import libraries
        from tqdm import tqdm 
        from skimage.measure import label, regionprops_table

        if self.metadata.navigation_scale == 1:

            warnings.warn(f"The real space calibration scale is equal to one.")

        if 'roundness' in remeasure_properties:
            
            if len((set(remeasure_properties) & set(['perimeter [um]', 
                                                     'perimeter', 
                                                     'Perimeter [um]', 
                                                     'Perimeter',]))) == 0:
                
                # To estimate a particle's roundness, we need both the area_convex from the 
                # property_translator and the perimeter.
                remeasure_properties.append('perimiter')
        
        from skimage.measure import regionprops_table

        # Translate property arguments:
        properties = []
        for prop in remeasure_properties: properties.append(property_translator[prop.lower()])

        if not hasattr(self.Images, 'relabeled_stitched_particle_masks'):
            
            self.Images.relabelled_stitched_particle_masks = dict()
        
        for l in tqdm(particle_labels):

            if l not in self.Images.unique_particle_labels: 
                
                warnings.warn(f"Particle label {l} is not found among the unique particle labels.")

            else: 

                pmask = self.Images._create_particle_mask(
                    pimage = self.get_depadded_particle_image(l),              
                    conditions = conditions,                                      
                    return_single_particle_only = return_single_particle_only
                )

                lab_image = label(pmask)

                self.Images.relabelled_stitched_particle_masks[l] = lab_image
                
                if len(np.unique(lab_image)) > 2:
                    
                    warnings.warn(f"\n\nThe particle labelled as {l} has {len(np.unique(lab_image))} labels.\nThe particle's geometric properties is recommended to be manually corrected.")

                else:
                
                    props = regionprops_table(label_image = lab_image, 
                                              intensity_image=self.get_depadded_particle_image(l), 
                                              properties=properties,
                                              spacing = self.metadata.navigation_scale)
    
                    for key in remeasure_properties:
    
                        if 'roundness' in key.lower():
                            
                            props['Roundness'] = props['area_convex'] / props['perimeter']
                        
                        elif 'orientation' in key.lower():
    
                            props['Orientation [degree]'] = np.rad2deg(props['orientation'])
                        
                        
                        self.Particles.particle_geometry[
                            class_translator[key.lower()]
                            ][np.where(self.Images.unique_particle_labels == l)] = props[
                                property_translator[key.lower()]]

    def update_phase_maps(self, background_val = 0):
        """Create phase maps according to the classes. The individual phase maps
        will be stored as a dictionary in the "Images" class.

        Note that the shape of the phase maps are stored as a 3D array. Reshaping is 
        necessary to make it compatible with a gridified signal.

        Example
        -------
        >>> import particle_analysis as pa
        >>> s = pa.load(filename)
        >>> s.load_images(image_path) # Necessary to identify the probed particles
        >>> s.gridify_SEM_images((5,4)) # Necessary to correctly shape the phase maps???
        >>> s.identify_particle_regions() # Get labelled particles
        >>> s.update_phase_maps() # Updating phase maps
        >>> s.plot() # Plotting phase maps with defualt or auto-generated rgb values
        """
        
        if not hasattr(self, 'Images'): 
            
            raise AttributeError("The class doesn't have Images attribute yet. See load_images function.")

        if not hasattr(self.Images, 'particle_map'): 
            
            raise AttributeError("The Images class attribute doesn't keep track of the particles' position. See *.identify_particle_regions()")

        # As we are iterating through self.get_particlse_image_id, we need to correct for the cropped particles
        # when making the phase maps.
        if hasattr(self.Images, 'cropped_particles_map'):

            correct_cropped_particles = True

        else: correct_cropped_particles = False

        num_classified_particles = len(self.Particles.classes) 
        
        num_unique_labels = len(self.Images.unique_particle_labels)

        if num_classified_particles != num_unique_labels:
            
            raise _errors.ShapeError(f"The number of classified particles ({num_classified_particles}) are different from the number of unique particle labels ({num_unique_labels}).")
        
        from tqdm import tqdm
        
        # Delete old phase maps
        if len(self.Images.phase_map.keys()) > 0: 
            
            del self.Images.phase_map

            self.Images.phase_map = dict()

        particle_map = self.Images.particle_map.data.copy()

        # Enable image iteration:
        if self.Images.is_gridified: 
            
            particle_map = _image_utils._gridify_4D_array_to_3D(particle_map)
        
        image_idxs = self.get_particles_image_id(print_warning = False) - 1

        classes = self.Particles.classes

        unique_classes = self.get_unique_particle_classes()
        
        for cl, counter in zip(unique_classes, np.arange(1, len(unique_classes) + 1)):

            print(counter,'/', len(unique_classes))

            # Temporary phase map
            phase_map = np.zeros_like(particle_map, bool)

            #class_indices = np.where(classes == cl)[0]
        
            #for img_idx in tqdm(image_idxs, desc=cl):
                
            #    unique_p_indices = np.unique(particle_map[img_idx])
                
                # Remove background
            #    unique_p_indices = np.delete(unique_p_indices, np.where(unique_p_indices == background_val))
                            
            #    for p_idx in unique_p_indices: 

            #        if (p_idx - 1) in class_indices:
                    
            #            phase_map[img_idx][np.where(particle_map[img_idx] == p_idx)] = True

            class_indices = np.where(classes == cl)
        
            for p_idx, img_idx in tqdm(zip(class_indices[0], 
                                           image_idxs[class_indices]), 
                                       desc=cl, 
                                       total = len(class_indices[0])):
                
                #phase_map[img_idx][np.where(particle_map[img_idx] == (p_idx + 1))] = True
                phase_map[img_idx][np.where(particle_map[img_idx] == self.Images.unique_particle_labels[p_idx])] = True
            
            self.Images.phase_map[cl] = phase_map.copy()

        # Correct for cropped particles:
        if correct_cropped_particles:

            print('NOT TESTED: Correcting for cropped particles')

            nav_shape = self.Images.navigation_shape

            self.degridify_SEM_images()

            relabelled_cropped_particles = self.metadata.Particles.relabelled_stitched_particle_labels[:,0]

            for lab in relabelled_cropped_particles:

                class_ = self.Particles.classes[np.where(self.Images.unique_particle_labels == lab)]

                self.Images.phase_map[class_[0]][np.where(self.Images.particle_map.data == lab)] = True

            self.gridify_SEM_images(navigation_shape = nav_shape)


        self.Images.update_phase_map_shape()

    def get_coloured_phase_map_on_SEM_images(self,
                                             colours,
                                             adjust_intensity = False,
                                             background_val = 0,
                                             iterate_through_phase_maps = True):
        """Tint gray-scale particle map with colours defined by the dictionary colours.

        Parameters
        ----------
        colours
            Colours dictionary
        background_val 
            Background value in labelled particle map (By default: 0)
        iterate_through_phase_maps
            Whether to mask array regions according to the phase_map arrays. 
            If the phase maps are large in size, it can be an idea to set it 
            to False to enable iteration through the SEM images that were found
            with particles.
            
        Returns 
        -------
        tinted particle rgb image
            a tinted gray-scale image according to the colours in colours argument

        Example
        -------

        >>> import particle_analysis as pa
        >>> s = pa.load('Results.csv')
        >>> s.load_images(image_path)
        
        >>> s.gridify_SEM_images(navigation_shape = (5,5))
        
        >>> s.identify_particle_regions(label_particles = True)
        Searching for the analysed particles in the SEM images...
        Overlapping particle regions will be shared.

        >>> phase_map = s.colour_particle_map(colours = ['red','blue','green'])
        >>> phase_map.shape 
        (5,5,768, 1024,3)
        """
        
        if type(colours) != type(dict()): raise TypeError("Provided colour argument is not a dictionary.")

        class_args =  self._check_class_arguments(list(colours.keys()))
        
        if not class_args[0]: 
            
            warnings.warn(f"The provided classes {class_args[1]} in colour argument is not recognised.")
        
        if not hasattr(self, 'Images'):
            
            raise AttributeError("The particle analysis images must be ead in order to get a map of the chemically mapped particles. See ")

        if not hasattr(self.Images, 'particle_map'):

            raise AttributeError("The particle images must be defined first to enable colouring...")

        from matplotlib import colors 
        from skimage.color import gray2rgb
        from tqdm import tqdm
        import sys

        gridified_signal = self.Images.is_gridified
        
        # Whether phase maps have been set
        updated_phase_maps = len(self.Images.phase_map)

        SEM_imgs = self.Images.navigation_signal.data.copy()

        if gridified_signal: SEM_imgs = _image_utils._gridify_4D_array_to_3D(SEM_imgs)
        
        # First option: make the coloured phase map according to phase_map arrays
        if gridified_signal and updated_phase_maps > 0 and updated_phase_maps == len(self.get_unique_particle_classes()) and iterate_through_phase_maps:

            from copy import deepcopy
            
            nav_shape = self.Images.navigation_shape        
    
            phase_maps = deepcopy(self.Images.phase_map)
    
            SEM_imgs = utils.stitch_images(SEM_imgs, nav_shape)
    
            for key in phase_maps.keys(): phase_maps[key] = utils.stitch_images(phase_maps[key], nav_shape)
    
            print('Getting SEM images as normalised rgb greyscale array ...')
            SEM_imgs = utils.greyscale_to_rgb(SEM_imgs)
    
            for class_ in tqdm(self.Images.phase_map.keys()): SEM_imgs[phase_maps[class_]] *= colors.to_rgb(colours[class_])

        else: # Second option: iterate through the images

            particle_map = self.Images.particle_map.data
            
            print('Getting SEM images as normalised rgb greyscale array ...')
            SEM_imgs = utils.greyscale_to_rgb(SEM_imgs)
            
            if gridified_signal: particle_map = _image_utils._gridify_4D_array_to_3D(particle_map)
            
            rgb_vals = np.zeros((self.number_of_particles, 3), dtype = np.float32)
            
            for cl in colours.keys(): rgb_vals[np.where(self.Particles.classes == cl)] = colors.to_rgb(colours[cl])
            
            image_idxs = self.get_particles_image_id(print_warning = False) - 1
    
            classes = self.Particles.classes
    
            unique_classes = self.get_unique_particle_classes()
            
            for cl, counter in zip(unique_classes, np.arange(1, len(unique_classes) + 1)):
    
                print(counter,'/', len(unique_classes))
    
                class_indices = np.where(classes == cl)
            
                for p_idx, img_idx in tqdm(zip(class_indices[0], image_idxs[class_indices]), desc=cl, total = len(class_indices[0])):
                    
                    SEM_imgs[img_idx][np.where(particle_map[img_idx] == (p_idx + 1))] *= rgb_vals[p_idx]

            print("\n\033[1mImage stitching can be done using the utils-function 'stitch_rgb_phase_map(phase_map, nav_shape = (X,Y).\033[0m")

        if adjust_intensity: 
            
            from skimage.exposure import rescale_intensity
            
            SEM_imgs = rescale_intensity(np.sqrt(SEM_imgs), out_range = (0,1))

        return SEM_imgs

    
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%% CHEMISTRY MANIPULATIONS %%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    def threshold_particles_composition(self, lower_threshold):
        """Apply a lower threshold on the particles' chemical composition. The upper
        limit is 100 (%). I.e. element compositions < lower_threshold is set to zero.
        
        Parameters
        ----------
        lower_threshold 
            Lower threshold value (float) in percentage.
        """
        # Make sure the concentration is in the range [0, 100]:
        self._update_particles_concentration()
        
        self.Particles.composition[self.Particles.composition < lower_threshold] = 0.0

        self._update_particles_concentration()

    def update_particles_composition_shape(self):
        """Update the attribute element_shape"""
        self.particles_composition_shape = (len(self.Particles.elements), self.number_of_particles)
        
    def remove_element(self, element):
        """Remove a specific element, or list of elements from the object.
        The function will update the particles' element concentration.

        Parameters
        ----------
        element
            A single element (string) or a list of elements
        """
        if type(element) == str: element = [element]

        print(f"Removing elements {element}")

        available_elements = list(ELEMENTS.as_dictionary().keys())[1:]
        
        for elem in element:

            elem_index = self.Particles.elements.index(elem)

            if elem not in available_elements: 
                
                print(f"Element {elem} is not recognised and will be ignored.")

            else:

                if elem in self.get_identified_elements():
    
                    self.Particles.elements.remove(elem)
                    
                    self.Particles.composition = np.delete(self.Particles.composition, elem_index, 0)
    
                else:
    
                    print(f'Element {elem} has not been identified during the data acquisition.')
                    
            #if hasattr(self.metadata.Particles, 'matrix_composition'): 

            #    remove_from_matrix = input(f"Remove element {elem} from the matrix composition? (y/[n])")

            #    if remove_from_matrix.upper() == 'Y': 
    
            #       self.metadata.Particles.matrix_composition = np.delete(self.metadata.Particles.matrix_composition, elem_index)

        self._update_particles_concentration()

        self._update_particles_composition_shape()
    
    def get_identified_elements(self):
        """Get the stored elements in 'self.Particles' elements attribute as a list"""
        return list(self.Particles.elements)
        
    def print_identified_elements(self):
        """Print the stored elements in 'self.Particles'"""
        print(self.get_identified_elements())

        
    def get_particles_with_elements(self, list_of_elements):
        """Identify particles contaning ALL the elements specified in the list_of_elements argument. 
        Note that other elements might also be present in the particles.
        
        Parameters
        ----------
        list_of_elements
            List of elements or a string of a single element

        Returns
        -------
            Array of particles with elements as specified in the list_of_elements 
        """
        if type(list_of_elements) == str: list_of_elements = [list_of_elements]

        particles = np.ones((self.number_of_particles), bool)

        for elem in list_of_elements:

            particles *= (self.Particles.composition[self.Particles.elements.index(elem)] > 0)

        return particles

    def get_particles_containing_only_elements(self, list_of_elements, ignore_elements = []):
        """Identify particles contaning the elements specified in the list_of_elements argument. 
        Other elements might also be present in the particles.
        
        Parameters
        ----------
        list_of_elements
            List of elements or a string of a single element
        ignore_elements
            List of elements to ignore (f.ex. C and/or O) - NOT FIXED

        Returns
        -------
        particles 
            Array of particles only containing the specified elements
        """
        list_type = type(list_of_elements) 
        
        if list_type == str: list_of_elements = [list_of_elements]

        elif list_type not in (list, tuple): raise TypeError(f"{list_of_elements} is unexpected. Provide a list/tuple of elements (strings), or a single element (string)")

        if type(ignore_elements) == str: ignore_elements = [ignore_elements]
        
        is_empty = True # Whether to ignore certain elements or not.

        if len(ignore_elements) > 0: 

            # Check if the same element is in both lists.
            common_elements = set(list_of_elements) & set(ignore_elements)

            if len(common_elements) > 0: warnings.warn(f"The elements to ignore {common_elements} are also found in the list of elements to search for.") 
            
            is_empty = False

        num_elements = len(list_of_elements)

        # Make an array template reflecting the elements we're interested in
        template = [0] * len(self.Particles.elements)

        for elem in list_of_elements: template[self.Particles.elements.index(elem)] = 1

        # template shape will match the Particle's attribute composition 
        template = np.reshape(np.asarray(template * self.number_of_particles, bool), self.particles_composition_shape[::-1]).T
        
        # Array of particles where the elements are found (in addition to other elements)
        array_of_fits = np.sum((template * (self.Particles.composition > 0)), axis = 0) == num_elements 
        
        # An array of particles where the exact number of elements are found
        reference_array = np.sum(self.Particles.composition > 0, axis = 0) == num_elements

        return array_of_fits * reference_array

    def get_unique_particle_classes(self, return_list = True):
        """Print the object's sorted unique particle labels.
        
        Parameters
        ----------
        return_list
            Whether to return the list of unique labels. (Default: False)

        Returns
        -------
        labels
            List of unique particle labels if return_list is True.
        """
        labels = np.unique(list(set(self.Particles.classes.astype(list))))
        
        if return_list: return labels
        
        else: print(labels)
    
    def print_unique_particle_classes(self):
        """Print the object's sorted unique particle labels.
        
        Parameters
        ----------
        return_list
            Whether to return the list of unique labels. (Default: False)

        Returns
        -------
        labels
            List of unique particle labels if return_list is True.
        """
        self.get_unique_particle_classes(return_list = False)

    def identify_false_particles_by_chemistry(self):
        """Map the match score between particles and the provided matrix composition by running template 
        matching. A high match scores represents a high probability for being part of the matrix.

        Returns
        ------
        ncc_scores : np.ndarray
            Array with normalised cross-correlation scores representing the particles' chemistry match with the 
            set matrix composition.

        Example
        -------
        >>> import particle_analysis as pa
        >>> s = pa.load(filename)
        >>> s.set_matrix_composition({'C' : 2.5, 'O' : 8.0, 'Al': 89.5}, unit = 'wt.%')
        >>> ncc_scores = s.identify_false_particles_by_chemistry()
        >>> ncc_scores
        array([0.97052249, 0.9584889 , 0.83429012, ..., 0.74499595, 0.5983186 ,
        0.92121971])

        >>> # Potential matches:
        >>> ncc_scores > 0.9999
        array([False, False, False, ..., False, False, False])
        """
        if not hasattr(self.metadata.Particles, 'matrix_composition'): raise ValueError("The class object doesn't contain information about the material's matrix composition")

        elif not hasattr(self.metadata.Particles, 'matrix_elements'): raise ValueError("The class object doesn't contain information about the material's matrix elements")

        else:

            #try: 
                
            #    from sklearn.metrics.pairwise import cosine_similarity

            #    print('Unfinished')
            #    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html

            #except ModuleNotFoundError: 

            print("Matching particles' chemical composition with the matrix composition using the NCC metric.")

            template = np.zeros((len(self.get_identified_elements())))
            
            matrix_elems = self.metadata.Particles.matrix_elements
            
            matrix_comp = self.metadata.Particles.matrix_composition
            
            identified_elems = self.get_identified_elements()
            
            for elem, conc in zip(matrix_elems, matrix_comp): template[identified_elems.index(elem)] = conc

            scores = _utils._template_match_1d(patterns = np.transpose(self.Particles.composition / 100),
                                                templates = template / 100,
                                                #num_particles = self.number_of_particles
                                               ).squeeze()

            return scores
            
            #false_positives = ncc_scores.squeeze() > (1 - (threshold / 100))

            #print(f"Number of identified false positives by chemistry: {np.sum(false_positives)}")

            #return false_positives

    def identify_false_particles_by_geometry(self,
                                             geometry = ['Feret diameter H. [um]',
                                                         'Feret diameter V. [um]']):
        """Identify potential false particles by clustering geometric properties defined by 
        geometry dictionary.

        Note that the data will be normalised according to the measurements largest value.

        Parameters
        ----------
        geometry
            dictionary with geometry key argument for the class' Particles attrubute particle_geometry

        Returns
        -------
        false_positives
            Array of false particles.

        Example
        -------
        >>> import particle_analysis as pa
        >>> s = pa.load(csv_filename)
        >>> s.identify_false_particles_by_geometry()
        
        """

        if not hasattr(self, 'Images'): raise ValueError("The class' attribute 'Images' is not yet defined. See *.load_images()")

        common_keys = set(geometry) & set(self.Particles.particle_geometry.keys())
        
        if set(geometry) == common_keys:

            from sklearn.cluster import DBSCAN

            # Normalisation constant
            max_ = max([np.max(self.Particles.particle_geometry[geom]) for geom in geometry])

            # Transposed normalised data
            X = np.asarray([self.Particles.particle_geometry[geom] / max_ for geom in geometry]).T

            clustering = DBSCAN(eps = 0.11, min_samples = 100).fit(X)
            
            if len(np.unique(clustering.labels_)) != 2: print('Multiple clusters were identified. Please adjust the clustering parameters.')
            
            else:
                arg_min = np.argmin([np.sum(clustering.labels_ == lab) for lab in np.unique(clustering.labels_)])

                label = np.unique(clustering.labels_)[arg_min]

                false_positives = clustering.labels_ == np.unique(clustering.labels_)[arg_min]
                
                print(f"Number of identified false positives by geometry: {np.sum(false_positives)}")

                return false_positives

        else: 
            
            uncommon_keys = []

            for key in geometry: 
                
                if key not in self.Particles.particle_geometry.keys(): uncommon_keys.append(key)
            
            raise TypeError(f"The keys {uncommon_keys} are not identified in the classæ Particles attribute particle_geometry- Valid arguments are: {list(self.Particles.particle_geometry.keys())}")

    #def identify_false_particles(self):
    #    """The function searches for false particles by chemistry and geometry"""
    #    print('\033[1m The function uses default values, and is therefore not recommended ...\033[0m')
    #    return self.identify_false_particles_by_chemistry() + self.identify_false_particles_by_geometry()

    def get_artificial_eds_map(self, 
                               bkgr_idx = 0,
                               Erange = 15.0, 
                               steps = 0.02,
                               stitch_signal = True):
        """Create a dummy EDS spectrum containing signals from the elements stored in element_list
        

        Parameters
        ----------
         bkgr_idx 
             Integer representing the background in the particle_map (0 by default)
         Erange
             float, Value of energy range. (20 keV by default)
         steps
             Energy resolution (0.02 keV by default)
         stitch_signal
             Whether to stitch the signal into a EDS map (True by default)
         Returns
         -------
         signal : exspy.signals.EDSSEMSpectrum
             Dummy EDS-SEM signal
         """

        if not hasattr(self.Images, 'particle_map'): raise AttributeError("The object doesn't have particle_map defined yet. See identify_particle_regions for instance.")
            
        #warnings.warn("The process is not optimised for large datasets...")
        
        from tqdm import tqdm

        im_ids = np.unique(self.get_particles_image_id(print_warning = False) - 1)
        
        regridify = False

        LINES = []

        if self.Images.is_gridified: 

            nav_shape = self.Images.navigation_shape

            regridify = True

            self.degridify_SEM_images()

        # Signal to store the artificial EDS map in
        signal = np.zeros((self.Images.navigation_shape) + self.Images.signal_shape + (int(Erange/steps),), dtype = np.uint8)

        for im in tqdm(im_ids):

            #labels = np.unique(self.Images.particle_map.data[im])
            #labels = self.Images.unique_particle_labels
            
            #if 0 in labels: labels = np.delete(labels, np.where(labels == 0))
            
            #labels -= 1 # Labelling commence at 1, but the index position starts at 0

            signal[im], lines = _utils._create_dummy_eds_spectra(labelled_image = self.Images.particle_map.data[im],
                                                                 elements = self.get_identified_elements(),
                                                                 label_concentrations = self.Particles.composition,#[:, labels],
                                                                 bkgr_idx = bkgr_idx,
                                                                 Erange = Erange,
                                                                 steps = steps)

            dissimilar_lines = list(set(LINES).symmetric_difference(set(lines)))

            if len(dissimilar_lines) > 0: 

                for line in dissimilar_lines: LINES.append(line)

        if regridify: 
            
            self.gridify_SEM_images(navigation_shape = nav_shape)

            signal = Signal1D(np.transpose(signal, (1,2,0,3)))

        else:

            signal = Signal1D(np.transpose(signal, (1,2,0)))
        
        # Bin the signal if the file size is too large
        if signal.data.nbytes / 1024**3 > 16:

            print(f'\nThe signal size ({signal.data.nbytes / 1024**3} Gb) is too large to do anything reasonable with... The signal will be downscaled.')
            
            signal = signal.rebin(scale = (1,2,2,1), dtype = "same") 

        if self.Images.is_gridified and stitch_signal: 

            print('Stitching the artificial signal')
            
            signal = Signal1D(_utils._reshape_artificial_eds_map(signal.data, nav_shape))

        signal.set_signal_type('EDS_SEM')

        signal.set_elements(self.get_identified_elements())

        signal.axes_manager[-1].name = 'Energy'
        
        signal.axes_manager[-1].units = 'keV'

        signal.axes_manager[-1].scale = steps

        return signal, LINES
                      
    
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    #&&&&&&&&&&&&&&&&&&&& PARTICLE CLASSIFICATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    
    def unclassify_particles(self, particle_arr):
        """Unclassify the particles defined by 'particles' array argument

        Parameters
        ----------
        particles 
            List of particles to unclassify. Note that the array must be of the
            same shape as the length of the classified particles.
            If particles is set to 'all', all particles will be unclassified.
        """
        ptype = type(particle_arr)

        if ptype == str: 
            
            if particle_arr == 'all': particle_arr = np.ones((self.number_of_particles), bool)
        
        if self._check_array_length(particle_arr): 
            
            # Set the particles of interest to not classified:
            self.is_classified[particle_arr] = False
            
            # Allocate corr. classification name 
            self.Particles.classes[particle_arr] = "Unclassified"

            if hasattr(self.Images):
                print("Resetting phase maps.")
                self.Images._phase_maps = {}
            
        else: print(f"Provided particle array of shape {np.shape(particle_arr)} do not match the total number of particles")
            
    def classify_particles(self, particle_arr, class_name):
        """Classify particles defined by argument 'particles'
        
        Parameters
        ----------
        particles
            array of particles to classify

        class_name 
            String that the particles will be labelled as
        """
        # Allocate a class name 
        if self._check_array_length(particle_arr):
            
            self.Particles.classes[particle_arr] = class_name
    
            # Update list of classified particles
            self.is_classified[particle_arr] = True

    def classify_noise(self, particle_arr):
        """
        Label specified particles as noise. I.e. particles defined by particle_arr
        is relabelled as Noise
        
        Parameters
        ----------
        particle_arr
            Array of particles to label as noise
        """
        
        # Allocate class name 
        
        if self._check_array_length(particle_arr):
            
            self.Particles.classes[particle_arr] = 'Noise'
    
            # Update list of classified particles
            self.is_classified[particle_arr] = True

    def tabulate_and_save_classified_particles_composition(self, path):
        """Create tables of the classified particles' composition.
        Note that the particle names will be the particles' label name.
        """
        
        path = str(path)

        if not os.path.isdir(directory_path):
            
            create_dir = input(f"The directory '{path}' does not exist. Create it? (y/[n])")

            if create_dir.upper() == 'Y':

                os.mkdir(path)

        classes = self.get_unique_particle_classes()

        for cl in classes:

            cl_arr = self.Particles.classes == cl
            
            utils.save_particles_property(
                self.Particles.composition[:,cl_arr],
                np.array(self.metadata.Particles.label_name)[cl_arr],
                self.get_identified_elements(),
                path = path,
                filename = f"{cl}_{self.metadata.chemical_unit}.txt"
            )

        


            

        