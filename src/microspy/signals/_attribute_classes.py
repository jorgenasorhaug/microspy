import numpy as np
from src.microspy.io import _read_metadata, _io

class General:
    """metadata sub-class storing general information s.a. acquisition date and 
    project name"""
    def __init__(self, arg):
        
        self.acquisition_date = read_metadata._read_acquisition_date(arg)
        
        self.project_name = read_metadata._read_project_name(arg)

    # When .metadata is called, print its instance arguments: (DOESN*T WORK YET)
    def __repr__(self): 

        print_string = "General\n"

        strings = []
        
        for key in self.__dict__.keys(): 

            strings.append(key)

        strings = np.sort(strings)

        for string, counter in zip(strings, np.arange(len(strings))):

            if counter != len(strings)-1: print_string += f" ├── {string}\n"

            else: print_string += f" └── {string}"
            
        return print_string

class Acquisition_settings:
    """metadata sub-class storing information s.a. the acquisition time, analysed area, 
    analysed views and used magnification from particle analysis"""
    def __init__(self, arg):
        # Acquisition time
        self.acquisition_time = read_metadata._get_acquisition_time(arg)
        # Analysed area [um2]
        self.analysed_area = read_metadata._read_analysed_area(arg)
        # Number of views
        self.analysed_views = read_metadata._get_analysed_views(arg)
        # Used magnification
        self.magnification = read_metadata._get_magnification(arg)
        # SEM image shape
        self.signal_shape = () # Shape of SEM images
        # Stage grid shape
        self.navigation_shape = ()

    # When .metadata is called, print its instance arguments: (DOESN*T WORK YET)
    def __repr__(self): 

        print_string = "Acquisition\n"

        strings = []
        
        for key in self.__dict__.keys(): 

            strings.append(key)

        strings = np.sort(strings)

        for string, counter in zip(strings, np.arange(len(strings))):

            if counter != len(strings)-1: print_string += f" ├── {string}\n"

            else: print_string += f" └── {string}"

        return print_string

class Particle_labels:
    """metadata sub-class storing information s.a. the label names from the data acquisition. This information can be used to read the correct order of particle images."""
    def __init__(self, arg):
        
        self.label_name = read_metadata._get_particle_label_names(arg)

    # When .metadata is called, print its instance arguments: (DOESN*T WORK YET)
    def __repr__(self): 

        print_string = "Particles\n"

        strings = []
        
        for key in self.__dict__.keys(): 

            # Ignore hidden attributes
            if key[0] != '_': strings.append(key)

        strings = np.sort(strings)

        for string, counter in zip(strings, np.arange(len(strings))):

            if counter != len(strings)-1: print_string += f" ├── {string}\n"

            else: print_string += f" └── {string}"

        return print_string

class metadata:
    """Metadata class"""
    def __init__(self, arg):
        #for metadata in _io.get_module_function_names(read_metadata):
        self.General = General(arg)
        self.Acquisition_settings = Acquisition_settings(arg)
        self.Particles = Particle_labels(arg)
        self.chemical_unit = '[Mass %]' # Default
        self.navigation_scale = 1
        self.navigation_unit = ''

    # When .metadata is called, print its instance arguments: (DOESN'T WORK YET)
    def __repr__(self): 

        print_string = "metadata\n"

        strings = []
        
        for key in self.__dict__.keys(): 

            strings.append(key)               

        strings = np.sort(strings)

        for string, counter in zip(strings, np.arange(len(strings))):

            if counter != len(strings)-1: print_string += f" ├── {string}\n"

            else: print_string += f" └── {string}"
        
        return print_string

class Particles:
    """Particles class that keeps track of elements, chemical composition and unit, 
    particles' classification and their geometries. 

    Parameters
    arg
        *.csv filename (string) or pandas DataFrame object (from *.csv) from
        particle analysis.

    See also
    --------
    particle_analysis.ParticleAnalysis
    
    Examples
    --------
    Load one of the example datasets and inspect some properties

    >>> import particle_analysis as pa
    >>> s = pa.data.test_data()
    >>> s
    <PA, title: 250327 - dark particles, dimensions: (21 | 548)>
    >>> s.metadata
    metadata
     |--Acquisition
     |--General
     |--Particles
     |--unit

     >>> s.Particles.composition.shape()
     (9, 1500) # (9 identified elements, 15000 particle)
    """
    
    def __init__(self, arg):
        # Get identified elements
        self.elements = _io.get_elements(arg)
        
        # By default, the unit in the csv file should be Mass
        self.chemical_unit = _io.get_unit(arg)

        # Array of element concentrations (per element)
        self.composition = _io.get_particles_composition(arg, self.chemical_unit)

        # Classes stored in file
        self.classes = _io.get_classes(arg)

        self.particle_geometry = _io._get_particles_geometry(arg)

    # When .metadata is called, print its instance arguments: (DOESN*T WORK YET)
    def __repr__(self): 

        print_string = "Particles\n"

        strings = []
        
        for key in self.__dict__.keys(): 

            # Ignore hidden attributes
            if key[0] != '_': strings.append(key)

        strings = np.sort(strings)

        for string, counter in zip(strings, np.arange(len(strings))):

            if counter != len(strings)-1: print_string += f" ├── {string}\n"

            else: print_string += f" └── {string}"

        return print_string


        