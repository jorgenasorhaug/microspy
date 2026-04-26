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
import warnings

from hyperspy.signals import Signal1D, Signal2D
from hyperspy.misc import utils
from hyperspy.utils.markers import Texts
from src.microspy._misc import exceptions 

from src.microspy._misc.material import (
    elements, 
    weight_to_atomic, 
    atomic_to_weight
)
from src.microspy.signals.utils.array_tools import (
    _3Darray_2_4Darray,
    _4Darray_2_3Darray
)

ELEMENTS = list(elements.keys())[1:]

ALLOWED_CHEMICAL_UNITS = [
    ['Atomic %', 'atomic %', 'Atomic%', 'atomic%', '[Atomic %]', '[atomic %]', '[Atomic%]', '[atomic%]',
    'At %', 'at %', 'At%', 'at%', '[At %]', '[at %]', '[At%]', '[at%]', 'At.%', 'at.%', '[At.%]','[at.%]'],
    ['Mass %', 'mass %', 'Mass%', 'mass%', '[Mass %]', '[mass %]', '[Mass%]', '[mass%]',
    'Wt %', 'wt %', 'Wt%', 'wt%', 'Wt. %', 'wt. %', 'Wt.%', 'wt.%', 
    '[Wt %]', '[wt %]', '[Wt%]', '[wt%]', '[Wt. %]', '[wt. %]', '[Wt.%]', '[wt.%]']
]

####################################################
################# PARENT CLASSES ###################
####################################################

class MicroSpySignal1D(Signal1D):
    """Class for tracking particles' chemistry using microspy, 
    extending HyperSpy's Signal1D class with some methods for 
    carrying over custom properties and others for manipulating 
    composition.

    Not meant to be used directly.

    Example
    -------
    >>> s = np.arange(1,10).reshape(3,3)
    >>> s = MicroSpySignal1D(s)
    >>> s
    <MicroSpySignal1D, title: Particle analysis, dimensions: (3|3)>
    """ 
    def __init__(self, *args, **kwargs) -> None:
        # Call the super constructor
        super().__init__(*args, **kwargs)
        self.metadata.General.title = 'Particle analysis'
        self.metadata.Signal.units = kwargs.get('units')

    @property 
    def unit(self) -> str:
        """Return the stored unit"""
        if not hasattr(self.metadata.Signal, 'units'):
            raise AttributeError("The signal unit is not set.")
        return self.metadata.Signal.units

    @property
    def prop(self) -> list:
        return self.metadata.Signal.props

    def set_unit(self, units : list | str) -> None:
        """Set the signal unit"""
        self.metadata.Signal.units = units

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the data shape: (num. particles | number of [properties])"""
        return self.data.shape

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%% PRIVATE FUNCTIONS %%%%%%%%%%%%%%%%%%%

    def _create_markers(self, **kwargs):
        """Create markers (chemical elements) to overlay when plotting 
        the particles' properties as a 1D signal.

        Parameters
        ----------
        shift_along_x
            float value controlling the shift of the markers along the 
            x-axis
        color
            color of the markers
        """
        offsets = np.zeros((self.data.shape[1],2))
        offsets[:,1] = 1
        offsets[:,0] = np.arange(self.data.shape[1]).reshape(
            self.data.shape[1]
        ) - kwargs.get('shift_along_x')
        offsets[0,0] += 0.25
        offsets[-1,0] -= 0.25
        
        return Texts(
            offsets = offsets,
            texts =  self.prop,
            sizes = 6, 
            offset_transform = 'relative',
            horizontalalignment="left",
            verticalalignment="top",
            color = kwargs.get('color'),
            shift = -0.01, # along y-axis
        )

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%% OPEN FUNCTIONS %%%%%%%%%%%%%%%%%%%%%

    def plot_with_markers(self,
                         shift_along_x : float = 0.10,
                         color = 'k'):
        """Plot the signal with markers. For some reason, 
        the metadata structures fails after plotting the 
        signal, so a deepcopy signal is instead made.
        
        
        """
        
        markers = self._create_markers(
            **{'shift_along_x' : shift_along_x,
              'color' : color}
        ) 
        self.plot(autoscale = 'x') 
        self.add_marker(markers)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%% SUB-CLASSES %%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class MicroSpySignal1D_Chemistry(MicroSpySignal1D):
    """General class for tracking particles' chemistry 
    using microspy, extending HyperSpy's Signal1D class.

    Not meant to be used directly.

    Example
    -------
    >>> s = np.arange(1,10).reshape(3,3)
    >>> s = MicroSpySignal1D(s)
    >>> s
    <MicroSpySignal1D_Chemistry, title: Particle analysis, dimensions: (3|3)>
    """  
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        elements = kwargs.get('props')
        self.metadata.Signal.signal_type = 'Chemistry'
        self.metadata.Signal.props = elements

        self._remove_nans() # Remove nan values
        self._update_concentration() # Set total composition to 100 

    @property
    def elements(self) -> list[str, ...]:
        """Return the stored elements"""
        return self.metadata.Signal.prop

    @property
    def matrix_elements(self) -> list[str, ...]:
        if not hasattr(self.metadata.Signal, "matrix_elements"):
            raise AttributeError("The matrix composition has not been set.")
        return self.metadata.Signal.matrix_elements

    @property
    def matrix_composition(self) -> list[float, ...]:
        if not hasattr(self.metadata.Signal, "matrix_composition"):
            raise AttributeError("The matrix composition has not been set.")
        return self.metadata.Signal.matrix_composition

    ##############################################
    #%%%%%%%%%%%% Private functions %%%%%%%%%%%%%%
        
    def _identify_single_element_particles(self, return_array = False):
        """Identify rows/cols? that only contain single elements"""
        arr = np.sum(self.data > 0, axis = 0) == 1
        
        print(f"Number of single element particles: {np.sum(arr)}")
        
        if return_array: return arr
    
    def _remove_nans(self):
        """Replace nan values with zeros"""
        self.data[np.isnan(self.data)] = 0

    def _update_concentration(self, decimals = 2):
        """Update/normalise the chemistry (to 100%)"""

        total = np.sum(self.data, axis = 1).transpose()

        self.data *= (100 / total[:, np.newaxis]) #percentage

        self.data = np.round(self.data, decimals = decimals)

    ##############################################
    #%%%%%%%%%%%% Open functions %%%%%%%%%%%%%%
        
    def set_matrix_composition(self, 
                               composition : dict,
                               unit : str):
        """Set the matrix chemcial composition for reference. 

        Parameters
        ----------
        matrix_composition
            Chemical composition of the matrix to facilitate identification of false particles. 
            A dictionary with elements as key arguments.
        unit
            Unit of the matrix composition. If there is mismatch between the matrix and the 
            particle's unit, the matrix unit will be updated.

        Example
        -------
        >>> import particle_analysis as pa
        >>> s = pa.load(filename)
        >>> s.print_identified_elements()
        ['C','O','Al','Fe']
        
        >>> s.set_matric_composition(matrix_composition = [0.2, 0.5, 95, 4.3] # Equivalent to {'C' : 0.2, 'O' : 0.5, ... 'Fe' : 4.3}
                                    )
        >>> s.get_matrix_composition()
        [0.2, 0.5, 95, 4.3]
        """

        current_unit = self.unit

        if type(current_unit) == type(None):
            
            raise AttributeError("The chemical unit has not been set. See *.set_unit()")
        
        elements = self.elements
        
        if type(composition) != dict:

            raise TypeError(f"Matrix composition argument type "
                            f"({type(composition)}) is not a valid "
                            "type. Provide a dictionary.")
        
        diff = set(composition) - set(ELEMENTS)

        if len(diff) > 0: 
            
            raise ValueError(f"Element(s) {diff} is not recognised.")
        
        m_comp = np.asarray(list(composition.values()), float)

        if np.sum(m_comp < 0) > 0: 
            
            raise ValueError("Negative values are not supported.")

        m_sum = m_comp.sum()
        
        if m_sum < 0.0 or m_sum > 100.0001: 
            
            raise ValueError(f"The provided composition is not valid: "
                             f"sum({m_comp}) \u2260 {m_sum}") 
            
        elif m_sum != 100.0: 

            warnings.warn(f"\nNormalising the provided matrix composition "
                          "from a total of {m_sum} % to 100%.\n")

            m_comp *= (100 / m_sum)

        diff2 = set(elements) - set(composition)
        
        if len(diff2) > 0:
        
            warnings.warn(f"\n{diff2} are not part of the particles' chemistry.")

        # Save the elements
        self.metadata.Signal.matrix_elements = list(set(composition))

        # Change matrix' unit if necessary
        p_unit = {
            (self.unit in ALLOWED_CHEMICAL_UNITS[0]) : 0,
            (self.unit in ALLOWED_CHEMICAL_UNITS[1]) : 1
        }[True]
        m_unit = {
            (unit in ALLOWED_CHEMICAL_UNITS[0]) : 0,
            (unit in ALLOWED_CHEMICAL_UNITS[1]) : 1
        }[True]

        status = (p_unit, m_unit)
        
        if p_unit != m_unit:

            if status == (0,1): 
                
                print("Changing the matrix composition from weight to atomic.")
                
                m_comp = np.round(weight_to_atomic(m_comp, self.metadata.Signal.matrix_elements), decimals = 2)

            else: 

                print("Changing the matrix composition from atomic to weight.")
                
                m_comp = np.round(atomic_to_weight(m_comp, self.metadata.Signal.matrix_elements), decimals = 2)

        # Save the matrix composition
        self.metadata.Signal.matrix_composition = m_comp

    def change_unit(self, **kwargs):
        """Change the chemical unit and thus the particles' 
        quantified composition.

        To specify a way to display the unit, a kwargs argument 
        can be provided

        Example
        -------
        # Read compositions
        >>> s = pd.read_csv(filename)
        >>> cr = np.expand_dims(np.asarray(s['Cr [Mass%]']), axis = 1) # [wt.%]
        >>> cu = np.expand_dims(np.asarray(s['Cu [Mass%]']), axis = 1)
        >>> zn = np.expand_dims(np.asarray(s['Zn [Mass%]']), axis = 1)
        >>> s = np.concatenate([cr,cu, zn], axis = 1) 
        
        # Set the signal
        >>> s = _microspy_signal.MicroSpySignal1D_Chemistry(s)
        
        # Change unit from [Mass%] to [at.%] (default string along with [wt.%])
        >>> s.change_unit() #
        
        # Alternatively set the unit: 
        >>> s.change_unit(**{'unit' : 'wt.%'})
        """
        
        current_unit = self.unit

        if type(current_unit) == type(None):
            
            raise AttributeError("The chemical unit has not been set. See *.set_unit()")

        # Unit index position in ALLOWED_CHEMICAL_UNITS
        _unit = {
            (current_unit in ALLOWED_CHEMICAL_UNITS[0]) : 0,
            (current_unit in ALLOWED_CHEMICAL_UNITS[1]) : 1
        }[True]

        unit_changer = (atomic_to_weight, weight_to_atomic)

        new_unit = ['[wt.%]', '[at.%]']

        kwarg_unit = kwargs.get('unit')

        if type(kwarg_unit) != type(None):
            _ = {
                (kwarg_unit in ALLOWED_CHEMICAL_UNITS[0]) : 0,
                (kwarg_unit in ALLOWED_CHEMICAL_UNITS[1]) : 1
            }[False]

            if _unit != _: raise AttributeError(f"The keyword {kwarg_unit} is the same as the existing unit: {self.unit}. See *.set_unit()")

            new_unit[_] = kwarg_unit

        # Convert the data
        self.data = np.transpose(
            np.round(
                unit_changer[_unit](
                    np.transpose(self.data), 
                    self.elements), 
                decimals = 2
            )
        )

        # Update the matrix composition too
        if hasattr(self.metadata.Signal, 'matrix_composition'): 
            
            self.metadata.Signal.matrix_composition = np.round(
                unit_changer[_unit](
                    self.metadata.Signal.matrix_composition,
                    self.metadata.Signal.matrix_elements),
                decimals = 2
            )

        # Set the unit
        self.set_unit(new_unit[_unit])

class MicroSpySignal1D_Geometry(MicroSpySignal1D):
    """General class for tracking particles' geometry 
    using microspy, extending HyperSpy's Signal1D class.

    Not meant to be used directly.

    Example
    -------
    >>> s = np.arange(1,10).reshape(3,3)
    >>> s = MicroSpySignal1D(s)
    >>> s
    <MicroSpySignal1D_Geometry, title: Particle analysis, dimensions: (3|3)>
    """  

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metadata.Signal.signal_type = 'Geometry'
        self.metadata.Signal.props = kwargs.get('props')





        

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%% IMAGE CLASSES %%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Images_signal_type = {
    "CompositeSig" : "Overview_image", #Overview/stitched im.
    "ParentSig" : "Acquisition", # Individual images
    "ChildSig" : "Cropped_ROIs"
}      


class MicroSpySignal2D(Signal2D):
    """Class for tracking particle images using microspy, 
    extending HyperSpy's Signal2D class.

    Not meant to be used directly.

    Example
    -------
    >>> s = np.arange(8).reshape(2,2,2)
    >>> s = MicroSpySignal2D(s)
    >>> s
    <MicroSpySignal2D, title: , dimensions: (2|2, 2)>
    """ 
    def __init__(self, *args, **kwargs) -> None:
        # Call the super constructor
        super().__init__(*args, **kwargs)
        self._redefine_metadata()

    def _redefine_metadata(self):
        self.metadata.add_node("Signal")
        self.metadata.add_node("Signal.signal_type")

class MicroSpySignal2D_Parent(MicroSpySignal2D):
    """Parent signal class with attribute functions that
    allows the data to be manipulated as if images are
    sequentually acquired in a grid.
    """
    def __init__(self, *args, **kwargs) -> None:
        # Call the super constructor
        super().__init__(*args, **kwargs)
        self._update_metadata()

    def _update_metadata(self):
        md = self.metadata
        md.General.title = "Acquisition"
        md.Signal.signal_type = "ParentSig"
    
    
    @property
    def is_gridified(self):
        """Check if the signal is gridified, 
        i.e. the data shape is 4D. The function 
        returns False if not.
        """
        return len(self.data.shape) == 4 
        

    def gridify(
        self, 
        grid_shape : tuple(int, int),
        flip_axis : int | list | None = 1
    ):
        """Gridify the images into a 4D grid

        Parameters
        ----------
        grid_shape
            2D grid shape

            Assuming the data shape is (X, W, H),
            the data will be reshaped into grid_shape 
            + (W, H)
        """

        if not self.is_gridified:
            
            N, H, W = np.shape(self.data)

            errtxt = "The grid shape is not compatible with data shape."
            assert N == np.prod(grid_shape), errtxt

            # Gridifying the signal:

            grid = _3Darray_2_4Darray(
                arr = self.data,
                to_shape = grid_shape,
                flip_axis = flip_axis
            )

            # Reset signal
            self.__init__(grid)

        else: print("The signal is already gridified.")

    def degridify(self):
        """Gridify the images into a 4D grid

        Parameters
        ----------
        grid_shape
            2D grid shape

            Assuming the data shape is (X, W, H),
            the data will be reshaped into grid_shape 
            + (W, H)
        """

        if self.is_gridified:

            # Degridifying the signal:
            degrid = _4Darray_2_3Darray(
                arr = self.data,
                flip_axis = flip_axis
            )

            # Reset signal
            self.__init__(degrid)

        else: print("The signal is already degridified.")
        

class Images:
    """A class to keep track of and manipulate acquired
    SEM images. The images are hyperspy 2D signals.
    """
    def __init__(self, images : list | np.ndarray) -> None:

        from src.microspy.io._images import _io

        # Images to MicroSpySignal2D
        images = _io._arrays2signals(images)

        sig_types = []
        
        for im in images:

            sig_types.append(im.metadata.Signal.signal_type)
            
            setattr(self, sig_types[-1], im)

        self.phase_maps = dict()

        self._create_metadata()

        for sig_type in sig_types:

            # Set a brief description of the signal type
            self.metadata.set_item(
                f"Signals.{sig_type}", 
                Images_signal_type.get(sig_type)
            )

        self.metadata.set_item("General.title", "")

    def __repr__(self):
        # General overview:
        dim = self.num_signals
        title = self.metadata.General.title
        print_string = f"<Images class, title: {title}," 
        print_string += f"dimensions: ({dim}|)>:\n"

        # Signal soverview:
        sig_types = self.metadata.Signals.as_dictionary()
        for counter, key in enumerate(sig_types.keys()):
            _sig = vars(self).get(key)
            title = _sig.metadata.General.title
            _dim = _sig.data.shape
            dim = f"({_dim[:-2]}|{_dim[-2:]})".replace(
                "(","").replace(")","")
            sig_string = f"<title: {title}, dimensions: ({dim})" 
            if counter < len(sig_types) - 1:
                print_string += f"├── {key}: {sig_string}\n"
            else:
                print_string += f"└── {key}: {sig_string}\n"
        return print_string

    def _create_metadata(self):
        self._metadata = utils.DictionaryTreeBrowser()
        md = self.metadata
        md.add_node("General")
        md.General.add_node("title")
        md.add_node("Signals")
        self._original_metadata = utils.DictionaryTreeBrowser()
        
    @property
    def is_gridified(self):
        """Check if the Parent signal is gridified
        into a 4D grid or not. The function returns
        True if the array shape is (Y,X,ky,kx) (numpy
        convention) or higher.

        Note:
            If the signal is a series of images taken
            along a row/column, the signal must be grid-
            ified into a 4D signal to be correctly inter-
            preted. 
            Example: A column of images with shape (c,X,Y) 
            is correctly shaped, hence interpreted, if the 
            data is gridified into shape (c,1,X,Y).
        """
        if hasattr(self, "ParentSig"):
            return self.ParentSig.is_gridified
        else: 
            raise AttributeError("The class has no parent signal.")

    def gridify_ParentSig(
        self,
        nav_shape : tuple | list | None = None,
        flip_axes : int | tuple | list = None
    ):
        """Gridify the Parent signal images

        Parameters
        ----------
        nav_shape
            Shape of the navigation grid
        """

        if not hasattr(self, "ParentSig"):

            raise AttributeError("The signal class does not keep "
                                "track of acqiusition images. "
                                "See *.setParentSig().")

        if self.is_gridified:

            print("The signal is already gridified.")

        else:

            self.ParentSig.gridify(
                grid_shape = nav_shape,
                flip_axis = flip_axes
            )

    @property
    def metadata(self):
        """The metadata of the signal."""
        return self._metadata

    @property
    def num_signals(self):
        """Get the total number of image signals"""
        num = 0
        for attr, value in vars(self).items():
            if isinstance(value, MicroSpySignal2D | MicroSpySignal2D_Parent):
                num += 1
        return num

    def setParentSig(
        self,
        current_name : str
    ):
        """NOT TESTED | Set the parent signal, i.e. the instance 
        attribute name will be changed to "ParentSig".

        Parameters
        ----------
        current_signal 
            Name of the current signal to be changed. 
        """
        if hasattr(self, current_name):
            # Get attribute and set correct signal type
            attr = MicroSpySignal2D_Parent(
                getattr(self, current_name)
                )

            # metadata:
            sigVal = list(Images_signal_type.values())[1]
            sigKey = list(Images_signal_type.keys())[1]
            
            attr.metadata.Signal.signal_type = sigKey
            attr.metadata.General.title = sigVal

            md = self._metadata.Signals.as_dictionary()
            del self.metadata.Signals
            del md[current_name]
            md[sigKey] = sigVal
            self._metadata.add_node("Signals")
            self._metadata.Signals.add_dictionary(md)

            # Set new attribute:
            setattr(self, "ParentSig", attr)
            delattr(self, current_name)
        else:
            raise AttributeError("The signal class doesn't have "
                                "an instance attribute named "
                                f"{current_name}, but \n{self.__repr__}")

    def setChildSig(
        self,
        current_name : str
    ):
        """NOT TESTED | Set the child signal, i.e. the instance 
        attribute name will be changed to "ChildSig".

        Parameters
        ----------
        current_signal
            Name of the current signal to be changed. 
        
        Note
        ----
        See setParentSig
        """
        if hasattr(self, current_name):
            # Get attribute and set correct signal type
            attr = MicroSpySignal2D(
                getattr(self, current_name)
                )

            # metadata:
            sigVal = list(Images_signal_type.values())[2]
            sigKey = list(Images_signal_type.keys())[2]
            
            attr.metadata.Signal.signal_type = sigKey
            attr.metadata.General.title = sigVal

            md = self._metadata.Signals.as_dictionary()
            del self.metadata.Signals
            del md[current_name]
            md[sigKey] = sigVal
            self._metadata.add_node("Signals")
            self._metadata.Signals.add_dictionary(md)

            # Set new attribute:
            setattr(self, "ChildSig", attr)
            delattr(self, current_name)
            
        else:
            raise AttributeError("The signal class doesn't have "
                                "an instance attribute named "
                                f"{current_name}, but \n{self.__repr__}")
    