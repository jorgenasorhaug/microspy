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
import warnings, os
from pathlib import Path

from hyperspy.signals import Signal1D, Signal2D
from hyperspy.misc import utils
from hyperspy.utils.markers import Texts

from microspy.misc import exceptions 
from .utils import _image_utils

from microspy.misc.material import (
    ELEMENTS, 
    weight_to_atomic, 
    atomic_to_weight
)
from .utils.array_tools import (
    _3Darray_2_4Darray,
    _4Darray_2_3Darray
)

ALLOWED_CHEMICAL_UNITS = (
    ('Atomic %', 'atomic %', 'Atomic%', 'atomic%', '[Atomic %]', '[atomic %]', '[Atomic%]', '[atomic%]',
    'At %', 'at %', 'At%', 'at%', '[At %]', '[at %]', '[At%]', '[at%]', 'At.%', 'at.%', '[At.%]','[at.%]'),
    ('Mass %', 'mass %', 'Mass%', 'mass%', '[Mass %]', '[mass %]', '[Mass%]', '[mass%]',
    'Wt %', 'wt %', 'Wt%', 'wt%', 'Wt. %', 'wt. %', 'Wt.%', 'wt.%', 
    '[Wt %]', '[wt %]', '[Wt%]', '[wt%]', '[Wt. %]', '[wt. %]', '[Wt.%]', '[wt.%]')
)


class MicroSpySignal1D(Signal1D):
    """General microspy class extending HyperSpy's Signal1D class 
    with some methods for carrying over custom properties and others 
    for manipulating units and plotting with marker.

    This class extends HyperSpy's Signal1D class. Some of the 
    docstrings are obtained from HyperSpy. See the docstring of
    :class:`~hyperspy._signals.signal1d.Signal1D` for the list of
    inherited attributes and methods.
    
    Not meant to be used directly.

    Parameters
    ----------
    *args
        See :class:`~hyperspy._signals.signal1d.Signal1D`.
    **kwargs 
        See :class:`~hyperspy._signals.signal1d.Signal1D`.
        
        Can also take the keywords
        - "title" : title to be stored in the metadata
        - "units" : either single unit or list of units 
                    (equally many as the number of columns)
        
    Examples
    --------
    >>> arr = np.arange(1,13).reshape(4,3)
    >>> s = MicroSpySignal1D(
            arr,
            **{
                "title" : "Test signal",
                "units" : "a",
                "props" : ["1","2","3"]
            }
        )
    >>> s
    <MicroSpySignal1D, title: Test signal, dimensions: (4|3)>
    >>> print(s.metadata)
    ├── General
    │   └── title = Test Signal
    └── Signal
        ├── props = ['1', '2', 3]
        ├── signal_type = None
        └── units = a
    """ 
    def __init__(self, *args, **kwargs) -> None:
        # Call the super constructor
        super().__init__(*args, **kwargs)
        
        # Set metadata
        if kwargs.get("metadata") is not None:
            self.metadata.add_dictionary(kwargs.get("metadata"))
        else:
            self.metadata.General.set_item("title" , value = kwargs.get("title"))
            self.metadata.Signal.set_item("units", value = kwargs.get("units"))
            self.metadata.Signal.set_item("props", value = kwargs.get("props"))
            self.metadata.Signal.set_item(
                "signal_type", value = kwargs.get("signal_type")
            )

    # ---------------------------------------------------------------- #
    # ---------------------- Custom attributes ----------------------- #
    # ---------------------------------------------------------------- #

    @property 
    def unit(self) -> str:
        """Return the signal's unit(s)"""
        if not hasattr(self.metadata.Signal, 'units'):
            raise AttributeError("The signal unit is not set.")
        return self.metadata.get_item("Signal.units")

    @property
    def prop(self) -> list:
        """Return the signal's properties"""
        return self.metadata.get_item("Signal.props")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the data shape: (# particles, # properties).
        """
        return self.data.shape

    # ---------------------------------------------------------------- #
    # ----------------------- Private methods ------------------------ #
    # ---------------------------------------------------------------- #
    
    def _remove_nans(self):
        """Replace nan with zeros"""
        self.data[np.isnan(self.data)] = 0

    def _create_markers(
        self, 
        **kwargs
    ) -> Texts:
        """Create markers from the property prop (in metadata)
        to overlay when plotting the signal.

        """
        offsets = np.zeros((self.data.shape[1],2))
        offsets[:,1] = 1
        offsets[:,0] = np.arange(
            self.data.shape[1]
        ).reshape(
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
            color = kwargs.get('colour'),
            shift = -0.01, # along y-axis
        )
        
    def _check_for_empty_data_columns(self):
        """Check if there exists any empty data columns. If found, the 
        function will ask the user whether to remove the data.
        """
        
        remove_columns = False
        empty_columns = self.data.sum(axis = 0) == 0.0
        
        if empty_columns.any():
            from tabulate import tabulate
            
            _props = np.asarray(
                self.metadata.get_item("Signal.props")
            )
            props = _props[empty_columns].reshape((1,empty_columns.sum()))
            
            print("Empty columns have been identified. The headers are:")
            print(
                tabulate(
                    tabular_data = props,
                    tablefmt = "heavy_grid"
                )
            )
            
            ans = input("Remove the empty columns? (y/[n])")
            if ans.lower() in ("yes", "y"):
                exceptions.formatted_warning(
                    f"Removing properties '{props[0]}' from the data"
                )
                shape = self.data.shape
                self.data = self.data[
                    np.tile(
                        A = ~empty_columns.reshape(shape[1],1),
                        reps = shape[0]
                    ).T
                ].reshape((shape[0], ~empty_columns.sum()))
                
                self.metadata.set_item(
                    item_path = "Signal.props", 
                    value = [str(x) for x in _props[~empty_columns]]
                )
                
            #Reinitialise the class:
            kwargs = {
                #"title" : self.metadata.get_item("General.title"),
                #"signal_type" : self.metadata.get_item("Signal.signal_type"),
                #"units" : self.metadata.get_item("Signal.units"),
                #"props" : self.metadata.get_item("Signal.props")
                "metadata" : self.metadata.as_dictionary()
            }
            self.__init__(self.data, **kwargs)

    # ---------------------------------------------------------------- #
    # ------------------------- Open methods ------------------------- #
    # ---------------------------------------------------------------- #
    
    def change_dtype(
        self, 
        dtype,
    ) -> None:
        """Change the signal's data type.
        
        Parameters
        ----------
        dtype
            data type to be changed to
        
        Note
        ----
        The function is needed to avoid the signal being set to 
        hyperspy Signal1D.
        """
        
        self.data = self.data.astype(dtype, copy=False)
        
    def set_units(
        self, 
        units : list | str
    ) -> None:
        """Set the signal unit(s)
        
        Parameters
        ----------
        units
            string for all properties, or list of strings 
            for each property.
        """
        
        if isinstance(units, list | tuple):
            expected = self.data.shape[-1]
            if expected != len(units):
                raise exceptions.InputError(
                    f"Number of units {len(units)} \u2260 {expected}."
                    )
        elif not isinstance(units, str):            
            raise exceptions.InputError(
                f"units of type {type(units)} is not supported."
                )
        self.metadata.Signal.units = units
    
    def threshold_data(
        self,
        threshold : int | float | list[float | int, ...] | tuple[float | int, ...] = 0.0
    ) -> None:
        """Set a lower threshold for all the data. If the threshold 
        argument is a list, tuple or an ndrarray, the threshold will be set 
        to all data according to property.
        
        Parameters
        ----------
        threshold
            threshold value.
        """
        
        if isinstance(threshold, int | float):
            if int(threshold) <= 0:
                exceptions.formatted_warning(
                    f"Data lower than {threshold!r} will be set to 0.0."
                )
        
        # Threshold all data equally:
        if isinstance(threshold, int | float):
            self.data[self.data < threshold] = 0.0
            
        # Threshold all data unequally:
        elif isinstance(threshold, list | tuple | np.ndarray):
            props_shape = self.axes_manager.signal_shape
            shape = np.shape(threshold)
            
            if shape == props_shape:
                repeats = self.axes_manager.navigation_shape
                thresholds_tile = np.tile(
                    A = threshold, 
                    reps = repeats + (1,)
                )
            else:
                thresholds_tile = threshold
            
            if thresholds_tile.shape != self.data.shape:
                raise exceptions.InputError(
                        f"Threshold shape '{thresholds_tile.shape}' is not "
                        f"compatible with the data shape '{self.data.shape}'."
                    )
                
            self.data[self.data < thresholds_tile] = 0.0
    
    def plot_with_markers(
        self,
        shift_along_x : float = 0.10,
        colour = 'k',
        markers : list | tuple | None = None
    ) -> None:
        """Plot the signal with overlayed markers.
        
        Parameters
        ----------
        shift_along_x
            Text shift along x-axis
        colour 
            colour of the text markers
        markers
            list or markers or None. If None, the object's 
            props will be used.
        
        Note
        ----
        For some reason, the metadata structures fails 
        after plotting the signal, so a deepcopy signal 
        is made instead.
        """
        
        if markers is None:
            if not isinstance(self.prop, list | tuple):
                raise AttributeError(
                    f"Can not overlay plot with data type {type(self.prop)}"
                )
            if not isinstance(self.prop[0], str):
                raise AttributeError(
                    f"Markers should be strings, {type(self.prop[0])} is not supported."
                )
            markers = self._create_markers(
                **{'shift_along_x' : shift_along_x,
                  'colour' : colour}
            ) 
        else:
            if len(markers) != self.data.shape[1]:
                raise exceptions.ShapeError(
                    f"Number of markers ({len(markers)}) must be {self.data.shape[1]}."
                )
            
        self.plot(autoscale = 'x')
        self.add_marker(markers)

class MicroSpySignal1D_Chemistry(MicroSpySignal1D):
    """General class for tracking particles' chemistry using 
    microspy.
    
    This class extends HyperSpy's Signal1D class. Some of the 
    docstrings are obtained from HyperSpy. See the docstring of
    :class:`~hyperspy._signals.signal1d.Signal1D` and :class:
    `~microspy.signals.MicroSpySignal1D` for the list of
    inherited attributes and methods.
    
    Not meant to be used directly.

    Examples
    --------
    >>> arr = np.arange(1,13).reshape(4,3)
    >>> s = MicroSpySignal1D_Chemistry(
            arr,
            **{
                "title" : "Test signal",
                "units" : "a",
                "props" : ["1","2","3"]
            }
        )
        
    >>> s
    <MicroSpySignal1D, title: Test signal, dimensions: (4|3)>
    >>> print(s.metadata)
    ├── General
    │   └── title = Test signal
    └── Signal
        ├── props = ['1', '2', 3]
        ├── signal_type = Chemistry
        └── units = a
    """  
    def __init__(
        self, 
        *args, 
        **kwargs
    ) -> None:
        
        kwargs.update({"signal_type" : "Chemistry"})
        
        super().__init__(*args, **kwargs)
        self._remove_nans() 
        self._update_concentration()  
        
    # ---------------------------------------------------------------- #
    # ---------------------- Custom attributes ----------------------- #
    # ---------------------------------------------------------------- #
    
    @property
    def elements(self) -> list[str, ...]:
        """Return the stored elements"""
        return self.metadata.get_item("Signal.props")

    @property
    def matrix_elements(
        self
    ) -> list[str, ...]:
        """Return the stored matrix elemenets.
        
        See :func:'Images.set_matrix_composition'
        """
        if not hasattr(self.metadata.Signal, "matrix_elements"):
            raise AttributeError("The matrix composition has not been set.")
        return self.metadata.get_item("Signal.matrix_elements")

    @property
    def matrix_composition(
        self
    ) -> list[float, ...]:
        """Return the matrix composition.
        
        See :func:'Images.set_matrix_composition'.
        
        For matrix elements, see :meth:'matrix_elements'
        """
        if not hasattr(self.metadata.Signal, "matrix_composition"):
            raise AttributeError("The matrix composition has not been set.")
        return self.metadata.get_item("Signal.matrix_composition")

    # ---------------------------------------------------------------- #
    # ----------------------- Private methods ------------------------ #
    # ---------------------------------------------------------------- #
        
    def _identify_single_elements(
        self, 
        return_array = False
    ) -> None | np.ndarray:
        """Identify single element columns.
        """
        arr = np.sum(self.data > 0, axis = 0) == 1
        print(f"Number of single element particles: {np.sum(arr)}")
        return arr

    def _update_concentration(
        self, 
        normalisation_value : int = 100,
    ):
        """Update/normalise the data.
        
        Parameters
        ----------
        normalisation_value
            The normalisation value, 100 (%) by default.
        """
        self.change_dtype(np.float32) 
        total = np.sum(self.data, axis = 1).transpose()
        self.data *= (normalisation_value / total[:, np.newaxis]) 
        #self.data = np.round(self.data, decimals = decimals)
        
    def _check_for_empty_data_columns(self) -> None:
        """Text
        """
        super()._check_for_empty_data_columns()

    # ---------------------------------------------------------------- #
    # ------------------------- Open methods ------------------------- #
    # ---------------------------------------------------------------- #
        
    def set_matrix_composition(
        self, 
        composition : dict | list | tuple,
        unit : str,
        normalisation_value : int | float = 100
    ):
        """Set the matrix composition and elements. 

        Parameters
        ----------
        matrix_composition
            Chemical composition of the matrix with the elements 
            as keys if given as a dictionary.
            If the composition is given as a list/tuple, the 
            number of values must match the number of elements.
        unit
            Unit of the matrix composition. If there is mismatch 
            between the matrix and the stored unit, the matrix unit 
            will be updated.

        Examples
        --------
        >>> import microspy as ms
        >>> s = ms.load(filename)
        >>> s.print_identified_elements()
        ['C','O','Al','Fe']
        
        >>> s.set_matric_composition(
                matrix_composition = [0.2, 0.5, 95, 4.3] 
                # Equivalent to {'C' : 0.2, 'O' : 0.5, ... 'Fe' : 4.3}
            )
        >>> s.get_matrix_composition()
        [0.2, 0.5, 95, 4.3]
        """
        from copy import deepcopy
        
        current_unit = self.unit
        elements = self.elements
        _composition = deepcopy(composition)
        
        if type(current_unit) == type(None):
            raise AttributeError(
                    "The chemical unit has not been set. See "
                    ":func:'set_units'"
                )
        
        if not isinstance(_composition, dict):
            if len(_composition) != len(self.elements):
                raise exceptions.ShapeError(
                        f"The matrix composition shape ({len(_composition)}) "
                        "must match the number of element ("
                        f"{len(self.elements)})."
                    )
            # Set the matrix composition as a dict:
            _composition = dict(zip(elements, composition))
        
        # Check for unrecognised elements:
        diff = set(_composition) - set(ELEMENTS)
        if len(diff) > 0: 
            raise ValueError(f"Element(s) {diff} is not recognised.")
        
        # Check for negative values:
        m_comp = np.asarray(list(_composition.values()), float)
        if np.sum(m_comp < 0) > 0: 
            raise ValueError("Negative values are not supported.")

        # Check composition:
        m_sum = m_comp.sum()
        if m_sum < 0.0 or m_sum > normalisation_value + 0.0001: 
            raise ValueError(
                    f"The provided composition is not valid: "
                    f"sum({m_comp}) \u2260 {m_sum}"
                ) 
            
        elif m_sum != normalisation_value: 
            # Normalise it:
            exceptions.formatted_warning(
                f"Normalising the provided matrix composition "
                f"from a total of {m_sum}% to 100%.\n"
            )
            m_comp *= (normalisation_value / m_sum)

        # Matrix elements vs. particles' elements
        diff2 = set(elements) - set(_composition)
        if len(diff2) > 0:
            exceptions.formatted_warning(
                f"Signal elements {diff2} are not part of the matrix "
                "composition."
            )

        # Set the elements
        self.metadata.Signal.matrix_elements = list(set(_composition))

        # Change matrix' chemical unit if necessary:
        p_unit = {
            (self.unit in ALLOWED_CHEMICAL_UNITS[0]) : 0,
            (self.unit in ALLOWED_CHEMICAL_UNITS[1]) : 1
        }[True]
        m_unit = {
            (unit in ALLOWED_CHEMICAL_UNITS[0]) : 0,
            (unit in ALLOWED_CHEMICAL_UNITS[1]) : 1
        }[True]

        # Unit status:
        status = (p_unit, m_unit)
        if p_unit != m_unit:
            if status == (0,1): 
                print("Changing matrix composition from weight to atomic %.")
                m_comp = weight_to_atomic(
                    m_comp, 
                    self.metadata.get_item("Signal.matrix_elements")
                )

            else: 
                print("Changing matrix composition from atomic to weight %.")
                m_comp = atomic_to_weight(
                    m_comp, 
                    self.metadata.get_item("Signal.matrix_elements")
                )

        # Save the matrix composition
        self.metadata.set_item(
            item_path = "Signal.matrix_composition", 
            value = m_comp
        )

    def change_unit(
        self, 
        **kwargs
    ) -> None:
        """Change the chemical unit of the signal and matrix.

        kwargs can be used to specify the unit, see Examples.

        Examples
        --------
        # Read compositions
        >>> s = pd.read_csv(filename)
        >>> cr = np.expand_dims(np.asarray(s['Cr [Mass%]']), axis = 1) # [wt.%]
        >>> cr
        array([[  nan],
        [0.291],
        ...
        [  nan]])
        
        >>> cr.shape
        (131,1)
        
        >>> # Add other elements
        >>> cu = np.expand_dims(np.asarray(s['Cu [Mass%]']), axis = 1)
        >>> zn = np.expand_dims(np.asarray(s['Zn [Mass%]']), axis = 1)
        >>> arr = np.concatenate([cr,cu, zn], axis = 1) 
        
        >>> # Set the signal
        >>> s = MicroSpySignal1D_Chemistry(arr)
        
        >>> # Change unit from [Mass%] to [at.%] (default string along with 
        >>> # [wt.%])
        >>> s.change_unit() #
        >>> s.unit
        '[at.%]'
        
        >>> # Alternatively set the unit: 
        >>> s.change_unit(**{'unit' : 'wt.%'})
        >>> s.unit
        'wt.%'
        """
        
        current_unit = self.unit

        if current_unit is None:
            raise AttributeError(
                    "The chemical unit has not been set. See *.set_units()"
                )

        # Current unit position in ALLOWED_CHEMICAL_UNITS
        _unit = {
            (current_unit in ALLOWED_CHEMICAL_UNITS[0]) : 0,
            (current_unit in ALLOWED_CHEMICAL_UNITS[1]) : 1
        }[True]

        unit_changer = (atomic_to_weight, weight_to_atomic)

        new_unit = ['[wt.%]', '[at.%]']

        kwarg_unit = kwargs.get('unit')

        if kwarg_unit is not None:
            # Identify the unit to be changed to in ALLOWED_CHEMICAL_UNITS:
            _ = {
                (kwarg_unit in ALLOWED_CHEMICAL_UNITS[0]) : 0,
                (kwarg_unit in ALLOWED_CHEMICAL_UNITS[1]) : 1
            }[False]
            
            if _unit != _ or (kwarg_unit == current_unit): 
                # Same unit, but we allow changing the format:
                exceptions.formatted_warning(
                        f"The keyword {kwarg_unit} is the same as the "
                        f"existing unit: {self.unit}."
                    )
                # Set the unit
                self.set_units(kwarg_unit)
                return
            
            new_unit[_] = kwarg_unit

        # Convert the data
        self.data = np.transpose(
            unit_changer[_unit](
                np.transpose(self.data), 
                self.elements
            ) 
        )

        # Update the matrix composition too
        if hasattr(self.metadata.Signal, 'matrix_composition'): 
            self.metadata.set_item(
                item_path = "Signal.matrix_composition", 
                value = unit_changer[_unit](
                    self.metadata.get_item("Signal.matrix_composition"),
                    self.metadata.get_item("Signal.matrix_elements")
                )
            )
        # Set unit
        self.set_units(new_unit[_unit])
        
    def get_particles_with_elements(
        self,
        elements : list | tuple | str,
        mode : str = "only",
    ) -> np.ndarray:
        """Get particles that match the provided conditions, like particles 
        containing a specific element combination, or particles containing
        specified elements (and possibly others).
        
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
        
        if type(elements) not in (list, tuple): 
            if isinstance(elements, str): 
                elements = [elements]
            else:
                raise TypeError(
                        f"{elements} is unexpected. Provide a list/tuple "
                        "of elements (strings), or a single element (string)"
                    )
        else:
            elements = list(elements)
            
        if mode not in ("all", "only"):
            raise exceptions.InputError(
                    f"mode '{mode}' is not recognised."
                )
        
        # Check if the provided elements exists:
        diff = set(elements) - set(ELEMENTS)
        if len(diff) > 0:
            raise exceptions.InputError(
                    f"Element(s) '{list(diff)}' does/do not exist."
                )
        
        # Check if the provided elements have been detected:
        particle_elements = list(self.elements)
        num_elements = len(particle_elements)
        diff = set(elements) - set(particle_elements)
        if len(diff) > 0:
            exceptions.formatted_warning(
                    f"Element(s) '{list(diff)}' has/have not been detected "
                    "and will be removed from the list."
                )
            for elem in diff: elements.remove(elem)

        # Make an element template:
        template = np.zeros(shape = (num_elements,), dtype = bool)
        for elem in elements: 
            template[particle_elements.index(elem)] = True

        if mode == "only":
            return ((self.data > 0.0) == template).all(axis = 1)
        else:
            return (self.data > 0.0).T[np.where(template)].T.all(axis=1)
            
    def threshold_data(
        self,
        threshold : int | float | list[float | int, ...] | tuple[float | int, ...] = 0.0,
        elements : list | tuple | None = None,
        **kwargs
    ) -> None:
        """
        
        Unless the elements are stated in the 'elements' argument, this 
        function uses the inherited function of same name from 
        :class:'MicroSpySignal1D' to threshold data, but updates also the 
        element concentrations afterwards.
        
        Parameters
        ----------
        threshold
            Lower threshold values.
        elements
            Specify elements to threshold.
        kwargs
            keyword arguments passed on to :func:'_update_concentration'
            if elements is None.
        """
        from copy import deepcopy
        threshold = deepcopy(threshold)
        
        # Specified elements 
        if elements is not None:
            
            num_elements = len(elements)
            particle_elements = self.metadata.get_item("Signal.props")
            num_particle_elements = len(particle_elements)
            
            # One threshold per specified element
            if isinstance(threshold, list | tuple | np.ndarray):
                if 0 < len(threshold) < num_particle_elements:
                    if len(threshold) != num_elements:
                        raise exceptions.InputError(
                                "The number of threhold values "
                                f"('{len(threshold)}') must match the number "
                                f"of elements ('{num_elements}')."
                            )
                else:
                    raise exceptions.InputError(
                            f"The threshold argument ('{len(threshold)}') "
                            "cannot exceed the element shape "
                            f"({num_elements})."
                        )
            
            if particle_elements is None:
                raise AttributeError(
                        "The class has no metadata about the detected "
                        "elements to make a template for the thresholding."
                    )
            
            # Make an element template:
            template = np.zeros(
                shape = (num_particle_elements,), 
                dtype = bool
            )
            for elem in elements: 
                index = particle_elements.index(elem)
                template[index] = True
                
            # Set the thresholds:
            _threshold = np.zeros_like(template, dtype = float)
            _threshold[template] = threshold
            
            # Raise a warning if a zero-valued threshold is detected:
            if (_threshold[template] == 0.0).any():
                from tabulate import tabulate
                exceptions.formatted_warning(
                    "The threshold value(s) for "
                    f"'{np.asarray(elements)[np.asarray(threshold) == 0]}' "
                    "is '0'. The minimum concentration will not change. "
                )
        else:
            _threshold = threshold
            
        # Set thresholds:
        super().threshold_data(
            threshold = _threshold
        )
        # Update the chemical compositions.
        self._update_concentration(**kwargs)
        self._check_for_empty_data_columns()
        

class MicroSpySignal1D_Geometry(MicroSpySignal1D):
    """General class for tracking particles' geometry using 
    microspy.

    This class extends HyperSpy's Signal1D class. Some of the 
    docstrings are obtained from HyperSpy. See the docstring of
    :class:`~hyperspy._signals.signal1d.Signal1D` and :class:
    `~microspy.signals.MicroSpySignal1D` for the list of
    inherited attributes and methods.
    
    Not meant to be used directly.

    Examples
    --------
    >>> s = np.arange(1,10).reshape(3,3)
    >>> s = MicroSpySignal1D(s)
    >>> s
    <MicroSpySignal1D_Geometry, title: , dimensions: (3|3)>
    """  

    def __init__(self, *args, **kwargs) -> None:
        kwargs.update({"signal_type" : "Geometry"})
        super().__init__(*args, **kwargs)

