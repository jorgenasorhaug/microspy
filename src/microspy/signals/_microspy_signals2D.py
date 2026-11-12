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

# OBS! DO NOT CHANGE
# signal_type : title 
#Images_signal_type = {
Images_signals = {
    "CompositeSig" : "Overview_image", #Overview/stitched im.
    "ParentSig" : "Acquisition", # Individual images
    "ChildSig" : "Cropped_ROIs", # Cropped child images
    "ChildMap" : "Particle_map", # Map of child images
}


class MicroSpySignal2D(Signal2D):
    """Class for tracking particle images using microspy.
    
    This class extends HyperSpy's Signal2D class for SEM images. Some
    of the docstrings are obtained from HyperSpy. See the docstring of
    :class:`~hyperspy._signals.signal2d.Signal2D` for the list of
    inherited attributes and methods.

    Not meant to be used directly.
    
    Parameters
    ----------
    *args
        See :class:`~hyperspy._signals.signal2d.Signal2D`.
    **kwargs 
        See :class:`~hyperspy._signals.signal2d.Signal2D`.
        
        Can also take the keywords
        - "title" : title to be stored in the metadata
        - "signal_type" : signal type to be stored in the
                          metadata

    Examples
    --------
    >>> arr = np.arange(8).reshape(2,2,2)
    >>> s = MicroSpySignal2D(arr)
    >>> s
    <MicroSpySignal2D, title: , dimensions: (2|2, 2)>
    """ 
    def __init__(self, *args, **kwargs) -> None:
        # Call the super constructor
        super().__init__(*args, **kwargs)
        
        # Set metadata
        if kwargs.get("metadata") is not None:
            self._metadata.add_dictionary(
                kwargs.get("metadata")
            )

    # ---------------------------------------------------------------- #
    # ------------------------- Open methods ------------------------- #
    # ---------------------------------------------------------------- #

    def set_scale(
        self,
        scale : int | float,
        unit : str = "NA"
    ):
        """Set navigation scale.

        Parameters
        ----------
        scale
            scale (unit per pixel)
        unit
            unit
        """
        
        self.axes_manager[-2].scale = scale
        self.axes_manager[-2].units = unit
        self.axes_manager[-1].scale = scale
        self.axes_manager[-1].units = unit

        ndim = np.ndim(self.data)

        if ndim == 3:
            # The images are rectangular -> cannot set a scale along unknown
            # direction.
            parentSig = self.metadata.Signal.signal_type == "ParentSig"
            if (self.axes_manager[-2].size != self.axes_manager[-1].size 
                and parentSig):
                warnings.warn(
                    "Unable to calibrate rectangular and ungridified parent "
                    "signal(s)."
                )
            else:
                cal = scale * self.axes_manager[-1].size
                self.axes_manager[0].scale = cal
                self.axes_manager[0].units = unit
            
        elif ndim == 4:
            cal0 = scale * self.axes_manager[-2].size
            cal1 = scale * self.axes_manager[-1].size
            self.axes_manager[0].scale = cal0
            self.axes_manager[0].units = unit
            self.axes_manager[1].scale = cal1
            self.axes_manager[1].units = unit

        elif ndim > 4:
            warnings.warn("Multidimensional signal. "
                          "Manually set the scale."
                          )
            

class MicroSpySignal2D_Parent(MicroSpySignal2D):
    """Class with attribute methods that allows the data 
    to be manipulated as if images are acquired sequentually
    in a grid.
    
    This class extends HyperSpy's Signal1D class. Some of the 
    docstrings are obtained from HyperSpy. See the docstring of
    :class:`~hyperspy._signals.signal2d.Signal2D` and :class:
    `~microspy.signals.MicroSpySignal2D` for the list of
    inherited attributes and methods.
    
    Not meant to be used directly.

    Examples
    --------
    >>> arr = np.arange(8).reshape(2,2,2)
    >>> s = MicroSpySignal2D_Parent(arr)
    >>> s
    <MicroSpySignal2D_Parent, title: Acquisition, dimensions: (2|2, 2)>
    
    >>> s = MicroSpySignal2D_Parent(
            arr,
            **{"metadata" : {
                    "General" : {"title" : "Title"},
                    "Signal" : {"signal_type" : "test_signal"}
                }
            }
        )
    >>> print(s.metadata)
    ├── General
    │   └── title = Title
    └── Signal
        └── signal_type = test_signal
    """
    def __init__(self, *args, **kwargs) -> None:
        # Update metadata
        default_type = list(Images_signals.keys())[1]
        default_title = Images_signals[default_type]
        
        md = kwargs.get("metadata")
        if md is not None:
            # Set title:
            try:
                title = md.get("General").get("title")
            except AttributeError:
                title = None

            if title is None:
                title = default_title
            
            # Set signal type:
            try:
                signal_type = md.get("Signal").get("signal_type")
            except AttributeError:
                signal_type = None
            if signal_type is None:
                signal_type = default_type
                
            md.update(
                {
                    "General" : {
                        "title" : title
                    },
                    "Signal" : {
                        "signal_type" : signal_type
                    }
                }
            )
        
        # Call the super constructor
        super().__init__(*args, **kwargs)
        
    # ---------------------------------------------------------------- #
    # ----------------------- Custom attributes ---------------------- #
    # ---------------------------------------------------------------- #
    
    @property
    def is_gridified(self):
        """Check if the signal is gridified, i.e. the data dimension is 4. 
        """
        return np.ndim(self.data) > 3 
        
    # ---------------------------------------------------------------- #
    # ------------------------- Open methods ------------------------- #
    # ---------------------------------------------------------------- #

    def gridify(
        self, 
        grid_shape : tuple(int, int),
        flip_axis : int | list | None = None
    ):
        """Gridify the images into a 4D grid.

        Parameters
        ----------
        grid_shape
            2D grid shape.
            Assuming the data shape is (X, W, H), the data 
            will be reshaped into grid_shape + (W, H)
            
            Note that the grid_shape takes (cols, rows) as
            argument.
        flip_axis 
            Depending on the image acquisition order,
            flip the images along specified axis/axes. 
            
        Examples
        --------
        >>> # Jeol takes SEM images from top to bottom, but right 
        >>> # to left:
        >>> arr = np.arange(24).reshape(6,2,2)
        >>> s = MicroSpySignal2D_Parent(arr)
        >>> s
        <MicroSpySignal2D_Parent, title: Acquisition, dimensions: (6|2, 2)>
        
        >>> s.gridify(grid_shape = (3,2), flip_axis = 1)
        >>> s
        <MicroSpySignal2D_Parent, title: Acquisition, dimensions: (2, 3|2, 2)>
        
        >>> s.data.shape
        (3, 2, 2, 2)
        """

        if not self.is_gridified:

            N, H, W = np.shape(self.data)

            errtxt = "The grid shape is not compatible with data shape."
            assert N == np.prod(grid_shape), errtxt
            print(flip_axis)
            # Gridifying the signal:
            grid = _3Darray_2_4Darray(
                arr = self.data,
                to_shape = grid_shape,
                flip_axis = flip_axis
            )
            # Reset signal
            md = self._metadata.as_dictionary()
            self.__init__(
                grid,
                **{"metadata" : md}
            )

        else: print("The signal is already gridified.")

    def degridify(
        self,
        flip_axis : int | list | None = None
    ):
        """Degridify the images into a 3D grid.
        See :func:'Images.gridify'

        Parameters
        ----------
        flip_axis 
            Depending on the image acquisition order,
            flip the images along specified axis/axes. 
        """

        if self.is_gridified:

            # Degridifying the signal:
            degrid = _4Darray_2_3Darray(
                arr = self.data,
                flip_axis = flip_axis
            )

            # Reset signal
            md = self._metadata.as_dictionary()
            self.__init__(
                degrid,
                **{"metadata" : md}
            )
            
        else: print("The signal is already degridified.")

class Images:
    """A class to keep track of and manipulate (SEM) images. 
    Signals are based on :class:`~microspy.signals.MicroSpySignal2D`.
    
    Not meant to be used directly.

    Parameters
    ----------
    images
        List of numpy ndarrays or a single numpy ndarray.
        
    Example 1
    ---------
    >>> Images(np.arange(8).reshape(2,2,2))
    <Images class, title: , dimensions: (1|)>:
    └── ND_Image_shape2x2x2: <title: None, dimensions: (2,|2, 2)
    
    >>> out = Images(np.arange(8).reshape(2,2,2), 
                     np.arange(16).reshape(2,2,2,2))
    >>> out
    <Images class, title: , dimensions: (2|)>:
    ├── ND_Image_shape2x2x2: <title: None, dimensions: (2,|2, 2)
    └── ND_Image_shape2x2x2x2: <title: None, dimensions: (2, 2|2, 2)
    """
    def __init__(self, images : list | np.ndarray) -> None:
        from microspy.io._images import _io

        # images to MicroSpySignal2D variants
        images = _io._arrays2signals(images)
        
        sig_types = []
        for im in images:
            if im is not None:
                sig_types.append(im.metadata.Signal.signal_type)
                setattr(self, sig_types[-1], im)

        self._phase_maps = dict()
        self._updated_phase_maps : bool = False
        self._create_metadata()

        for sig_type in sig_types:
            # Set a brief description of the signal type
            self.metadata.set_item(
                f"Signals.{sig_type}", 
                Images_signals.get(sig_type)
            )
        
        # Images' title:
        self.metadata.set_item("General.title", "")

    def __repr__(self):
        # General overview:
        dim = self.num_signals
        title = self.metadata.General.title
        print_string = f"<Images class, title: {title}, " 
        print_string += f"dimensions: ({dim}|)>:\n"

        # Signals overview:
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

    # ---------------------------------------------------------------- #
    # ---------------------- Custom attributes ----------------------- #
    # ---------------------------------------------------------------- #

    @property
    def metadata(self):
        """The metadata of the signal."""
        return self._metadata

    @property
    def phase_maps_updated(self) -> bool:
        """Check if phase maps are updated."""
        return self._updated_phase_maps
        
    @property
    def phase_maps(self) -> dict:
        """Return the phase maps.

        Returns
        -------
        phase_maps
            Dictionary with ndarrays representing each class (key) 
            positions.
        """
        from copy import deepcopy
        return deepcopy(self._phase_maps)

    @property
    def num_signals(self) -> int:
        """The total number of Signal2Ds"""
        num = 0
        for attr, value in vars(self).items():
            if isinstance(value, MicroSpySignal2D | MicroSpySignal2D_Parent):
                num += 1
        return num
            
    @property
    def navigation_unit(self) -> str:
        """Get the calibration unit"""
        unit = self.metadata.get_item(
            "Acquisition_instrument.Acquisition.unit"
        )
        if unit is not None:
            return unit
        else:
            return "NA"

    @property
    def navigation_scale(self) -> str:
        """Get the calibration scale."""
        scale = self.metadata.get_item(
            "Acquisition_instrument.Acquisition.scale"
        )
        if scale is not None:
            return scale
        else:
            return 1

    @property
    def is_calibrated(self):
        """Check if the signals are calibrated"""
        scaled = self.metadata.get_item(
            "Acquisition_instrument.Acquisition.scale"
        )
        set_unit = self.metadata.get_item(
            "Acquisition_instrument.Acquisition.unit"
        )
        if scaled and set_unit is not None:

            return True
        
        else: 
            # It might be the case that the signals have been calibrated,
            # but no information about this exists in the metadata. If so,
            # set it by asking the user:
            parent_scale = self.ParentSig.axes_manager[-1].scale
            
            if parent_scale != 1:
                calibrate = input(
                    "The Parent signal was found with a navigation scale "
                    f"of '{parent_scale}'. Calibrate the signals using "
                    "this? ([y]/n)"
                )
                if calibrate.lower() in ("y", "yes", ""):
                    print("\nUpdating navigation calibration")
                    scale = parent_scale
                    unit = self.ParentSig.axes_manager[-1].units
                    is_cal = True
                else: 
                    print("\nResetting navigation calibration.")
                    scale = 1
                    unit = "NA"
                    is_cal = False
                
                self.calibrate_signals(
                    scale = scale,
                    unit = unit
                )
                
                return is_cal
                
            else: return False
    
    @property
    def is_gridified(self):
        """Check if the signal is gridified or not. 
        The function returns True if the array shape 
        is (Y,X,ky,kx) >= 4.

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
            _gridified = True
            # Iterate through all Parent signal types:
            for attr, value in self.__dict__.items():
                if isinstance(value, MicroSpySignal2D_Parent):
                    _gridified *= self.__dict__[attr].is_gridified
            return bool(_gridified)
        else: 
            raise AttributeError("The class has no parent signal.")
            
    # ---------------------------------------------------------------- #
    # ----------------------- Private methods ------------------------ #
    # ---------------------------------------------------------------- #

    def _create_metadata(self):
        """Create metadata structure using hyperspy's DictionaryTreeBrowser."""
        self._metadata = utils.DictionaryTreeBrowser()
        md = self.metadata
        md.add_node("General")
        md.General.add_node("title")
        md.add_node("Signals")
        self._original_metadata = utils.DictionaryTreeBrowser()
        
    def _reset_phase_maps(self):
        """Reset phase maps to an empty dictionary"""
        if len(self._phase_maps) > 0:
            self._phase_maps = {}
            
    def _gridify_phase_maps(
        self,
        grid_shape : tuple,
        flip_axis : int | list | tuple | None = None
    ) -> None:
        """Gridify the phase maps.

        Parameters
        ----------
        grid_shape
            2D grid shape.
            Assuming the data shape is (X, W, H), the data will be reshaped 
            into grid_shape + (W, H)
        flip_axis 
            Depending on the image acquisition order, flip the images along 
            specified axis/axes. 
        """
        if len(self._phase_maps) > 0: 
            _phase_maps = self.phase_maps
            self._reset_phase_maps()
            for key, val in _phase_maps.items():
                _phase_maps[key] = _3Darray_2_4Darray(
                    arr = val,
                    to_shape = grid_shape,
                    flip_axis = flip_axis
                )
            self._phase_maps = _phase_maps

    def _degridify_phase_maps(
        self,
        flip_axes : int | list | tuple | None = None
    ):
        """Degridify phase maps.
        
        Parameters
        ----------
        flip_axes
            axes to flip to make the image acquisition order
            compatible with other signals.
        """

        if len(self._phase_maps) > 0:
            _phase_maps = self.phase_maps
            self._reset_phase_maps()
            
            for key, val in _phase_maps.items():
                _phase_maps[key] = _4Darray_2_3Darray(
                    arr = val,
                    flip_axis = flip_axes
                )
            self._phase_maps = _phase_maps
            
    # ---------------------------------------------------------------- #
    # ------------------------- Open methods ------------------------- #
    # ---------------------------------------------------------------- #

    def set_phase_maps(
        self,
        classes : list | np.ndarray | tuple,
        background_label : int = -1,
        **kwargs
    ):
        """Set the phase maps (a boolean ndarray per class).
        
        Note!
        See 
            :meth:'Images.phase_maps'
            :func:'Images.get_phaseMap'

        Parameters
        ----------
        classes
            List or tuple of particle class names
        label_maps
            boolean ndarray of particle locations  
        kwargs
            keyword arguments
            The function reads 'acquisition_order' and 'vendor'.
        """
        from microspy.draw import imaging
        gridify = False
        
        if not hasattr(self, list(Images_signals.keys())[3]):
            raise AttributeError(
                "The class has no map of particle locations. See "
                ":func:'Images.map_ChildSig_onto_ParentSig'"
            )
        
        if isinstance(
            kwargs.get("acquisition_order"), list | np.ndarray):
                
            from microspy.misc._misc import _vendor2ImFlipAxes
            vendor = kwargs.get("vendor")
            flip_axes = _vendor2ImFlipAxes(vendor)
            
            # Degridify to simplify the mapping
            grid_shape = self.ParentSig.axes_manager.navigation_shape[::-1]
            gridify = True if len(grid_shape) > 1 else False
            self.degridify_ParentSig(flip_axes = flip_axes)

        __attr = getattr(self, list(Images_signals.keys())[3])
        self._reset_phase_maps()
        
        phase_maps = imaging._get_phase_maps_from_label_map(
            classes = classes,
            label_map = __attr.data,
            background_label = background_label,
            **kwargs
        ) 
        
        # Set phase maps
        self._phase_maps = phase_maps
        self._updated_phase_maps = True
        
        # Re-gridify:
        if gridify: 
            #self._gridify_phase_maps(grid_shape)
            # Should include phase_maps
            self.gridify_ParentSig(
                nav_shape = grid_shape,
                flip_axes = flip_axes
            )
            
    def get_phase_map(
        self,
        bkgr_label : int = -1,
        class_order : dict | list[str, ...] | tuple[str, ...] | None = None
    ) -> np.ndarray:
        """Return a single phase map with labelled regions representing 
        the phases/classes. 

        Parameters
        ----------
        bkgr_label
            Label of the bakcground, -1 by default.
        class_order
            Order of the phase map labels according to class order

        Returns
        -------
        classes
            Dictionary with phase map classes as keys and phase map labels 
            as values.
        PM
            phase map as ndarray with labelled regions according to class.
            
        Examples
        --------
        >>> classes, PM = get_phase_map()
        >>> classes.values()
        dict_values([np.int64(0), np.int64(1), np.int64(2), np.int64(3)])
        
        >>> classes, PM = get_phase_map(bkgr_label = 2)
        >>> classes.values()
        dict_values([np.int64(3), np.int64(4), np.int64(5), np.int64(6)])
        
        >>> classes, PM = get_phase_map(
                class_order = list_of_unique_classes
            )
        >>> classes.values()
        dict_values([np.int64(0), np.int64(1), np.int64(2), np.int64(3)])
        """
        
        if self.phase_maps_updated:
            
            from copy import deepcopy
            
            phase_maps = self.phase_maps
            keys = list(self.phase_maps.keys())
            num_phase_maps = len(phase_maps)
            
            # If class_order contains the order of the classes (and it matches 
            # the number of phase_maps):
            if isinstance(class_order, list | tuple):
                if not list(map(type, class_order)) == [str]*num_phase_maps:
                    raise exceptions.InputError(
                        f"The number of classes ({len(class_order)}) do not match "
                        f"the number of phase maps ({num_phase_maps})."
                    )
                
                diff = set(class_order) - set(keys)
                if len(diff) > 0:
                    raise exceptions.InputError(
                            f"'{diff}' is/are not among the phase map classes "
                            f"('{keys}')."
                        )
                else:
                    pm_keys = deepcopy(class_order)
                    num_input_classes = len(class_order)
                    enumerator = np.arange(
                        bkgr_label + 1,
                        stop = bkgr_label + 1 + num_input_classes,
                        step = 1
                    )
                    classes = dict(
                        zip(
                            pm_keys,
                            enumerator
                        )
                    )
            elif isinstance(class_order, dict):
                
                _keys = list(class_order.keys())
                Unrec = set(_keys) - set(keys) # Unrecognised key
                Missing = set(keys) - set(_keys) # "Missing" key
                
                if len(Unrec) > 0:
                    raise exceptions.InputError(
                            f"'{Unrec}' is/are not part of the phase maps' key "
                            "argument(s) ('{keys}'). See *.set_phase_maps()"
                        )
                else:
                    if len(Missing):
                        exceptions.formatted_warning(
                            f"Class(es) '{Missing}' is/are not included in the "
                            "'class_order' argument. It will be missing in the "
                            "returned phase map."
                        )
                    
                    # "~zip":
                    pm_keys, enumerator = zip(*class_order.items())
                    
                    if bkgr_label in list(enumerator):
                        index = list(enumerator).index(bkgr_label)
                        raise exceptions.InputError(
                                f"The background label ({bkgr_label}) is found "
                                "in class_order. Define a different key for "
                                f"'bkgr_label' or '{pm_keys[index]}'."
                            )
                    elif np.min(enumerator) < bkgr_label:
                        index = np.argmin(enumerator)
                        exceptions.formatted_warning(
                            f"'{pm_keys[index]}' is labelled with a smaller "
                            f"value ('{np.min(enumerator)}') than the background "
                            f"label ('{bkgr_label}').\nThis can cause issues "
                            "for attribute functions that interprets the minimum "
                            "phase\n map label as background label."
                        )
                    
                    classes = dict(
                        zip(
                            pm_keys,
                            enumerator
                        )
                    )
                    
            elif class_order is None:
                # Using the order as written in the phase_maps
                num_classes = len(phase_maps)
                enumerator = np.arange(
                        bkgr_label + 1,
                        stop = bkgr_label + 1 + num_classes,
                        step = 1
                    )
                classes = dict(
                    zip(
                        phase_maps.keys(),
                        enumerator
                    )
                )
                
            else:
                raise exceptions.InputError(
                        f"Input argument 'class_order' ({class_order}) is not "
                        "recognised."
                    )
            
            PM = np.full_like(
                self.ParentSig.data,
                fill_value = bkgr_label,
                dtype = int
            )
            
            for key, val in classes.items():
                PM[phase_maps[key]] = val

            return classes, PM
            
        elif not hasattr(self, "ChildMap"):
            raise AttributeError(
                    "The 'Images' class has no 'ChildMap' attrubte."
                    "See :func:'Images.map_ChildSig_onto_ParentSig'."
                )
        
        else:
            exceptions.formatted_warning(
                "Phase maps are not updated. See :func:'set_phase_maps'."
            )
            return {}, np.asarray([])
        
    def calibrate_signals(
        self,
        scale : int | float,
        unit : str = "NA",
        upsamplingChildSigFac : int | float = 1
    ):
        """Calibrate attribute MicroSpy2D signals.

        Parameters
        ----------
        scale
            ParentSig scale (unit per pixel)
            
            Note!
            The CompositeSig might be binned, so a factor is used
            when calibrating the CompositeSig. 
        unit
            scale unit. If "NA", the calibration is unsuccessful.
        upscaledChildSig
            Whether the child signal is upsampled or not, i.e.
            the childSig has a higher resolution than the ParentSig.
            If so, the argument > 1.
        """
    
        if hasattr(self, "CompositeSig"):
            # Adjust scale if CompositeSig has a different size than 
            # ParentSig.
            fac = self.ParentSig.data.size / self.CompositeSig.data.size
            self.CompositeSig.set_scale(
                scale = fac * scale,
                unit = unit
            )
            
        if hasattr(self, "ParentSig"):
            # Iterate through all Parent signal types:
            for attr, value in self.__dict__.items():
                if isinstance(value, MicroSpySignal2D_Parent):
                    self.__dict__[attr].set_scale(
                        scale = scale,
                        unit = unit
                        )
                        
        if hasattr(self, "ChildSig"):
            exceptions.formatted_warning(
                "Check if downsampling is possible. If not, check arg."
            )
            self.ChildSig.set_scale(
                scale = scale / upsamplingChildSigFac,
                unit = unit
            )
        
        if unit != "NA":
            self._metadata.set_item(
                item_path = "Acquisition_instrument.Acquisition.scale", 
                value = scale
            )
            self._metadata.set_item(
                item_path = "Acquisition_instrument.Acquisition.unit", 
                value = unit
            )

    def gridify_ParentSig(
        self,
        nav_shape : tuple | list | None = None,
        flip_axes : int | tuple | list | None = None#(1,)
    ):
        """Gridify the Parent signal(s).
        
        Note!
        If phase_maps exists, these too will become gridified.

        Parameters
        ----------
        nav_shape
            Shape of the navigation grid
        flip_axes 
            Depending on the image acquisition order,
            flip the images along specified axis/axes. 
        """

        if not hasattr(self, "ParentSig"):
            raise AttributeError("The signal class does not keep "
                                "track of acqiusition images. "
                                "See :func:'Images.setParentSig'.")
        
        gridify = False
        if self.is_gridified:
            _nav_shape = self.ParentSig.axes_manager.navigation_shape[::-1]
            if nav_shape != _nav_shape:
                ans = input("The signal is already gridified. Regridify? ([y]/n)")
                if ans.lower() in ("", "y", "yes"):
                    gridify = True
                    self.degridify_ParentSig(
                        flip_axes = flip_axes
                    )
            else:
                print(f"The signal is already gridified to shape {nav_shape}.")
        else:    
            gridify = True
            
        if gridify:
            # Iterate through all the MicroSpySignals2D_Parent
            for attr, value in self.__dict__.items():
                if isinstance(value, MicroSpySignal2D_Parent):
                    self.__dict__[attr].gridify(
                        grid_shape = nav_shape,
                        flip_axis = flip_axes
                    )
                    
            # Update calibration:
            if self.is_calibrated:
                self.calibrate_signals(
                    scale = self.navigation_scale,
                    unit = self.navigation_unit
                )

            if len(self._phase_maps) > 0:
                #print("Gridifying phase maps.")
                self._gridify_phase_maps(
                    grid_shape = nav_shape,
                    flip_axis = flip_axes
                )
        
    def degridify_ParentSig(
        self,
        flip_axes : int | tuple | list = None
    ):
        """Degridify the Parent signal(s).

        Parameters
        ----------
        flip_axes 
            Depending on the image acquisition order,
            flip the images along specified axis/axes. 
        """
        print(flip_axes)
        if not hasattr(self, "ParentSig"):
            raise AttributeError("The signal class does not keep "
                                "track of acqiusition images. "
                                "See *.setParentSig().")

        if not self.is_gridified:
            print("The signal is already degridified.")
        else:
            for attr, value in self.__dict__.items():
                if isinstance(value, MicroSpySignal2D_Parent):
                    self.__dict__[attr].degridify(
                        flip_axis = flip_axes
                    )
            
        # Update calibration:
        if self.is_calibrated:
            self.calibrate_signals(
                scale = self.navigation_scale,
                unit = self.navigation_unit
            )
        
        # Degridify phase maps if any:
        if len(self._phase_maps) > 0:
            self._degridify_phase_maps()

    def setParentSig(
        self,
        current_name : str
    ):
        """ Set the parent signal, i.e. the instance attribute name given by
        'current_name' argument will be changed to "ParentSig".

        Note!
        The class' metadata will be changed accordingly (see :meth:'metadata
        .Signals').

        Parameters
        ----------
        current_signal 
            Name of the current signal to be changed. 
        """
        
        proceed = True
        
        # signal types:
        sigVal = list(Images_signals.values())[1]
        sigKey = list(Images_signals.keys())[1]
        
        if hasattr(self, sigKey):
            ans = input(f"'{sigKey}' is already defined. Proceed? (y/[n])")
            if ans.lower() in ("no", "n", ""):
                proceed = False
                
        if hasattr(self, current_name):
            if proceed:
                # Get attribute and set correct signal type
                attr = MicroSpySignal2D_Parent(
                    getattr(self, current_name)
                )
                
                from copy import deepcopy
                md = deepcopy(attr.metadata.as_dictionary())

                delattr(self, current_name)
                
                # Set attribute metadata
                attr.metadata.add_dictionary(md)
                attr.metadata.set_item("Signal.signal_type", sigKey)
                attr.metadata.set_item("General.title", sigVal)
                
                # Set new attribute as ParentSig:
                setattr(self, sigKey, attr)
                
                # Set Images' metadata:
                sig_md = self.metadata.Signals.as_dictionary()
                del self.metadata.Signals
                del sig_md[f"{current_name}"]
                self.metadata.add_node("Signals")
                self.metadata.Signals.add_dictionary(sig_md)
                self.metadata.set_item(f"Signals.{sigKey}", sigVal)
                
        else:
            raise AttributeError("The signal class doesn't have "
                                "an instance attribute named "
                                f"{current_name}, but \n{self.__repr__}")

    def setChildSig(
        self,
        current_name : str
    ):
        """Set the child signal, i.e. the instance attribute name given by
        'current_name' will be changed to "ChildSig".
        
        Note!
        The class' metadata will be changed accordingly (see :meth:'metadata
        .Signals').

        Parameters
        ----------
        current_signal
            Name of the current signal to be changed. 
        """
        proceed = True
        
        # signal types:
        sigVal = list(Images_signals.values())[2]
        sigKey = list(Images_signals.keys())[2]
        
        if hasattr(self, sigKey):
            ans = input(f"'{sigKey}' is already defined. Proceed? (y/[n])")
            if ans.lower() in ("no", "n", ""):
                proceed = False
        
        if hasattr(self, current_name):
            if proceed:
                # Get attribute and set correct signal type
                attr = MicroSpySignal2D_Parent(
                    getattr(self, current_name)
                )
                
                from copy import deepcopy
                md = deepcopy(attr.metadata.as_dictionary())

                delattr(self, current_name)
                
                attr.metadata.add_dictionary(md)
                attr.metadata.set_item("Signal.signal_type", sigKey)
                attr.metadata.set_item("General.title", sigVal)

                # Set new attribute as ParentSig:                
                setattr(self, sigKey, attr)
                
                # Set Images' metadata:
                sig_md = self.metadata.Signals.as_dictionary()
                del self.metadata.Signals
                del sig_md[f"{current_name}"]
                self.metadata.add_node("Signals")
                self.metadata.Signals.add_dictionary(sig_md)
                self.metadata.set_item(f"Signals.{sigKey}", sigVal)
            
        else:
            raise AttributeError("The signal class doesn't have "
                                "an instance attribute named "
                                f"{current_name}, but \n{self.__repr__}")

    def map_ChildSig_onto_ParentSig(
        self,
        vendor : str,
        matrix_label : int = -1,
        return_map : bool = False,
        **kwargs
    ):
        """Map the Child images onto the Parent images.
        
        The function uses :func:'skimage.feature.match_template' to 
        map the Child images/arrays onto the parent array.

        Parameters
        ----------
        vendor
            Vendor of the data acquisition, as the image acquisition order 
            can be different between different manufacturers.
            
            Example: Jeol's particle analysis solution acquires images from
            right to left and top to bottom.
        matrix_label 
            Label of the matrix, i.e. non-particles.
        """
        exceptions.formatted_warning(
            "Currently not supporting upsampled images."
        )
        if not hasattr(self, "ParentSig") or not hasattr(self, "ChildSig"):
            raise AttributeError("Parent and/or Child signal has not been"
                                "set. See :func:'Images.setParentSig'/"
                                 ":func:'Images.setChildSig'.")
        
        exceptions.formatted_warning(
            "Employ vendor plugins?"
        )
        """
        vendor plugins?
        """

        if vendor.lower() in ["jeol"]:

            # SEM images:
            is_gridified = self.is_gridified
            if is_gridified:
                from microspy.misc._misc import _vendor2ImFlipAxes
                flip_axes = _vendor2ImFlipAxes(vendor)
                to_grid = self.ParentSig.axes_manager.navigation_shape[::-1]
                self.degridify_ParentSig(
                    flip_axes = flip_axes
                )
                parentArr = self.ParentSig.data.copy()
                self.gridify_ParentSig(
                    nav_shape = to_grid, 
                    flip_axes = flip_axes
                )
            else:
                parentArr = self.ParentSig.data.copy()
            
            # e.g. np.asarray([0, 0, 0, 1, 1, ... 16, 16])
            parentOrder = kwargs.get("acquisition_order")

        if not isinstance(parentOrder, np.ndarray):
            raise ValueError(
                "CAN ONE EXPECT ALL VENDORS TO PROVIDE IMAGE NUMBER "
                "INFORMATION?"
            )
            # Stitch the entire parent signal and the search for the 
            # particles? Then de-stitch the particle map?
            """from microspy.draw import get_stitched_grid_signal
            from microspy.misc._misc import _vendor2ImAquisitionOrder
            
            # Stitching directions:
            hor, ver = _vendor2ImAquisitionOrder(vendor)
            
            # Stitch the parent signal:
            parentArr = get_stitched_grid_signal(
                array = self.ParentSig.data.copy(),
                grid_shape = self.ParentSig.axes_manager.navigation_shape[::-1],
                horisontal_direction = hor,
                vertical_direction = vert,
            )"""

        # Depadded particle images
        childArr = self.ChildSig.data.copy()

        # Whether to use nested progressbar or not
        nested_progressbar = kwargs.get("nested_progressbar")
        if not nested_progressbar:
            if parentOrder is not None:
                # Mean number of particles per image:
                nested_progressbar = len(childArr) / len(
                    np.unique(parentOrder)
                ) > 200
            else:
                # Tot. number of particles
                nested_progressbar = len(childArr) > 500

        label_maps = _image_utils._map_particle_regions(
            childArray = childArr,
            parentArray = parentArr,
            parentOrder = parentOrder,
            background_label = matrix_label,
            nested_progressbar = nested_progressbar
        )   

        attribute_name = list(Images_signals.keys())[3]
        sig_type = Images_signals[attribute_name]
        print(f"Allocating MicroSpySignal2D_Parent '{attribute_name}' "
              "to 'Images' attribute.")
        
        setattr(self, 
                attribute_name, 
                MicroSpySignal2D_Parent(
                    label_maps,
                    **{"metadata" : {
                            "General" : {"title" : sig_type},
                            "Signal" : {"signal_type" : attribute_name}
                        }
                    }
                )
            )
        
        __attr = getattr(self, attribute_name)
        
        # Gridify signal:
        if is_gridified: 
            __attr.gridify(
                grid_shape = to_grid,
                flip_axis = flip_axes
            )
            
        # Calibrate signal:
        if self.is_calibrated:
            __attr.set_scale(
                scale = self.navigation_scale,
                unit = self.navigation_unit
            )
        
        self._metadata.Signals.set_item(
            item_path = attribute_name, 
            value = sig_type
        )
