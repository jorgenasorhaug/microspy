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
import pytest

import microspy as ms

class TestParticleAnalysis:
    def test_init(self):
        from microspy.signals._microspy_signals1D import (
            MicroSpySignal1D_Chemistry,
            MicroSpySignal1D_Geometry,
        )
        # Signal shape
        chemArr0 = np.arange(1, 13).reshape(3,4)
        geomArr0 = np.zeros(shape = (15,)).reshape(3,5)
        
        chem0 = MicroSpySignal1D_Chemistry(chemArr0)
        geom0 = MicroSpySignal1D_Geometry(geomArr0)
        
        assert chemArr0.shape == chem0.data.shape
        assert geomArr0.shape == geom0.data.shape
        
        s0 = ms.signals.ParticleAnalysis([chem0, geom0])
        
        assert s0.Chemistry.data.shape == chem0.data.shape
        assert s0.Geometry.data.shape == geom0.data.shape
        assert s0.Chemistry.axes_manager.shape == chemArr0.shape
        assert s0.Geometry.axes_manager.shape == geomArr0.shape
        
        # Signal types
        assert isinstance(s0.Chemistry, MicroSpySignal1D_Chemistry)
        assert isinstance(s0.Geometry, MicroSpySignal1D_Geometry)
        
        # Particle classes
        assert len(np.unique(s0.particle_classes)) == 1
        assert ("Unclassified" == 
            np.unique(s0.particle_classes)[0] == 
            np.unique(s0._original_classes)[0]
        )
        assert not s0.is_classified.all()
        
        # Attribute properties
        assert s0.elements == s0.Chemistry.elements
        assert s0.num_particles == chemArr0.shape[0] == geomArr0.shape[0]
        
    def test_Images_init(self, dummy_signal):
        assert isinstance(dummy_signal.Images.phase_maps, dict)
        assert not dummy_signal.Images.phase_maps_updated
        assert dummy_signal.Images.num_signals == 2
        assert dummy_signal.Images.navigation_unit == "NA"
        assert dummy_signal.Images.navigation_scale == 1
        assert not dummy_signal.Images.is_calibrated
        assert not dummy_signal.Images.is_gridified
        
    def test_gridify_parent_sig(self, dummy_signal):
        signal_shape = dummy_signal.Images.ParentSig.axes_manager.signal_shape[::-1]
        to_shape = (3,2)
        dummy_signal.gridify_ParentSig(
            navigation_shape = to_shape
        )
        assert dummy_signal.Images.is_gridified
        assert dummy_signal.Images.ParentSig.data.shape == to_shape + signal_shape
    
    def test_shallow_copy_deepcopy(self, dummy_signal):
        from copy import deepcopy
        
        s0 = dummy_signal
        assert not np.may_share_memory(dummy_signal, s0)
        
        chem = dummy_signal.Chemistry.data
        chem0 = s0.Chemistry.data
        geom = dummy_signal.Geometry.data
        geom0 = s0.Geometry.data
        
        assert np.may_share_memory(chem, chem0)
        assert np.allclose(chem, chem0)
        assert np.may_share_memory(geom, geom0)
        assert np.allclose(geom, geom0)
        
        parent = dummy_signal.Images.ParentSig.data
        parent0 = s0.Images.ParentSig.data
        
        child = dummy_signal.Images.ChildSig.data
        child0 = s0.Images.ChildSig.data
        
        assert np.may_share_memory(parent, parent0)
        assert np.allclose(parent, parent0)
        assert np.may_share_memory(child, child0)
        assert np.allclose(child, child0)
        
        s1 = deepcopy(s0)
        assert not np.may_share_memory(s0, s1)
        
        chem1 = s1.Chemistry.data
        geom1 = s1.Geometry.data
        
        assert not np.may_share_memory(chem, chem1)
        assert np.allclose(chem, chem1)
        assert not np.may_share_memory(geom, geom1)
        assert np.allclose(geom, geom1)
        
        parent1 = s1.Images.ParentSig.data
        child1 = s1.Images.ChildSig.data
        
        assert not np.may_share_memory(parent, parent1)
        assert np.allclose(parent, parent1)
        assert not np.may_share_memory(child, child1)
        assert np.allclose(child, child1)


class TestChemistryProperty:
    def test_change_dtype(self, dummy_signal):
        from microspy.signals._microspy_signals1D import (
            MicroSpySignal1D_Chemistry
        )
        dummy_signal.Chemistry.change_dtype("uint16")
        assert isinstance(dummy_signal, ms.signals.ParticleAnalysis)
        assert isinstance(dummy_signal.Chemistry, MicroSpySignal1D_Chemistry)
        
    def test_set_and_change_units(self, dummy_signal):
        dummy_signal.Chemistry.set_units("wt.%")
        assert dummy_signal.Chemistry.unit == "wt.%"
        dummy_signal.Chemistry.set_units("Wt.%")
        assert dummy_signal.Chemistry.unit == "Wt.%"
        dummy_signal.Chemistry.change_unit()
        assert dummy_signal.Chemistry.unit == "[at.%]"
        dummy_signal.Chemistry.change_unit()
        assert dummy_signal.Chemistry.unit == "[wt.%]"
        
    def test_properties(self, dummy_signal):
        assert (dummy_signal.elements 
            == dummy_signal.Chemistry.elements 
            == dummy_signal.Chemistry.metadata.Signal.props
        )

        
class TestGeometryProperty:
    def test_change_dtype(self, dummy_signal):
        from microspy.signals._microspy_signals1D import (
            MicroSpySignal1D_Geometry
        )
        dummy_signal.Geometry.change_dtype("uint16")
        assert isinstance(dummy_signal, ms.signals.ParticleAnalysis)
        assert isinstance(dummy_signal.Geometry, MicroSpySignal1D_Geometry)
    
    def test_set_and_change_units(self, dummy_signal):
        dummy_signal.Geometry.set_units(
            [
                f"Unit{i}" for i in range(
                    dummy_signal.Geometry.data.shape[-1]
                )
            ]
        )
        assert dummy_signal.Geometry.unit == [
            "Unit0", "Unit1", "Unit2", "Unit3", "Unit4"
        ]
        
    def test_set_scan_calibration(self, dummy_signal):
        to_shape = (3,2)
        scale = 2
        unit = "Arb.Unit"
        
        dummy_signal.gridify_ParentSig(
            navigation_shape = to_shape
        )
        
        dummy_signal.calibrate_navigation(
            scale = scale,
            unit = unit
        )
        assert dummy_signal.Images.is_calibrated
        
        assert dummy_signal.Images.metadata.get_item(
            "Acquisition_instrument.Acquisition.scale"
        ) == 2
        
        assert dummy_signal.Images.metadata.get_item(
            "Acquisition_instrument.Acquisition.unit"
        ) == "Arb.Unit"
        
        assert (
            dummy_signal.Images.ParentSig.axes_manager[0].units
            == dummy_signal.Images.ParentSig.axes_manager[1].units
            == dummy_signal.Images.ParentSig.axes_manager[2].units
            == dummy_signal.Images.ParentSig.axes_manager[3].units
            == unit
        )
        assert (
            dummy_signal.Images.ParentSig.axes_manager[0].scale 
            == scale * dummy_signal.Images.ParentSig.axes_manager[2].size
            and 
            dummy_signal.Images.ParentSig.axes_manager[1].scale 
            == scale * dummy_signal.Images.ParentSig.axes_manager[3].size
        )
        assert (
            dummy_signal.Images.ParentSig.axes_manager[2].scale
            == dummy_signal.Images.ParentSig.axes_manager[3].scale
            == scale
        )
        
class TestImages:
    def test_gridify_parentSig(self, dummy_signal):
        from copy import deepcopy
        s1 = deepcopy(dummy_signal)
        
        to_shape = (3,2)
        
        dummy_signal.gridify_ParentSig(
            navigation_shape = to_shape
        )
        s1.Images.gridify_ParentSig(
            nav_shape = to_shape
        )
        assert s1.Images.is_gridified
        assert (
            dummy_signal.Images.ParentSig.axes_manager.shape
            == s1.Images.ParentSig.axes_manager.shape
        )
    def test_gridify_carry_over_from_copy(self, dummy_signal):
        from copy import deepcopy
        to_shape = (3,2)
        dummy_signal.gridify_ParentSig(
            navigation_shape = to_shape
        )
        shape = dummy_signal.Images.ParentSig.data.shape
        s0 = deepcopy(dummy_signal)
        assert s0.Images.is_gridified
        assert s0.Images.ParentSig.data.shape == shape
        
    def test_calibration_carry_over_from_copy(self, dummy_signal):
        from copy import deepcopy
        to_shape = (3,2)
        dummy_signal.gridify_ParentSig(
            navigation_shape = to_shape
        )
        dummy_signal.calibrate_navigation(
            unit = "Arb.",
            scale = 2
        )
        
        s0 = deepcopy(dummy_signal)
        assert s0.Images.is_calibrated
        
        dummy_axes_dict = dummy_signal.Images.ParentSig.axes_manager.as_dictionary()
        copy_axes_dict = s0.Images.ParentSig.axes_manager.as_dictionary()
        assert dummy_axes_dict == copy_axes_dict
        
        dummy_meta_dict = dummy_signal.Images.metadata.as_dictionary()
        copy_meta_dict = s0.Images.metadata.as_dictionary()
        assert dummy_meta_dict == copy_meta_dict
    
    def test_gridify_degridify(self, dummy_signal):
        from copy import deepcopy
        s0 = deepcopy(dummy_signal)
        dummy_signal.gridify_ParentSig(
            navigation_shape = (3,2)
        )
        dummy_signal.degridify_ParentSig()
        assert (
            dummy_signal.Images.ParentSig.axes_manager.shape 
            == s0.Images.ParentSig.axes_manager.shape
        )
    
    def test_gridify_degridify_axis_flip(self, dummy_signal):
        from copy import deepcopy
        to_shape = (3,2)
        
        s0 = deepcopy(dummy_signal)
        s0.gridify_ParentSig(
            navigation_shape = to_shape,
            flip_axes = 0
        )
        assert (
            s0.Images.ParentSig.axes_manager.navigation_shape[::-1]
            == to_shape
        )
        
        s1 = deepcopy(dummy_signal)
        s1.gridify_ParentSig(
            navigation_shape = to_shape,
            flip_axes = 1
        )
        