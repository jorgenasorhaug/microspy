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
        # Signal shape
        chemArr0 = np.arange(1, 13).reshape(3,4)
        geomArr0 = np.zeros(shape = (15,)).reshape(3,5)
        chem0 = ms.signals._microspy_signals.MicroSpySignal1D_Chemistry(
            chemArr0
        )
        geom0 = ms.signals._microspy_signals.MicroSpySignal1D_Geometry(
            geomArr0
        )

        s0 = ms.signals.ParticleAnalysis([chem0, geom0])
        assert chemArr0.shape == chem0.data.shape
        assert s0.Chemistry.data.shape == chem0.data.shape