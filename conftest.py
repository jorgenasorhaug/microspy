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
import os
from pathlib import Path
import pytest
from typing import Callable, Generator, Literal

import hyperspy.api as hs
import microspy as ms

#DATA_PATH = Path(ms.data.__file__).parent.resolve()

# --------------------------- pytest hooks --------------------------- #

def pytest_sessionstart(session):
    # Download material data files
    ms.misc.element_database
    downloaded_files = Path(ms.misc.__file__).parent.resolve()
    lines_file = downloaded_files / "xray_lines.json"
    props_file = downloaded_files / "elements_general_properties.json"
    
    assert lines_file.is_file()
    assert props_file.is_file()
    
    # Delete materials data files
    os.remove(str(lines_file))
    os.remove(str(props_file))
    plt.rcParams.update({"backend": "agg", "figure.max_open_warning": False})

@pytest.fixture(autouse=True)
def doctest_setup_teardown(request):
    # Temporarily turn off interactive plotting with Matplotlib
    plt.ioff()

    # Temporarily suppress HyperSpy's progressbar
    hs.preferences.General.show_progressbar = False

    # Temporary directory for saving files in
    #temporary_directory = tempfile.TemporaryDirectory()
    #original_directory = os.getcwd()
    #os.chdir(temporary_directory.name)
    #yield

    # Teardown
    #os.chdir(original_directory)

#@pytest.fixture(autouse=True)
#def import_to_namespace(doctest_namespace) -> None:
#    doctest_namespace["DATA_PATH"] = DATA_PATH / "microspy_hdf5"

@pytest.fixture
def dummy_signal(
    data: np.ndarray,
) -> Generator[ms.signals.ParticleAnalysis]:
    """Dummy signal of shape ... If this is changed, all 
    tests using this signal will fail as they compare the output
    from methods using this signal as input to hard-coded outputs.
    """
    num_particles = 3
    num_elements = 4
    num_geometric_props = 5
    
    # 3 particles with 4 elements each
    chemArr = np.array(
        [
            69, 30, 1, 13, 80, 7, 52, 9, 39, 10, 69, 21
        ]
    ).reshape(num_particles, num_elements)
    # 3 particles with 5 geometric properties each
    geomArr = np.array(
        [
            [9, 1, 44, 63, 29, 45, 18, 83, 7, 84, 83, 10, 95, 23, 23]
        ]
    ).reshape(num_particles, num_geometric_props)
    
    chem = ms.signals.MicroSpySignal1D_Chemistry(
        chemArr,
        **{"title" : "test_chemistry"}
    )
    
    chem.metadata.set_item(
        "Sample.elements", ["C", "O", "Al", "Fe"]
    )
    
    geom = ms.signals._microspy_signals.MicroSpySignal1D_Geometry(
        geomArr,
        **{"title" : "test_geometry"}
    )
    
    s = ms.signals.ParticleAnalysis(
        [chem, geom]
    )
    
    yield s

