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
    dummy_images: np.ndarray
) -> Generator[microspy.signals.ParticleAnalysis]:
    """Dummy signal with 3 particles, 4 elements and 5 geometric 
    proeprties. If this is changed, all tests using this signal 
    will fail as they compare the output from methods using this 
    signal as input to hard-coded outputs.
    """
    from microspy.signals._microspy_signals1D import (
        MicroSpySignal1D_Chemistry,
        MicroSpySignal1D_Geometry,
    )
    
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
            9, 1, 44, 63, 29, 45, 18, 83, 7, 84, 83, 10, 95, 23, 23
        ]
    ).reshape(num_particles, num_geometric_props)
    
    chem = MicroSpySignal1D_Chemistry(
        chemArr,
        **{"title" : "test_chemistry"}
    )
    
    chem.metadata.set_item(
        "Signal.props", ["C", "O", "Al", "Fe"]
    )
    
    geom = MicroSpySignal1D_Geometry(
        geomArr,
        **{"title" : "test_geometry"}
    )

    s = ms.signals.ParticleAnalysis(
        [chem, geom], **{"images" : dummy_images}
    )
    
    yield s


@pytest.fixture
def dummy_parent() -> Generator[np.ndarray]:
    """Dummy parent image for the dummy signal. If this is 
    changed, all tests using this signal will fail since they
    compare the output from methods using this signal (as input) 
    to hard-coded outputs."""
    yield np.array(
        [
            121,  99, 112,  69, 246,  88, 190,  52,  21,  19, 200, 193, 151,
            242, 204,  15, 155,   0, 148, 211,  27, 107, 250, 201, 250, 198,
            164, 237, 132,  18, 142,  12,  97,  79, 119,  76,  18, 188, 178,
            217,  81, 164, 240,  16, 108, 148, 182, 127,  88, 137, 139, 199,
            101,   4,  96,  21, 201,  55, 196,  75, 106,  11, 221, 227, 215,
            236,  70, 119,   8, 136, 166, 131,
         ], dtype = np.uint8
     ).reshape(6,3,4)


@pytest.fixture
def dummy_children() -> Generator[np.ndarray]:
    """Dummy children images for the dummy signal. If this is 
    changed, all tests using this signal will fail since they
    compare the output from methods using this signal (as input) 
    to hard-coded outputs."""
    yield np.array(
        [
            121,  99, 121,  99, 121,  99, 148, 211, 250, 201,   0,   0, 250,
            198, 132,  18,  97,  79, 178, 217, 240,  16,   0,   0, 196,  75,
              0,   0,   0,   0, 106,  11, 106,  11, 106,  11
        ], 
        dtype=np.uint8
    ).reshape((6, 3, 2))


@pytest.fixture
def dummy_images(
    dummy_parent: np.ndarray,
    dummy_children: np.ndarray,
) -> Generator[list]:
    """Dummy children images for the dummy signal. If this is 
    changed, all tests using this signal will fail since they
    compare the output from methods using this signal (as input) 
    to hard-coded outputs."""
    from microspy.signals._microspy_signals2D import (
        MicroSpySignal2D, 
        MicroSpySignal2D_Parent,
        Images_signals
    )

    sig_type = "ParentSig"
    pmd = {
        "metadata" : {
            "General" : {"title" : Images_signals.get(sig_type)},
            "Signal" : {"signal_type" : sig_type}
        }
    }

    sig_type = "ChildSig"
    cmd = {
        "metadata" : {
            "General" : {"title" : Images_signals.get(sig_type)},
            "Signal" : {"signal_type" : sig_type}
        }
    }

    dummy_images = [
        [
            MicroSpySignal2D_Parent(dummy_parent, **pmd),
            MicroSpySignal2D(dummy_children, **cmd),
        ]
    ]
    
    yield dummy_images