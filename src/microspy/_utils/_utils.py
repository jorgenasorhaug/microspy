import numpy as np
from hyperspy.api import model
from src import _errors
from exspy import material
from hyperspy.signals import Signal1D
import warnings

element_dict = material.elements.as_dictionary()

numpy_image_datatypes = [
    np.bool_, np.byte, np.ubyte, 
    np.int_, np.int8, np.int16, np.int32, np.int64,
    np.uint, np.uint8, np.uint16, np.uint32, np.uint64,
    np.float16, np.float32, np.float64
]

