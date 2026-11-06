"""Strict output conversion: preserve values, reject nonfinite/complex data."""
import json

import numpy as np


def numpy_default(value):
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(f'Unsupported JSON output type: {type(value).__name__}')


def dumps(value,**kwargs):
    return json.dumps(value,default=numpy_default,allow_nan=False,**kwargs)
