import os
import warnings

use_gpu = bool(os.environ.get('USE_GPU') == '1')

if use_gpu:
    import cupy as np
    np.add.reduce = np.ElementwiseKernel(
        'float32 x, float32 y, float32 z',
        'z = (x + y)',
        'add_reduce'
    )
    np.subtract.reduce = np.ElementwiseKernel(
        'float32 x, float32 y, float32 z',
        'z = (x - y)',
        'subtract_reduce'
    )
else:
    import numpy as np
    np.seterr(divide='ignore')
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

np