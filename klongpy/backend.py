import os
import warnings

use_gpu = bool(os.environ.get('USE_GPU') == '1')

if use_gpu:
    import cupy as np
else:
    import numpy as np
    np.seterr(divide='ignore')
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

np