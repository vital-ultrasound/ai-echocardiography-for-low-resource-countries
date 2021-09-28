# flake8: noqa

import sys
import platform
import torch
import numpy

print('Platform:  ', platform.platform())
print('PyTorch:   ', torch.__version__)
print('NumPy:     ', numpy.__version__)
print('Python:    ', sys.version)
