import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qcnet'))

from exp_methods.qcnet.predictors.qcnet import QCNet
