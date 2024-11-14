import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo'))

from exp_methods.demo.src.model.model_forecast import ModelForecast as DeMo
