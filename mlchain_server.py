"""
THE BASE MLCHAIN SERVER 
"""
# Import mlchain 
from mlchain.base import ServeModel
from mlchain import mlconfig 


# IMPORT YOUR CLASS HERE - YOU ONLY CARE THIS
from infer import CarDetection # Import your class here 

model = CarDetection() # Init your class first 
# END YOUR WORK HERE


# Wrap your class by mlchain ServeModel
serve_model = ServeModel(model)

# THEN GO TO CONSOLE: 
# mlchain run -c mlconfig.yaml 