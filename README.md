# YOLOv7_Inference
Create a yolov7 model detecting cars in images, using model YOLOv7

## Preparation
To run it, you need to download the weight (and a sample image incase you need it) by:
``` 
pip install 'dvc[gdrive]' && dvc pull
```

## How to run
- To test the model, run:
``` 
python test.py
```

- To run it on MLChain server, run:
``` 
mlchain run -c mlconfig.yaml 
```
An API server will be hosted at http://0.0.0.0:8001

---
If you have any question or encouter any problem regarding this repo. Please open an issue and cc me. Thank you.

