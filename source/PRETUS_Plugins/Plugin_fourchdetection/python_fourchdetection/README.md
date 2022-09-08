# python fourdetection with demos, models and utils

## Running demo/worker

1. Switch on US devices and connect USB framegrabber 
2. Activate conda VE and run demo:
``` 
cd $HOME/repositories/echocardiography/source/PRETUS_Plugins/Plugin_fourchdetection/python_fourchdetection
export PYTHONPATH=$HOME/repositories/echocardiography/ #set PYTHONPATH environment variable
conda activate pretus # conda activate VE 
# Alternatively
# conda activate rt-ai-echo-VE
python FourChDetection_demo.py --InputVideoID 2
```