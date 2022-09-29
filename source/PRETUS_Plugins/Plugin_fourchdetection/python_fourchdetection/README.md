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
python FourChDetection_demo.py --InputVideoID 2 --modelfilename metric_model_SqueezeNet_source0_for_06-subjects_with_NUMBER_OF_FRAMES_30BATCH_SIZE_OF_CLIPS30EPOCHS_500_train00.pth
python FourChDetection_demo.py --InputVideoID ~/datasets/echocardiography-vital/videos-echo-annotated-06-subjects/01NVb-003-042/T1/01NVb-003-042-1-echo.mp4 --modelfilename metric_model_SqueezeNet_source0_for_06-subjects_with_NUMBER_OF_FRAMES_30BATCH_SIZE_OF_CLIPS30EPOCHS_500_train00.pth
```
