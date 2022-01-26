# Pytorch dataloaders for echochardiography datasets
A Pytorch dataloader to preprocess and serve data samples extracted from videos using annotations in json format.


## Generate list txt files for train / validate sets

### [split_train_validate_test.py](split_train_validate_test.py)
Open a terminal and load your conda environment 
```
cd $HOME/repositories/echocardiography/scripts/dataloaders
export PYTHONPATH=$HOME/repositories/echocardiography/ #set PYTHONPATH environment variable
conda activate rt-ai-echo-VE 
python split_train_validate_test.py --echodataset_path $HOME/datasets/vital-us/echocardiography/videos-echo-test/ --data_list_output_path $HOME/repositories/echocardiography/scripts/config_files/data_lists/ --ntraining 0.8
```

Then, text files looks like as follows:
```
../config_files/data_lists/annotation_list_full.txt
../config_files/data_lists/annotation_list_train.txt
../config_files/data_lists/video_list_full.txt
../config_files/data_lists/video_list_train.txt
```

## Dataloaders

### [echo_classes.py](echo_classes.py)
Open a terminal and load your conda environment 
```
cd $HOME/repositories/echocardiography/scripts/dataloaders
export PYTHONPATH=$HOME/repositories/echocardiography/ #set PYTHONPATH environment variable
conda activate rt-ai-echo-VE
python echo_classes.py --config ../config_files/config_echo_classes.yml
```


### dataloader_4CV.py
Open a terminal and load your conda environment 
```
cd $HOME/repositories/echocardiography/scripts/dataloaders
export PYTHONPATH=$HOME/repositories/echocardiography/ #set PYTHONPATH environment variable
conda activate rt-ai-echo-VE 
python dataloader_4CV.py --config ../config_files/config_4cv.yml
```

### Setting config file 
For datapaths of other users, you can edit ../config_files/config_4cv.yml and add respective participant video path and json files path. 
``` 
### Datapaths
#### 01NVb-003-072
participant_videos_path: '/home/mx19/datasets/vital-us/echocardiography/videos-echo/01NVb-003-072'
participant_path_json_files: '/home/mx19/datasets/vital-us/echocardiography/json/01NVb-003-072'
```

### Jupyter Notebooks
Open a terminal and load your conda environment 
```
cd $HOME/repositories/echocardiography/scripts/dataloaders
export PYTHONPATH=$HOME/repositories/echocardiography/ #set PYTHONPATH environment variable
conda activate rt-ai-echo-VE 
jupyter notebook
```


