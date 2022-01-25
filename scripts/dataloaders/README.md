# Pytorch dataloaders for echochardiography datasets
A Pytorch dataloader to preprocess and serve data samples extracted from videos using annotations in json format.


# Generate train / validate sets

Use the script [split_train_validate_test.py](split_train_validate_test.py), which generates the following files:
```commandline
<data folder>/annotation_list_full.txt
<data folder>/annotation_list_train.txt
<data folder>/annotation_list_validate.txt
<data folder>/video_list_full.txt
<data folder>/video_list_train.txt
<data folder>/video_list_validate.txt
```


## dataloader_4CV.py
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


## echo_classes.py
Open a terminal and load your conda environment 
```
cd $HOME/repositories/echocardiography/scripts/dataloaders
export PYTHONPATH=$HOME/repositories/echocardiography/ #set PYTHONPATH environment variable
conda activate rt-ai-echo-VE
python echo_classes.py --config ../config_files/config_echo_classes.yml
jupyter notebook
```
