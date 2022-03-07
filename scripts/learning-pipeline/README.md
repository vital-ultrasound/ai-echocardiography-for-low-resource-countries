# Learning pipeline
## Introduction 
The learning pipeline for the echochardiography datasets will be based in the following elements: 
Data-selection and management; Model training and tuning; Model validation (performance evaluation and clinical evaluation); AI-based device modification, and (perhaps) AI-based production model.
See Figure 1 that illustrates [Good Machine Learning Practices](https://www.fda.gov/media/122535/download).  
![fig](../../figures/fig2-good-ml-dl-practices.png)   
_Fig 1. Total product lifecycle (TPLC) approach on AI/ML workflow from [Good Machine Learning Practices](https://www.fda.gov/media/122535/download)_

## 1. Generate list txt files for train / validate sets

### [split_train_validate_test.py](split_train_validate_test.py)
Open a terminal and load your conda environment 
```
cd $HOME/repositories/echocardiography/scripts/learning-pipeline
export PYTHONPATH=$HOME/repositories/echocardiography/ #set PYTHONPATH environment variable
conda activate rt-ai-echo-VE
python split_train_validate_test.py --config ../config_files/users_paths_files/config_users_paths_files_username_$USER.yml 
```
Edit [config_users_paths_files_username_$USER.yml](../config_files/users_paths_files/config_users_paths_files_username_template.yml) with the right paths and percentage of `ntraining`:  
```
echodataset_path: !join [*HOME_DIR, /datasets/vital-us/echocardiography/videos-echo-test]
data_list_output_path: !join [*HOME_DIR, /repositories/echocardiography/scripts/config_files/data_lists/]
ntraining: 0.5
```

## 2. Learning pipeline scripts 
### [learning_pipeline_notebook.ipynb](learning_pipeline_notebook.ipynb)
* Open a terminal, load your conda environment and run the script 
```
cd $HOME/repositories/echocardiography/scripts/learning-pipeline
export PYTHONPATH=$HOME/repositories/echocardiography/ #set PYTHONPATH environment variable
conda activate rt-ai-echo-VE 
jupyter notebook # to open learning_pipeline_notebook.ipynb in your web-browser
```

* Description when using echo_classes.py
``` 
* 'participant 072 with T1-01clips; T2-03clips; T3-02clips' with`echo_classes.py` generates 12 clips
* 'participant 074 - T1-02clips; T2-02clips; T3-00clips' with `echo_classes.py` generates 8 clips
* 'participant 072 with T1-01clips;' with `echo_classes.py` generate 2 clips 
* 'participant 072 with T2-03clips;' with `echo_classes.py` generate 6 clips
* 'participant 072 with T3-02clips;' with `echo_classes.py` generate 4 clips
```

* Temporal files 
`EchoClassesDataset()` creates a temporal tamp at `$HOME/tmp/echoviddata_{K}frames` where K are the `number_of_frames_per_segment_in_a_clip`.  
Example:
```
mx19@sie133-lap:~/tmp/echoviddata_10frames$ ll
total 269M
10133820 drwxrwxr-x 2 mx19 mx19 4.0K Feb  9 12:25 .
10133817 drwxrwxr-x 3 mx19 mx19 4.0K Feb  9 12:25 ..
10094468 -rw-rw-r-- 1 mx19 mx19 9.4M Feb  9 12:25 videoID_0_label_0_train.pth
10101267 -rw-rw-r-- 1 mx19 mx19 5.2M Feb  9 12:25 videoID_10_label_0_train.pth
10101269 -rw-rw-r-- 1 mx19 mx19 9.4M Feb  9 12:25 videoID_11_label_0_train.pth
10101278 -rw-rw-r-- 1 mx19 mx19 9.5M Feb  9 12:25 videoID_12_label_0_train.pth
```


### Echochardiography classes
The following figure illustrate the pipeline to create classes for background and 4CH; segments of random clips; segment sampling and frame sliding window techniques.
![fig](../../figures/classes-windowing-sampling.png)  
_Fig 2. Description of clips, videos and classes for 4CV_

