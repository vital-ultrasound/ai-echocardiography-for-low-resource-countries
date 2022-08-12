# Learning pipeline
## Introduction 
The learning pipeline for the echochardiography datasets is based in the following elements: 
Data-selection and management; Model training and tuning; Model validation (performance evaluation and clinical evaluation); AI-based device modification, and (perhaps) AI-based production model (Fig. 1).   
![fig](../../docs/figures/fig2-good-ml-dl-practices.png)   
_Fig 1. Total product lifecycle (TPLC) approach on AI/ML workflow from [Good Machine Learning Practices](https://www.fda.gov/media/122535/download)_

## 1. Setting up your datasets and labels
Prepare your mp4 files and annotations files as suggested [here](../curation-selection-validation). 
It is suggested that files follow this organisation and localisation.
**NOTE**. extra video files can be renamed with a different extension to avoid taking them into account.

<details>
  <summary>Click to expand and see files organisation and location! </summary>
  
```
$ cd /media/mx19/vitaluskcl/datasets/echocardiography/videos-echo-annotated-ALL
$ tree -s
.
├── [       4096]  01NVb-003-050
│   ├── [       4096]  T1
│   │   ├── [        986]  01NVb-003-050-1-4CV.json
│   │   └── [ 1803334463]  01NVb-003-050-1 echo.mp4
│   ├── [       4096]  T2
│   │   ├── [        988]  01NVb-003-050-2-4CV.json
│   │   └── [ 1752445210]  01NVb-003-050-2 echo.mp4
│   └── [       4096]  T3
│       ├── [        987]  01NVb-003-050-3-4CV.json
│       └── [ 1062609410]  01NVb-003-050-3 echo.mp4
├── [       4096]  01NVb-003-051
│   ├── [       4096]  T1
│   │   ├── [        986]  01NVb-003-051-1-4CV.json
│   │   └── [  826247505]  01NVb-003-051-1 echo.mp4
│   ├── [       4096]  T2
│   │   ├── [        988]  01NVb-003-051-2-4CV.json
│   │   └── [ 1234164657]  01NVb-003-051-2 echo.mp4
│   └── [       4096]  T3
│       ├── [        906]  01NVb-003-051-3-4CV.json
│       └── [ 1198707159]  01NVb-003-051-3 echo.mp4
```
</details> 

See more details on data curation [here](../../data)

## 2. Learning pipeline notebook [:notebook:](learning_pipeline.py)
The jupyter nobebook [learning_pipeline_notebook.ipynb](learning_pipeline_notebook.ipynb) involves pre-processing, segment sampling, model and hyperparameter tunning pipeline (Fig. 1).

![fig](../../docs/figures/DL-pipeline.png)       
_**Fig 1.** Deep learning pipeline of the AI-empowered echocardiography._

* config_files/users_paths_files
See [README](../config_files/users_paths_files)

* Open a terminal, load your conda environment and run the script.
```
cd $HOME/repositories/echocardiography/scripts/learning-pipeline
export PYTHONPATH=$HOME/repositories/echocardiography/ #set PYTHONPATH environment variable
conda activate rt-ai-echo-VE 
jupyter notebook # to open *.ipynb in your web-browser
```

* Temporal files
`EchoClassesDataset()` creates a temporal path at 
`$HOME/datasets/vital-us/echocardiography/temporal-files/echovideodatafiles_FRAMESPERCLIP{$K}_PIXELSIZE_{$NW}W{$NH}H` 
where K are the `number_of_frames_per_segment_in_a_clip` and `{$NW}` and `{$NH}` are pixel size of the ultrasound image.

Example:
```
mx19@sie133-lap:~/datasets/vital-us/echocardiography/temporal-files/echovideodatafiles_FRAMESPERCLIP500_PIXELSIZE_200W200H$ ll
total 1.9G
13370667 drwxrwxr-x 2 mx19 mx19  12K May 16 13:50 .
12599341 drwxrwxr-x 3 mx19 mx19 4.0K May 16 13:48 ..
13373904 -rw-rw-r-- 1 mx19 mx19  24M May 16 13:49 videoID_00_040-1_label_00.pth
13374254 -rw-rw-r-- 1 mx19 mx19  23M May 16 13:49 videoID_00_043-2_label_00.pth
13373934 -rw-rw-r-- 1 mx19 mx19  23M May 16 13:49 videoID_01_040-2_label_00.pth
13374255 -rw-rw-r-- 1 mx19 mx19  23M May 16 13:49 videoID_01_041-3_label_00.pth
13373958 -rw-rw-r-- 1 mx19 mx19  23M May 16 13:49 videoID_02_040-1_label_00.pth
```

## 3. Heuristics for learning pipeline [:notebook:](heuristics_learning_pipeline_notebook.ipynb)
* Open a terminal, load your conda environment and run the script.
```
cd $HOME/repositories/echocardiography/scripts/learning-pipeline
export PYTHONPATH=$HOME/repositories/echocardiography/ #set PYTHONPATH environment variable
conda activate rt-ai-echo-VE 
jupyter notebook # to open *.ipynb in your web-browser
```

## 4. Evaluation of learning pipeline [:notebook:](evaluation_of_learning_pipeline_notebook.ipynb)
* Open a terminal, load your conda environment and run the script.
```
cd $HOME/repositories/echocardiography/scripts/learning-pipeline
export PYTHONPATH=$HOME/repositories/echocardiography/ #set PYTHONPATH environment variable
conda activate rt-ai-echo-VE 
jupyter notebook # to open *.ipynb in your web-browser
```