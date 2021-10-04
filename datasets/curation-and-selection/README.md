# Curation and selection of US imaging datasets

## `video2sliding-video.py`
Run the script with the use of your virtual environment
```
conda activate ve-AICO 
cd $HOME/vital-us/echocardiography/datasets/curation-and-selection
python video_to_sliding_video.py --videofile_in $HOME/datasets/vital-us/raw-datasets/01NVb-003-001/T1/01NVb-003-001-echo.mp4 --videofile_out $HOME/datasets/vital-us/preprocessed-datasets/tmp/01NVb-003-001-echo-sliced.mp4 --bounds 100 100  
```


## `video_channel_measurement.py`
This script helps identify good pairs of images/labels and save them to a folder.
``` 
conda activate ve-AICO
cd $HOME/vital-us/echocardiography/datasets/curation-and-selection
python video_channel_measurement.py --videofile_in $HOME/datasets/vital-us/raw-datasets/01NVb-003-001/T1/01NVb-003-001-echo.mp4 --image_frames_path $HOME/datasets/vital-us/preprocessed-datasets/tmp/nframes_ --bounds 331 107 1477 823
```

