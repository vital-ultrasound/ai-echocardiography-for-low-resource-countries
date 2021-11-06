# Pytorch dataloaders for echo videos

A Pytorch dataloader to preprocess and serve data samples extracted from videos using annotations in json format, as generated with the VGG annotator tool.

# Usage: 

The dataloader takes as input a list of the videos, and a list of annotations, in two text files. As follows:

```
import datasets.dataloaders.EchocardiographicVideoDataset as EchoDatasets
...
dataset = EchoDatasets.EchoViewVideoDataset(
            root='/home/ag09/data/VITAL/echo', 
            video_list_file='video_list.txt', 
            annotation_list_file='annotation_list.txt')
```

So the files `video_list.txt` and `annotation_list.txt` should be located in the root folder.

The contents of those files should be, for example, as follows:

`video_list.txt`

```
01NVb-003-072/T1/01NVb-003-072-1 echo.mp4
01NVb-003-072/T2/01NVb-003-072-2 echo cont.mp4
01NVb-003-072/T3/01NVb-003-072-3 echo.mp4
```

`annotation_list.txt`

```
01NVb-003-072/T1/01NVb_003_072_T1_4CV.json
01NVb-003-072/T2/01NVb_003_072_T2_4CV.json
01NVb-003-072/T3/01NVb_003_072_T3_4CV.json

```

Containing the paths to the files within the root folder.