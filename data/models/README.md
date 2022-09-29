# Models
Path to save temporal files of trained models (e.g. VGG-based models).

See the following terminal log showing few example:
``` 
mx19@sie133-lap:~/repositories/echocardiography/data/models$ tree -s
.
├── [ 2215719935]  basicVGG2D_04layers_model_trained_with_05subjects_and_BATCH_SIZE_OF_CLIPS_10.pth
├── [  167776863]  metric_model_old_version.pth
├── [    3114843]  metric_model_SqueezeNet_source0_for_06-subjects_with_NUMBER_OF_FRAMES_30BATCH_SIZE_OF_CLIPS30EPOCHS_500_train00.pth

## 05-subjects 
├── [    3114843]  metric_model_SqueezeNet_source0_for_05-subjects_with_NUMBER_OF_FRAMES_30BATCH_SIZE_OF_CLIPS30EPOCHS_500_train00.pth
├── [    3114843]  metric_model_SqueezeNet_source0_for_05-subjects_with_NUMBER_OF_FRAMES_30BATCH_SIZE_OF_CLIPS30EPOCHS_500_train01.pth
├── [    3114843]  metric_model_SqueezeNet_source0_for_05-subjects_with_NUMBER_OF_FRAMES_30BATCH_SIZE_OF_CLIPS30EPOCHS_500_train02.pth

## 31-subjects
├── [    3014619]  metric_model_SqueezeNet_source0_for_31-subjects_with_NUMBER_OF_FRAMES_01BATCH_SIZE_OF_CLIPS01EPOCHS_500_train00.pth
├── [    3014619]  metric_model_SqueezeNet_source0_for_31-subjects_with_NUMBER_OF_FRAMES_01BATCH_SIZE_OF_CLIPS01EPOCHS_500_train01.pth
├── [    3014619]  metric_model_SqueezeNet_source0_for_31-subjects_with_NUMBER_OF_FRAMES_01BATCH_SIZE_OF_CLIPS01EPOCHS_500_train02.pth
├── [    3014619]  metric_model_SqueezeNet_source0_for_31-subjects_with_NUMBER_OF_FRAMES_01BATCH_SIZE_OF_CLIPS30EPOCHS_500_train00.pth
├── [    3014619]  metric_model_SqueezeNet_source0_for_31-subjects_with_NUMBER_OF_FRAMES_01BATCH_SIZE_OF_CLIPS30EPOCHS_500_train01.pth
├── [    3014619]  metric_model_SqueezeNet_source0_for_31-subjects_with_NUMBER_OF_FRAMES_01BATCH_SIZE_OF_CLIPS30EPOCHS_500_train02.pth
├── [    3114843]  metric_model_SqueezeNet_source0_for_31-subjects_with_NUMBER_OF_FRAMES_30BATCH_SIZE_OF_CLIPS01EPOCHS_500_train00.pth
├── [    3114843]  metric_model_SqueezeNet_source0_for_31-subjects_with_NUMBER_OF_FRAMES_30BATCH_SIZE_OF_CLIPS01EPOCHS_500_train01.pth
├── [    3114843]  metric_model_SqueezeNet_source0_for_31-subjects_with_NUMBER_OF_FRAMES_30BATCH_SIZE_OF_CLIPS01EPOCHS_500_train02.pth
├── [    3114843]  metric_model_SqueezeNet_source0_for_31-subjects_with_NUMBER_OF_FRAMES_30BATCH_SIZE_OF_CLIPS30EPOCHS_500_train00.pth
├── [    3114843]  metric_model_SqueezeNet_source0_for_31-subjects_with_NUMBER_OF_FRAMES_30BATCH_SIZE_OF_CLIPS30EPOCHS_500_train01.pth
├── [    3114843]  metric_model_SqueezeNet_source0_for_31-subjects_with_NUMBER_OF_FRAMES_30BATCH_SIZE_OF_CLIPS30EPOCHS_500_train02.pth

## others
├── [    3014619]  metric_model_SqueezeNet_source0_NUMBER_OF_FRAMES_01BATCH_SIZE_OF_CLIPS01EPOCHS_500_train00.pth
├── [    3014619]  metric_model_SqueezeNet_source0_NUMBER_OF_FRAMES_01BATCH_SIZE_OF_CLIPS01EPOCHS_500_train01.pth
├── [    3014619]  metric_model_SqueezeNet_source0_NUMBER_OF_FRAMES_01BATCH_SIZE_OF_CLIPS01EPOCHS_500_train02.pth
```
