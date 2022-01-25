import os
import random
from pathlib import Path

# To use, just replace this with your local root folder
root = '/home/ag09/data/VITAL/echo/'

ntraining = 0.8
nvalidation = 1 - ntraining


all_videos_file = 'video_list_full.txt'
all_labels_file = 'annotation_list_full.txt'

# ----------------------------------

imagelist = '{}{}'.format(root, all_videos_file)
labellist = '{}{}'.format(root, all_labels_file)

# list all  files
result = list(Path(root+'train_validate/').rglob("*echo.[mM][pP][4]"))
with open(imagelist, 'w') as f:
    for fn in result:
        fn_nopath = str(fn).replace(root, '')
        f.write(fn_nopath+'\n')

result = list(Path(root+'train_validate/').rglob("*4CV.[jJ][sS][oO][nN]"))
with open(labellist, 'w') as f:
    for fn in result:
        fn_nopath = str(fn).replace(root, '')
        f.write(fn_nopath+'\n')



# load filenames into list

image_filenames = [line.strip() for line in open(imagelist)]
label_filenames = [line.strip() for line in open(labellist)]

#randomly shuffle them
c = list(zip(image_filenames, label_filenames))
random.shuffle(c)
image_filenames, label_filenames = zip(*c)

# now split and save
N = len(image_filenames)
image_filenames_t = image_filenames[:int(N*ntraining)]
label_filenames_t = label_filenames[:int(N*ntraining)]
image_filenames_v = image_filenames[int(N*ntraining):]
label_filenames_v = label_filenames[int(N*ntraining):]

def write_to_file(l, filename, root):
    textfile = open('{}{}'.format(root, filename), "w")
    for element in l:
        textfile. write(element + "\n")

write_to_file(image_filenames_t, 'video_list_train.txt', root)
write_to_file(label_filenames_t, 'annotation_list_train.txt', root)
write_to_file(image_filenames_v, 'video_list_validate.txt', root)
write_to_file(label_filenames_v, 'annotation_list_validate.txt', root)