#! /bin/bash
# Usage: bash anonymize_video.sh <videoname in> <videoname out> <xmin xmax ymin ymax>
# Example: bash anonymize_video.sh 01NVb_sample_short.avi 01NVb_sample_test1_anonym.mp4 300 1550 60 850

# bounding box xmin xmax ymin ymax
bb=($3 $4 $5 $6)

# Get video height and width
array=`ffprobe -v error -show_entries stream=width,height -of csv=p=0:s=x $1 | awk 'BEGIN{ FS="x"}{print $1" "$2}'`
IFS=' ' read -r -a imsize <<< "$array"
echo ${imsize[1]}

ffmpeg -y -i $1 -vf "drawbox=x=0:y=0:w=${imsize[0]}:h=${bb[2]}:color=black:t=fill, \
drawbox=x=0:y=0:w=${bb[0]}:h=${imsize[1]}:color=black:t=fill, \
drawbox=x=${bb[1]}:y=0:w=${imsize[0]}:h=${imsize[1]}:color=black:t=fill, \
drawbox=x=0:y=${bb[3]}:w=${imsize[0]}:h=${imsize[1]}:color=black:t=fill"  $2