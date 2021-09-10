#!/usr/bin/env python
# coding: utf-8

import re
import os, os.path
from os.path import splitext
import pydicom as dicom
import numpy as np
from pydicom.uid import UID, generate_uid
import shutil
from multiprocessing import dummy as multiprocessing
import time
import subprocess
import datetime
from datetime import date
import sys
import cv2
#from scipy.misc import imread
import matplotlib.pyplot as plt
import sys
from shutil import copy
import math


def mask(output):
    dimension = output.shape[0]
    
    # Mask pixels outside of scanning sector
    m1, m2 = np.meshgrid(np.arange(dimension), np.arange(dimension))

    mask = ((m1+m2)>int(dimension/2) + int(dimension/10)) 
    mask *=  ((m1-m2)<int(dimension/2) + int(dimension/10))
    mask = np.reshape(mask, (dimension, dimension)).astype(np.int8)
    maskedImage = cv2.bitwise_and(output, output, mask = mask)

    return maskedImage


def makeVideo(fileToProcess, destinationFolder):
    try:
        fileName = fileToProcess.split('/')[-1]  # \\ if GNU/linux OS
        # fileName = fileToProcess.split('\\')[-1] #\\ if Windows OS
        # fileName = hex(abs(hash(fileToProcess.split('/')[-1]))).upper() ##/ if on mac or sherlock

        if not os.path.isdir(os.path.join(destinationFolder,fileName)):

            cropSize = (112, 112)

            dataset = dicom.dcmread(fileToProcess, force=True)
            testarray = dataset.pixel_array

            ## yCrop mean
            frame0 = testarray[0]
            mean = np.mean(frame0, axis=1)
            mean = np.mean(mean)
            yCrop = np.where(mean<90)[0][0]
            testarray = testarray[:, yCrop:, :, :]

            bias = int(np.abs(testarray.shape[2] - testarray.shape[1])/2)
            if bias>0:
                if testarray.shape[1] < testarray.shape[2]:
                    testarray = testarray[:, :, bias:-bias, :]
                else:
                    testarray = testarray[:, bias:-bias, :, :]

            print(testarray.shape)
            frames,height,width,channels = testarray.shape

            fps = 30

            try:
                fps = dataset[(0x18, 0x40)].value
            except:
                print("couldn't find frame rate, default to 30")

            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            video_filename = os.path.join(destinationFolder, fileName + '.avi')
            out = cv2.VideoWriter(video_filename, fourcc, fps, cropSize)


            for i in range(frames):

                outputA = testarray[i,:,:,0]
                smallOutput = outputA[int(height/10):(height - int(height/10)), int(height/10):(height - int(height/10))]

                # Resize image
                output = cv2.resize(smallOutput, cropSize, interpolation = cv2.INTER_CUBIC)

                finaloutput = mask(output)


                finaloutput = cv2.merge([finaloutput,finaloutput,finaloutput])
                out.write(finaloutput)

            out.release()

        else:
            print(fileName,"hasAlreadyBeenProcessed")
    except:
        print("something filed, not sure what, have to debug", fileName)
    return 0


def main():
    # destinationFolder = "C:\\Users\\huynh\\OneDrive\\Máy tính\\GEMS_IMG\\AVI"
    destinationFolder = "/home/mx19/datasets/vital/01NVb-003-001/T1/preprocessed-data/DICOM2AVI"
    # fileToProcess = "C:\\Users\\huynh\\OneDrive\\Máy tính\\GEMS_IMG\\JD141706\\a4c\\K13EA882"
    # fileToProcess = "/home/mx19/datasets/vital/01NVb-003-001/T1/_T145245/K61FES84" # shape (846, 1538, 3), mean 85.36389250108367
    fileToProcess = "/home/mx19/datasets/vital/01NVb-003-001/T1/_T145245/K61G7OG2"  # shape (96, 846, 1538, 3), mean 88.75089433843549

    makeVideo(fileToProcess, destinationFolder)

if __name__=='__main__':
    main()

    # Question: How to iterates through a folder, including subfolders,
