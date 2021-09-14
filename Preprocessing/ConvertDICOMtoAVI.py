import os, os.path
import pydicom as dicom
import numpy as np
import cv2

def mask(output):
    dimension = output.shape[0]
    # Mask pixels outside of scanning sector
    m1, m2 = np.meshgrid(np.arange(dimension), np.arange(dimension))
    mask = ((m1 + m2) > int(dimension / 2) + int(dimension / 10))
    mask *= ((m1 - m2) < int(dimension / 2) + int(dimension / 10))
    mask = np.reshape(mask, (dimension, dimension)).astype(np.int8)
    maskedImage = cv2.bitwise_and(output, output, mask = mask)
    return maskedImage

def makeVideo(fileToProcess, destinationFolder):
    try:
        fileName = fileToProcess.split('/')[-1]  # \\ if GNU/linux OS
        # fileName = fileToProcess.split('\\')[-1] #\\ if Windows OS
        # fileName = hex(abs(hash(fileToProcess.split('/')[-1]))).upper() ##/ if on mac or sherlock

        if not os.path.isdir(os.path.join(destinationFolder,fileName)):

            cropSize = (512, 512) ## Resolution for the frames

            dataset = dicom.dcmread(fileToProcess, force=True)
            # print(dataset) # Print metadata
            print(f' {dataset[0x0008, 0x1030]}')# e.g. Study Description  LO: 'Lung / Cons/Eff'
            testarray = dataset.pixel_array

            frames,height,width,channels = testarray.shape
            print(f'  frames,height,width,channels = {testarray.shape}')

            ## yCrop mean
            frame0 = testarray[0]
            mean = np.mean(frame0, axis=1)
            mean = np.mean(mean)
            #TODO: how the yCrop mean is impacting on the file conversion?
            yCrop = np.where(mean<200)[0][0]
            testarray = testarray[:, yCrop:, :, :]

            bias = int(np.abs(testarray.shape[2] - testarray.shape[1])/2)
            if bias>0:
                if testarray.shape[1] < testarray.shape[2]:
                    testarray = testarray[:, :, bias:-bias, :]
                else:
                    testarray = testarray[:, bias:-bias, :, :]

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
                #smallOutput = outputA[int(height/10):(height - int(height/10)), int(height/10):(height - int(height/10))]
                ifactor=60 # default=10
                smallOutput = outputA[int(height / ifactor):(height - int(height / ifactor)), int(height / ifactor):(height - int(height / ifactor))]
                # Resize image
                output = cv2.resize(smallOutput, cropSize, interpolation = cv2.INTER_CUBIC)
                finaloutput = mask(output)
                finaloutput = cv2.merge([finaloutput,finaloutput,finaloutput])
                out.write(finaloutput)

            out.release()

        else:
            print(fileName,"hasAlreadyBeenProcessed")
    except:
        print("   Something failed, not sure what, have to debug", fileName)
    return 0


def main():
    print(r"""\
                      ___                ___                                     
                .-.  (   )              (   )                                    
     ___  ___  ( __)  | |_       .---.   | |               ___  ___      .--.    
    (   )(   ) (''") (   __)    / .-, \  | |              (   )(   )   /  _  \   
     | |  | |   | |   | |      (__) ; |  | |    .------.   | |  | |   . .' `. ;  
     | |  | |   | |   | | ___    .'`  |  | |   (________)  | |  | |   | '   | |  
     | |  | |   | |   | |(   )  / .'| |  | |               | |  | |   _\_`.(___) 
     | |  | |   | |   | | | |  | /  | |  | |               | |  | |  (   ). '.   
     ' '  ; '   | |   | ' | |  ; |  ; |  | |               | |  ; '   | |  `\ |  
      \ `' /    | |   ' `-' ;  ' `-'  |  | |               ' `-'  /   ; '._,' '  
       '_.'    (___)   `.__.   `.__.'_. (___)               '.__.'     '.___.'   
                                                                                                                                                           
                    """)


    ## Set paths of datasets
    ## DAY 01
    #pathToProcess = "/home/mx19/datasets/vital/01NVb-003-001/T1/_T145245/"
    #destinationFolder = "/home/mx19/datasets/vital/01NVb-003-001/T1/preprocessed-data/DICOM2AVI"
    ## DAY 02
    # pathToProcess = "/home/mx19/datasets/vital/01NVb-003-001/T2/_T145245/"
    # destinationFolder = "/home/mx19/datasets/vital/01NVb-003-001/T2/preprocessed-data/DICOM2AVI"
    # ## DAY 03
    # pathToProcess = "/home/mx19/datasets/vital/01NVb-003-001/T3/_T144314"
    # destinationFolder = "/home/mx19/datasets/vital/01NVb-003-001/T3/preprocessed-data/DICOM2AVI"

    ## TEST PATHS
    pathToProcess = "/home/mx19/datasets/vital/01NVb-003-001/T3/preprocessed-data/tests-dicoms"
    destinationFolder = "/home/mx19/datasets/vital/01NVb-003-001/T3/preprocessed-data/DICOM2AVI"

    # fileToProcess = "/home/mx19/datasets/vital/01NVb-003-001/T1/_T145245/K61FES84" # shape (846, 1538, 3), mean 85.36389250108367
    # fileToProcess = "/home/mx19/datasets/vital/01NVb-003-001/T1/_T145245/K61G7OG2"  # shape (96, 846, 1538, 3), mean 88.75089433843549

    DICOMfiles_path = os.listdir(pathToProcess)
    total_number_of_DICOMfiles = len(DICOMfiles_path)

    count = 0
    for DICOMfile_i in DICOMfiles_path:
        count += 1
        VideoPath_DICOMfile_i = os.path.join(pathToProcess, DICOMfile_i)
        print(f'* DICOMfile_i = {count}/ {total_number_of_DICOMfiles} , Path: {VideoPath_DICOMfile_i}')
        if not os.path.exists(os.path.join(destinationFolder, DICOMfile_i + ".avi")):
            makeVideo(VideoPath_DICOMfile_i, destinationFolder)
        else:
             print("Already did this file", VideoPath_DICOMfile_i)

if __name__=='__main__':
    main()
