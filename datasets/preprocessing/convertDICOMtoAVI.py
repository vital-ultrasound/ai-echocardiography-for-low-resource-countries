import os
import pydicom as dicom
import numpy as np
import cv2
import argparse

def asciiart():
    print(r"""

                                                           _..._                 .-'''-.     
                                                        .-'_..._''.             '   _    \   
             .--.                     __.....__       .' .'      '.\  .       /   /` '.   \  
             |__|                 .-''         '.    / .'           .'|      .   |     \  '  
             .--.                /     .-''"'-.  `. . '            <  |      |   '      |  ' 
        __   |  | ,.----------. /     /________\   \| |             | |      \    \     / /  
     .:--.'. |  |//            \|                  || |             | | .'''-.`.   ` ..' /   
    / |   \ ||  |\\            /\    .-------------'. '             | |/.'''. \  '-...-'`    
    `" __ | ||  | `'----------'  \    '-.____...---. \ '.          .|  /    | |              
     .'.''| ||__|                 `.             .'   '. `._____.-'/| |     | |              
    / /   | |_                      `''-...... -'       `-.______ / | |     | |              
    \ \._,\ '/                                                   `  | '.    | '.             
     `--'  `"                                                       '---'   '---'            


                    """)
    # Reference: https://patorjk.com/software/taag/#p=testall&f=JS%20Stick%20Letters&t=ai-echo%0A

def mask(output):
    dimension = output.shape[0]
    # Mask pixels outside of scanning sector
    m1, m2 = np.meshgrid(np.arange(dimension), np.arange(dimension))
    mask = ((m1 + m2) > int(dimension / 2) + int(dimension / 10))
    mask *= ((m1 - m2) < int(dimension / 2) + int(dimension / 10))
    mask = np.reshape(mask, (dimension, dimension)).astype(np.int8)
    maskedImage = cv2.bitwise_and(output, output, mask = mask)
    return maskedImage

def ycrop_mean(testarray):
    ## yCrop mean
    # TODO: how the yCrop mean is impacting on the file conversion?
    frame0 = testarray[0]
    mean = np.mean(frame0, axis=1)
    mean = np.mean(mean)

    yCrop = np.where(mean<200)[0][0]
    testarray = testarray[:, yCrop:, :, :]

    bias = int(np.abs(testarray.shape[2] - testarray.shape[1])/2)
    if bias>0:
        if testarray.shape[1] < testarray.shape[2]:
            testarray = testarray[:, :, bias:-bias, :]
        else:
            testarray = testarray[:, bias:-bias, :, :]

    return testarray

def makeVideo(fileToProcess, destinationFolder, cropSize):
    try:
        fileName = fileToProcess.split('/')[-1]  # \\ if GNU/linux OS
        # fileName = fileToProcess.split('\\')[-1] #\\ if Windows OS
        # fileName = hex(abs(hash(fileToProcess.split('/')[-1]))).upper() ##/ if on mac or sherlock

        if not os.path.isdir(os.path.join(destinationFolder,fileName)):

            dataset = dicom.dcmread(fileToProcess, force=True)
            # print(dataset) # Print metadata
            print(f' {dataset[0x0008, 0x1030]}')# e.g. Study Description  LO: 'Lung / Cons/Eff'
            testarray = dataset.pixel_array

            frames,height,width,channels = testarray.shape
            print(f'  frames,height,width,channels = {testarray.shape}')

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
                smallOutput = outputA[0:int(height), 0:int(width)] #outputA[0:846, 0:1538]

                # # # Resize image
                output = cv2.resize(smallOutput, cropSize, interpolation=cv2.INTER_CUBIC)
                # output = mask(output) # With mask
                out.write(cv2.merge([output, output,output]))


            out.release()

        else:
            print(fileName,"hasAlreadyBeenProcessed")
    except:
        print("   :warning: Something failed, not sure what, have to debug", fileName)
    return 0


def main():
    asciiart()
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--homedatapath', required=True, help='Specify the dataset path.')
    args = parser.parse_args()

    ## Data Paths
    pathToProcess = os.path.join(args.homedatapath, 'raw-datasets/01NVb-003-001/T1/_T145245')
    destinationFolder = os.path.join(args.homedatapath, 'preprocessed-datasets/01NVb-003-001/T1')

    DICOMfiles_path = os.listdir(pathToProcess)
    total_number_of_DICOMfiles = len(DICOMfiles_path)

    cropSize = (1538, 846)  ## Frame Resolution

    count = 0
    for DICOMfile_i in DICOMfiles_path:
        count += 1
        VideoPath_DICOMfile_i = os.path.join(pathToProcess, DICOMfile_i)
        print(f' + DICOMfile_i (', {DICOMfile_i}, '=' , {count} , '/' , {total_number_of_DICOMfiles})
        if not os.path.exists(os.path.join(destinationFolder, DICOMfile_i + ".avi")):
            makeVideo(VideoPath_DICOMfile_i, destinationFolder, cropSize)
        else:
             print("  :warning: Already did this file", DICOMfile_i)

if __name__=='__main__':
    main()
