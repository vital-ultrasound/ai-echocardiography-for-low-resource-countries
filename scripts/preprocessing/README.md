# Preprocessing Echochardiography Ultrasound data

## Cropping and Masking Datasets

### Running script
Open a terminal and load your conda environment 
```
cd $HOME/repositories/echocardiography/scripts/preprocessing
export PYTHONPATH=$HOME/repositories/echocardiography/ #set PYTHONPATH environment variable
conda activate rt-ai-echo-VE
python cropping_and_masking_4CV.py --config ../config_files/config_echodatasets.yml
jupyter notebook
```

## DICOM datasets

### convertDICOMtoAVI.py
The script converts DICOM files into AVI files.
```
conda activate ve-AICU
cd $HOME/repositories/echocardiography/scripts/preprocessing
python convertDICOMtoAVI.py --datapath $HOME/datasets/vital-us --participant_ID 01NVb-003-001
```

### DICOM Metadata 

#### DICOM file_i: 01NVb-003-001/T3/_T144314/K65EO7G2
```
Dataset.file_meta -------------------------------
(0002, 0000) File Meta Information Group Length  UL: 202
(0002, 0001) File Meta Information Version       OB: b'\x00\x01'
(0002, 0002) Media Storage SOP Class UID         UI: Ultrasound Multi-frame Image Storage
(0002, 0003) Media Storage SOP Instance UID      UI: 1.2.840.113619.2.446.364.1591368511.9.1.512
(0002, 0010) Transfer Syntax UID                 UI: JPEG Baseline (Process 1)
(0002, 0012) Implementation Class UID            UI: 1.2.840.113619.6.446
(0002, 0013) Implementation Version Name         SH: 'VENUE_302'
(0002, 0016) Source Application Entity Title     AE: 'VENGO-VGB000364'
-------------------------------------------------
(0008, 0008) Image Type                          CS: ['DERIVED', 'PRIMARY', '', '0001']
(0008, 0016) SOP Class UID                       UI: Ultrasound Multi-frame Image Storage
(0008, 0018) SOP Instance UID                    UI: 1.2.840.113619.2.446.364.1591368511.9.1.512
(0008, 0020) Study Date                          DA: '20200605'
(0008, 0021) Series Date                         DA: '20200605'
(0008, 0023) Content Date                        DA: '20200605'
(0008, 002a) Acquisition DateTime                DT: '20200605144830.000'
(0008, 0030) Study Time                          TM: '144314'
(0008, 0031) Series Time                         TM: '144314'
(0008, 0033) Content Time                        TM: '144830'
(0008, 0050) Accession Number                    SH: ''
(0008, 0060) Modality                            CS: 'US'
(0008, 0070) Manufacturer                        LO: 'GE Healthcare Ultrasound'
(0008, 0080) Institution Name                    LO: 'GE Healthcare'
(0008, 0090) Referring Physician's Name          PN: ''
(0008, 1010) Station Name                        SH: 'VENGO-VGB000364'
(0008, 1030) Study Description                   LO: 'Lung / Cons/Eff'
(0008, 1070) Operators' Name                     PN: '01NVB'
(0008, 1090) Manufacturer's Model Name           LO: 'VenueGo'
(0008, 2142) Start Trim                          IS: '1'
(0008, 2143) Stop Trim                           IS: '95'
(0008, 2144) Recommended Display Frame Rate      IS: '22'
(0010, 0010) Patient's Name                      PN: 'TVH'
(0010, 0020) Patient ID                          LO: '01nvb-003-001-3'
(0010, 0030) Patient's Birth Date                DA: '19350101'
(0010, 0032) Patient's Birth Time                TM: '000000'
(0010, 0040) Patient's Sex                       CS: 'M'
(0010, 1000) Other Patient IDs                   LO: '01nvb-003-001-3'
(0010, 1002)  Other Patient IDs Sequence  1 item(s) ---- 
   (0010, 0020) Patient ID                          LO: '01nvb-003-001-3'
   (0010, 0022) Type of Patient ID                  CS: 'TEXT'
   ---------
(0018, 0040) Cine Rate                           IS: '22'
(0018, 0072) Effective Duration                  DS: '4.17255'
(0018, 1020) Software Versions                   LO: 'VenueGo:302.74.1'
(0018, 1063) Frame Time                          DS: '44.3889'
(0018, 1066) Frame Delay                         DS: '0.0'
(0018, 1088) Heart Rate                          IS: '-1'
(0018, 1242) Actual Frame Duration               IS: '44'
(0018, 1244) Preferred Playback Sequencing       US: 0
(0018, 5020) Processing Function                 LO: 'Lung / Cons/Eff'
(0018, 6011)  Sequence of Ultrasound Regions  1 item(s) ---- 
   (0018, 6012) Region Spatial Format               US: 1
   (0018, 6014) Region Data Type                    US: 1
   (0018, 6016) Region Flags                        UL: 0
   (0018, 6018) Region Location Min X0              UL: 182
   (0018, 601a) Region Location Min Y0              UL: 50
   (0018, 601c) Region Location Max X1              UL: 1356
   (0018, 601e) Region Location Max Y1              UL: 838
   (0018, 6020) Reference Pixel X0                  SL: 587
   (0018, 6022) Reference Pixel Y0                  SL: 60
   (0018, 6024) Physical Units X Direction          US: 3
   (0018, 6026) Physical Units Y Direction          US: 3
   (0018, 602c) Physical Delta X                    FD: 0.01236996
   (0018, 602e) Physical Delta Y                    FD: 0.01236996
   (0018, 6030) Transducer Frequency                UL: 2000
   ---------
(0020, 000d) Study Instance UID                  UI: 1.2.840.113619.2.446.364.1591368194.1.1
(0020, 000e) Series Instance UID                 UI: 1.2.840.113619.2.446.364.1591368194.2.1
(0020, 0010) Study ID                            SH: '144314'
(0020, 0011) Series Number                       IS: '1'
(0020, 0013) Instance Number                     IS: '7'
(0020, 0020) Patient Orientation                 CS: ''
(0028, 0002) Samples per Pixel                   US: 3
(0028, 0004) Photometric Interpretation          CS: 'YBR_FULL_422'
(0028, 0006) Planar Configuration                US: 0
(0028, 0008) Number of Frames                    IS: '95'
(0028, 0009) Frame Increment Pointer             AT: (0018, 1063)
(0028, 0010) Rows                                US: 846
(0028, 0011) Columns                             US: 1538
(0028, 0014) Ultrasound Color Data Present       US: 0
(0028, 0100) Bits Allocated                      US: 8
(0028, 0101) Bits Stored                         US: 8
(0028, 0102) High Bit                            US: 7
(0028, 0103) Pixel Representation                US: 0
(0028, 0301) Burned In Annotation                CS: 'YES'
(0028, 2110) Lossy Image Compression             CS: '01'
(0028, 2112) Lossy Image Compression Ratio       DS: '95.0'
(0032, 1060) Requested Procedure Description     LO: ''
(0040, 0275)  Request Attributes Sequence  1 item(s) ---- 
   (0020, 000d) Study Instance UID                  UI: 1.2.840.113619.2.446.364.1591368194.1.1
   ---------
(6007, 0010) Private Creator                     LO: 'nhiemB'
(7fe0, 0010) Pixel Data                          OB: Array of 16337558 elements
(fffc, fffc) Data Set Trailing Padding           OB: b'\x00\x00'
```

### Reference
* David Ouyang is the author of [`ConvertDICOMtoAVI.py`](https://github.com/echonet/dynamic/blob/master/scripts/ConvertDICOMToAVI.ipynb)  
* See more https://github.com/echonet/   