# Demodata

Five videos of anonimised participants 

## File organisation
```
mx19@sie133-lap:~/datasets/echocardiography/videos-echo-annotated-05-subjects$ tree
.
├── 01NVb-003-042
│   ├── T1
│   │   ├── 01nvb-003-042-1-4cv.json
│   │   └── 01NVb-003-042-1 echo.mp4
│   ├── T2
│   │   ├── 01nvb-003-042-2-4cv.json
│   │   └── 01NVb-003-042-2 echo.mp4
│   └── T3
│       ├── 01nvb-003-042-3-4cv.json
│       └── 01NVb-003-042-3 echo.mp4
├── 01NVb-003-043
│   ├── T1
│   │   ├── 01nvb-003-043-1-4cv.json
│   │   ├── 01NVb-003-043-1 echo.mp4
│   │   ├── 01nvb-003-043-2-4cv.json
│   │   └── 01NVb-003-043-2 echo.mp4
│   ├── T2
│   └── T3
│       ├── 01nvb-003-043-3-4cv.json
│       └── 01NVb-003-043-3 echo.mp4
├── 01NVb-003-044
│   ├── T1
│   │   ├── 01nvb-003-044-2-4cv.json
│   │   └── 01NVb-003-044-2 echo.mp4
│   ├── T2
│   └── T3
│       ├── 01nvb-003-044-3-4cv.json
│       └── 01NVb-003-044-3 echo.mp4
├── 01NVb-003-045
│   ├── T1
│   │   ├── 01nvb-003-045-1-4cv.json
│   │   └── 01NVb-003-045-1 echo.mp4
│   ├── T2
│   │   ├── 01nvb-003-045-2-4cv.json
│   │   └── 01NVB-003-045-2 echo.mp4
│   └── T3
│       ├── 01nvb-003-045-3-4cv.json
│       └── 01NVb-003-045-3 echo.mp4
├── 01NVb-003-046
│   ├── T1
│   │   ├── 01nvb-003-046-1-4cv.json
│   │   └── 01NVb-003-046-1 echo.mp4
│   ├── T2
│   │   ├── 01nvb-003-046-2-4cv.json
│   │   └── 01NVb-003-046-2 echo.mp4
│   └── T3
│       ├── 01nvb-003-046-3-4cv.json
│       └── 01NVb-003-046-3 echo.mp4
└── 01NVb-003-047
    ├── T1
    │   ├── 01nvb-003-047-1-4cv.json
    │   └── 01NVb-003-047-1 echo.mp4
    ├── T2
    │   ├── 01nvb-003-047-2-4cv.json
    │   └── 01NVb-003-047-2 echo.mp4
    └── T3
        ├── 01nvb-003-047-3-4cv.json
        └── 01NVb-003-047-3 echo.mp4

24 directories, 34 files
``` 

## Demographics of six participants 
``` 
EVENT,USUBJID,USUBJID,LABELLED,VIDEO_PATH,VIDEOS,4CV_CLIPS_T1,4CV_CLIPS_T2,4CV_CLIPS_T3,STUDYID,SUBJID,ADMDTC,ICUDTC,ICUTIME,HYPERTENSION,CHRONIC,MILDLIVER,MODERATELIVER,MODERATEKIDNEY,DIABETES,DIABETESCHRONIC,SMOKER,SMOKERDAY,TAKINGMED,TAKINGHERBAL,WEIGHT,HEIGHT,POINTCARE,LABORATORY,YEAROFBIRTH,AGE,SEX,GROUP1,SEPSIS,DENGUE,TETANUS,GROUP2,TETANUSAG,ADMITTEDICU,DAYSOFICU
All,003-042,003-042,Y,YES,YES,1,1,3,01NVb,42,01/10/2020 00:00:00,01/10/2020 00:00:00,01/01/1900 16:40:00,Y,N,N,N,N,Y,N,N,,N,Y,67,167,True,False,1976,44,M,True,False,True,False,False,False,False,8
All,003-043,003-043,Y,YES,YES,1,2,2,01NVb,43,06/10/2020 00:00:00,,,N,N,N,Y,N,Y,N,N,,Y,N,114,178,True,False,2004,16,M,True,False,True,False,False,False,False,0
All,003-044,003-044,Y,YES,YES,0,3,1,01NVb,44,07/10/2020 00:00:00,07/10/2020 00:00:00,01/01/1900 05:55:00,N,N,N,N,N,Y,N,N,,Y,N,78,173,True,False,1956,64,M,True,True,False,False,False,False,False,0
All,003-045,003-045,Y,YES,YES,2,1,3,01NVb,45,07/10/2020 00:00:00,07/10/2020 00:00:00,01/01/1900 17:05:00,N,N,Y,N,N,Y,N,N,,N,N,62,157,True,False,2003,17,F,True,False,True,False,False,False,False,2
All,003-046,003-046,Y,YES,YES,2,2,2,01NVb,46,12/10/2020 00:00:00,,,N,N,N,N,N,Y,N,N,,Y,N,42,153,False,True,1964,56,F,True,True,False,False,False,False,False,0
All,003-047,003-047,Y,YES,YES,3,2,2,01NVb,47,11/10/2020 00:00:00,11/10/2020 00:00:00,01/01/1900 21:30:00,N,N,Y,N,N,N,N,N,,N,N,53,150,True,False,1975,45,F,True,True,False,False,False,False,False,14
```