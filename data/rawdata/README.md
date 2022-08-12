# Dataset path and files
## Raw Data
Raw datasets in local drive path (ScanDisk1TB):
```
cd /media/$USER/vitaluskcl/datasets/echocardiography
```
MP4 video data files were collected with [Hauppauge HD PVR Rocket Portable Stand Alone HD 1080p Video Recorder](https://www.hauppauge.co.uk/site/products/data_hdpvr-rocket.html#main) which videos have 1080P30 with H.264 video compression.

## Raw video datasets tree
``` 

mx19@sie133-lap:/media/mx19/vitaluskcl/datasets/echocardiography/videos-echo-annotated-33-subjects$ tree -s
.
├── [      32768]  01NVb-003-040
│   ├── [      32768]  T1
│   │   ├── [       1070]  01nvb-003-040-1-4cv.json
│   │   └── [  985197916]  01NVb-003-040-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [        990]  01nvb-003-040-2-4cv.json
│   │   └── [ 1378718854]  01NVb-003-040-2 echo.mp4
│   └── [      32768]  T3
├── [      32768]  01NVb-003-041
│   ├── [      32768]  T1
│   │   ├── [        990]  01nvb-003-041-1-4cv.json
│   │   └── [ 1146822377]  01NVb-003-041-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [        990]  01nvb-003-041-2-4cv.json
│   │   └── [  654004200]  01NVb-003-041-2 echo.mp4
│   └── [      32768]  T3
│       ├── [       1148]  01nvb-003-041-3-4cv.json
│       ├── [  106323225]  01NVb-003-041-3 echo cont_mp4_
│       └── [  621610199]  01NVb-003-041-3 echo.mp4
├── [      32768]  01NVb-003-042
│   ├── [      32768]  T1
│   │   ├── [        906]  01nvb-003-042-1-4cv.json
│   │   └── [ 1169134134]  01NVb-003-042-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [        906]  01nvb-003-042-2-4cv.json
│   │   └── [  725567484]  01NVb-003-042-2 echo.mp4
│   └── [      32768]  T3
│       ├── [       1072]  01nvb-003-042-3-4cv.json
│       └── [  809756791]  01NVb-003-042-3 echo.mp4
├── [      32768]  01NVb-003-043
│   ├── [      32768]  T1
│   │   ├── [        904]  01nvb-003-043-1-4cv.json
│   │   ├── [  950961689]  01NVb-003-043-1 echo.mp4
│   │   ├── [        990]  01nvb-003-043-2-4cv.json
│   │   └── [  829466615]  01NVb-003-043-2 echo.mp4
│   ├── [      32768]  T2
│   └── [      32768]  T3
│       ├── [        988]  01nvb-003-043-3-4cv.json
│       └── [  525102198]  01NVb-003-043-3 echo.mp4
├── [      32768]  01NVb-003-044
│   ├── [      32768]  T1
│   │   ├── [       1072]  01nvb-003-044-2-4cv.json
│   │   └── [  888797757]  01NVb-003-044-2 echo.mp4
│   ├── [      32768]  T2
│   └── [      32768]  T3
│       ├── [        906]  01nvb-003-044-3-4cv.json
│       └── [  659842801]  01NVb-003-044-3 echo.mp4
├── [      32768]  01NVb-003-045
│   ├── [      32768]  T1
│   │   ├── [        988]  01nvb-003-045-1-4cv.json
│   │   └── [  580984997]  01NVb-003-045-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [       1072]  01nvb-003-045-2-4cv.json
│   │   └── [  909521183]  01NVB-003-045-2 echo.mp4
│   └── [      32768]  T3
│       ├── [        904]  01nvb-003-045-3-4cv.json
│       └── [ 1020975213]  01NVb-003-045-3 echo.mp4
├── [      32768]  01NVb-003-046
│   ├── [      32768]  T1
│   │   ├── [        983]  01nvb-003-046-1-4cv.json
│   │   └── [  949499594]  01NVb-003-046-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [        988]  01nvb-003-046-2-4cv.json
│   │   └── [  824773453]  01NVb-003-046-2 echo.mp4
│   └── [      32768]  T3
│       ├── [        986]  01nvb-003-046-3-4cv.json
│       └── [  887400408]  01NVb-003-046-3 echo.mp4
├── [      32768]  01NVb-003-047
│   ├── [      32768]  T1
│   │   ├── [       1070]  01nvb-003-047-1-4cv.json
│   │   └── [ 1118539796]  01NVb-003-047-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [        988]  01nvb-003-047-2-4cv.json
│   │   └── [  928784267]  01NVb-003-047-2 echo.mp4
│   └── [      32768]  T3
│       ├── [        988]  01nvb-003-047-3-4cv.json
│       └── [  896513816]  01NVb-003-047-3 echo.mp4
├── [      32768]  01NVb-003-048
│   ├── [      32768]  T1
│   │   ├── [        988]  01nvb-003-048-1-4cv.json
│   │   └── [  929199309]  01NVb-003-048-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [        906]  01nvb-003-048-2-4cv.json
│   │   └── [  946927133]  01NVb-003-048-2 echo.mp4
│   └── [      32768]  T3
│       ├── [        986]  01nvb-003-048-3-4cv.json
│       └── [ 1681028126]  01NVb-003-048-3 echo.mp4
├── [      32768]  01NVb-003-050
│   ├── [      32768]  T1
│   │   ├── [        986]  01NVb-003-050-1-4CV.json
│   │   └── [ 1803334463]  01NVb-003-050-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [        988]  01NVb-003-050-2-4CV.json
│   │   └── [ 1752445210]  01NVb-003-050-2 echo.mp4
│   └── [      32768]  T3
│       ├── [        987]  01NVb-003-050-3-4CV.json
│       └── [ 1062609410]  01NVb-003-050-3 echo.mp4
├── [      32768]  01NVb-003-051
│   ├── [      32768]  T1
│   │   ├── [        986]  01NVb-003-051-1-4CV.json
│   │   └── [  826247505]  01NVb-003-051-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [        988]  01NVb-003-051-2-4CV.json
│   │   └── [ 1234164657]  01NVb-003-051-2 echo.mp4
│   └── [      32768]  T3
│       ├── [        906]  01NVb-003-051-3-4CV.json
│       └── [ 1198707159]  01NVb-003-051-3 echo.mp4
├── [      32768]  01NVb-003-052
│   ├── [      32768]  T1
│   │   ├── [        990]  01NVb-003-052-1-4CV.json
│   │   └── [  953257316]  01NVb-003-052-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [        906]  01NVb-003-052-2-4CV.json
│   │   ├── [  594644608]  01NVb-003-052-2 echo (2)_mp_4_
│   │   └── [ 1917434840]  01NVb-003-052-2 echo.mp4
│   └── [      32768]  T3
│       ├── [        989]  01NVb-003-052-3-4CV__json__
│       └── [  594644608]  01NVb-003-052-3 echo__mp4__
├── [      32768]  01NVb-003-053
│   ├── [      32768]  T1
│   │   ├── [       1150]  01NVb-003-053-1-4CV.json
│   │   └── [ 1437769809]  01NVb-003-053-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [        986]  01NVb-003-053-2-4CV.json
│   │   └── [ 1281841685]  01NVb-003-053-2 echo.mp4
│   └── [      32768]  T3
│       ├── [       1070]  01NVb-003-053-3-4CV.json
│       ├── [  286798819]  01NVb-003-053-3 echo cont__mp4__
│       └── [  515984189]  01NVb-003-053-3 echo.mp4
├── [      32768]  01NVb-003-054
│   ├── [      32768]  T1
│   │   ├── [        988]  01NVb-003-054-1-4CV.json
│   │   └── [  999313763]  01NVb-003-054-1 echo.mp4
│   ├── [      32768]  T2
│   └── [      32768]  T3
│       ├── [        987]  01NVb-003-054-3-4CV.json
│       └── [  948032732]  01NVb-003-054-3 echo.mp4
├── [      32768]  01NVb-003-056
│   ├── [      32768]  T1
│   ├── [      32768]  T2
│   │   ├── [        905]  01NVb-003-056-2-4CV.json
│   │   ├── [  278549244]  01NVb-003-056-2 echo cont_mp4_
│   │   └── [ 2101040630]  01NVb-003-056-2 echo.mp4
│   └── [      32768]  T3
│       ├── [        906]  01NVb-003-056-3-4CV.json
│       ├── [  370831920]  01NVb-003-056-3 echo cont_mp4_
│       └── [ 2101400984]  01NVb-003-056-3 echo.mp4
├── [      32768]  01NVb-003-057
│   ├── [      32768]  T1
│   ├── [      32768]  T2
│   │   ├── [        908]  01NVb-003-057-2-4CV.json
│   │   └── [ 2097924623]  01NVb-003-057-2 echo.mp4
│   └── [      32768]  T3
│       ├── [        994]  01NVb-003-057-3-4CV.json
│       └── [ 1488090627]  01NVb-003-057-3 echo.mp4
├── [      32768]  01NVb-003-058
│   ├── [      32768]  T1
│   │   ├── [        904]  01NVb-003-058-1-4CV.json
│   │   └── [ 1903280524]  01NVb-003-058-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [        986]  01NVb-003-058-2-4CV.json
│   │   └── [ 1093631405]  01NVb-003-058-2 echo.mp4
│   └── [      32768]  T3
│       ├── [       1068]  01NVb-003-058-3-4CV.json
│       └── [  928612373]  01NVb-003-058-3 echo.mp4
├── [      32768]  01NVb-003-060
│   ├── [      32768]  T1
│   │   ├── [        904]  01NVb-003-060-1-4CV.json
│   │   └── [  925627181]  01NVb-003-060-1 echo.mp4
│   ├── [      32768]  T2
│   │   └── [      32768]  extras
│   │       ├── [  197737394]  01NVb-003-060-2 echo cont_mp4_
│   │       └── [ 2100486583]  01NVb-003-060-2 echo_mp4_
│   └── [      32768]  T3
├── [      32768]  01NVb-003-061
│   ├── [      32768]  T1
│   ├── [      32768]  T2
│   └── [      32768]  T3
│       ├── [        904]  01NVb-003-061-3-4CV.json
│       └── [ 1536436836]  01NVb-003-061-3 echo.mp4
├── [      32768]  01NVb-003-063
│   ├── [      32768]  T1
│   │   ├── [        988]  01NVb-003-063-1-4CV.json
│   │   └── [ 1224566105]  01NVb-003-063-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [        986]  01NVb-003-063-2-4CV.json
│   │   └── [ 2099290213]  01NVb-003-063-2 echo.mp4
│   └── [      32768]  T3
│       ├── [        904]  01NVb-003-063-3-4CV.json
│       └── [  745764679]  01NVb-003-063-3 echo.mp4
├── [      32768]  01NVb-003-064
│   ├── [      32768]  T1
│   │   ├── [        990]  01NVb-003-064-1-4CV.json
│   │   └── [  873630172]  01NVb-003-064-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [       1072]  01NVb-003-064-2-4CV.json
│   │   └── [ 2011565463]  01NVb-003-064-2 echo.mp4
│   └── [      32768]  T3
├── [      32768]  01NVb-003-065
│   ├── [      32768]  T1
│   │   ├── [       1074]  01NVb-003-065-1-4CV.json
│   │   └── [  843344200]  01NVb-003-065-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [       1153]  01NVb-003-065-2-4CV.json
│   │   └── [  898836427]  01NVb-003-065-2 echo.mp4
│   └── [      32768]  T3
│       ├── [       1239]  01NVb-003-065-3-4CV.json
│       └── [  980305284]  01NVb-003-065-3 echo.mp4
├── [      32768]  01NVb-003-066
│   ├── [      32768]  T1
│   │   ├── [        906]  01NVb-003-066-1-4CV.json
│   │   └── [  963141960]  01NVb-003-066-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [        990]  01NVb-003-066-2-4CV.json
│   │   └── [  705532801]  01NVb-003-066-2 echo.mp4
│   └── [      32768]  T3
├── [      32768]  01NVb-003-068
│   ├── [      32768]  T1
│   │   ├── [        990]  01NVb-003-068-1-4CV.json
│   │   └── [ 1007127711]  01NVb-003-068-1 echo.mp4
│   ├── [      32768]  T2
│   └── [      32768]  T3
│       ├── [        998]  01NVb-003-068-3-4CV.json
│       └── [ 1104881573]  01NVb-003-068-3 echo.mp4
├── [      32768]  01NVb-003-069
│   ├── [      32768]  T1
│   │   ├── [        987]  01NVb-003-069-1-4CV.json
│   │   └── [ 1358342013]  01NVb-003-069-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [        987]  01NVb-003-069-2-4CV.json
│   │   └── [ 1783210718]  01NVb-003-069-2 echo.mp4
│   └── [      32768]  T3
│       ├── [        988]  01NVb-003-069-3-4CV.json
│       └── [ 1178134931]  01NVb-003-069-3 echo.mp4
├── [      32768]  01NVb-003-070
│   ├── [      32768]  T1
│   │   ├── [       1263]  01NVb-003-070-1-4CV.json
│   │   └── [ 1787863710]  01NVb-003-070-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [       1017]  01NVb-003-070-2-4CV.json
│   │   └── [ 2041918386]  01NVb-003-070-2 echo.mp4
│   └── [      32768]  T3
│       ├── [       1017]  01NVb-003-070-3-4CV.json
│       └── [ 1240743015]  01NVb-003-070-3 echo.mp4
├── [      32768]  01NVb-003-071
│   ├── [      32768]  T1
│   │   ├── [       1017]  01NVb-003-071-1-4CV.json
│   │   └── [ 1364777706]  01NVb-003-071-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [       1097]  01NVb-003-071-2-4CV.json
│   │   └── [ 1298512277]  01NVb-003-071-2 echo.mp4
│   └── [      32768]  T3
│       ├── [       1017]  01NVb-003-071-3-4CV.json
│       └── [ 1301733199]  01NVb-003-071-3 echo.mp4
├── [      32768]  01NVb-003-073
│   ├── [      32768]  T1
│   │   ├── [       1101]  01NVb-003-073-1-4CV.json
│   │   └── [ 1484818675]  01NVb-003-073-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [       1017]  01NVb-003-073-2-4CV.json
│   │   └── [ 1195148922]  01NVb-003-073-2 echo.mp4
│   └── [      32768]  T3
├── [      32768]  01NVb-003-074
│   ├── [      32768]  T1
│   │   ├── [       1101]  01NVb-003-074-1-4CV.json
│   │   └── [ 1092197139]  01NVb-003-074-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [       1094]  01NVb-003-074-2-4CV.json
│   │   └── [ 1123518452]  01NVb-003-074-2 echo.mp4
│   └── [      32768]  T3
│       ├── [        934]  01NVb-003-074-3-4CV.json
│       └── [ 1383799102]  01NVb-003-074-3 echo.mp4
├── [      32768]  01NVb-003-075
│   ├── [      32768]  T1
│   │   ├── [       1017]  01NVb-003-075-1-4CV.json
│   │   └── [ 1400306121]  01NVb-003-075-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [       1101]  01NVb-003-075-2-4CV.json
│   │   └── [  849445313]  01NVb-003-075-2 echo.mp4
│   └── [      32768]  T3
│       ├── [       1098]  01NVb-003-075-3-4CV.json
│       └── [  745483429]  01NVb-003-075-3 echo.mp4
├── [      32768]  01NVb-003-076
│   ├── [      32768]  T1
│   │   ├── [       1016]  01NVb-003-076-1-4CV.json
│   │   └── [ 1766650850]  01NVb-003-076-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [       1014]  01NVb-003-076-2-4CV.json
│   │   └── [ 1731627481]  01NVb-003-076-2 echo.mp4
│   └── [      32768]  T3
│       ├── [       1098]  01NVb-003-076-3-4CV.json
│       └── [ 1092363409]  01NVb-003-076-3 echo.mp4
├── [      32768]  01NVb-003-078
│   ├── [      32768]  T1
│   │   ├── [       1017]  01NVb-003-078-1-4CV.json
│   │   └── [ 1222683293]  01NVb-003-078-1 echo.mp4
│   ├── [      32768]  T2
│   │   ├── [       1014]  01NVb-003-078-2-4CV.json
│   │   └── [  789414594]  01NVb-003-078-2 echo.mp4
│   └── [      32768]  T3
│       ├── [       1015]  01NVb-003-078-3-4CV.json
│       └── [  934204569]  01NVb-003-078-3 echo.mp4
└── [      32768]  01NVb-003-079
    ├── [      32768]  T1
    │   ├── [       1017]  01NVb-003-079-1-4CV.json
    │   └── [  956782521]  01NVb-003-079-1 echo.mp4
    ├── [      32768]  T2
    │   ├── [       1099]  01NVb-003-079-2-4CV.json
    │   └── [  999452084]  01NVb-003-079-2 echo.mp4
    └── [      32768]  T3
        ├── [       1099]  01NVb-003-079-3-4CV.json
        └── [  854992210]  01NVb-003-079-3 echo.mp4

133 directories, 179 files


 ```


## Raw datasets tree in the server 
```
* 01NVb
	* Group 1-ECHO+LUS
		* ...
		* 01NVb-003-002
			* T1
				* 01NVB-003-002-1
					* 20200604043114
						* Image001.avi
						* ...
						* Image001.jpg
						* ...
				* _P163114
					* K64GR282
					* ...
				* 01NVb-003-002-1.mp4
			* T2	
				* 01NVb-003-002-2
					* 20200608110757	
						* Image001.avi
						* ...
						* Image001.jpg
						* ...					
				* _P110757
					* K68B7G82
					* ...
			* T3 
				* 01NVb-003-002-3
					* 20200608040258
						* Image001.avi
						* ...
						* Image001.jpg
						* ...					
				* _P160258
					* K68B7302
					* ...
		* 01NVb-003-003
			* T1
			* T2
			* T3
			
			
		*  ... 
		
		
		
		* 01NVb-003-079
			* T1
			* T2
			* T3
		* 01NVb-003-080
			* T1
		* 01NVb-003-081
			* T1
			* T2
``` 

## Raw datasets in local machine 
Data path, filename and size are shown below for 8 out of 81 participants (%10 of the dataset). 
```
mx19@sie133-lap:~/datasets/vital/rawdatasets$ ls -lR
.:
total 32
drwxrwxr-x 5 mx19 mx19 4096 Sep 21 01:57 01NVb-003-001
drwxrwxr-x 5 mx19 mx19 4096 Sep 21 04:53 01NVb-003-002
drwxrwxr-x 5 mx19 mx19 4096 Sep 21 07:23 01NVb-003-003
drwxrwxr-x 5 mx19 mx19 4096 Sep 22 00:00 01NVb-003-004
drwxrwxr-x 4 mx19 mx19 4096 Sep 21 07:33 01NVb-003-005
drwxrwxr-x 5 mx19 mx19 4096 Sep 21 07:30 01NVb-003-006
drwxrwxr-x 5 mx19 mx19 4096 Sep 21 07:28 01NVb-003-007
drwxrwxr-x 5 mx19 mx19 4096 Sep 21 07:27 01NVb-003-008

./01NVb-003-001:
total 12
drwxrwxr-x 3 mx19 mx19 4096 Sep 21 23:56 T1
drwxrwxr-x 3 mx19 mx19 4096 Sep 21 23:56 T2
drwxrwxr-x 3 mx19 mx19 4096 Sep 21 23:57 T3

./01NVb-003-001/T1:
total 1291328
-rw-r--r-- 1 mx19 mx19 1322309160 Sep 21 00:46 01NVb-003-001-echo.mp4
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 00:46 _T145245

./01NVb-003-001/T1/_T145245:
total 790672
-rw-r--r-- 1 mx19 mx19 55577160 Sep 21 00:47 K61FC702
-rw-r--r-- 1 mx19 mx19 46695404 Sep 21 00:26 K61FC704
-rw-r--r-- 1 mx19 mx19   389250 Sep 21 00:39 K61FCHO2
-rw-r--r-- 1 mx19 mx19   415714 Sep 21 00:36 K61FCHO4
-rw-r--r-- 1 mx19 mx19   380010 Sep 21 00:25 K61FCHO6
-rw-r--r-- 1 mx19 mx19 34832108 Sep 21 00:30 K61FE4G2
-rw-r--r-- 1 mx19 mx19 31447378 Sep 21 00:29 K61FE4G4
-rw-r--r-- 1 mx19 mx19   379832 Sep 21 00:36 K61FE782
-rw-r--r-- 1 mx19 mx19   390382 Sep 21 00:25 K61FE784
-rw-r--r-- 1 mx19 mx19   373770 Sep 21 00:36 K61FE786
-rw-r--r-- 1 mx19 mx19 29220734 Sep 21 00:28 K61FEQO2
-rw-r--r-- 1 mx19 mx19 26751938 Sep 21 00:33 K61FEQO4
-rw-r--r-- 1 mx19 mx19   207074 Sep 21 00:25 K61FES82
-rw-r--r-- 1 mx19 mx19   187356 Sep 21 00:29 K61FES84
-rw-r--r-- 1 mx19 mx19 57753372 Sep 21 00:35 K61FRE82
-rw-r--r-- 1 mx19 mx19 43820598 Sep 21 00:32 K61FRSG2
-rw-r--r-- 1 mx19 mx19 61813632 Sep 21 00:41 K61FS282
-rw-r--r-- 1 mx19 mx19 32150516 Sep 21 00:38 K61FSIG2
-rw-r--r-- 1 mx19 mx19 15681770 Sep 21 00:28 K61FSNG2
-rw-r--r-- 1 mx19 mx19   238904 Sep 21 00:46 K61FTBG2
-rw-r--r-- 1 mx19 mx19   272580 Sep 21 00:31 K61FTBG4
-rw-r--r-- 1 mx19 mx19 79654138 Sep 21 00:43 K61G1582
-rw-r--r-- 1 mx19 mx19 39104060 Sep 21 00:46 K61G1KG2
-rw-r--r-- 1 mx19 mx19 38592266 Sep 21 00:45 K61G1M02
-rw-r--r-- 1 mx19 mx19 32886066 Sep 21 00:36 K61G3EG2
-rw-r--r-- 1 mx19 mx19   357916 Sep 21 00:25 K61G3RO2
-rw-r--r-- 1 mx19 mx19 11998650 Sep 21 00:32 K61G6282
-rw-r--r-- 1 mx19 mx19 12938978 Sep 21 00:30 K61G6682
-rw-r--r-- 1 mx19 mx19 13585808 Sep 21 00:27 K61G6LO2
-rw-r--r-- 1 mx19 mx19 11458860 Sep 21 00:36 K61G6TO2
-rw-r--r-- 1 mx19 mx19 13601210 Sep 21 00:39 K61G7B02
-rw-r--r-- 1 mx19 mx19 11666180 Sep 21 00:38 K61G7OG2
-rw-r--r-- 1 mx19 mx19 11548256 Sep 21 00:41 K61G80G2
-rw-r--r-- 1 mx19 mx19 11306158 Sep 21 00:39 K61G8I02
-rw-r--r-- 1 mx19 mx19 12988806 Sep 21 00:32 K61G8T02
-rw-r--r-- 1 mx19 mx19 14376596 Sep 21 00:44 K61G9382
-rw-r--r-- 1 mx19 mx19 13201648 Sep 21 00:27 K61G9702
-rw-r--r-- 1 mx19 mx19 16129536 Sep 21 00:37 K61G9B02
-rw-r--r-- 1 mx19 mx19 11969892 Sep 21 00:37 K61G9DO2
-rw-r--r-- 1 mx19 mx19 13203980 Sep 21 00:43 K61G9K82

./01NVb-003-001/T2:
total 1184768
-rw-r--r-- 1 mx19 mx19 1213190810 Sep 21 02:29 01NVb-003-001-2-echo.mp4
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 02:26 _T154312

./01NVb-003-001/T2/_T154312:
total 208428
-rw-r--r-- 1 mx19 mx19  6742442 Sep 21 02:24 K64FMK02
-rw-r--r-- 1 mx19 mx19  6641530 Sep 21 02:21 K64FN9G2
-rw-r--r-- 1 mx19 mx19  7888862 Sep 21 02:24 K64FNJG2
-rw-r--r-- 1 mx19 mx19  7752876 Sep 21 02:26 K64FO002
-rw-r--r-- 1 mx19 mx19  6870316 Sep 21 02:22 K64FOD02
-rw-r--r-- 1 mx19 mx19  7657296 Sep 21 02:24 K64FORO2
-rw-r--r-- 1 mx19 mx19  7757806 Sep 21 02:25 K64FP082
-rw-r--r-- 1 mx19 mx19  6906882 Sep 21 02:21 K64FPLO2
-rw-r--r-- 1 mx19 mx19  7905636 Sep 21 02:26 K64FPU02
-rw-r--r-- 1 mx19 mx19  6147144 Sep 21 02:22 K64FQ3O2
-rw-r--r-- 1 mx19 mx19  6465490 Sep 21 02:24 K64FQE02
-rw-r--r-- 1 mx19 mx19  6810060 Sep 21 02:24 K64FQN02
-rw-r--r-- 1 mx19 mx19  7290244 Sep 21 02:20 K64FR282
-rw-r--r-- 1 mx19 mx19  7327524 Sep 21 02:26 K64FRB82
-rw-r--r-- 1 mx19 mx19  7300588 Sep 21 02:22 K64FRLO2
-rw-r--r-- 1 mx19 mx19  6690672 Sep 21 02:21 K64FRRO2
-rw-r--r-- 1 mx19 mx19  7653842 Sep 21 02:25 K64FS082
-rw-r--r-- 1 mx19 mx19   209264 Sep 21 02:20 K64G2OG2
-rw-r--r-- 1 mx19 mx19 11029716 Sep 21 02:26 K64G34O2
-rw-r--r-- 1 mx19 mx19   236378 Sep 21 02:20 K64G3PO2
-rw-r--r-- 1 mx19 mx19   222840 Sep 21 02:21 K64G3PO4
-rw-r--r-- 1 mx19 mx19 10235970 Sep 21 02:21 K64G5182
-rw-r--r-- 1 mx19 mx19 10609044 Sep 21 02:23 K64G5PG2
-rw-r--r-- 1 mx19 mx19 11531350 Sep 21 02:23 K64G67O2
-rw-r--r-- 1 mx19 mx19 11380060 Sep 21 02:25 K64G6C02
-rw-r--r-- 1 mx19 mx19 10989426 Sep 21 02:26 K64G6I82
-rw-r--r-- 1 mx19 mx19 12569590 Sep 21 02:23 K64G7P82
-rw-r--r-- 1 mx19 mx19   199204 Sep 21 02:20 K64G8EO2
-rw-r--r-- 1 mx19 mx19   174160 Sep 21 02:20 K64G8EO4
-rw-r--r-- 1 mx19 mx19 12176684 Sep 21 02:22 K64G95G2

./01NVb-003-001/T3:
total 1596028
-rw-r--r-- 1 mx19 mx19 1634321254 Sep 21 01:29 01NVb-003-001-3-echo.mp4
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 01:57 _T144314

./01NVb-003-001/T3/_T144314:
total 564716
-rw-r--r-- 1 mx19 mx19 12766390 Sep 21 01:49 K65EM882
-rw-r--r-- 1 mx19 mx19 11805232 Sep 21 01:57 K65EMEO2
-rw-r--r-- 1 mx19 mx19 15566632 Sep 21 01:56 K65EMU02
-rw-r--r-- 1 mx19 mx19 11171238 Sep 21 01:53 K65ENGO2
-rw-r--r-- 1 mx19 mx19 15248614 Sep 21 01:53 K65ENL02
-rw-r--r-- 1 mx19 mx19 18930688 Sep 21 01:56 K65EO002
-rw-r--r-- 1 mx19 mx19 16339230 Sep 21 01:51 K65EO7G2
-rw-r--r-- 1 mx19 mx19 16104426 Sep 21 01:53 K65EO8O2
-rw-r--r-- 1 mx19 mx19 16655516 Sep 21 01:53 K65EOD02
-rw-r--r-- 1 mx19 mx19 14414416 Sep 21 01:57 K65EP3G2
-rw-r--r-- 1 mx19 mx19 14762006 Sep 21 01:55 K65EQ2G2
-rw-r--r-- 1 mx19 mx19 19222336 Sep 21 01:53 K65EQIG2
-rw-r--r-- 1 mx19 mx19 18248146 Sep 21 01:52 K65EQQ82
-rw-r--r-- 1 mx19 mx19 18619234 Sep 21 01:50 K65ER102
-rw-r--r-- 1 mx19 mx19 16213864 Sep 21 01:52 K65ERD02
-rw-r--r-- 1 mx19 mx19 17754128 Sep 21 01:57 K65ERIG2
-rw-r--r-- 1 mx19 mx19  9062980 Sep 21 01:52 K65F04G2
-rw-r--r-- 1 mx19 mx19 42301828 Sep 21 01:55 K65F2JG2
-rw-r--r-- 1 mx19 mx19   396322 Sep 21 01:54 K65F2Q82
-rw-r--r-- 1 mx19 mx19   408732 Sep 21 01:56 K65F2Q84
-rw-r--r-- 1 mx19 mx19   383336 Sep 21 01:56 K65F41G2
-rw-r--r-- 1 mx19 mx19   409798 Sep 21 01:52 K65F41G4
-rw-r--r-- 1 mx19 mx19 43346040 Sep 21 01:52 K65F5N02
-rw-r--r-- 1 mx19 mx19   201480 Sep 21 01:56 K65F5QG2
-rw-r--r-- 1 mx19 mx19 47298368 Sep 21 01:51 K65F6L82
-rw-r--r-- 1 mx19 mx19   244634 Sep 21 01:56 K65F6L84
-rw-r--r-- 1 mx19 mx19   237462 Sep 21 01:53 K65F6L86
-rw-r--r-- 1 mx19 mx19 41694354 Sep 21 01:57 K65F6PO2
-rw-r--r-- 1 mx19 mx19   267742 Sep 21 01:50 K65F76G2
-rw-r--r-- 1 mx19 mx19   409950 Sep 21 01:52 K65F9602
-rw-r--r-- 1 mx19 mx19   416092 Sep 21 01:53 K65FA3O2
-rw-r--r-- 1 mx19 mx19 44523814 Sep 21 01:54 K65FAU82
-rw-r--r-- 1 mx19 mx19   324958 Sep 21 01:54 K65FCI02
-rw-r--r-- 1 mx19 mx19 46876586 Sep 21 01:51 K65FD182
-rw-r--r-- 1 mx19 mx19 45559932 Sep 21 01:55 K65FD802

./01NVb-003-002:
total 12
drwxrwxr-x 3 mx19 mx19 4096 Sep 21 23:57 T1
drwxrwxr-x 3 mx19 mx19 4096 Sep 21 23:57 T2
drwxrwxr-x 3 mx19 mx19 4096 Sep 21 23:58 T3

./01NVb-003-002/T1:
total 1897068
-rw-r--r-- 1 mx19 mx19 1942587232 Sep 21 03:36 01NVb-003-002-1.mp4
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 02:55 _P163114

./01NVb-003-002/T1/_P163114:
total 253160
-rw-r--r-- 1 mx19 mx19 14931470 Sep 21 02:51 K64GR282
-rw-r--r-- 1 mx19 mx19  7404472 Sep 21 02:54 K64GR4O2
-rw-r--r-- 1 mx19 mx19   243548 Sep 21 02:51 K64GR982
-rw-r--r-- 1 mx19 mx19   214338 Sep 21 02:54 K64GR984
-rw-r--r-- 1 mx19 mx19 14933398 Sep 21 02:50 K64GRUO2
-rw-r--r-- 1 mx19 mx19   225604 Sep 21 02:55 K64GT8G2
-rw-r--r-- 1 mx19 mx19   211344 Sep 21 02:48 K64GT8G4
-rw-r--r-- 1 mx19 mx19 14131888 Sep 21 02:51 K64H0202
-rw-r--r-- 1 mx19 mx19 13091624 Sep 21 02:52 K64H0EO2
-rw-r--r-- 1 mx19 mx19 15053228 Sep 21 02:49 K64H0Q82
-rw-r--r-- 1 mx19 mx19 15312706 Sep 21 02:55 K64H12G2
-rw-r--r-- 1 mx19 mx19 14365010 Sep 21 02:53 K64H1882
-rw-r--r-- 1 mx19 mx19  7726398 Sep 21 02:49 K64H1AG2
-rw-r--r-- 1 mx19 mx19   202230 Sep 21 02:48 K64H1H02
-rw-r--r-- 1 mx19 mx19   216890 Sep 21 02:54 K64H1H04
-rw-r--r-- 1 mx19 mx19   100910 Sep 21 02:54 K64H2302
-rw-r--r-- 1 mx19 mx19   194354 Sep 21 02:52 K64H2TG2
-rw-r--r-- 1 mx19 mx19   221484 Sep 21 02:54 K64H2TG4
-rw-r--r-- 1 mx19 mx19 15727846 Sep 21 02:50 K64H3H02
-rw-r--r-- 1 mx19 mx19   163862 Sep 21 02:51 K64H4482
-rw-r--r-- 1 mx19 mx19 13549664 Sep 21 02:53 K64H5902
-rw-r--r-- 1 mx19 mx19  7286502 Sep 21 02:53 K64H6AO2
-rw-r--r-- 1 mx19 mx19  8295666 Sep 21 02:52 K64H6J82
-rw-r--r-- 1 mx19 mx19  8459804 Sep 21 02:55 K64H70O2
-rw-r--r-- 1 mx19 mx19  8243846 Sep 21 02:52 K64H7EO2
-rw-r--r-- 1 mx19 mx19  8513204 Sep 21 02:54 K64H7K82
-rw-r--r-- 1 mx19 mx19  8295202 Sep 21 02:48 K64H7P82
-rw-r--r-- 1 mx19 mx19  7786906 Sep 21 02:54 K64H83G2
-rw-r--r-- 1 mx19 mx19  7366906 Sep 21 02:48 K64H8AO2
-rw-r--r-- 1 mx19 mx19  6974018 Sep 21 02:55 K64H8MO2
-rw-r--r-- 1 mx19 mx19  8118550 Sep 21 02:48 K64H8UG2
-rw-r--r-- 1 mx19 mx19  8201798 Sep 21 02:49 K64H95O2
-rw-r--r-- 1 mx19 mx19  8097000 Sep 21 02:56 K64H9E02
-rw-r--r-- 1 mx19 mx19  7767214 Sep 21 02:51 K64H9MG2
-rw-r--r-- 1 mx19 mx19  7524294 Sep 21 02:52 K64H9SO2

./01NVb-003-002/T2:
total 2966420
-rw-r--r-- 1 mx19 mx19  938921061 Sep 21 05:19 '01NVb-003-002-2 (cont).mp4'
-rw-r--r-- 1 mx19 mx19 2098677758 Sep 21 05:46  01NVb-003-002-2.mp4
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 06:39  _P110757

./01NVb-003-002/T2/_P110757:
total 898560
-rw-r--r-- 1 mx19 mx19 47928280 Sep 21 06:30 K68B7G82
-rw-r--r-- 1 mx19 mx19 51043158 Sep 21 06:37 K68B8D02
-rw-r--r-- 1 mx19 mx19   374664 Sep 21 06:34 K68B92G2
-rw-r--r-- 1 mx19 mx19 49028648 Sep 21 06:40 K68B9KG2
-rw-r--r-- 1 mx19 mx19   371644 Sep 21 06:37 K68BABO2
-rw-r--r-- 1 mx19 mx19   371154 Sep 21 06:34 K68BAD82
-rw-r--r-- 1 mx19 mx19   397260 Sep 21 06:33 K68BAD84
-rw-r--r-- 1 mx19 mx19 51566622 Sep 21 06:28 K68BBQG2
-rw-r--r-- 1 mx19 mx19   370596 Sep 21 06:28 K68BBUO2
-rw-r--r-- 1 mx19 mx19   371410 Sep 21 06:27 K68BC182
-rw-r--r-- 1 mx19 mx19   382418 Sep 21 06:37 K68BC184
-rw-r--r-- 1 mx19 mx19 47015266 Sep 21 06:31 K68BCUO2
-rw-r--r-- 1 mx19 mx19 55726542 Sep 21 06:33 K68BD4O2
-rw-r--r-- 1 mx19 mx19 52516130 Sep 21 06:35 K68BD8G2
-rw-r--r-- 1 mx19 mx19 50173728 Sep 21 06:35 K68BE302
-rw-r--r-- 1 mx19 mx19   221410 Sep 21 06:29 K68BE304
-rw-r--r-- 1 mx19 mx19 53840502 Sep 21 06:38 K68BE6G2
-rw-r--r-- 1 mx19 mx19 46831664 Sep 21 06:29 K68BEGG2
-rw-r--r-- 1 mx19 mx19   229568 Sep 21 06:27 K68BEGG4
-rw-r--r-- 1 mx19 mx19 45479916 Sep 21 06:31 K68BEL02
-rw-r--r-- 1 mx19 mx19 47865348 Sep 21 06:37 K68BFBG2
-rw-r--r-- 1 mx19 mx19   207458 Sep 21 06:34 K68BFN02
-rw-r--r-- 1 mx19 mx19   216068 Sep 21 06:33 K68BGH82
-rw-r--r-- 1 mx19 mx19   257332 Sep 21 06:33 K68BGH84
-rw-r--r-- 1 mx19 mx19   277504 Sep 21 06:29 K68BGH86
-rw-r--r-- 1 mx19 mx19   219274 Sep 21 06:35 K68BGK02
-rw-r--r-- 1 mx19 mx19 53422340 Sep 21 06:39 K68BH882
-rw-r--r-- 1 mx19 mx19   199342 Sep 21 06:39 K68BH9G2
-rw-r--r-- 1 mx19 mx19 15490510 Sep 21 06:34 K68BHQG2
-rw-r--r-- 1 mx19 mx19   373372 Sep 21 06:32 K68BKG82
-rw-r--r-- 1 mx19 mx19   350370 Sep 21 06:28 K68BKI02
-rw-r--r-- 1 mx19 mx19   354456 Sep 21 06:34 K68BKI04
-rw-r--r-- 1 mx19 mx19 49599766 Sep 21 06:33 K68BMK02
-rw-r--r-- 1 mx19 mx19   293556 Sep 21 06:34 K68BO4G2
-rw-r--r-- 1 mx19 mx19   307030 Sep 21 06:29 K68BO4G4
-rw-r--r-- 1 mx19 mx19 14774558 Sep 21 06:37 K68BPD82
-rw-r--r-- 1 mx19 mx19 16728998 Sep 21 06:34 K68BPMG2
-rw-r--r-- 1 mx19 mx19 14633012 Sep 21 06:29 K68BR8O2
-rw-r--r-- 1 mx19 mx19 14826358 Sep 21 06:39 K68BRJ02
-rw-r--r-- 1 mx19 mx19 13263128 Sep 21 06:37 K68BRUO2
-rw-r--r-- 1 mx19 mx19 15437818 Sep 21 06:29 K68BS3O2
-rw-r--r-- 1 mx19 mx19 14509652 Sep 21 06:29 K68BSAG2
-rw-r--r-- 1 mx19 mx19 13911166 Sep 21 06:30 K68BSL82
-rw-r--r-- 1 mx19 mx19 16390924 Sep 21 06:32 K68BT1O2
-rw-r--r-- 1 mx19 mx19 16694080 Sep 21 06:28 K68BTBG2
-rw-r--r-- 1 mx19 mx19 15757210 Sep 21 06:32 K68BTG82
-rw-r--r-- 1 mx19 mx19 14001080 Sep 21 06:35 K68BTQO2
-rw-r--r-- 1 mx19 mx19 15418204 Sep 21 06:31 K68C00O2

./01NVb-003-002/T3:
total 2050144
-rw-r--r-- 1 mx19 mx19 2099338067 Sep 21 04:32 01NVb-003-002-3.mp4
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 04:03 _P160258

./01NVb-003-002/T3/_P160258:
total 577824
-rw-r--r-- 1 mx19 mx19 55010758 Sep 21 03:50 K68G73O2
-rw-r--r-- 1 mx19 mx19   382918 Sep 21 03:58 K68G7D02
-rw-r--r-- 1 mx19 mx19   392672 Sep 21 04:03 K68G7D04
-rw-r--r-- 1 mx19 mx19   205534 Sep 21 04:03 K68G7TO2
-rw-r--r-- 1 mx19 mx19   381974 Sep 21 03:58 K68GA302
-rw-r--r-- 1 mx19 mx19   408634 Sep 21 03:47 K68GAIG2
-rw-r--r-- 1 mx19 mx19   381720 Sep 21 03:48 K68GC0O2
-rw-r--r-- 1 mx19 mx19   408092 Sep 21 04:02 K68GC0O4
-rw-r--r-- 1 mx19 mx19 60305090 Sep 21 03:58 K68GDEG2
-rw-r--r-- 1 mx19 mx19 51477288 Sep 21 04:00 K68GE202
-rw-r--r-- 1 mx19 mx19 51310704 Sep 21 03:53 K68GEA82
-rw-r--r-- 1 mx19 mx19 51260912 Sep 21 03:56 K68GEK82
-rw-r--r-- 1 mx19 mx19 49486800 Sep 21 03:54 K68GEUO2
-rw-r--r-- 1 mx19 mx19   222516 Sep 21 03:47 K68GF8G2
-rw-r--r-- 1 mx19 mx19   253070 Sep 21 03:58 K68GF8G4
-rw-r--r-- 1 mx19 mx19   253070 Sep 21 03:47 K68GF8G6
-rw-r--r-- 1 mx19 mx19   379172 Sep 21 03:55 K68GHAO2
-rw-r--r-- 1 mx19 mx19   382172 Sep 21 04:00 K68GHAO4
-rw-r--r-- 1 mx19 mx19 56754980 Sep 21 03:51 K68GIRO2
-rw-r--r-- 1 mx19 mx19   284508 Sep 21 03:58 K68GJ4O2
-rw-r--r-- 1 mx19 mx19   298382 Sep 21 04:03 K68GJ4O4
-rw-r--r-- 1 mx19 mx19 14320764 Sep 21 04:02 K68GK682
-rw-r--r-- 1 mx19 mx19 16870294 Sep 21 04:01 K68GKCO2
-rw-r--r-- 1 mx19 mx19 15906188 Sep 21 03:57 K68GKT02
-rw-r--r-- 1 mx19 mx19 14505306 Sep 21 04:03 K68GLCG2
-rw-r--r-- 1 mx19 mx19 16001172 Sep 21 04:02 K68GLLG2
-rw-r--r-- 1 mx19 mx19 12506200 Sep 21 03:52 K68GM402
-rw-r--r-- 1 mx19 mx19 12191310 Sep 21 03:55 K68GM782
-rw-r--r-- 1 mx19 mx19 14979608 Sep 21 03:48 K68GMEO2
-rw-r--r-- 1 mx19 mx19 15229280 Sep 21 04:01 K68GMLG2
-rw-r--r-- 1 mx19 mx19 16590224 Sep 21 04:00 K68GN202
-rw-r--r-- 1 mx19 mx19 16463038 Sep 21 03:48 K68GNA82
-rw-r--r-- 1 mx19 mx19 16581396 Sep 21 04:03 K68GNN82
-rw-r--r-- 1 mx19 mx19 15961528 Sep 21 04:01 K68GNUO2
-rw-r--r-- 1 mx19 mx19 13256536 Sep 21 03:50 K68GO402

./01NVb-003-003:
total 12
drwxrwxr-x 3 mx19 mx19 4096 Sep 21 23:58 T1
drwxrwxr-x 3 mx19 mx19 4096 Sep 21 23:58 T2
drwxrwxr-x 3 mx19 mx19 4096 Sep 21 23:59 T3

./01NVb-003-003/T1:
total 3700056
-rw-r--r-- 1 mx19 mx19 1690810839 Sep 21 07:23 '01NVb-003-003-1 (cont).mp4'
-rw-r--r-- 1 mx19 mx19 2098030205 Sep 21 07:35  01NVb-003-003-1.mp4
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 09:32  _T105047

./01NVb-003-003/T1/_T105047:
total 1156584
-rw-r--r-- 1 mx19 mx19   358550 Sep 21 07:44 K6AB0282
-rw-r--r-- 1 mx19 mx19 34618710 Sep 21 08:30 K6AB0KG2
-rw-r--r-- 1 mx19 mx19 33445410 Sep 21 08:28 K6AB0M82
-rw-r--r-- 1 mx19 mx19 63221876 Sep 21 08:22 K6AB2102
-rw-r--r-- 1 mx19 mx19   344018 Sep 21 07:43 K6AB2A82
-rw-r--r-- 1 mx19 mx19   375278 Sep 21 07:46 K6AB2A84
-rw-r--r-- 1 mx19 mx19   330492 Sep 21 07:41 K6AB3MG2
-rw-r--r-- 1 mx19 mx19   333188 Sep 21 07:42 K6AB3PG2
-rw-r--r-- 1 mx19 mx19   342854 Sep 21 07:43 K6AB3PG4
-rw-r--r-- 1 mx19 mx19 73687756 Sep 21 08:22 K6AB4HO2
-rw-r--r-- 1 mx19 mx19   194496 Sep 21 07:30 K6AB4HO4
-rw-r--r-- 1 mx19 mx19 66636186 Sep 21 09:31 K6AB5U82
-rw-r--r-- 1 mx19 mx19 72042550 Sep 21 09:34 K6AB63G2
-rw-r--r-- 1 mx19 mx19 70467930 Sep 21 09:33 K6AB6AO2
-rw-r--r-- 1 mx19 mx19 63599484 Sep 21 09:23 K6AB6LG2
-rw-r--r-- 1 mx19 mx19 58390782 Sep 21 09:13 K6AB6T02
-rw-r--r-- 1 mx19 mx19 65468992 Sep 21 09:29 K6AB97G2
-rw-r--r-- 1 mx19 mx19   255044 Sep 21 07:38 K6AB97G4
-rw-r--r-- 1 mx19 mx19   264200 Sep 21 07:38 K6AB97G6
-rw-r--r-- 1 mx19 mx19 66109622 Sep 21 09:30 K6ABA6O2
-rw-r--r-- 1 mx19 mx19 63660936 Sep 21 09:23 K6ABAPG2
-rw-r--r-- 1 mx19 mx19   196754 Sep 21 07:30 K6ABAUO2
-rw-r--r-- 1 mx19 mx19   195510 Sep 21 07:30 K6ABB102
-rw-r--r-- 1 mx19 mx19   182172 Sep 21 07:28 K6ABBIG2
-rw-r--r-- 1 mx19 mx19   187520 Sep 21 07:29 K6ABBJG2
-rw-r--r-- 1 mx19 mx19   255468 Sep 21 07:38 K6ABBJG4
-rw-r--r-- 1 mx19 mx19   179398 Sep 21 07:27 K6ABCDG2
-rw-r--r-- 1 mx19 mx19 55762130 Sep 21 09:05 K6ABCS02
-rw-r--r-- 1 mx19 mx19 53842180 Sep 21 08:56 K6ABCUG2
-rw-r--r-- 1 mx19 mx19   366750 Sep 21 07:45 K6ABDP82
-rw-r--r-- 1 mx19 mx19   373860 Sep 21 07:46 K6ABDUO2
-rw-r--r-- 1 mx19 mx19 63972244 Sep 21 09:25 K6ABEEO2
-rw-r--r-- 1 mx19 mx19   318080 Sep 21 07:40 K6ABFB82
-rw-r--r-- 1 mx19 mx19   336144 Sep 21 07:42 K6ABFB84
-rw-r--r-- 1 mx19 mx19 59011006 Sep 21 09:16 K6ABG382
-rw-r--r-- 1 mx19 mx19 10308548 Sep 21 07:56 K6ABH582
-rw-r--r-- 1 mx19 mx19  9011810 Sep 21 07:51 K6ABHCO2
-rw-r--r-- 1 mx19 mx19  9724108 Sep 21 07:53 K6ABHEO2
-rw-r--r-- 1 mx19 mx19 11291150 Sep 21 08:03 K6ABHQ02
-rw-r--r-- 1 mx19 mx19 11240430 Sep 21 08:02 K6ABJ382
-rw-r--r-- 1 mx19 mx19 13828570 Sep 21 08:24 K6ABJEO2
-rw-r--r-- 1 mx19 mx19 13876392 Sep 21 08:24 K6ABJJG2
-rw-r--r-- 1 mx19 mx19 14691428 Sep 21 08:25 K6ABJT82
-rw-r--r-- 1 mx19 mx19 13711574 Sep 21 08:24 K6ABKE82
-rw-r--r-- 1 mx19 mx19 13822054 Sep 21 08:24 K6ABKGG2
-rw-r--r-- 1 mx19 mx19 12497394 Sep 21 08:23 K6ABKMG2
-rw-r--r-- 1 mx19 mx19 12497762 Sep 21 08:23 K6ABKPO2
-rw-r--r-- 1 mx19 mx19 10102188 Sep 21 07:54 K6ABLRG2
-rw-r--r-- 1 mx19 mx19  9861354 Sep 21 07:54 K6ABM5G2
-rw-r--r-- 1 mx19 mx19 11166310 Sep 21 08:01 K6ABNGO2
-rw-r--r-- 1 mx19 mx19 14099466 Sep 21 08:24 K6ABOHG2
-rw-r--r-- 1 mx19 mx19 11485760 Sep 21 08:05 K6ABP902
-rw-r--r-- 1 mx19 mx19 11645710 Sep 21 08:07 K6ABPLG2

./01NVb-003-003/T2:
total 3113020
-rw-r--r-- 1 mx19 mx19 1089265090 Sep 21 12:40 '01NVb-003-003-2 (cont).mp4'
-rw-r--r-- 1 mx19 mx19 2098452805 Sep 21 13:01  01NVb-003-003-2.mp4
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 09:13  _T102613

./01NVb-003-003/T2/_T102613:
total 980448
-rw-r--r-- 1 mx19 mx19 58957540 Sep 21 09:15 K6BAF882
-rw-r--r-- 1 mx19 mx19   356860 Sep 21 07:44 K6BAFGO2
-rw-r--r-- 1 mx19 mx19   383002 Sep 21 07:47 K6BAFGO4
-rw-r--r-- 1 mx19 mx19   358036 Sep 21 07:44 K6BAH5O2
-rw-r--r-- 1 mx19 mx19   367552 Sep 21 07:45 K6BAH5O4
-rw-r--r-- 1 mx19 mx19 54201024 Sep 21 08:57 K6BAHQ82
-rw-r--r-- 1 mx19 mx19   191250 Sep 21 07:29 K6BAHR02
-rw-r--r-- 1 mx19 mx19 54281370 Sep 21 08:58 K6BAJ5O2
-rw-r--r-- 1 mx19 mx19 55546800 Sep 21 09:05 K6BAJC82
-rw-r--r-- 1 mx19 mx19 57514420 Sep 21 09:09 K6BAJEG2
-rw-r--r-- 1 mx19 mx19 57133194 Sep 21 09:09 K6BAJK02
-rw-r--r-- 1 mx19 mx19 50805780 Sep 21 08:49 K6BAKOO2
-rw-r--r-- 1 mx19 mx19   251908 Sep 21 07:37 K6BAKOO4
-rw-r--r-- 1 mx19 mx19   251908 Sep 21 07:37 K6BAKOO6
-rw-r--r-- 1 mx19 mx19 49336238 Sep 21 08:48 K6BAKTO2
-rw-r--r-- 1 mx19 mx19   179308 Sep 21 07:27 K6BALEO2
-rw-r--r-- 1 mx19 mx19   268610 Sep 21 07:38 K6BALEO4
-rw-r--r-- 1 mx19 mx19   231052 Sep 21 07:36 K6BALT02
-rw-r--r-- 1 mx19 mx19 54748242 Sep 21 09:00 K6BANC02
-rw-r--r-- 1 mx19 mx19 56586724 Sep 21 09:07 K6BANP82
-rw-r--r-- 1 mx19 mx19   315726 Sep 21 07:40 K6BANSO2
-rw-r--r-- 1 mx19 mx19   333830 Sep 21 07:42 K6BANSO4
-rw-r--r-- 1 mx19 mx19 51605200 Sep 21 08:50 K6BAQ8O2
-rw-r--r-- 1 mx19 mx19 52758262 Sep 21 08:53 K6BAQC02
-rw-r--r-- 1 mx19 mx19 52598664 Sep 21 08:53 K6BAT9G2
-rw-r--r-- 1 mx19 mx19 45915390 Sep 21 08:43 K6BATEG2
-rw-r--r-- 1 mx19 mx19 45705338 Sep 21 08:43 K6BB0MO2
-rw-r--r-- 1 mx19 mx19 41836778 Sep 21 08:36 K6BB0OG2
-rw-r--r-- 1 mx19 mx19   381322 Sep 21 07:46 K6BB0RO2
-rw-r--r-- 1 mx19 mx19 10244180 Sep 21 07:55 K6BB21G2
-rw-r--r-- 1 mx19 mx19  8919860 Sep 21 07:51 K6BB2JO2
-rw-r--r-- 1 mx19 mx19  9533502 Sep 21 07:52 K6BB3802
-rw-r--r-- 1 mx19 mx19  9505854 Sep 21 07:52 K6BB3BG2
-rw-r--r-- 1 mx19 mx19 11292402 Sep 21 08:03 K6BB4402
-rw-r--r-- 1 mx19 mx19 11400812 Sep 21 08:04 K6BB4BG2
-rw-r--r-- 1 mx19 mx19 11524466 Sep 21 08:05 K6BB4D82
-rw-r--r-- 1 mx19 mx19 10650246 Sep 21 07:58 K6BB4P02
-rw-r--r-- 1 mx19 mx19 11337728 Sep 21 08:17 K6BB4UO2
-rw-r--r-- 1 mx19 mx19 10281180 Sep 21 07:56 K6BB57O2
-rw-r--r-- 1 mx19 mx19  9225516 Sep 21 07:52 K6BB5C82
-rw-r--r-- 1 mx19 mx19  9193568 Sep 21 07:52 K6BB7082
-rw-r--r-- 1 mx19 mx19 10460210 Sep 21 07:57 K6BB7782
-rw-r--r-- 1 mx19 mx19 10655508 Sep 21 07:58 K6BB78O2
-rw-r--r-- 1 mx19 mx19  9561992 Sep 21 07:53 K6BB7JO2
-rw-r--r-- 1 mx19 mx19  6684672 Sep 21 08:17 K6BB7TO2

./01NVb-003-003/T3:
total 1513704
-rw-r--r-- 1 mx19 mx19   89010056 Sep 21 09:35 '01NVb-003-003-3 (cont1).mp4'
-rw-r--r-- 1 mx19 mx19 1461009044 Sep 21 13:33  01NVb-003-003-3.mp4
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 09:30  _T100023

./01NVb-003-003/T3/_T100023:
total 1347764
-rw-r--r-- 1 mx19 mx19 54805174 Sep 21 09:02 K6CA2P02
-rw-r--r-- 1 mx19 mx19 53763120 Sep 21 08:56 K6CA2RG2
-rw-r--r-- 1 mx19 mx19   370886 Sep 21 07:45 K6CA3202
-rw-r--r-- 1 mx19 mx19   401234 Sep 21 07:49 K6CA3204
-rw-r--r-- 1 mx19 mx19   370006 Sep 21 07:45 K6CA3HO2
-rw-r--r-- 1 mx19 mx19   383196 Sep 21 07:47 K6CA3HO4
-rw-r--r-- 1 mx19 mx19 64740570 Sep 21 09:28 K6CA3M82
-rw-r--r-- 1 mx19 mx19   185332 Sep 21 07:28 K6CA3OO2
-rw-r--r-- 1 mx19 mx19   197128 Sep 21 07:30 K6CA3OO4
-rw-r--r-- 1 mx19 mx19 66858354 Sep 21 09:32 K6CA4902
-rw-r--r-- 1 mx19 mx19 64486978 Sep 21 09:27 K6CA4IO2
-rw-r--r-- 1 mx19 mx19 57540440 Sep 21 09:10 K6CA4K82
-rw-r--r-- 1 mx19 mx19 53088412 Sep 21 08:55 K6CA4TO2
-rw-r--r-- 1 mx19 mx19 31156632 Sep 21 08:27 K6CA4UO2
-rw-r--r-- 1 mx19 mx19 64587294 Sep 21 09:27 K6CA6002
-rw-r--r-- 1 mx19 mx19 61428272 Sep 21 09:18 K6CA6202
-rw-r--r-- 1 mx19 mx19 57803752 Sep 21 09:10 K6CA64G2
-rw-r--r-- 1 mx19 mx19 58108064 Sep 21 09:12 K6CA6802
-rw-r--r-- 1 mx19 mx19 61155320 Sep 21 09:17 K6CA6TG2
-rw-r--r-- 1 mx19 mx19 56384874 Sep 21 09:07 K6CA7082
-rw-r--r-- 1 mx19 mx19   273460 Sep 21 07:38 K6CA7084
-rw-r--r-- 1 mx19 mx19   291270 Sep 21 07:39 K6CA7086
-rw-r--r-- 1 mx19 mx19   397374 Sep 21 07:48 K6CA7D02
-rw-r--r-- 1 mx19 mx19 58505826 Sep 21 09:15 K6CA7UO2
-rw-r--r-- 1 mx19 mx19   279130 Sep 21 07:38 K6CA7UO4
-rw-r--r-- 1 mx19 mx19 54850226 Sep 21 09:02 K6CA87O2
-rw-r--r-- 1 mx19 mx19 54301486 Sep 21 08:59 K6CA8AO2
-rw-r--r-- 1 mx19 mx19 52396864 Sep 21 08:52 K6CA8CG2
-rw-r--r-- 1 mx19 mx19   222162 Sep 21 07:35 K6CA8K02
-rw-r--r-- 1 mx19 mx19   222162 Sep 21 07:35 K6CA8K04
-rw-r--r-- 1 mx19 mx19   308260 Sep 21 07:40 K6CA8K06
-rw-r--r-- 1 mx19 mx19   198454 Sep 21 07:30 K6CA9382
-rw-r--r-- 1 mx19 mx19   204658 Sep 21 07:32 K6CA9582
-rw-r--r-- 1 mx19 mx19   204658 Sep 21 07:32 K6CA9584
-rw-r--r-- 1 mx19 mx19   283420 Sep 21 07:39 K6CA9SO2
-rw-r--r-- 1 mx19 mx19 52262614 Sep 21 08:52 K6CAACO2
-rw-r--r-- 1 mx19 mx19 38397842 Sep 21 08:35 K6CAAE02
-rw-r--r-- 1 mx19 mx19   351976 Sep 21 07:43 K6CAALO2
-rw-r--r-- 1 mx19 mx19   351976 Sep 21 07:43 K6CAALO4
-rw-r--r-- 1 mx19 mx19   370804 Sep 21 07:45 K6CAALO6
-rw-r--r-- 1 mx19 mx19   370804 Sep 21 07:45 K6CAALO8
-rw-r--r-- 1 mx19 mx19 10553032 Sep 21 07:58 K6CABBO2
-rw-r--r-- 1 mx19 mx19 11354818 Sep 21 08:03 K6CABJ02
-rw-r--r-- 1 mx19 mx19  9539980 Sep 21 07:53 K6CAC0G2
-rw-r--r-- 1 mx19 mx19 14309066 Sep 21 08:25 K6CAC882
-rw-r--r-- 1 mx19 mx19 15547634 Sep 21 08:26 K6CACJO2
-rw-r--r-- 1 mx19 mx19 16071140 Sep 21 08:26 K6CACQO2
-rw-r--r-- 1 mx19 mx19 16794316 Sep 21 08:26 K6CACT82
-rw-r--r-- 1 mx19 mx19 15544140 Sep 21 08:26 K6CAD7O2
-rw-r--r-- 1 mx19 mx19 15958782 Sep 21 08:26 K6CAD9G2
-rw-r--r-- 1 mx19 mx19 15542332 Sep 21 08:25 K6CADGG2
-rw-r--r-- 1 mx19 mx19 14271684 Sep 21 08:24 K6CADLG2
-rw-r--r-- 1 mx19 mx19 12969936 Sep 21 08:23 K6CAE302
-rw-r--r-- 1 mx19 mx19 10150248 Sep 21 07:55 K6CAEHG2
-rw-r--r-- 1 mx19 mx19 11424466 Sep 21 08:04 K6CAF282
-rw-r--r-- 1 mx19 mx19 11534422 Sep 21 08:06 K6CAF702
-rw-r--r-- 1 mx19 mx19 11842124 Sep 21 08:10 K6CAFT82
-rw-r--r-- 1 mx19 mx19 12991270 Sep 21 08:23 K6CAGRG2
-rw-r--r-- 1 mx19 mx19 14357214 Sep 21 08:25 K6CAH1G2
-rw-r--r-- 1 mx19 mx19 16326194 Sep 21 08:26 K6CAH6G2

./01NVb-003-004:
total 12
drwxrwxr-x 3 mx19 mx19 4096 Sep 21 23:59 T1
drwxrwxr-x 3 mx19 mx19 4096 Sep 22 00:00 T2
drwxrwxr-x 3 mx19 mx19 4096 Sep 21 16:06 T3

./01NVb-003-004/T1:
total 4150484
-rw-r--r-- 1 mx19 mx19  873683196 Sep 21 14:54 '01NVb-003-004-1 (cont).mp4'
-rw-r--r-- 1 mx19 mx19 1275333332 Sep 21 14:10 '01NVb-003-004-1 mea.mp4'
-rw-r--r-- 1 mx19 mx19 2101055776 Sep 21 14:32  01NVb-003-004-1.mp4
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 15:26  _P105845

./01NVb-003-004/T1/_P105845:
total 527000
-rw-r--r-- 1 mx19 mx19 19578956 Sep 21 15:24 K6CB77G2
-rw-r--r-- 1 mx19 mx19 19495272 Sep 21 15:24 K6CB7902
-rw-r--r-- 1 mx19 mx19   356766 Sep 21 07:44 K6CB7G02
-rw-r--r-- 1 mx19 mx19   384890 Sep 21 07:47 K6CB7G04
-rw-r--r-- 1 mx19 mx19 18116554 Sep 21 15:22 K6CBAC82
-rw-r--r-- 1 mx19 mx19   363694 Sep 21 07:44 K6CBB2G2
-rw-r--r-- 1 mx19 mx19   363694 Sep 21 07:44 K6CBB2G4
-rw-r--r-- 1 mx19 mx19 20908196 Sep 21 15:26 K6CBCGO2
-rw-r--r-- 1 mx19 mx19 20783964 Sep 21 15:26 K6CBCI82
-rw-r--r-- 1 mx19 mx19   373976 Sep 21 07:46 K6CBCQG2
-rw-r--r-- 1 mx19 mx19   375068 Sep 21 07:46 K6CBD382
-rw-r--r-- 1 mx19 mx19 20035050 Sep 21 15:24 K6CBE282
-rw-r--r-- 1 mx19 mx19 17675714 Sep 21 15:21 K6CBEO82
-rw-r--r-- 1 mx19 mx19 18699332 Sep 21 15:22 K6CBEPO2
-rw-r--r-- 1 mx19 mx19 19037856 Sep 21 15:23 K6CBETG2
-rw-r--r-- 1 mx19 mx19 18321266 Sep 21 15:22 K6CBEUG2
-rw-r--r-- 1 mx19 mx19 19414538 Sep 21 15:23 K6CBFA82
-rw-r--r-- 1 mx19 mx19 19474414 Sep 21 15:23 K6CBFC82
-rw-r--r-- 1 mx19 mx19 20544734 Sep 21 15:25 K6CBG3O2
-rw-r--r-- 1 mx19 mx19 20403812 Sep 21 15:25 K6CBG502
-rw-r--r-- 1 mx19 mx19   221454 Sep 21 07:35 K6CBGH82
-rw-r--r-- 1 mx19 mx19   215608 Sep 21 07:34 K6CBI9G2
-rw-r--r-- 1 mx19 mx19   200674 Sep 21 07:31 K6CBIEO2
-rw-r--r-- 1 mx19 mx19 20100296 Sep 21 15:25 K6CBITO2
-rw-r--r-- 1 mx19 mx19 20748026 Sep 21 15:26 K6CBJB82
-rw-r--r-- 1 mx19 mx19 20266820 Sep 21 15:25 K6CBJE02
-rw-r--r-- 1 mx19 mx19 16850450 Sep 21 15:19 K6CBKCO2
-rw-r--r-- 1 mx19 mx19   204598 Sep 21 07:32 K6CBL7G2
-rw-r--r-- 1 mx19 mx19   204598 Sep 21 07:32 K6CBL7G4
-rw-r--r-- 1 mx19 mx19 18710722 Sep 21 15:23 K6CBLG82
-rw-r--r-- 1 mx19 mx19 15483820 Sep 21 15:15 K6CBLU02
-rw-r--r-- 1 mx19 mx19 14622746 Sep 21 15:15 K6CBM4G2
-rw-r--r-- 1 mx19 mx19 17060256 Sep 21 15:19 K6CBNI02
-rw-r--r-- 1 mx19 mx19 16831474 Sep 21 15:19 K6CBNS02
-rw-r--r-- 1 mx19 mx19 16650172 Sep 21 15:18 K6CBNT82
-rw-r--r-- 1 mx19 mx19 17334502 Sep 21 15:20 K6CBOJ82
-rw-r--r-- 1 mx19 mx19 17581224 Sep 21 15:21 K6CBOTO2
-rw-r--r-- 1 mx19 mx19 17568256 Sep 21 15:21 K6CBP9O2
-rw-r--r-- 1 mx19 mx19 17750110 Sep 21 15:21 K6CBPI02
-rw-r--r-- 1 mx19 mx19 16256690 Sep 21 15:17 K6CBPT02

./01NVb-003-004/T2:
total 5422428
-rw-r--r-- 1 mx19 mx19 2098868242 Sep 21 17:11  01NVb-003-004-2-echo+lus.mp4
-rw-r--r-- 1 mx19 mx19 2100338859 Sep 21 17:36  01NVb-003-004-2-echo.mp4
-rw-r--r-- 1 mx19 mx19  210410806 Sep 21 16:13 '01NVb-003-004-2 EF%.mp4'
-rw-r--r-- 1 mx19 mx19 1142918794 Sep 21 16:42  01NVb-003-004-2-mea.mp4
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 15:43  _P112205

./01NVb-003-004/T2/_P112205:
total 1356372
-rw-r--r-- 1 mx19 mx19 21037724 Sep 21 15:27 K6FBC8G2
-rw-r--r-- 1 mx19 mx19 21613452 Sep 21 15:27 K6FBCG82
-rw-r--r-- 1 mx19 mx19 16442776 Sep 21 15:17 K6FBCJ02
-rw-r--r-- 1 mx19 mx19 43448884 Sep 21 15:37 K6FBCN82
-rw-r--r-- 1 mx19 mx19   388270 Sep 21 07:48 K6FBD302
-rw-r--r-- 1 mx19 mx19   390650 Sep 21 07:48 K6FBD304
-rw-r--r-- 1 mx19 mx19   419882 Sep 21 07:49 K6FBD306
-rw-r--r-- 1 mx19 mx19 44902322 Sep 21 15:43 K6FBDDO2
-rw-r--r-- 1 mx19 mx19   387754 Sep 21 07:47 K6FBDIG2
-rw-r--r-- 1 mx19 mx19   414886 Sep 21 07:49 K6FBDIG4
-rw-r--r-- 1 mx19 mx19 44125772 Sep 21 15:41 K6FBDP02
-rw-r--r-- 1 mx19 mx19   392946 Sep 21 07:48 K6FBDS82
-rw-r--r-- 1 mx19 mx19   386308 Sep 21 07:47 K6FBE8G2
-rw-r--r-- 1 mx19 mx19   397582 Sep 21 07:48 K6FBE8G4
-rw-r--r-- 1 mx19 mx19 43643198 Sep 21 15:38 K6FBF482
-rw-r--r-- 1 mx19 mx19   207438 Sep 21 07:33 K6FBF484
-rw-r--r-- 1 mx19 mx19 43093412 Sep 21 15:35 K6FBF9G2
-rw-r--r-- 1 mx19 mx19   207582 Sep 21 07:33 K6FBF9G4
-rw-r--r-- 1 mx19 mx19 43693606 Sep 21 15:39 K6FBFB82
-rw-r--r-- 1 mx19 mx19 40148422 Sep 21 15:30 K6FBFR82
-rw-r--r-- 1 mx19 mx19 41937600 Sep 21 15:32 K6FBG002
-rw-r--r-- 1 mx19 mx19 41689236 Sep 21 15:31 K6FBG282
-rw-r--r-- 1 mx19 mx19 43754026 Sep 21 15:39 K6FBG982
-rw-r--r-- 1 mx19 mx19   252948 Sep 21 07:38 K6FBICO2
-rw-r--r-- 1 mx19 mx19   245148 Sep 21 07:37 K6FBIDG2
-rw-r--r-- 1 mx19 mx19   248242 Sep 21 07:37 K6FBJ982
-rw-r--r-- 1 mx19 mx19   249540 Sep 21 07:37 K6FBJ984
-rw-r--r-- 1 mx19 mx19   331660 Sep 21 07:41 K6FBJ986
-rw-r--r-- 1 mx19 mx19 45196640 Sep 21 15:44 K6FBJNG2
-rw-r--r-- 1 mx19 mx19   235796 Sep 21 07:37 K6FBJNG4
-rw-r--r-- 1 mx19 mx19 44491174 Sep 21 15:43 K6FBJPO2
-rw-r--r-- 1 mx19 mx19 44235732 Sep 21 15:41 K6FBJS02
-rw-r--r-- 1 mx19 mx19 43319114 Sep 21 15:36 K6FBK502
-rw-r--r-- 1 mx19 mx19   235440 Sep 21 07:37 K6FBK504
-rw-r--r-- 1 mx19 mx19 44436102 Sep 21 15:42 K6FBKB02
-rw-r--r-- 1 mx19 mx19   235040 Sep 21 07:36 K6FBKB04
-rw-r--r-- 1 mx19 mx19   247796 Sep 21 07:37 K6FBKB06
-rw-r--r-- 1 mx19 mx19   235426 Sep 21 07:36 K6FBL102
-rw-r--r-- 1 mx19 mx19   241014 Sep 21 07:37 K6FBLJ82
-rw-r--r-- 1 mx19 mx19   225818 Sep 21 07:36 K6FBLPG2
-rw-r--r-- 1 mx19 mx19   385462 Sep 21 07:47 K6FBM7O2
-rw-r--r-- 1 mx19 mx19   392722 Sep 21 07:48 K6FBM7O4
-rw-r--r-- 1 mx19 mx19   388660 Sep 21 07:48 K6FBMDG2
-rw-r--r-- 1 mx19 mx19   405828 Sep 21 07:49 K6FBMDG4
-rw-r--r-- 1 mx19 mx19 43233484 Sep 21 15:35 K6FBMT02
-rw-r--r-- 1 mx19 mx19   382466 Sep 21 07:47 K6FBN102
-rw-r--r-- 1 mx19 mx19 43799398 Sep 21 15:40 K6FBN502
-rw-r--r-- 1 mx19 mx19   234444 Sep 21 07:36 K6FBN504
-rw-r--r-- 1 mx19 mx19 43468292 Sep 21 15:37 K6FBN702
-rw-r--r-- 1 mx19 mx19   226428 Sep 21 07:36 K6FBN704
-rw-r--r-- 1 mx19 mx19   232660 Sep 21 07:36 K6FBN706
-rw-r--r-- 1 mx19 mx19 42619520 Sep 21 15:34 K6FBN8G2
-rw-r--r-- 1 mx19 mx19 42066366 Sep 21 15:33 K6FBP0G2
-rw-r--r-- 1 mx19 mx19 42481034 Sep 21 15:33 K6FBP202
-rw-r--r-- 1 mx19 mx19   291254 Sep 21 07:39 K6FBP6G2
-rw-r--r-- 1 mx19 mx19   285228 Sep 21 07:39 K6FBPO82
-rw-r--r-- 1 mx19 mx19   295366 Sep 21 07:39 K6FBQ0G2
-rw-r--r-- 1 mx19 mx19   295366 Sep 21 07:39 K6FBQ0G4
-rw-r--r-- 1 mx19 mx19   312066 Sep 21 07:40 K6FBQ0G6
-rw-r--r-- 1 mx19 mx19   313910 Sep 21 07:40 K6FBQ0G8
-rw-r--r-- 1 mx19 mx19 42598050 Sep 21 15:34 K6FBQCG2
-rw-r--r-- 1 mx19 mx19 13794446 Sep 21 15:14 K6FBSAO2
-rw-r--r-- 1 mx19 mx19 14263000 Sep 21 15:15 K6FBSE82
-rw-r--r-- 1 mx19 mx19 17433096 Sep 21 15:20 K6FBSQ82
-rw-r--r-- 1 mx19 mx19 16788070 Sep 21 15:18 K6FBTDO2
-rw-r--r-- 1 mx19 mx19 16768270 Sep 21 15:18 K6FBTUO2
-rw-r--r-- 1 mx19 mx19 17200840 Sep 21 15:20 K6FC0AO2
-rw-r--r-- 1 mx19 mx19 16462516 Sep 21 15:17 K6FC0PG2
-rw-r--r-- 1 mx19 mx19 17302548 Sep 21 15:20 K6FC1582
-rw-r--r-- 1 mx19 mx19 16628658 Sep 21 15:18 K6FC1OO2
-rw-r--r-- 1 mx19 mx19 18297746 Sep 21 15:22 K6FC2702
-rw-r--r-- 1 mx19 mx19 14688084 Sep 21 15:15 K6FC2M02
-rw-r--r-- 1 mx19 mx19 14059884 Sep 21 15:14 K6FC2N02
-rw-r--r-- 1 mx19 mx19 16548906 Sep 21 15:17 K6FC2RG2
-rw-r--r-- 1 mx19 mx19 17188672 Sep 21 15:19 K6FC3N02
-rw-r--r-- 1 mx19 mx19 13301982 Sep 21 15:13 K6FC3T82
-rw-r--r-- 1 mx19 mx19 16508372 Sep 21 15:17 K6FC4D02
-rw-r--r-- 1 mx19 mx19 15157264 Sep 21 15:15 K6FC4KG2
-rw-r--r-- 1 mx19 mx19 13413792 Sep 21 15:13 K6FC4SO2
-rw-r--r-- 1 mx19 mx19 16902108 Sep 21 15:19 K6FC5PO2
-rw-r--r-- 1 mx19 mx19 16241910 Sep 21 15:16 K6FC63O2
-rw-r--r-- 1 mx19 mx19 15948848 Sep 21 15:16 K6FC6582
-rw-r--r-- 1 mx19 mx19 15503698 Sep 21 15:16 K6FC69G2
-rw-r--r-- 1 mx19 mx19 15844482 Sep 21 15:16 K6FC6GG2

./01NVb-003-004/T3:
total 135988
-rw-r--r-- 1 mx19 mx19 139239567 Sep 21 16:10 '01NVb-003-004-3 EF%.mp4'
drwxrwxr-x 2 mx19 mx19      4096 Sep 21 15:37  _P160321

./01NVb-003-004/T3/_P160321:
total 296672
-rw-r--r-- 1 mx19 mx19 34785596 Sep 21 15:28 K6FGCO82
-rw-r--r-- 1 mx19 mx19   327590 Sep 21 07:41 K6FGCR82
-rw-r--r-- 1 mx19 mx19   333948 Sep 21 07:42 K6FGCR84
-rw-r--r-- 1 mx19 mx19 35267472 Sep 21 15:28 K6FGD602
-rw-r--r-- 1 mx19 mx19   162470 Sep 21 07:26 K6FGDBG2
-rw-r--r-- 1 mx19 mx19 36431128 Sep 21 15:29 K6FGDR82
-rw-r--r-- 1 mx19 mx19   350436 Sep 21 07:43 K6FGEEG2
-rw-r--r-- 1 mx19 mx19   372382 Sep 21 07:45 K6FGEEG4
-rw-r--r-- 1 mx19 mx19 38994146 Sep 21 15:30 K6FGFEO2
-rw-r--r-- 1 mx19 mx19 43473196 Sep 21 15:38 K6FGG202
-rw-r--r-- 1 mx19 mx19 36354550 Sep 21 15:29 K6FGG782
-rw-r--r-- 1 mx19 mx19   226828 Sep 21 07:36 K6FGGN82
-rw-r--r-- 1 mx19 mx19   268662 Sep 21 07:38 K6FGGN84
-rw-r--r-- 1 mx19 mx19 33585134 Sep 21 15:28 K6FGHT02
-rw-r--r-- 1 mx19 mx19   254734 Sep 21 07:38 K6FGHT04
-rw-r--r-- 1 mx19 mx19 41946592 Sep 21 15:32 K6FGIEG2
-rw-r--r-- 1 mx19 mx19   303804 Sep 21 07:40 K6FGIU82
-rw-r--r-- 1 mx19 mx19   317794 Sep 21 07:40 K6FGIU84

./01NVb-003-005:
total 8
drwxrwxr-x 3 mx19 mx19 4096 Sep 22 00:01 T1
drwxrwxr-x 3 mx19 mx19 4096 Sep 22 00:01 T2

./01NVb-003-005/T1:
total 2724548
-rw-r--r-- 1 mx19 mx19 2098626099 Sep 21 18:28 '01Nvb-003-005-1 - echo.mp4'
-rw-r--r-- 1 mx19 mx19  217494711 Sep 21 17:49 '01NVb-003-005-1 EF%.mp4'
-rw-r--r-- 1 mx19 mx19  473793575 Sep 21 18:10 '01NVb-003-005-1 - mea.mp4'
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 18:25  _N104812

./01NVb-003-005/T1/_N104812:
total 770992
-rw-r--r-- 1 mx19 mx19 12257674 Sep 21 08:15 K6JAON02
-rw-r--r-- 1 mx19 mx19 16458388 Sep 21 18:17 K6JAOUG2
-rw-r--r-- 1 mx19 mx19 15148348 Sep 21 18:10 K6JAPAO2
-rw-r--r-- 1 mx19 mx19 13416962 Sep 21 18:11 K6JAPIG2
-rw-r--r-- 1 mx19 mx19 14810638 Sep 21 18:21 K6JAPQG2
-rw-r--r-- 1 mx19 mx19 12929500 Sep 21 18:15 K6JAQ5G2
-rw-r--r-- 1 mx19 mx19 13344680 Sep 21 18:22 K6JAQ982
-rw-r--r-- 1 mx19 mx19 11756330 Sep 21 08:09 K6JAQG82
-rw-r--r-- 1 mx19 mx19 14071192 Sep 21 18:22 K6JAQQG2
-rw-r--r-- 1 mx19 mx19 13672094 Sep 21 18:18 K6JAR202
-rw-r--r-- 1 mx19 mx19 15203828 Sep 21 18:22 K6JAREG2
-rw-r--r-- 1 mx19 mx19 15577042 Sep 21 18:10 K6JARLG2
-rw-r--r-- 1 mx19 mx19 12796824 Sep 21 18:17 K6JARP82
-rw-r--r-- 1 mx19 mx19 13671960 Sep 21 18:23 K6JARTG2
-rw-r--r-- 1 mx19 mx19 23537796 Sep 21 18:17 K6JATT02
-rw-r--r-- 1 mx19 mx19   247754 Sep 21 07:37 K6JATT04
-rw-r--r-- 1 mx19 mx19   365212 Sep 21 07:45 K6JB0282
-rw-r--r-- 1 mx19 mx19   350590 Sep 21 07:43 K6JB0CG2
-rw-r--r-- 1 mx19 mx19   349318 Sep 21 07:43 K6JB0CG4
-rw-r--r-- 1 mx19 mx19   377214 Sep 21 07:46 K6JB0CG6
-rw-r--r-- 1 mx19 mx19 22375276 Sep 21 18:15 K6JB0MG2
-rw-r--r-- 1 mx19 mx19   183254 Sep 21 07:28 K6JB0MG4
-rw-r--r-- 1 mx19 mx19 22825760 Sep 21 18:11 K6JB0PG2
-rw-r--r-- 1 mx19 mx19   360154 Sep 21 07:44 K6JB14G2
-rw-r--r-- 1 mx19 mx19   360422 Sep 21 07:44 K6JB1MG2
-rw-r--r-- 1 mx19 mx19   370566 Sep 21 07:45 K6JB1MG4
-rw-r--r-- 1 mx19 mx19 20834428 Sep 21 18:18 K6JB2282
-rw-r--r-- 1 mx19 mx19 24342944 Sep 21 18:26 K6JB2N82
-rw-r--r-- 1 mx19 mx19 24441018 Sep 21 18:14 K6JB2P82
-rw-r--r-- 1 mx19 mx19 24224128 Sep 21 18:15 K6JB2QG2
-rw-r--r-- 1 mx19 mx19 50380588 Sep 21 18:16 K6JB3502
-rw-r--r-- 1 mx19 mx19 48178452 Sep 21 18:25 K6JB3CG2
-rw-r--r-- 1 mx19 mx19   241506 Sep 21 07:37 K6JB3CG4
-rw-r--r-- 1 mx19 mx19 46922828 Sep 21 18:19 K6JB3G02
-rw-r--r-- 1 mx19 mx19   246392 Sep 21 07:37 K6JB3G04
-rw-r--r-- 1 mx19 mx19   263534 Sep 21 07:38 K6JB3G06
-rw-r--r-- 1 mx19 mx19   202600 Sep 21 07:32 K6JB3P02
-rw-r--r-- 1 mx19 mx19   198650 Sep 21 07:31 K6JB3SO2
-rw-r--r-- 1 mx19 mx19   202770 Sep 21 07:32 K6JB43O2
-rw-r--r-- 1 mx19 mx19   214846 Sep 21 07:34 K6JB48G2
-rw-r--r-- 1 mx19 mx19   214846 Sep 21 07:34 K6JB48G4
-rw-r--r-- 1 mx19 mx19   297788 Sep 21 07:39 K6JB48G6
-rw-r--r-- 1 mx19 mx19   216238 Sep 21 07:34 K6JB4DG2
-rw-r--r-- 1 mx19 mx19   353982 Sep 21 07:43 K6JB52G2
-rw-r--r-- 1 mx19 mx19   345278 Sep 21 07:43 K6JB5GO2
-rw-r--r-- 1 mx19 mx19   368658 Sep 21 07:45 K6JB5SG2
-rw-r--r-- 1 mx19 mx19   368658 Sep 21 07:45 K6JB5SG4
-rw-r--r-- 1 mx19 mx19 39860690 Sep 21 18:20 K6JB6PG2
-rw-r--r-- 1 mx19 mx19 42384802 Sep 21 18:12 K6JB6U02
-rw-r--r-- 1 mx19 mx19 42247930 Sep 21 18:13 K6JB7A82
-rw-r--r-- 1 mx19 mx19   231980 Sep 21 07:36 K6JB7A84
-rw-r--r-- 1 mx19 mx19   230862 Sep 21 07:36 K6JB7A86
-rw-r--r-- 1 mx19 mx19 41339772 Sep 21 18:14 K6JB8IG2
-rw-r--r-- 1 mx19 mx19 38298162 Sep 21 18:21 K6JBA4G2
-rw-r--r-- 1 mx19 mx19 14157326 Sep 21 18:23 K6JBCRG2
-rw-r--r-- 1 mx19 mx19 15298072 Sep 21 18:21 K6JBD8G2
-rw-r--r-- 1 mx19 mx19 15527204 Sep 21 18:25 K6JBDA82
-rw-r--r-- 1 mx19 mx19 14958092 Sep 21 18:23 K6JBDJ82
-rw-r--r-- 1 mx19 mx19 14968528 Sep 21 18:18 K6JBDP02

./01NVb-003-005/T2:
total 898112
-rw-r--r-- 1 mx19 mx19 919658496 Sep 21 18:43 '01NVb-003-005-2 (vent-difficult to do).mp4'
drwxrwxr-x 2 mx19 mx19      4096 Sep 21 19:25  _N112809

./01NVb-003-005/T2/_N112809:
total 372192
-rw-r--r-- 1 mx19 mx19 12643366 Sep 21 18:46 K6MBEIO2
-rw-r--r-- 1 mx19 mx19 14976274 Sep 21 18:45 K6MBEP82
-rw-r--r-- 1 mx19 mx19 16438890 Sep 21 18:47 K6MBFE82
-rw-r--r-- 1 mx19 mx19 13545988 Sep 21 18:56 K6MBFL02
-rw-r--r-- 1 mx19 mx19 15125552 Sep 21 19:08 K6MBG002
-rw-r--r-- 1 mx19 mx19 12010722 Sep 21 08:12 K6MBG982
-rw-r--r-- 1 mx19 mx19 12942122 Sep 21 18:47 K6MBGDG2
-rw-r--r-- 1 mx19 mx19 14434694 Sep 21 18:44 K6MBGR82
-rw-r--r-- 1 mx19 mx19 13749112 Sep 21 18:58 K6MBH282
-rw-r--r-- 1 mx19 mx19 14442512 Sep 21 18:44 K6MBHB02
-rw-r--r-- 1 mx19 mx19 11891364 Sep 21 08:10 K6MBHI02
-rw-r--r-- 1 mx19 mx19 14921914 Sep 21 19:07 K6MBHN02
-rw-r--r-- 1 mx19 mx19 10915890 Sep 21 08:00 K6MBHS82
-rw-r--r-- 1 mx19 mx19 12784038 Sep 21 18:47 K6MBI182
-rw-r--r-- 1 mx19 mx19   328868 Sep 21 07:41 K6MBL002
-rw-r--r-- 1 mx19 mx19 28439476 Sep 21 18:45 K6MBL4G2
-rw-r--r-- 1 mx19 mx19 30467998 Sep 21 19:21 K6MBNTO2
-rw-r--r-- 1 mx19 mx19 33040630 Sep 21 18:46 K6MBP2G2
-rw-r--r-- 1 mx19 mx19 32659692 Sep 21 18:48 K6MBP502
-rw-r--r-- 1 mx19 mx19   218740 Sep 21 07:35 K6MBRD82
-rw-r--r-- 1 mx19 mx19   210606 Sep 21 07:33 K6MBROG2
-rw-r--r-- 1 mx19 mx19   217348 Sep 21 07:34 K6MBSUO2
-rw-r--r-- 1 mx19 mx19   215002 Sep 21 07:34 K6MBT7G2
-rw-r--r-- 1 mx19 mx19 32962170 Sep 21 19:26 K6MBTJO2
-rw-r--r-- 1 mx19 mx19   223344 Sep 21 07:35 K6MC05O2
-rw-r--r-- 1 mx19 mx19   216010 Sep 21 07:34 K6MC2382
-rw-r--r-- 1 mx19 mx19 31032466 Sep 21 18:44 K6MC3KO2

./01NVb-003-006:
total 12
drwxrwxr-x 4 mx19 mx19 4096 Sep 21 22:19 T1
drwxrwxr-x 3 mx19 mx19 4096 Sep 21 20:56 T2
drwxrwxr-x 3 mx19 mx19 4096 Sep 22 00:02 T3

./01NVb-003-006/T1:
total 2051116
-rw-r--r-- 1 mx19 mx19 2100330176 Sep 21 23:16 '01NVb-003-006-1 echo.mp4'
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 19:29  _D101703
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 19:18  _D105452

./01NVb-003-006/T1/_D101703:
total 776684
-rw-r--r-- 1 mx19 mx19 35676110 Sep 21 19:30 K6NAAM82
-rw-r--r-- 1 mx19 mx19 34424878 Sep 21 19:29 K6NAB3O2
-rw-r--r-- 1 mx19 mx19   360132 Sep 21 07:44 K6NAB6O2
-rw-r--r-- 1 mx19 mx19   361632 Sep 21 07:44 K6NABE82
-rw-r--r-- 1 mx19 mx19 33447802 Sep 21 19:27 K6NABQ02
-rw-r--r-- 1 mx19 mx19   364316 Sep 21 07:45 K6NADC02
-rw-r--r-- 1 mx19 mx19 33594822 Sep 21 19:28 K6NADU82
-rw-r--r-- 1 mx19 mx19 33426832 Sep 21 19:27 K6NAE682
-rw-r--r-- 1 mx19 mx19   362548 Sep 21 07:44 K6NAE982
-rw-r--r-- 1 mx19 mx19   348934 Sep 21 07:43 K6NAEHO2
-rw-r--r-- 1 mx19 mx19   355254 Sep 21 07:44 K6NAEP82
-rw-r--r-- 1 mx19 mx19   364574 Sep 21 07:45 K6NAEUO2
-rw-r--r-- 1 mx19 mx19 32251392 Sep 21 19:24 K6NAFA82
-rw-r--r-- 1 mx19 mx19 33253036 Sep 21 19:27 K6NAFH02
-rw-r--r-- 1 mx19 mx19 31125808 Sep 21 19:22 K6NAFN82
-rw-r--r-- 1 mx19 mx19 30370424 Sep 21 19:21 K6NAFPG2
-rw-r--r-- 1 mx19 mx19 31434924 Sep 21 19:23 K6NAG5O2
-rw-r--r-- 1 mx19 mx19   222290 Sep 21 07:35 K6NAH7O2
-rw-r--r-- 1 mx19 mx19 32312524 Sep 21 19:25 K6NAHCG2
-rw-r--r-- 1 mx19 mx19 31595928 Sep 21 19:24 K6NAHE02
-rw-r--r-- 1 mx19 mx19 30220740 Sep 21 19:20 K6NAHI02
-rw-r--r-- 1 mx19 mx19 30346050 Sep 21 19:21 K6NAJ182
-rw-r--r-- 1 mx19 mx19 29620666 Sep 21 19:20 K6NAJ2G2
-rw-r--r-- 1 mx19 mx19 24067448 Sep 21 19:19 K6NAK9G2
-rw-r--r-- 1 mx19 mx19   177852 Sep 21 07:27 K6NANKG2
-rw-r--r-- 1 mx19 mx19 29425864 Sep 21 19:20 K6NAOEO2
-rw-r--r-- 1 mx19 mx19 28498256 Sep 21 19:19 K6NAOQ82
-rw-r--r-- 1 mx19 mx19   316300 Sep 21 07:40 K6NAP7O2
-rw-r--r-- 1 mx19 mx19 11525704 Sep 21 08:05 K6NAQT02
-rw-r--r-- 1 mx19 mx19 16161006 Sep 21 19:14 K6NARAG2
-rw-r--r-- 1 mx19 mx19 13044366 Sep 21 18:53 K6NARKG2
-rw-r--r-- 1 mx19 mx19 12270862 Sep 21 08:15 K6NARQ82
-rw-r--r-- 1 mx19 mx19 15149848 Sep 21 19:08 K6NB09G2
-rw-r--r-- 1 mx19 mx19 13525478 Sep 21 18:56 K6NB0D82
-rw-r--r-- 1 mx19 mx19 15659404 Sep 21 19:11 K6NB0U82
-rw-r--r-- 1 mx19 mx19 15112890 Sep 21 19:08 K6NB1A82
-rw-r--r-- 1 mx19 mx19 14052336 Sep 21 19:00 K6NB1G02
-rw-r--r-- 1 mx19 mx19 14595648 Sep 21 19:04 K6NB1OO2
-rw-r--r-- 1 mx19 mx19 12558452 Sep 21 18:51 K6NB1U02
-rw-r--r-- 1 mx19 mx19 15037442 Sep 21 19:08 K6NB2CG2
-rw-r--r-- 1 mx19 mx19 14572424 Sep 21 19:03 K6NB2I82
-rw-r--r-- 1 mx19 mx19 15445176 Sep 21 19:11 K6NB2L02
-rw-r--r-- 1 mx19 mx19 14224348 Sep 21 19:01 K6NB2U02
-rw-r--r-- 1 mx19 mx19 13981546 Sep 21 19:00 K6NB31G2

./01NVb-003-006/T1/_D105452:
total 173932
-rw-r--r-- 1 mx19 mx19 22890046 Sep 21 19:19 K6OB1702
-rw-r--r-- 1 mx19 mx19   207700 Sep 21 07:33 K6OB1704
-rw-r--r-- 1 mx19 mx19   228150 Sep 21 07:36 K6OB1706
-rw-r--r-- 1 mx19 mx19 22186488 Sep 21 19:19 K6OB1G82
-rw-r--r-- 1 mx19 mx19   200760 Sep 21 07:31 K6OB1G84
-rw-r--r-- 1 mx19 mx19   223060 Sep 21 07:35 K6OB1G86
-rw-r--r-- 1 mx19 mx19 21858420 Sep 21 19:18 K6OB1LG2
-rw-r--r-- 1 mx19 mx19   199684 Sep 21 07:31 K6OB1LG4
-rw-r--r-- 1 mx19 mx19   199132 Sep 21 07:31 K6OB1LG6
-rw-r--r-- 1 mx19 mx19   222350 Sep 21 07:35 K6OB1LG8
-rw-r--r-- 1 mx19 mx19 21642614 Sep 21 19:18 K6OB28O2
-rw-r--r-- 1 mx19 mx19   198998 Sep 21 07:31 K6OB28O4
-rw-r--r-- 1 mx19 mx19   226642 Sep 21 07:36 K6OB28O6
-rw-r--r-- 1 mx19 mx19 21676214 Sep 21 19:18 K6OB3502
-rw-r--r-- 1 mx19 mx19   198696 Sep 21 07:31 K6OB3504
-rw-r--r-- 1 mx19 mx19   219404 Sep 21 07:35 K6OB3506
-rw-r--r-- 1 mx19 mx19 21405194 Sep 21 19:17 K6OB39O2
-rw-r--r-- 1 mx19 mx19   190110 Sep 21 07:29 K6OB39O4
-rw-r--r-- 1 mx19 mx19   210294 Sep 21 07:33 K6OB39O6
-rw-r--r-- 1 mx19 mx19 21227680 Sep 21 19:17 K6OB3DG2
-rw-r--r-- 1 mx19 mx19   192848 Sep 21 07:29 K6OB3DG4
-rw-r--r-- 1 mx19 mx19   212968 Sep 21 07:34 K6OB3DG6
-rw-r--r-- 1 mx19 mx19 21624300 Sep 21 19:18 K6OB3IG2
-rw-r--r-- 1 mx19 mx19   196916 Sep 21 07:30 K6OB3IG4
-rw-r--r-- 1 mx19 mx19   217036 Sep 21 07:34 K6OB3IG6

./01NVb-003-006/T2:
total 1621396
-rw-r--r-- 1 mx19 mx19  255937083 Sep 21 20:26 '01NVb-003-006-2 echo.mp4'
-rw-r--r-- 1 mx19 mx19 1148413538 Sep 21 21:19  01NVb-003-006-2-echo.mp4
-rw-r--r-- 1 mx19 mx19  255937083 Sep 21 20:27 '01NVb-003-006-2 mea.mp4'
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 19:46  _T152537

./01NVb-003-006/T2/_T152537:
total 410136
-rw-r--r-- 1 mx19 mx19 42584948 Sep 21 19:45 K6PFE382
-rw-r--r-- 1 mx19 mx19   383716 Sep 21 07:47 K6PFE5O2
-rw-r--r-- 1 mx19 mx19   355572 Sep 21 07:44 K6PFEO02
-rw-r--r-- 1 mx19 mx19   381828 Sep 21 07:47 K6PFEO04
-rw-r--r-- 1 mx19 mx19   383698 Sep 21 07:47 K6PFEO06
-rw-r--r-- 1 mx19 mx19   398182 Sep 21 07:48 K6PFEO08
-rw-r--r-- 1 mx19 mx19   391234 Sep 21 07:48 K6PFEO0A
-rw-r--r-- 1 mx19 mx19   384148 Sep 21 07:47 K6PFEO0C
-rw-r--r-- 1 mx19 mx19   194118 Sep 21 07:30 K6PFFC82
-rw-r--r-- 1 mx19 mx19 42099580 Sep 21 19:41 K6PFFTO2
-rw-r--r-- 1 mx19 mx19 40433386 Sep 21 19:37 K6PFG182
-rw-r--r-- 1 mx19 mx19   361070 Sep 21 07:44 K6PFGRG2
-rw-r--r-- 1 mx19 mx19   368620 Sep 21 07:45 K6PFGRG4
-rw-r--r-- 1 mx19 mx19 39509766 Sep 21 19:34 K6PFHQ02
-rw-r--r-- 1 mx19 mx19 39382094 Sep 21 19:34 K6PFII82
-rw-r--r-- 1 mx19 mx19   222106 Sep 21 07:35 K6PFIO02
-rw-r--r-- 1 mx19 mx19   253058 Sep 21 07:38 K6PFIO04
-rw-r--r-- 1 mx19 mx19   357278 Sep 21 07:44 K6PFJUG2
-rw-r--r-- 1 mx19 mx19   372920 Sep 21 07:45 K6PFJUG4
-rw-r--r-- 1 mx19 mx19 43233510 Sep 21 19:48 K6PFKJ02
-rw-r--r-- 1 mx19 mx19   297590 Sep 21 07:39 K6PFKJ04
-rw-r--r-- 1 mx19 mx19   319102 Sep 21 07:41 K6PFKUO2
-rw-r--r-- 1 mx19 mx19 11385190 Sep 21 08:03 K6PFLPO2
-rw-r--r-- 1 mx19 mx19 14801796 Sep 21 19:05 K6PFM282
-rw-r--r-- 1 mx19 mx19 12463198 Sep 21 18:51 K6PFM882
-rw-r--r-- 1 mx19 mx19 14404628 Sep 21 19:02 K6PFMC02
-rw-r--r-- 1 mx19 mx19 11387850 Sep 21 08:04 K6PFMK82
-rw-r--r-- 1 mx19 mx19  9434216 Sep 21 07:52 K6PFMQ82
-rw-r--r-- 1 mx19 mx19 11563008 Sep 21 08:06 K6PFN002
-rw-r--r-- 1 mx19 mx19 11243760 Sep 21 08:02 K6PFN502
-rw-r--r-- 1 mx19 mx19 13106218 Sep 21 18:53 K6PFN7O2
-rw-r--r-- 1 mx19 mx19 11565206 Sep 21 08:06 K6PFNG02
-rw-r--r-- 1 mx19 mx19  9986842 Sep 21 07:54 K6PFNK82
-rw-r--r-- 1 mx19 mx19 14096880 Sep 21 19:01 K6PFNNO2
-rw-r--r-- 1 mx19 mx19  9732286 Sep 21 07:53 K6PFNSO2
-rw-r--r-- 1 mx19 mx19 12070766 Sep 21 08:12 K6PFO582

./01NVb-003-006/T3:
total 1222396
-rw-r--r-- 1 mx19 mx19 1251725099 Sep 21 21:43 01NVb-003-006-3-echo.mp4
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 19:45 _T144343

./01NVb-003-006/T3/_T144343:
total 573184
-rw-r--r-- 1 mx19 mx19 13630810 Sep 21 18:57 K6QEMD82
-rw-r--r-- 1 mx19 mx19 16374910 Sep 21 19:15 K6QEMJ82
-rw-r--r-- 1 mx19 mx19 15237682 Sep 21 19:09 K6QEMO82
-rw-r--r-- 1 mx19 mx19 14924158 Sep 21 19:07 K6QEN0O2
-rw-r--r-- 1 mx19 mx19 14818170 Sep 21 19:06 K6QENB02
-rw-r--r-- 1 mx19 mx19 14776816 Sep 21 19:05 K6QENJ82
-rw-r--r-- 1 mx19 mx19 16237138 Sep 21 19:14 K6QENLO2
-rw-r--r-- 1 mx19 mx19 14923110 Sep 21 19:07 K6QENP02
-rw-r--r-- 1 mx19 mx19 13468326 Sep 21 18:56 K6QENTG2
-rw-r--r-- 1 mx19 mx19 13940088 Sep 21 18:59 K6QEO7G2
-rw-r--r-- 1 mx19 mx19 14846466 Sep 21 19:06 K6QEODO2
-rw-r--r-- 1 mx19 mx19 17230376 Sep 21 19:16 K6QEOJO2
-rw-r--r-- 1 mx19 mx19 13183800 Sep 21 18:54 K6QEOKO2
-rw-r--r-- 1 mx19 mx19 14764378 Sep 21 19:05 K6QEOQG2
-rw-r--r-- 1 mx19 mx19 13984448 Sep 21 19:00 K6QEOU82
-rw-r--r-- 1 mx19 mx19   355514 Sep 21 07:44 K6QERR02
-rw-r--r-- 1 mx19 mx19 42401594 Sep 21 19:42 K6QESLG2
-rw-r--r-- 1 mx19 mx19 38788432 Sep 21 19:32 K6QESS02
-rw-r--r-- 1 mx19 mx19 38771180 Sep 21 19:32 K6QET482
-rw-r--r-- 1 mx19 mx19 35196420 Sep 21 19:30 K6QET702
-rw-r--r-- 1 mx19 mx19 39236326 Sep 21 19:33 K6QETA82
-rw-r--r-- 1 mx19 mx19   308496 Sep 21 07:40 K6QETU82
-rw-r--r-- 1 mx19 mx19 41977206 Sep 21 19:41 K6QF20O2
-rw-r--r-- 1 mx19 mx19   334228 Sep 21 07:42 K6QF2DG2
-rw-r--r-- 1 mx19 mx19   357350 Sep 21 07:44 K6QF2DG4
-rw-r--r-- 1 mx19 mx19   214236 Sep 21 07:34 K6QF4NG2
-rw-r--r-- 1 mx19 mx19   372516 Sep 21 07:45 K6QF5L02
-rw-r--r-- 1 mx19 mx19   385524 Sep 21 07:47 K6QF5L04
-rw-r--r-- 1 mx19 mx19   380760 Sep 21 07:46 K6QF5L06
-rw-r--r-- 1 mx19 mx19   199780 Sep 21 07:31 K6QF6C02
-rw-r--r-- 1 mx19 mx19 41309298 Sep 21 19:39 K6QF6LO2
-rw-r--r-- 1 mx19 mx19 43161454 Sep 21 19:46 K6QF6P82
-rw-r--r-- 1 mx19 mx19 40784906 Sep 21 19:37 K6QF7T02

./01NVb-003-007:
total 12
drwxrwxr-x 4 mx19 mx19 4096 Sep 21 21:08 T1
drwxrwxr-x 4 mx19 mx19 4096 Sep 22 00:02 T2
drwxrwxr-x 3 mx19 mx19 4096 Sep 21 07:28 T3

./01NVb-003-007/T1:
total 2313788
-rw-r--r-- 1 mx19 mx19 1139852359 Sep 21 21:08 '01NVb-003-007-1 echo.mp4'
-rw-r--r-- 1 mx19 mx19 1153759712 Sep 21 21:30  01NVb-003-007-1-echo.mp4
-rw-r--r-- 1 mx19 mx19   75680110 Sep 21 20:17 '01NVb-003-007-1 mea.mp4'
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 20:12  2N145238
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 20:14  2N145859

./01NVb-003-007/T1/2N145238:
total 835424
-rw-r--r-- 1 mx19 mx19 14530722 Sep 21 19:03 K6OERGG2
-rw-r--r-- 1 mx19 mx19 13258392 Sep 21 18:54 K6OERMO2
-rw-r--r-- 1 mx19 mx19 16653880 Sep 21 19:15 K6OES9G2
-rw-r--r-- 1 mx19 mx19 15362106 Sep 21 19:10 K6OESGO2
-rw-r--r-- 1 mx19 mx19 17585876 Sep 21 19:17 K6OESMG2
-rw-r--r-- 1 mx19 mx19 13416250 Sep 21 18:55 K6OET482
-rw-r--r-- 1 mx19 mx19 15382064 Sep 21 19:10 K6OET702
-rw-r--r-- 1 mx19 mx19 14526308 Sep 21 19:03 K6OETG82
-rw-r--r-- 1 mx19 mx19 15680458 Sep 21 19:11 K6OETQG2
-rw-r--r-- 1 mx19 mx19 15896480 Sep 21 19:13 K6OF0I82
-rw-r--r-- 1 mx19 mx19 15883392 Sep 21 19:12 K6OF0K02
-rw-r--r-- 1 mx19 mx19 14768752 Sep 21 19:05 K6OF0S02
-rw-r--r-- 1 mx19 mx19 15854108 Sep 21 19:12 K6OF1682
-rw-r--r-- 1 mx19 mx19 15350394 Sep 21 19:10 K6OF1JO2
-rw-r--r-- 1 mx19 mx19 17302428 Sep 21 19:17 K6OF1O82
-rw-r--r-- 1 mx19 mx19 46921908 Sep 21 20:01 K6OF3O82
-rw-r--r-- 1 mx19 mx19   341388 Sep 21 07:42 K6OF3TO2
-rw-r--r-- 1 mx19 mx19 51646052 Sep 21 20:10 K6OF4GO2
-rw-r--r-- 1 mx19 mx19   369538 Sep 21 07:45 K6OF4KO2
-rw-r--r-- 1 mx19 mx19 47045420 Sep 21 20:02 K6OF5402
-rw-r--r-- 1 mx19 mx19 43684700 Sep 21 19:51 K6OF5882
-rw-r--r-- 1 mx19 mx19 49731408 Sep 21 20:09 K6OF5CG2
-rw-r--r-- 1 mx19 mx19 48419642 Sep 21 20:07 K6OF5E02
-rw-r--r-- 1 mx19 mx19 41569434 Sep 21 19:40 K6OF5GO2
-rw-r--r-- 1 mx19 mx19 49042022 Sep 21 20:07 K6OF6J02
-rw-r--r-- 1 mx19 mx19 54656294 Sep 21 20:14 K6OF6RO2
-rw-r--r-- 1 mx19 mx19 52323718 Sep 21 20:11 K6OF7702
-rw-r--r-- 1 mx19 mx19 39361390 Sep 21 19:33 K6OF7HG2
-rw-r--r-- 1 mx19 mx19 52426762 Sep 21 20:12 K6OF7P02
-rw-r--r-- 1 mx19 mx19 46142352 Sep 21 19:57 K6OF8DG2
-rw-r--r-- 1 mx19 mx19   272704 Sep 21 07:38 K6OF9002

./01NVb-003-007/T1/2N145859:
total 494784
-rw-r--r-- 1 mx19 mx19 13457852 Sep 21 18:55 K6PF0102
-rw-r--r-- 1 mx19 mx19 13115206 Sep 21 18:53 K6PF0582
-rw-r--r-- 1 mx19 mx19 15733932 Sep 21 19:12 K6PF0E82
-rw-r--r-- 1 mx19 mx19 13571484 Sep 21 18:56 K6PF0I82
-rw-r--r-- 1 mx19 mx19 15399526 Sep 21 19:10 K6PF0NO2
-rw-r--r-- 1 mx19 mx19 12243640 Sep 21 08:14 K6PF0S02
-rw-r--r-- 1 mx19 mx19 12740926 Sep 21 18:52 K6PF0UG2
-rw-r--r-- 1 mx19 mx19 12626738 Sep 21 18:51 K6PF1582
-rw-r--r-- 1 mx19 mx19 13805812 Sep 21 18:58 K6PF1BG2
-rw-r--r-- 1 mx19 mx19  4963764 Sep 21 07:51 K6PF1CG2
-rw-r--r-- 1 mx19 mx19 15966118 Sep 21 19:13 K6PF1NO2
-rw-r--r-- 1 mx19 mx19 13035442 Sep 21 18:53 K6PF1SG2
-rw-r--r-- 1 mx19 mx19 14893716 Sep 21 19:06 K6PF2282
-rw-r--r-- 1 mx19 mx19 15021744 Sep 21 19:07 K6PF2682
-rw-r--r-- 1 mx19 mx19 13943322 Sep 21 19:00 K6PF28O2
-rw-r--r-- 1 mx19 mx19 33539322 Sep 21 19:27 K6PF4SO2
-rw-r--r-- 1 mx19 mx19   332022 Sep 21 07:41 K6PF52O2
-rw-r--r-- 1 mx19 mx19   342274 Sep 21 07:42 K6PF52O4
-rw-r--r-- 1 mx19 mx19   337878 Sep 21 07:42 K6PF5IO2
-rw-r--r-- 1 mx19 mx19   337878 Sep 21 07:42 K6PF5IO4
-rw-r--r-- 1 mx19 mx19   384226 Sep 21 07:47 K6PF5IO6
-rw-r--r-- 1 mx19 mx19   162278 Sep 21 07:26 K6PF6D82
-rw-r--r-- 1 mx19 mx19   162278 Sep 21 07:26 K6PF6D84
-rw-r--r-- 1 mx19 mx19 43330310 Sep 21 19:48 K6PF6O82
-rw-r--r-- 1 mx19 mx19 46752612 Sep 21 19:58 K6PF6R82
-rw-r--r-- 1 mx19 mx19 41301928 Sep 21 19:38 K6PF7OO2
-rw-r--r-- 1 mx19 mx19   206852 Sep 21 07:33 K6PF7OO4
-rw-r--r-- 1 mx19 mx19   224464 Sep 21 07:36 K6PF7OO6
-rw-r--r-- 1 mx19 mx19   208228 Sep 21 07:33 K6PF7OO8
-rw-r--r-- 1 mx19 mx19   204300 Sep 21 07:32 K6PF7OOA
-rw-r--r-- 1 mx19 mx19   204300 Sep 21 07:32 K6PF7OOC
-rw-r--r-- 1 mx19 mx19 36146214 Sep 21 19:30 K6PF8OO2
-rw-r--r-- 1 mx19 mx19   197796 Sep 21 07:30 K6PF8T82
-rw-r--r-- 1 mx19 mx19   229848 Sep 21 07:36 K6PF8T84
-rw-r--r-- 1 mx19 mx19 46053030 Sep 21 19:57 K6PFAEO2
-rw-r--r-- 1 mx19 mx19   341056 Sep 21 07:42 K6PFAO02
-rw-r--r-- 1 mx19 mx19   343032 Sep 21 07:43 K6PFAO04
-rw-r--r-- 1 mx19 mx19 54726144 Sep 21 20:15 K6PFC082

./01NVb-003-007/T2:
total 2296784
-rw-r--r-- 1 mx19 mx19  252555785 Sep 21 20:22 '01NVb-003-007-2 echo cont (2).mp4'
-rw-r--r-- 1 mx19 mx19 2099331035 Sep 21 22:50 '01NVb-003-007-2 echo.mp4'
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 20:14  2N145859
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 20:11  2N152617

./01NVb-003-007/T2/2N145859:
total 494788
-rw-r--r-- 1 mx19 mx19 13457852 Sep 21 18:56 K6PF0102
-rw-r--r-- 1 mx19 mx19 13115206 Sep 21 18:53 K6PF0582
-rw-r--r-- 1 mx19 mx19 15733932 Sep 21 19:12 K6PF0E82
-rw-r--r-- 1 mx19 mx19 13571484 Sep 21 18:57 K6PF0I82
-rw-r--r-- 1 mx19 mx19 15399526 Sep 21 19:11 K6PF0NO2
-rw-r--r-- 1 mx19 mx19 12243640 Sep 21 08:14 K6PF0S02
-rw-r--r-- 1 mx19 mx19 12740926 Sep 21 18:52 K6PF0UG2
-rw-r--r-- 1 mx19 mx19 12626738 Sep 21 18:51 K6PF1582
-rw-r--r-- 1 mx19 mx19 13805812 Sep 21 18:58 K6PF1BG2
-rw-r--r-- 1 mx19 mx19  4963764 Sep 21 07:51 K6PF1CG2
-rw-r--r-- 1 mx19 mx19 15966118 Sep 21 19:14 K6PF1NO2
-rw-r--r-- 1 mx19 mx19 13035442 Sep 21 18:53 K6PF1SG2
-rw-r--r-- 1 mx19 mx19 14893716 Sep 21 19:06 K6PF2282
-rw-r--r-- 1 mx19 mx19 15021744 Sep 21 19:07 K6PF2682
-rw-r--r-- 1 mx19 mx19 13943322 Sep 21 19:00 K6PF28O2
-rw-r--r-- 1 mx19 mx19 33539322 Sep 21 19:28 K6PF4SO2
-rw-r--r-- 1 mx19 mx19   332022 Sep 21 07:41 K6PF52O2
-rw-r--r-- 1 mx19 mx19   342274 Sep 21 07:42 K6PF52O4
-rw-r--r-- 1 mx19 mx19   337878 Sep 21 07:42 K6PF5IO2
-rw-r--r-- 1 mx19 mx19   337878 Sep 21 07:42 K6PF5IO4
-rw-r--r-- 1 mx19 mx19   384226 Sep 21 07:47 K6PF5IO6
-rw-r--r-- 1 mx19 mx19   162278 Sep 21 07:26 K6PF6D82
-rw-r--r-- 1 mx19 mx19   162278 Sep 21 07:26 K6PF6D84
-rw-r--r-- 1 mx19 mx19 43330310 Sep 21 19:49 K6PF6O82
-rw-r--r-- 1 mx19 mx19 46752612 Sep 21 19:59 K6PF6R82
-rw-r--r-- 1 mx19 mx19 41301928 Sep 21 19:38 K6PF7OO2
-rw-r--r-- 1 mx19 mx19   206852 Sep 21 07:33 K6PF7OO4
-rw-r--r-- 1 mx19 mx19   224464 Sep 21 07:36 K6PF7OO6
-rw-r--r-- 1 mx19 mx19   208228 Sep 21 07:33 K6PF7OO8
-rw-r--r-- 1 mx19 mx19   204300 Sep 21 07:32 K6PF7OOA
-rw-r--r-- 1 mx19 mx19   204300 Sep 21 07:32 K6PF7OOC
-rw-r--r-- 1 mx19 mx19 36146214 Sep 21 19:31 K6PF8OO2
-rw-r--r-- 1 mx19 mx19   197796 Sep 21 07:30 K6PF8T82
-rw-r--r-- 1 mx19 mx19   229848 Sep 21 07:36 K6PF8T84
-rw-r--r-- 1 mx19 mx19 46053030 Sep 21 19:57 K6PFAEO2
-rw-r--r-- 1 mx19 mx19   341056 Sep 21 07:42 K6PFAO02
-rw-r--r-- 1 mx19 mx19   343032 Sep 21 07:43 K6PFAO04
-rw-r--r-- 1 mx19 mx19 54726144 Sep 21 20:15 K6PFC082

./01NVb-003-007/T2/2N152617:
total 632124
-rw-r--r-- 1 mx19 mx19 49413156 Sep 21 20:08 K6QFGN82
-rw-r--r-- 1 mx19 mx19   344394 Sep 21 07:43 K6QFGS02
-rw-r--r-- 1 mx19 mx19   353804 Sep 21 07:43 K6QFGS04
-rw-r--r-- 1 mx19 mx19 53827458 Sep 21 20:12 K6QFHJO2
-rw-r--r-- 1 mx19 mx19 48159234 Sep 21 20:05 K6QFHQO2
-rw-r--r-- 1 mx19 mx19   373238 Sep 21 07:45 K6QFI502
-rw-r--r-- 1 mx19 mx19   385170 Sep 21 07:47 K6QFJTG2
-rw-r--r-- 1 mx19 mx19 45537100 Sep 21 19:54 K6QFL4O2
-rw-r--r-- 1 mx19 mx19   194394 Sep 21 07:30 K6QFL5G2
-rw-r--r-- 1 mx19 mx19 50391496 Sep 21 20:09 K6QFMK82
-rw-r--r-- 1 mx19 mx19 47261634 Sep 21 20:02 K6QFNDG2
-rw-r--r-- 1 mx19 mx19 42460036 Sep 21 19:43 K6QFOQ82
-rw-r--r-- 1 mx19 mx19 42566158 Sep 21 19:44 K6QFPL02
-rw-r--r-- 1 mx19 mx19   246926 Sep 21 07:37 K6QFRCG2
-rw-r--r-- 1 mx19 mx19 43340400 Sep 21 19:49 K6QG11O2
-rw-r--r-- 1 mx19 mx19   305996 Sep 21 07:40 K6QG1882
-rw-r--r-- 1 mx19 mx19 13576052 Sep 21 18:57 K6QG2882
-rw-r--r-- 1 mx19 mx19 14740498 Sep 21 19:04 K6QG2HG2
-rw-r--r-- 1 mx19 mx19 13386838 Sep 21 18:54 K6QG2O02
-rw-r--r-- 1 mx19 mx19 15834878 Sep 21 19:12 K6QG3382
-rw-r--r-- 1 mx19 mx19 14166978 Sep 21 19:01 K6QG3A82
-rw-r--r-- 1 mx19 mx19 15204916 Sep 21 19:08 K6QG3GO2
-rw-r--r-- 1 mx19 mx19 13759204 Sep 21 18:58 K6QG3QO2
-rw-r--r-- 1 mx19 mx19 16523940 Sep 21 19:15 K6QG44G2
-rw-r--r-- 1 mx19 mx19   186518 Sep 21 07:29 K6QG4882
-rw-r--r-- 1 mx19 mx19 14270100 Sep 21 19:02 K6QG58O2
-rw-r--r-- 1 mx19 mx19 15252526 Sep 21 19:09 K6QG5HG2
-rw-r--r-- 1 mx19 mx19 15912000 Sep 21 19:13 K6QG5Q82
-rw-r--r-- 1 mx19 mx19 13819382 Sep 21 18:58 K6QG64G2
-rw-r--r-- 1 mx19 mx19 16913724 Sep 21 19:16 K6QG68G2
-rw-r--r-- 1 mx19 mx19 14513434 Sep 21 19:02 K6QG6IO2
-rw-r--r-- 1 mx19 mx19 13999650 Sep 21 19:00 K6QG6PO2

./01NVb-003-007/T3:
total 4
drwxrwxr-x 2 mx19 mx19 4096 Sep 21 20:12 2N152617

./01NVb-003-007/T3/2N152617:
total 633696
-rw-r--r-- 1 mx19 mx19 49484500 Sep 21 20:08 K6QFGN82
-rw-r--r-- 1 mx19 mx19   344714 Sep 21 07:43 K6QFGS02
-rw-r--r-- 1 mx19 mx19   353804 Sep 21 07:43 K6QFGS04
-rw-r--r-- 1 mx19 mx19 53901070 Sep 21 20:14 K6QFHJO2
-rw-r--r-- 1 mx19 mx19 48226124 Sep 21 20:06 K6QFHQO2
-rw-r--r-- 1 mx19 mx19   373532 Sep 21 07:45 K6QFI502
-rw-r--r-- 1 mx19 mx19   385170 Sep 21 07:47 K6QFJTG2
-rw-r--r-- 1 mx19 mx19 45605128 Sep 21 19:55 K6QFL4O2
-rw-r--r-- 1 mx19 mx19   194394 Sep 21 07:30 K6QFL5G2
-rw-r--r-- 1 mx19 mx19 50483466 Sep 21 20:10 K6QFMK82
-rw-r--r-- 1 mx19 mx19 47356892 Sep 21 20:03 K6QFNDG2
-rw-r--r-- 1 mx19 mx19 42533588 Sep 21 19:44 K6QFOQ82
-rw-r--r-- 1 mx19 mx19 42643248 Sep 21 19:45 K6QFPL02
-rw-r--r-- 1 mx19 mx19   246926 Sep 21 07:37 K6QFRCG2
-rw-r--r-- 1 mx19 mx19 43423412 Sep 21 19:50 K6QG11O2
-rw-r--r-- 1 mx19 mx19   306266 Sep 21 07:40 K6QG1882
-rw-r--r-- 1 mx19 mx19 13638230 Sep 21 18:57 K6QG2882
-rw-r--r-- 1 mx19 mx19 14803404 Sep 21 19:06 K6QG2HG2
-rw-r--r-- 1 mx19 mx19 13449774 Sep 21 18:55 K6QG2O02
-rw-r--r-- 1 mx19 mx19 15897768 Sep 21 19:13 K6QG3382
-rw-r--r-- 1 mx19 mx19 14229918 Sep 21 19:02 K6QG3A82
-rw-r--r-- 1 mx19 mx19 15267352 Sep 21 19:09 K6QG3GO2
-rw-r--r-- 1 mx19 mx19 13822196 Sep 21 18:59 K6QG3QO2
-rw-r--r-- 1 mx19 mx19 16565096 Sep 21 19:15 K6QG44G2
-rw-r--r-- 1 mx19 mx19   186518 Sep 21 07:28 K6QG4882
-rw-r--r-- 1 mx19 mx19 14311066 Sep 21 19:02 K6QG58O2
-rw-r--r-- 1 mx19 mx19 15315596 Sep 21 19:09 K6QG5HG2
-rw-r--r-- 1 mx19 mx19 15974740 Sep 21 19:14 K6QG5Q82
-rw-r--r-- 1 mx19 mx19 13882524 Sep 21 18:59 K6QG64G2
-rw-r--r-- 1 mx19 mx19 16976024 Sep 21 19:16 K6QG68G2
-rw-r--r-- 1 mx19 mx19 14575906 Sep 21 19:04 K6QG6IO2
-rw-r--r-- 1 mx19 mx19 14062920 Sep 21 19:01 K6QG6PO2

./01NVb-003-008:
total 12
drwxrwxr-x 4 mx19 mx19 4096 Sep 21 21:30 T1
drwxrwxr-x 3 mx19 mx19 4096 Sep 22 00:03 T2
drwxrwxr-x 3 mx19 mx19 4096 Sep 22 00:03 T3

./01NVb-003-008/T1:
total 2263420
-rw-r--r-- 1 mx19 mx19 1544500661 Sep 21 21:58 '01NVb-003-008-1 echo.mp4'
-rw-r--r-- 1 mx19 mx19  247869058 Sep 21 20:21 '01NVb-003-008-1 EF%.mp4'
-rw-r--r-- 1 mx19 mx19  525344793 Sep 21 20:37 '01NVb-003-008-1 mea.mp4'
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 19:59  _N143126
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 20:15  _N150418

./01NVb-003-008/T1/_N143126:
total 785992
-rw-r--r-- 1 mx19 mx19 10106356 Sep 21 07:55 K6UEGA82
-rw-r--r-- 1 mx19 mx19 10937042 Sep 21 08:00 K6UEGH82
-rw-r--r-- 1 mx19 mx19 13449090 Sep 21 18:55 K6UEGTG2
-rw-r--r-- 1 mx19 mx19 13373586 Sep 21 18:54 K6UEH202
-rw-r--r-- 1 mx19 mx19 12670702 Sep 21 18:52 K6UEHHO2
-rw-r--r-- 1 mx19 mx19 11875858 Sep 21 08:10 K6UEHPO2
-rw-r--r-- 1 mx19 mx19 14683880 Sep 21 19:04 K6UEHTO2
-rw-r--r-- 1 mx19 mx19 12050230 Sep 21 08:12 K6UEI402
-rw-r--r-- 1 mx19 mx19 10324554 Sep 21 07:56 K6UEIDG2
-rw-r--r-- 1 mx19 mx19 10352978 Sep 21 07:57 K6UEII82
-rw-r--r-- 1 mx19 mx19 10467014 Sep 21 07:57 K6UEIP02
-rw-r--r-- 1 mx19 mx19 11536210 Sep 21 08:06 K6UEJ1G2
-rw-r--r-- 1 mx19 mx19 11775580 Sep 21 08:09 K6UEJB02
-rw-r--r-- 1 mx19 mx19 15327896 Sep 21 19:09 K6UEJE82
-rw-r--r-- 1 mx19 mx19   325354 Sep 21 07:41 K6UEL8O2
-rw-r--r-- 1 mx19 mx19 46819086 Sep 21 19:59 K6UEMJO2
-rw-r--r-- 1 mx19 mx19   176044 Sep 21 07:27 K6UEMJO4
-rw-r--r-- 1 mx19 mx19 39635164 Sep 21 19:35 K6UEQ2O2
-rw-r--r-- 1 mx19 mx19 44526334 Sep 21 19:53 K6UEQ582
-rw-r--r-- 1 mx19 mx19 42094876 Sep 21 19:41 K6UEQDO2
-rw-r--r-- 1 mx19 mx19 41621188 Sep 21 19:40 K6UEQIG2
-rw-r--r-- 1 mx19 mx19   226978 Sep 21 07:36 K6UEQIG4
-rw-r--r-- 1 mx19 mx19   228576 Sep 21 07:36 K6UEQIG6
-rw-r--r-- 1 mx19 mx19   229294 Sep 21 07:36 K6UEQIG8
-rw-r--r-- 1 mx19 mx19   230228 Sep 21 07:36 K6UEQIGA
-rw-r--r-- 1 mx19 mx19 38503294 Sep 21 19:31 K6UERS82
-rw-r--r-- 1 mx19 mx19   349448 Sep 21 07:43 K6UES702
-rw-r--r-- 1 mx19 mx19   386418 Sep 21 07:47 K6UES704
-rw-r--r-- 1 mx19 mx19   336258 Sep 21 07:42 K6UESH02
-rw-r--r-- 1 mx19 mx19   350182 Sep 21 07:43 K6UESH04
-rw-r--r-- 1 mx19 mx19 40232916 Sep 21 19:36 K6UESP82
-rw-r--r-- 1 mx19 mx19 39282792 Sep 21 19:33 K6UET002
-rw-r--r-- 1 mx19 mx19 40300680 Sep 21 19:36 K6UET4O2
-rw-r--r-- 1 mx19 mx19 32773332 Sep 21 19:25 K6UETA02
-rw-r--r-- 1 mx19 mx19 44685052 Sep 21 19:54 K6UETG02
-rw-r--r-- 1 mx19 mx19 46920038 Sep 21 20:00 K6UF0802
-rw-r--r-- 1 mx19 mx19 44425750 Sep 21 19:52 K6UF0EG2
-rw-r--r-- 1 mx19 mx19 43599510 Sep 21 19:51 K6UF1182
-rw-r--r-- 1 mx19 mx19   242838 Sep 21 07:37 K6UF1184
-rw-r--r-- 1 mx19 mx19   207896 Sep 21 07:33 K6UF1186
-rw-r--r-- 1 mx19 mx19   247676 Sep 21 07:37 K6UF1188
-rw-r--r-- 1 mx19 mx19   247676 Sep 21 07:37 K6UF118A
-rw-r--r-- 1 mx19 mx19 46614354 Sep 21 19:58 K6UF1682

./01NVb-003-008/T1/_N150418:
total 201092
-rw-r--r-- 1 mx19 mx19 52261908 Sep 21 20:11 K6UF2KG2
-rw-r--r-- 1 mx19 mx19 46840216 Sep 21 20:00 K6UF3A82
-rw-r--r-- 1 mx19 mx19   203308 Sep 21 07:32 K6UF3M02
-rw-r--r-- 1 mx19 mx19   203308 Sep 21 07:32 K6UF3M04
-rw-r--r-- 1 mx19 mx19   270278 Sep 21 07:38 K6UF3M06
-rw-r--r-- 1 mx19 mx19   399230 Sep 21 07:49 K6UF3U82
-rw-r--r-- 1 mx19 mx19   413178 Sep 21 07:49 K6UF3U84
-rw-r--r-- 1 mx19 mx19 57753106 Sep 21 20:16 K6UF5D02
-rw-r--r-- 1 mx19 mx19 46986666 Sep 21 20:01 K6UF5R02
-rw-r--r-- 1 mx19 mx19   275172 Sep 21 07:38 K6UF5SG2
-rw-r--r-- 1 mx19 mx19   292796 Sep 21 07:39 K6UF5SG4

./01NVb-003-008/T2:
total 2689704
-rw-r--r-- 1 mx19 mx19 1693822511 Sep 21 22:19 '01NVb-003-008-2 echo.mp4'
-rw-r--r-- 1 mx19 mx19  463750402 Sep 21 20:34 '01NVb-003-008-2 EF%.mp4'
-rw-r--r-- 1 mx19 mx19  596663844 Sep 21 20:45 '01NVb-003-008-2 mea.mp4'
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 20:03  _N105209

./01NVb-003-008/T2/_N105209:
total 947856
-rw-r--r-- 1 mx19 mx19 47685838 Sep 21 20:04 K71AQSG2
-rw-r--r-- 1 mx19 mx19   200752 Sep 21 07:31 K71AQSG4
-rw-r--r-- 1 mx19 mx19   365534 Sep 21 07:45 K71AR882
-rw-r--r-- 1 mx19 mx19   366542 Sep 21 07:45 K71ARP02
-rw-r--r-- 1 mx19 mx19   380984 Sep 21 07:46 K71ARP04
-rw-r--r-- 1 mx19 mx19 39441430 Sep 21 19:34 K71ASBG2
-rw-r--r-- 1 mx19 mx19   201728 Sep 21 07:31 K71ASBG4
-rw-r--r-- 1 mx19 mx19   372466 Sep 21 07:45 K71ASG82
-rw-r--r-- 1 mx19 mx19   409412 Sep 21 07:49 K71ASG84
-rw-r--r-- 1 mx19 mx19   372288 Sep 21 07:45 K71ASKG2
-rw-r--r-- 1 mx19 mx19 44239120 Sep 21 19:52 K71ASQO2
-rw-r--r-- 1 mx19 mx19 47150562 Sep 21 20:02 K71ASU82
-rw-r--r-- 1 mx19 mx19 47352104 Sep 21 20:03 K71AT1G2
-rw-r--r-- 1 mx19 mx19 48060926 Sep 21 20:04 K71B0J02
-rw-r--r-- 1 mx19 mx19 45714106 Sep 21 19:56 K71B0LG2
-rw-r--r-- 1 mx19 mx19 45870888 Sep 21 19:56 K71B0N82
-rw-r--r-- 1 mx19 mx19 39844758 Sep 21 19:36 K71B19G2
-rw-r--r-- 1 mx19 mx19 42157310 Sep 21 19:42 K71B1C82
-rw-r--r-- 1 mx19 mx19   243748 Sep 21 07:37 K71B1C84
-rw-r--r-- 1 mx19 mx19 39518484 Sep 21 19:35 K71B1HG2
-rw-r--r-- 1 mx19 mx19   204692 Sep 21 07:32 K71B1O82
-rw-r--r-- 1 mx19 mx19   198482 Sep 21 07:30 K71B1RO2
-rw-r--r-- 1 mx19 mx19   198482 Sep 21 07:30 K71B1RO4
-rw-r--r-- 1 mx19 mx19   193660 Sep 21 07:29 K71B2282
-rw-r--r-- 1 mx19 mx19   272562 Sep 21 07:38 K71B2284
-rw-r--r-- 1 mx19 mx19   377752 Sep 21 07:46 K71B2IG2
-rw-r--r-- 1 mx19 mx19   380802 Sep 21 07:46 K71B2NO2
-rw-r--r-- 1 mx19 mx19   394204 Sep 21 07:48 K71B2NO4
-rw-r--r-- 1 mx19 mx19 41319558 Sep 21 19:39 K71B3902
-rw-r--r-- 1 mx19 mx19 41654788 Sep 21 19:41 K71B3BG2
-rw-r--r-- 1 mx19 mx19 40476330 Sep 21 19:37 K71B3QO2
-rw-r--r-- 1 mx19 mx19 43190160 Sep 21 19:47 K71B4DO2
-rw-r--r-- 1 mx19 mx19 43437434 Sep 21 19:50 K71B4NO2
-rw-r--r-- 1 mx19 mx19 42489150 Sep 21 19:43 K71B4TO2
-rw-r--r-- 1 mx19 mx19 36834228 Sep 21 19:31 K71B5002
-rw-r--r-- 1 mx19 mx19   307558 Sep 21 07:40 K71B5602
-rw-r--r-- 1 mx19 mx19   308166 Sep 21 07:40 K71B5A82
-rw-r--r-- 1 mx19 mx19   326512 Sep 21 07:41 K71B5A84
-rw-r--r-- 1 mx19 mx19 13401646 Sep 21 18:54 K71B6002
-rw-r--r-- 1 mx19 mx19 11780918 Sep 21 08:09 K71B65G2
-rw-r--r-- 1 mx19 mx19 12516770 Sep 21 18:51 K71B6EG2
-rw-r--r-- 1 mx19 mx19 12459322 Sep 21 18:50 K71B6MO2
-rw-r--r-- 1 mx19 mx19 10864202 Sep 21 07:59 K71B6T82
-rw-r--r-- 1 mx19 mx19 12259686 Sep 21 08:15 K71B72O2
-rw-r--r-- 1 mx19 mx19  9695774 Sep 21 07:53 K71B7AO2
-rw-r--r-- 1 mx19 mx19 12829952 Sep 21 18:52 K71B7G02
-rw-r--r-- 1 mx19 mx19 10252658 Sep 21 07:55 K71B7NG2
-rw-r--r-- 1 mx19 mx19 11057564 Sep 21 08:00 K71B8982
-rw-r--r-- 1 mx19 mx19 10306752 Sep 21 07:56 K71B9302
-rw-r--r-- 1 mx19 mx19  9487350 Sep 21 07:52 K71B98G2
-rw-r--r-- 1 mx19 mx19 11242636 Sep 21 08:02 K71B9E82
-rw-r--r-- 1 mx19 mx19 13025004 Sep 21 18:52 K71B9NO2
-rw-r--r-- 1 mx19 mx19 13676358 Sep 21 18:58 K71BA082
-rw-r--r-- 1 mx19 mx19 13119364 Sep 21 18:54 K71BA582

./01NVb-003-008/T3:
total 979700
-rw-r--r-- 1 mx19 mx19 1003204489 Sep 21 20:56 01NVb-003-008-3-echo.mp4
drwxrwxr-x 2 mx19 mx19       4096 Sep 21 19:28 _N145321

./01NVb-003-008/T3/_N145321:
total 417140
-rw-r--r-- 1 mx19 mx19 11519386 Sep 21 08:05 K72EQT82
-rw-r--r-- 1 mx19 mx19 12523222 Sep 21 18:51 K72ER502
-rw-r--r-- 1 mx19 mx19 14556924 Sep 21 19:03 K72ERA82
-rw-r--r-- 1 mx19 mx19 12866644 Sep 21 18:52 K72ERGG2
-rw-r--r-- 1 mx19 mx19 15577784 Sep 21 19:11 K72EROG2
-rw-r--r-- 1 mx19 mx19 13928044 Sep 21 18:59 K72ES1G2
-rw-r--r-- 1 mx19 mx19 16793596 Sep 21 19:16 K72ES5G2
-rw-r--r-- 1 mx19 mx19 13581956 Sep 21 18:57 K72ESBO2
-rw-r--r-- 1 mx19 mx19 13436570 Sep 21 18:55 K72ESHO2
-rw-r--r-- 1 mx19 mx19 12314974 Sep 21 08:16 K72ESMG2
-rw-r--r-- 1 mx19 mx19 14757206 Sep 21 19:04 K72ET002
-rw-r--r-- 1 mx19 mx19 16084266 Sep 21 19:14 K72ET482
-rw-r--r-- 1 mx19 mx19 14213838 Sep 21 19:01 K72ET9G2
-rw-r--r-- 1 mx19 mx19 14652792 Sep 21 19:04 K72ETEO2
-rw-r--r-- 1 mx19 mx19 34170282 Sep 21 19:29 K72F2LO2
-rw-r--r-- 1 mx19 mx19   349954 Sep 21 07:43 K72F2SO2
-rw-r--r-- 1 mx19 mx19   377792 Sep 21 07:46 K72F2SO4
-rw-r--r-- 1 mx19 mx19   180652 Sep 21 07:28 K72F3L82
-rw-r--r-- 1 mx19 mx19   186686 Sep 21 07:29 K72F3L84
-rw-r--r-- 1 mx19 mx19 31249510 Sep 21 19:22 K72F3T02
-rw-r--r-- 1 mx19 mx19 31521548 Sep 21 19:23 K72F4282
-rw-r--r-- 1 mx19 mx19   350430 Sep 21 07:43 K72F4GG2
-rw-r--r-- 1 mx19 mx19   354632 Sep 21 07:43 K72F4GG4
-rw-r--r-- 1 mx19 mx19 32852034 Sep 21 19:26 K72F5B82
-rw-r--r-- 1 mx19 mx19 30594964 Sep 21 19:22 K72F5E82
-rw-r--r-- 1 mx19 mx19   222354 Sep 21 07:35 K72F5U82
-rw-r--r-- 1 mx19 mx19   198580 Sep 21 07:31 K72F7282
-rw-r--r-- 1 mx19 mx19   222584 Sep 21 07:35 K72F7284
-rw-r--r-- 1 mx19 mx19   384702 Sep 21 07:47 K72F7IG2
-rw-r--r-- 1 mx19 mx19   399166 Sep 21 07:49 K72F7IG4
-rw-r--r-- 1 mx19 mx19 31639806 Sep 21 19:24 K72F8D82
-rw-r--r-- 1 mx19 mx19 34403854 Sep 21 19:29 K72F9GO2
-rw-r--r-- 1 mx19 mx19   300284 Sep 21 07:39 K72F9O82
-rw-r--r-- 1 mx19 mx19   313264 Sep 21 07:40 K72F9O84

```
