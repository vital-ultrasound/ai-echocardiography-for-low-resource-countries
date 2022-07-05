# USB Frame grabbers

## 1. Epiphan AV.io family by Epiphan Systems Inc. Canada.

* Compare USB capture cards: https://www.epiphan.com/compare-usb-video-grabbers/
* Forum: https://www.epiphan.com/forum/c/avio 
* Support: https://www.epiphan.com/support/avio-hd-software-documentation/ 
* Drivers: https://ssl.epiphan.com/downloads/linux/?dir=Ubuntu 
* Publications using AV.io devices: 
  * "the system receives its input directly from theDVI port of the ultrasound machine, using an Epiphan AV.IO frame grabber tocapture and convert the raw video output to a serial data stream. The framegrabber output is then adapted from USB-A to USB-C with a standard On-The-
Mobile Quantitative Echocardiography 3Go (OTG) adapter, allowing us to pipe the ultrasound machine’s video outputdirectly into the Android device and through a neural network running on itsCPU, using TensorFlow’s Java inference interface." https://www.researchgate.net/publication/327659832_Quantitative_Echocardiography_Real-Time_Quality_Estimation_and_View_Classification_Implemented_on_a_Mobile_Android_Device_International_Workshops_POCUS_2018_BIVPCS_2018_CuRIOUS_2018_and_CPM_2018_Held_,
  * "standard ultra-
sound machine (MyLab 25, Esaote, Florence, Italy, 10-MHz
linear array probe) linked through a frame-grabber to the Ca-
rotid Studio software on an external PC" https://www.heartlungcirc.org/article/S1443-9506(21)01144-6/fulltext VIDEO using AV.io HD: https://www.youtube.com/watch?v=4q-hseWGBUk Epiphan AV.io HD Frame Grabber: http://www.quipu.eu/cardiovascular-suite/equipments/
  * ""A dynamic video stream from the video graphics array port of the portable ultrasound device was digitally broadcast onto the internet utilizing an Epiphan DVI-2USB device (Epiphan, Ottawa, Canada). The real-time video stream was routed to a secure, encrypted web site for viewing by remote guidance experts at HFH in Detroit, MI. https://doi.org/10.1016/j.amjsurg.2006.11.009 
  * "thelaptopwasconnectedtothedigitalvideooutputoftheUSsystemviaaframegrabber(EpiphanDVI2USB3.0)."  https://obgyn.onlinelibrary.wiley.com/doi/epdf/10.1002/pd.6059 
  * "The brachial artery was imaged by author KH approximately 3–5 cm proximal from the antecubital fossa using a linear array high resolution ultrasound transducer (9L-RS; 3–10 MHz) using Duplex ultrasound to concurrently measure blood flow velocity and vessel diameter (Vivid i, GE Healthcare Systems, Mississauga, Canada). Continuous ultrasound images were recorded using a video grabber device (AV.io HD, Epiphan Video) and the blood velocity, brachial artery diameter, and resultant shear rates (baseline and maximal) were analyzed using automated edge-detection software (Cardiovascular Suite, Quipu, Italy)." https://www.frontiersin.org/articles/10.3389/fphys.2022.846229/full?utm_source=dlvr.it&utm_medium=twitter
  * "ltrasound images were acquired using a USB frame grabber (Epiphan Systems, USA) at a size of 312×714 pixels at an imaging depth of 10 cm." https://link.springer.com/chapter/10.1007/978-3-030-87202-1_35

### 1.1 AV.io 4K [4K30fps, HD60fps]
> Capture 4K in perfect fidelity, or use hardware scaling to meet your application needs at any resolution.
* HDMI to USB 4K capture card £504.95
* Color space YUY2 (4:2:2); Color spaces NV12, YV12 and I420 (4:2:0)
* LATENCY: "The hardware latency of the [AV.io](http://av.io/) series is approximately 2 frames. Typically that’s somewhere in the ballpark of 66ms. The software you’re using and even the display you’re connected to will also increase the potential latency." "Basically there is a 2 frame hardware buffer and after that the [AV.io](http://av.io/) sends data as fast as it can. There should however be a reduction in latency from using a 60fps source as opposed to a 30fps source."
https://www.epiphan.com/forum/t/latency-for-av-io-4k-and-av-io-hd/1698
* Operating System support: Windows 8.1, Windows 10, Mac OS X 10.10 and up, Linux distribution with kernel 3.5.0 or higher 
* Online help: https://www.epiphan.com/userguides/avio-4k/Content/Home-AVio4K.htm
* User guide: https://www.epiphan.com/userguides/pdfs/UserGuide_AVio4K.pdf 
* Drawings https://www.epiphan.com/brochures/capture-card-family/AV.io-4K_Mechanical-drawing.pdf 
* Linux AV.io configuration tool Operating system: Ubuntu 16.04 LTS 64-bit and similar: https://www.epiphan.com/downloads/avio/ConfigTool_4.0.0_Linux.tar.bz2 
* HDMI cables: As a practical test, try using a shorter 3ft or 6ft cable in the workflow to see if it corrects the issue. If the signal is seen with a shorter cable, it could be signal degradation. I’d recommend using an active/directional HDMI cable.
https://www.epiphan.com/forum/t/av-io-sdi-compatibility-with-blackmagic-hdmi-to-sdi-converter/3588 
* VIDEOS: https://www.youtube.com/watch?v=AVBRCF37Qb4 https://www.youtube.com/watch?v=nn36cT14iMg 
* more https://www.epiphan.com/products/avio-4k/
* This device complies with ICES-003 of the ISED rules
* CE Compliance Statement: [Directive 2014/30/EU - Electromagnetic Compatibility], [Directive 2011/65/EU - RoHS, restriction of the use of certain hazardous substances in electrical and electronic
equipment]
* Resolutions, capture rates and aspect ratios
![Screenshot from 2022-04-07 11-54-56](https://user-images.githubusercontent.com/11370681/162183708-ce51e5af-eb20-4764-a008-065ba962e155.png)

### 1.2 AV.io HD [HD 60fps]

> The simplest way to capture HDMI, VGA or DVI video sources at resolutions up to 1080p.

* The simplest way to share HD video £389.95
* LATENCY: "The hardware latency of the [AV.io](http://av.io/) series is approximately 2 frames. Typically that’s somewhere in the ballpark of 66ms. The software you’re using and even the display you’re connected to will also increase the potential latency." "Basically there is a 2 frame hardware buffer and after that the [AV.io](http://av.io/) sends data as fast as it can. There should however be a reduction in latency from using a 60fps source as opposed to a 30fps source."
https://www.epiphan.com/forum/t/latency-for-av-io-4k-and-av-io-hd/1698
* User guide: https://www.epiphan.com/userguides/pdfs/UserGuide_AVioHD.pdf 
* VIDEOS: https://www.youtube.com/watch?v=4q-hseWGBUk 
* This device complies with ICES-003 of the ISED rules
* CE Compliance Statement: Directive 2014/30/EU - Electromagnetic Compatibility
Directive 2011/65/EU - RoHS, restriction of the use of certain hazardous substances in electrical and electronic
equipment
* [Cardiovascular Suite 4 User Manual](http://server2.quipu.eu:8090/cs4um/latest/en)
  * There's no software to install to use the AV.io HD; simply connect the cables and go. It works on Microsoft Windows computers and Apple Mac OS X computers.
  * The Epiphan AV.io HD supports resolution from 640x360 up to 1920x1200. Performance may be limited by your computer features. The Epiphan AV.io HD supports both USB 3.0 and USB 2.0.
  * CAUTION: the video converter must be connected directly to a USB port on your computer. Do not use hubs or the USB socket on the external keyboard. Use USB 3.0 to maximize performances.
  * CAUTION: the AV.io HD must be updated with the last firmware from Epiphan System Inc.
  * CAUTION: verify that the video output type and resolution of the ultrasound scanner are compatible with this video converter. 
* Resolutions, capture rates and aspect ratios
![Screenshot from 2022-04-07 11-54-40](https://user-images.githubusercontent.com/11370681/162183768-361b9cf7-494f-44ae-bfd9-aaeb798b62ca.png)



### 1.3 DVI2USB 3.0
* Precise control for DVI, HDMI, and VGA video capture £694.95 
* Video capt ure workst at ion operat ing syst em: Windows 10
* DVI2USB 3.0 on Linux: https://www.epiphan.com/userguides/dvi2usb-3-0/Content/UserGuides/VideoGrabber/pro/startHere/linux.htm 
* "DVI2USB 3.0 doesn't capture on Ubuntu 18.04 and 16.04 systems with AMD Ryzen5 (1600) AMD
USB 3.0 controllers when audio is enabled.
Workaround: In the Epiphan Capture Tool, disable Enable Audio Capture under Capture or
update your USB 3.0 controllers to AMD Ryzen5 (2600)." from https://www.epiphan.com/userguides/pdfs/Epiphan_DVI2USB30_userguide.pdf
* More linux drivers: https://www.epiphan.com/forum/t/ubuntu-18-04-lts-support-for-dvi2usb-3-0/1741/3  
* Userguide: https://www.epiphan.com/userguides/pdfs/Epiphan_DVI2USB30_userguide.pdf
* This device complies with ICES-003 of the ISED rules. 
* CE Compliance Statement: Directive 2014/30/EU - Electromagnetic Compatibility, Directive 2011/65/EU - RoHS, restriction of the use of certain hazardous substances in electrical and electronic
equipment
* More: https://www.epiphan.com/userguides/dvi2usb-3-0/Content/Home-DVI2USB30.htm  https://www.epiphan.com/products/dvi2usb-3-0/ 


## 2. USB Capture HDMI 4K Plus by Nanjing Magewell Electronics Co., Ltd, China 
* £525 from amazon https://www.amazon.co.uk/USB-Capture-HDMI-4K-Plus/dp/B0754C5XLW 
* No driver needs to be installed. USB Capture (Plus) cards are plug-and-play devices. They are compatible with Windows, Mac, Linux and Chrome OS.
* USB Capture HDMI/SDI 4K Plus cards can capture at up to 30fps when capturing 4K (3840x2160 NV12) videos.
* Loop-through HDMI signal
* Linux-Ubuntu: V3.0.3.4202 2018/03/1
* 3-year warranty
* User Manual:  https://www.magewell.com/files/20170817USB_Capture_Plus_User_Manual_EN.pdf
* FAQs: https://www.magewell.com/files/documents/FAQs/Short%20FAQs%20about%20USB%20Capture%20(Plus).pdf 
* [Cardiovascular Suite 4 User Manual](http://server2.quipu.eu:8090/cs4um/latest/en): 
  * Usages in US devices: Magewell USB capture AIO (to connect your computer to DVI, VGA, HDMI, S-video and C-video outputs) http://server2.quipu.eu:8090/cs4um/latest/en/installation/system-requirements 
  * About Magewell USB Capture AIO: 
    * The USB Capture AIO is a USB2.0/USB3.0 video capture device from Nanjing Magewell Electronics Co., Ltd, China.
    * There's no software to install to use USB Capture AIO; simply connect the cables and go. It works on Microsoft Windows computers and Apple Mac OS X computers.
    * The USB Capture AIO supports resolution up to 2048x2160. Performance may be limited by your computer features.
    * The Magewell USB Capture AIO supports both USB 3.0 and USB 2.0.
* More: https://www.magewell.com/products/usb-capture-hdmi-4k-plus 
* Publications: 
  * "the ultrasound unit XarioTM 2000 and the convex transducer for abdominal, fetal and pediatric imaging PVU-375BT, both marketed by Canon Medical (Japan, formerly known as Toshiba Medical), are used.. To capture and convert the ultrasound images, the frame-grabber XI100XUSB-PRO by Magewell (China) is used. Provided with ultrasound images via DVI-input, it forwards the captured images to the computer via USB." https://www.degruyter.com/document/doi/10.1515/cdbme-2020-0025/html
  * "Experiments setup. (a) The ultrasound machine - Mindray DC-70. (b) The video capture device - MAGEWELL USB Capture AIO. (c) Data-acquisition probe holder. (d) The computer for data collection with Intel i5 CPU and Nvidia GTX 1650 GPU, Ubuntu16.04 LTS." https://arxiv.org/pdf/2111.09739.pdf 
  * "For ultrasound images, we used the MAGEWELL USB Capture AIO to capture videos from the DAWEI DW-580 ultrasound machine" https://arxiv.org/pdf/2111.01625.pdf 
* NOTES: There is no forum; I cannot find CE marks in the user manual  https://www.magewell.com/files/20170817USB_Capture_Plus_User_Manual_EN.pdf 


## 3. HDMI-to-USB 3.0 converter by The Imaging Source Europe GmbH Germany
* £270.00 unit price, vat 357.00, shipping £15.00. Total £342.00
* "The DFG/HDMI can be used in Linux, it is v4l2 compatible. With Imaging Source cameras you will get some properties like Exposure and Whitebalance. White Balance should be disabled. However, they can be ignored. With q4l2 you will get a nice image too".
* Video formats @ frame rate (maximum) 1,920×1,080 (2.1 MP) YUY2 @ 60 fps3,840×2,160 (8.3 MP) Y411 @ 30 fps4,096×2,160 (8.8 MP) Bayer8 @ 30 fps 
* Color formats : RGB24, YUY2, Y411, Y800, Bayer8 
* Latest driver release version and date: 5.1.0.1719 February 4, 2022 
* Requirements: Windows 7 (32 & 64 bit), Windows 8 (32 & 64  bit), Windows 10 (32 & 64 bit), Windows 11 
* Few QAs: 
> * Is there any place where I can see how the DFG/HDMI is used in Linux (in https://github.com/TheImagingSource)? 
> No, because it is used like an USB camera by v4l2. The DFG/HDMI is v4l2 compatible Therefore, it will work with guvcview, qv4l2 and so on. 
> * Has the device been used in Ubuntu distros? 
>  In Ubuntu 20.04 and Debian. 
> * Can you share the schematic of  HDMI-to-USB converter as I don't see it in https://s1-dl.theimagingsource.com/api/2.5/packages/publications/schematics/sdconvfg/a572e37f-6556-5448-b391-9ff8211888ac/sdconvfg_1.2.en_US.pdf? 
> Sorry this is not available 
> * It would also help if you provide details on latency performance of the DFG/HDMI . 
> There are no details available for this, sorry for the inconvenience
> * Does the device require an external supply voltage? or does it work with only USB and HDMI cables?
> No power supply needed. Only the HDMI and USB 3 A-C cable. The DFG/HDMI is powered by the USB port of the computer.
> * Do you know if the device been used with Ultrasound clinical devices or any other medical devices?
> No. But HDMI is independent of the video source, regardless, whether it is Ultra Sound, IR, X-Ray or daylight.
> * Does the device support "HDMI Pass Through"?
> No, because the DFG/HDMI has no HDMI output. 
> * Can you test the device with Ubuntu 22.04?
> This has been tested. It works fine on Ubuntu 16.04 and above. It is v4l2 compatible. 
> You can use any v4l2 using software like qv4l2, guvcview and of course our tiscamera. 
* Schematic Diagrams: https://s1-dl.theimagingsource.com/api/2.5/packages/publications/schematics/sdconvfg/a572e37f-6556-5448-b391-9ff8211888ac/sdconvfg_1.2.en_US.pdf 
* MORE: https://www.theimagingsource.com/products/converters-grabbers/hdmi-to-usb-converters/dfghdmi/
* **NOTES**: Documentation only share schematic diagrams not containing the HDMI-to-USB 3.0 converter; there is no forum, and I couldn't find any CE marks in the the catalog https://s1-dl.theimagingsource.com/api/2.5/packages/documentation/factsheets/fsproductcatalog/cacb23c8-11d5-5eb0-851b-bcfa3763ccac/fsproductcatalog_202203.en_US.pdf nor  Schematic Diagrams: https://s1-dl.theimagingsource.com/api/2.5/packages/publications/schematics/sdconvfg/a572e37f-6556-5448-b391-9ff8211888ac/sdconvfg_1.2.en_US.pdf 

## 4.  MiraBox Video Capture by  VXIS Inc [https://miraboxbuy.com/pages/about-us]
* HSV3211 1080P HDMI USB3.0 game Video Capture card for gamers: Miraboxbuy Sale price $49.99  Tax included.
* TreasLin Gaming USB3.0 Capture Card, 1080P 60FPS Video HDMI Capture Card   
>  【HDMI USB3.0 Capture with Strong compatibility】The 1080P HDMi Capture Card Support for all 1080P 720P HDMI device, such as Wii U, PS4, PS3, Xbox One, Xbox 360, Wii, Switch, DVD, camera,Mobile phone(Apple iphone,Huawei), ZOSI security camera, DSLR and set top box etc. It is compatible with Linux, Mac OS, windows 7/8/10, very easy to setup, there is Plug and Play,No Need to Install Driver.  **Price $39.99**  https://www.amazon.com/dp/B08Q7CG9PZ/ref=emc_b_5_t?th=1   
* Input Interface: HDMI interface
* Host Interface: USB3.0, *300-350MB/s,USB2.0,*40MB/s
* Size 95x63x22 mm
* Storage Humidity 	5%-90%
* Power consumption 	<= 2.5W
* Frame Rate: 25/29.97/30/50/59.94/60 fps
* Audio and Video capture: standards of UVC and UAC
* HDMI Input Formats: 480i,480p,576i,576p,720p50,720p60,1080i50,1080i60,1080p24/25/30/50/60;
* Output Formats Resolutions: 640x480/720x480/720x576/768x576/800x600/1024x768/1280x720/1280x800/
1280x960/1280x1024/1368x768/1440x900/1600x1200/1680x1050/1920x1080/
1920x1200/640x360/856x480/960x540/1024x576 
* Working Temperature 	0-50 deg C
* Package 1x HDMI Capture Card , 1 x USB3.0 cable , 1x Manual .
* Support of OS:Windows 7; Windows Server 2008, R2 Linux (Kernel version 2.6.38 and above).
* Software Compatibility:Windows Media Encoder (Windows), Adobe Flash Media Live Encoder (Windows), Real Producer Plus (Windows), VLC (Windows, Linux),Wirecast (Windows), Potplayer(Windows)and etc.
* Development interface compatibility: DirectShow (Windows), DirectSound (Windows), V4L2 (Linux), ALSA (Linux)
* More details from miraboxbuy:  https://miraboxbuy.com/collections/new-arrivals/products/hsv3211-video-capture-card
* See further hardware tests https://github.com/vital-ultrasound/echocardiography/tree/main/scripts/hardware#frame-grabber-mirabox-video-capture  
* **NOTES**: There is no manual, forum or CE marked signs. 
_Originally posted by @mxochicale in https://github.com/vital-ultrasound/echocardiography/issues/39#issuecomment-1091613496_