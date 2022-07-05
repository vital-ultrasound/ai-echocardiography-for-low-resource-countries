# Real-time AI-empowered echocardiography

## Summary 
This repository contains documentation and code for an experimental clinical system to perform real-time artificial-intelligence-empowered echocardiography.
The machine learning pipeline are based on the total product lifecycle (TPLC) approach on AI/ML workflow from Good Machine Learning Practices by the U.S. Food and Drug Administration , Health Canada, and the UK's Medicines and Healthcare products Regulatory Agency (MHRA).
See more [here](scripts/learning-pipeline).

## Data collection, validation and management 
1. Video data from GE Venue Go GE using Probe 3SC-RS was collected with a Portable Video Recorder. See [more](data).
2. Creation and verification of annotations with VGG Image Annotator (VIA) software. See [more](data/labelling).
3. Jupyter notebook :notebook: for [data curation, selection and validation](scripts/curation-selection-validation); 
![fig](docs/figures/data-workflow.png)     
_**Fig 1.** Data workflow._

## Deep learning pipeline
1. Model training and tuning (Fig. 2). See [more](scripts/learning-pipeline).
2. Model validation (performance evaluation and clinical evaluation). 
3. AI-based device modification, and (perhaps) AI-based production model . 
![fig](docs/figures/DL-pipeline.png)     
_**Fig 2.** Deep learning pipeline of the AI-empowered echocardiography._

## Clinical system  
Figure 3 illustrates the real-time AI-empowered clinical system based on [EPIQ 7 ultrasound](https://www.usa.philips.com/healthcare/product/HC795200C/epiq-7-ultrasound-system-for-cardiology), [X5-1 xMATRIX array transducer ](https://www.philips.co.uk/healthcare/product/HC989605400801/x5-1) and [USB framegrabber MiraBox Video Capture](https://miraboxbuy.com/collections/new-arrivals/products/hsv3211-video-capture-card).
The software of the system is based on [Machine learning pipeline](scripts/learning-pipeline) and [Plug-in based, Real-time Ultrasound](source/PRETUS_Plugins).  
See further details on the system [here](docs/system). 

![fig](docs/figures/rt-ai-system.png)    
_**Fig 3.** Real-time AI-empowered clinical system._  

## Clone repository
After generating your SSH keys as suggested [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent), you can then clone the repository by typing (or copying) the following lines in a terminal:
```
mkdir -p $HOME/repositories/ && cd $HOME/repositories/ ## suggested path
git clone git@github.com:vital-ultrasound/echocardiography.git
```

## Contributors
Thanks goes to all these people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):  
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<table>
  <tr>
    <td align="center"><a href="https://github.com/mxochicale"><img src="https://avatars1.githubusercontent.com/u/11370681?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Miguel Xochicale</b></sub>           </a><br /><a href="https://github.com/vital-ultrasound/echocardiography/commits?author=fepegar" title="Code">💻</a> <a href="https://github.com/fepegar/torchio/commits?author=mxochicale" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/hamidehkerdegari"><img src="https://avatars1.githubusercontent.com/u/30697849?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Hamideh Kerdegari </b></sub>   </a><br /><a href="https://github.com/vital-ultrasound/echocardiography/commits?author=hamidehkerdegari" title="Code">💻</a> </td>
    <td align="center"><a href="https://github.com/huynhatd13"><img src="https://avatars1.githubusercontent.com/u/33121364?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Nhat Phung Tran Huy</b></sub>        </a><br /><a href="https://github.com/vital-ultrasound/echocardiography/commits?author=huynhatd13" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/"><img src="https://avatars1.githubusercontent.com/u/23114020?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Linda Denehy</b></sub>        </a><br /><a href="https://github.com/vital-ultrasound/echocardiography/commits?author=" title="Research">  🔬 🤔  </a></td>
    <td align="center"><a href="https://github.com/"><img src="https://avatars1.githubusercontent.com/u/23114020?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Louise Thwaites</b></sub>        </a><br /><a href="https://github.com/vital-ultrasound/echocardiography/commits?author=" title="Research">  🔬 🤔  </a></td>
    <td align="center"><a href="https://github.com/"><img src="https://avatars1.githubusercontent.com/u/23114020?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Sophie Yacoub</b></sub>        </a><br /><a href="https://github.com/vital-ultrasound/echocardiography/commits?author=" title="Research">  🔬 🤔  </a></td>
    <td align="center"><a href="https://github.com/atoandy"><img src="https://avatars1.githubusercontent.com/u/38954988?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Andrew King</b></sub>        </a><br /><a href="https://github.com/vital-ultrasound/nnUNet-for-PRETUS/commits?author=atoandy" title="Research">  🔬🤔  </a></td>
    <td align="center"><a href="https://github.com/gomezalberto"><img src="https://avatars1.githubusercontent.com/u/1684834?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Alberto Gomez</b></sub>             </a><br /><a href="https://github.com/vital-ultrasound/echocardiography/commits?author=gomezalberto" title="Code">💻</a></td>
  </tr>
</table>
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This work follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. 
Contributions of any kind welcome!

## Contact and issue report
If you have specific questions about the content of this repository, you can contact [Miguel Xochicale](mailto:miguel.xochicale@kcl.ac.uk?subject="[ai-echochardiography]"). 
If your question might be relevant to other people, please instead [open an issue](https://github.com/vital-ultrasound/echocardiography/issues).  