# Plug-in based, Real-time Ultrasound (PRETUS) Plugins

## Authors
Author: Alberto Gomez (alberto.gomez@kcl.ac.uk)   
Contributors: Miguel Xochicale (miguel.xochicale@kcl.ac.uk)   

## Summary
PRETUS Plug-ins for the echocardiography ultrasound. 
These plug-ins work with the PRETUS software which can be retrieved  [here](https://github.com/gomezalberto/pretus).

## Content
* [Plugin_fourchdetection](Plugin_fourchdetection) which contains the four chamber detection plug-in that differentiates between 4 chamber and background classes in real-time.

## Build instructions

## Dependencies
The minimum requirements are:
* VTK. You need to fill in the VTK
* ITK (for video inputs, built with the `ITKVideoBridgeOpencv` option `ON`).  You need to fill in the VTK
* Boost
* Qt 5 (tested with 5.12). You need to fill in the `QT_DIR` variable in CMake
* c++11
* Python (you need the python libraries, include dirs)

Additionally, for this plug-in: 

* Python 3 (tested on Python 3.7) 
* Python 3 library
* [PyBind11](https://pybind11.readthedocs.io/en/stable/advanced/cast/overview.html) (for the python interface if required), with python 3.
* [matplotlib]()
* [numpy]()
* [opencv-python]()
* [scikit-learn]()
* [scipy]()
* [torch]() 1.14.0

The python include and binary should be the same used for pybind11. For example, if the python distribution comes from Anaconda, your `PYTHON_INCLUDE_DIR` in the CMake will be something like `<HOME_FOLDER>/anaconda3/include/python3.7m` and your `PYTHON_LIBRARY` will be something like `<HOME_FOLDER>/anaconda3/lib/libpython3.7m.so`.

## Build and install

Launch CMake configure and generate. Then make and install
``` 
$ make && make install
$ export  PYTHONPATH=<your selected install folder\>:"$PYTHONPATH"
```
And launch.

## Troubleshooting
* You will need to set `CMAKE_INSTALL_PREFIX` to a path where your python scripts will go, e.g. <your selected install folder\> .
* Make sure that `PYTHON_INCLUDE_DIR` and `PYTHON_LIBRARY` are correctly set to Python 3.

## References
```
@misc{https://doi.org/10.48550/arxiv.2109.06519,
  doi = {10.48550/ARXIV.2109.06519},
  url = {https://arxiv.org/abs/2109.06519},
  author = {Gomez, Alberto and Zimmer, Veronika A. and Wheeler, Gavin and Toussaint, Nicolas and Deng, Shujie and Wright, Robert and Skelton, Emily and Matthew, Jackie and Kainz, Bernhard and Hajnal, Jo and Schnabel, Julia},
  keywords = {Medical Physics (physics.med-ph), Software Engineering (cs.SE), FOS: Physical sciences, FOS: Physical sciences, FOS: Computer and information sciences, FOS: Computer and information sciences, 65-04 (Primary), 92C55 (Secondary)},
  title = {PRETUS: A plug-in based platform for real-time ultrasound imaging research},
  publisher = {arXiv},
  year = {2021}, 
  copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
}
```