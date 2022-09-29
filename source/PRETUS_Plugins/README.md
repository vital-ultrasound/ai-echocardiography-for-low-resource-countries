# Plug-in based, Real-time Ultrasound (PRETUS) Plugins for 4 chamber view
[PRETUS](https://github.com/gomezalberto/pretus) [Plugin_fourchdetection](Plugin_fourchdetection) contains the four chamber detection plug-in that differentiates between 4 chamber and background classes in real-time.

## Building plug-in
* Install PRETUS covering all dependencies and same packages versions:    
See https://github.com/gomezalberto/pretus 

* Open terminal with conda environment
``` 
conda activate pretus
```

* Creating building paths
``` 
mkdir -p $HOME/build/pretus/4cv/ && cd $HOME/build/pretus/4cv/ 
```

* Config PLUGIN_* PATHS
``` 
* /home/mx19/build/pretus/release
PLUGIN FOLDER: /home/mx19/build/pretus/release/lib

* /home/mx19/build/pretus/4cv
PLUGIN_INCLUDE_DIR ~/local/pretus/include
MAKE_INSTALL_PREFIX ~/local/pretus-vital-echo

```

* build pretus and plugins
``` 
* ~/build/pretus/release$ make
* ~/build/pretus/4cv$ make; make install
```

* Edit `~/.config/iFIND/PRETUS.conf`
```
[MainApp]
plugin_folder="/home/mx19/build/pretus/release/lib;/home/mx19/local/pretus-vital-echo/lib"
```

* Open cmake-gui
```
cd $HOME/repositories/echocardiography/source/PRETUS_Plugins
cmake-gui .
```

* Creating building paths
``` 
Source code: $HOME/repositories/echocardiography/source/PRETUS_Plugins
Where to build binaries: $HOME/build/pretus/4cv
```

* CMake tags in PRETUS
``` 
    CMAKE_INSTALL_PREFIX set to $HOME/local/pretus   (Press configure)
    PLUGIN_INCLUDE_DIR set to $HOME/local/pretus/include (Press configure)
    PLUGIN_LIBRARY set to $HOME/local/pretus/lib/libPlugin.so (Press configure)
    VTK_DIR set to $HOME/workspace/VTK/release  (Press configure)
    ITK_DIR set to $HOME/workspace/ITK/release (Press configure) 
    
    PYTHON_LIBRARY set to $HOME/anaconda3/envs/pretus/lib/libpython3.7m.so (Press configure) 
    PYTHON_INCLUDE_DIR set to $HOME/anaconda3/envs/pretus/include/python3.7m (Press configure) 
    pybind11_DIR set to $HOME/local/pybind11/share/cmake/pybind11 (Press configure) 
    
    Qt settings
        Qt5Concurrent_DIR set to $HOME/Qt/5.12.5/gcc_64/lib/cmake/Qt5Concurrent
        Qt5Core_DIR set to $HOME/Qt/5.12.5/gcc_64/lib/cmake/Qt5Core
        Qt5Gui_DIR set to $HOME/Qt/5.12.5/gcc_64/lib/cmake/Qt5Gui
        Qt5OpenGL_DIR set to $HOME/Qt/5.12.5/gcc_64/lib/cmake/Qt5OpenGL
        Qt5PrintSupport_DIR set to $HOME/Qt/5.12.5/gcc_64/lib/cmake/Qt5PrintSupport
        Qt5Sql_DIR set to $HOME/Qt/5.12.5/gcc_64/lib/cmake/Qt5Sql
        Qt5Widgets_DIR set to $HOME/Qt/5.12.5/gcc_64/lib/cmake/Qt5Widgets
        Qt5X11Extras_DIR set to $HOME/Qt/5.12.5/gcc_64/lib/cmake/Qt5X11Extras
        Qt5Xml_DIR set to $HOME/Qt/5.12.5/gcc_64/lib/cmake/Qt5Xml
```

* Make project 
``` 
cd $HOME/build/pretus/4cv
conda activate pretus
make
make install 
```

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