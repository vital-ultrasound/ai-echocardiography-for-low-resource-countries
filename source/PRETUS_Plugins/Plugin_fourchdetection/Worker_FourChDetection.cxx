#include "Worker_FourChDetection.h"

#include <iostream>
#include <thread>

#include <QDebug>
#include <QThread>
#include <QDir>

#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pylifecycle.h>
#include <itkImportImageFilter.h>
#include <chrono>

Worker_FourChDetection::Worker_FourChDetection(QObject *parent) : Worker (parent){
    this->model = "models/model_001";
    this->buffer_capacity = 5;
    this->output_filename = "output.txt";

    mAbsoluteCropBounds = true; // switch to false to make relative
    mAratio = {1130, 810}; // for Sonosite
    //mCropBounds = {0.25, 0.1, 0.6, 0.75}; // for venugo 1920 × 1080
    mCropBounds = {480, 120, 1130, 810}; // for venugo 1920 × 1080
    mDesiredSize = {64, 64};
}

void Worker_FourChDetection::Initialize()
{
    if (this->params.verbose){
        std::cout << "[VERBOSE][Worker_FourChDetection::Initialize] initializing worker"<<std::endl;
    }

    if (!this->PythonInitialized)
    {
        try {
            py::initialize_interpreter();
        }
        catch (py::error_already_set const &python_err) {
            std::cout << python_err.what();
            return;
        }
        if (this->params.verbose){
            std::cout << "[VERBOSE][Worker_FourChDetection::Initialize] python interpreter initialized"<<std::endl;
        }
    }

    this->image_buffer.set_capacity(this->buffer_capacity);
    this->pixel_buffer.resize(0);

    if (this->params.verbose){
        std::cout << "[VERBOSE][Worker_FourChDetection::Initialize] Import python modules from "<<this->params.python_folder<<std::endl;
    }


    py::object getClassesFunction;

    try {
        py::exec("import sys");
        std::string command = "sys.path.append('" + this->params.python_folder + "')";
        py::exec(command.c_str());

        py::object processing = py::module::import("FourChDetection_worker");
        if (this->params.verbose){
            std::cout << "[VERBOSE][Worker_FourChDetection::Initialize] imported inference."<<std::endl;
        }
        this->PyImageProcessingFunction = processing.attr("dowork");
        this->PyPythonInitializeFunction = processing.attr("initialize");
        getClassesFunction = processing.attr("get_classes");

    }
    catch (std::exception const &python_err) {
        std::cout << "[ERROR][Worker_FourChDetection::Initialize] "<< python_err.what();
        return;
    }

    if (this->params.verbose){
        std::cout << "[VERBOSE][Worker_FourChDetection::Initialize] initialize LUS model"<<std::endl;
    }
    py::tuple sz = py::make_tuple(1, this->buffer_capacity, mDesiredSize[0], mDesiredSize[1]); // batch size, nframes, H, W
    this->PyPythonInitializeFunction(sz, this->params.python_folder + QString(QDir::separator()).toStdString() + this->model, bool(this->params.verbose));


    py::list pyclasses  = py::list(getClassesFunction());
    this->labels.clear();
    for (auto& el : pyclasses) this->labels.push_back(QString(el.cast<std::string>().data()));

    this->PythonInitialized = true;
}

Worker_FourChDetection::~Worker_FourChDetection(){
    py::finalize_interpreter();
}

void Worker_FourChDetection::doWork(ifind::Image::Pointer image){

    if (!this->PythonInitialized){
        return;
    }

    if (!Worker::gil_init) {
        this->set_gil_init(1);
        PyEval_SaveThread();

        ifind::Image::Pointer configuration = ifind::Image::New();
        configuration->SetMetaData<std::string>("Python_gil_init","True");
        Q_EMIT this->ConfigurationGenerated(configuration);
    }

    if (image == nullptr){
        if (this->params.verbose){
            std::cout << "Worker_FourChDetection::doWork() - input image was null" <<std::endl;
        }
        return;
    }

    if (this->params.verbose){
        std::cout << "[VERBOSE][Worker_FourChDetection::doWork()] process image"<<std::endl;
    }
    ifind::Image::Pointer image_ratio_adjusted;

    std::vector<int> absoluteCropBounds(4);
    if (this->absoluteCropBounds() == true){
        for (int i=0; i<mCropBounds.size(); i++) absoluteCropBounds[i] = mCropBounds[i];
    } else {
        // get the image size
        ifind::Image::SizeType imsize = image->GetLargestPossibleRegion().GetSize();
        absoluteCropBounds[0] = int(mCropBounds[0] * imsize[0]); // x0
        absoluteCropBounds[1] = int(mCropBounds[1] * imsize[1]); // y0
        absoluteCropBounds[2] = int(mCropBounds[2] * imsize[0]); // w
        absoluteCropBounds[3] = int(mCropBounds[3] * imsize[1]); // h

        if (this->params.verbose){
            std::cout << "\tWorker_RFSeg::doWork() computing absolute crop bounds"<<std::endl;

        }
    }
    if (this->params.verbose){
        ifind::Image::SizeType imsize = image->GetLargestPossibleRegion().GetSize();
        std::cout << "\tWorker_RFSeg::doWork() use bounds"<<std::endl;
        std::cout << "\t\timage size is "<< imsize[0] << "x" << imsize[1]<<std::endl;
        std::cout << "\t\tinput crop bounds are "<< mCropBounds[0] << ":" << mCropBounds[1]<< ":" << mCropBounds[2]<< ":" << mCropBounds[3]<<std::endl;
        std::cout << "\t\tabsolute crop bounds are "<< absoluteCropBounds[0] << ":" << absoluteCropBounds[1]<< ":" << absoluteCropBounds[2]<< ":" << absoluteCropBounds[3]<<std::endl;
    }

    /// Use the appropriate layer
    std::vector<std::string> layernames = image->GetLayerNames();
    int layer_idx = this->params.inputLayer;
    if (this->params.inputLayer <0){
        /// counting from the end
        layer_idx = image->GetNumberOfLayers() + this->params.inputLayer;
    }
    ifind::Image::Pointer layerImage = ifind::Image::New();
    layerImage->Graft(image->GetOverlay(layer_idx), layernames[layer_idx]);
    image_ratio_adjusted = this->CropImageToFixedAspectRatio(layerImage, &mAratio[0], &absoluteCropBounds[0]);

    //png::save_ifind_to_png_file<ifind::Image>(image_ratio_adjusted, "/home/ag09/data/VITAL/cpp_in_adjusted.png");
    // now resample to 64 64
    if (this->params.verbose){
        std::cout << "Worker_RFSeg::doWork() - resample"<<std::endl;
    }
    ifind::Image::Pointer image_ratio_adjusted_resampled  = this->ResampleToFixedSize(image_ratio_adjusted, &mDesiredSize[0]);
    //png::save_ifind_to_png_file<ifind::Image>(image_ratio_adjusted_resampled, "/home/ag09/data/VITAL/cpp_in_adjusted_resampled.png");
    this->params.out_spacing[0] = this->params.out_spacing[0] * (this->params.out_size[0] - 1 )/ (mDesiredSize[0] - 1);
    this->params.out_spacing[1] = this->params.out_spacing[1] * (this->params.out_size[1] - 1 )/ (mDesiredSize[1] - 1);
    this->params.out_size[0] = mDesiredSize[0];
    this->params.out_size[1] = mDesiredSize[1];
    /// extract central slice and crop

    ///----------------------------------------------------------------------------

    GrayImageType2D::Pointer image_2d = this->get2dimage(image_ratio_adjusted_resampled);

    //GrayImageType2D::Pointer image_2db = this->get2dimage(image);  /// Extract central slice
    std::vector <unsigned long> dims = {image_2d->GetLargestPossibleRegion().GetSize()[1],
                                        image_2d->GetLargestPossibleRegion().GetSize()[0]};
    if (!image_2d->GetBufferPointer() || (dims[0] < 5) || (dims[1] < 5))
    {
        qWarning() << "[VERBOSE][Worker_FourChDetection::doWork()] image buffer is invalid";
        return;
    }

    /// Here starts. Fill the buffer
    if (this->params.verbose){
        std::cout << "[VERBOSE][Worker_FourChDetection::doWork()] add image in the buffer"<<std::endl;
    }
    this->image_buffer.push_back(image_2d);

    if (this->image_buffer.full() == false){
        /// We can't do anything until the buffer is full
        if (this->params.verbose){
            std::cout << "[VERBOSE][Worker_FourChDetection::slot_Work] the image buffer only has "<< this->image_buffer.size() << " out of "<< this->buffer_capacity<< ", waiting to have more images"<<std::endl;
        }
        return;
    }
    if (this->params.verbose){
        std::cout << "[VERBOSE][Worker_FourChDetection::doWork()] the buffer is full, we can now analyse the sequence"<<std::endl;
    }

    /// The buffer is now full, we can create our 3D np array
    /// Input dimensions are swapped as ITK and numpy have inverted orders

    std::vector <unsigned long> dims_buffer = {this->buffer_capacity, image_2d->GetLargestPossibleRegion().GetSize()[1],
                                               image_2d->GetLargestPossibleRegion().GetSize()[0]};

    if (this->params.verbose){
        std::cout << "[VERBOSE][Worker_FourChDetection::doWork()] the buffer has a size of "<< dims_buffer[0]<<", "<< dims_buffer[1]<<", "<< dims_buffer[2]<<std::endl;
    }

    this->UpdatePixelBuffer();

    GrayImageType2D::Pointer attention_map;


    std::vector<float> detection_result;
    std::vector<float> localisation_result;

    /// This must be only around the python stuff! otherwise the c++ will not run on a separate thread
    this->gstate = PyGILState_Ensure();
    {
        //py::array numpyarray(dims_buffer, static_cast<GrayImageType2D::PixelType*>(&(this->pixel_buffer[0])));
        py::array numpyarray(dims_buffer, static_cast<float*>(&(this->pixel_buffer[0])));
        //py::tuple predictions = this->PyImageProcessingFunction(numpyarray);
        //py::array dr = py::array(predictions[0]);
        py::array prediction = this->PyImageProcessingFunction(numpyarray);

        this->array_to_vector<float>(prediction, detection_result);
    }
    PyGILState_Release(this->gstate);

    if (this->params.verbose){
        std::cout << "[VERBOSE][Worker_FourChDetection::doWork()]  prediction done"<<std::endl;
    }


    /// Find max min and prediction
    double max_val = 0;
    int max_index = -1;
    std::vector<float>::const_iterator cit;
    for (cit = detection_result.begin(); cit != detection_result.end(); cit++){
        if (*cit > max_val){
            max_val = *cit;
            max_index = cit - detection_result.begin();
        }
    }
    double max_val_l = 0;
    int max_index_l = -1;
    int count = 0;
    if (max_index ==1){
        /// there is a b-line
        for (cit = localisation_result.begin(); cit != localisation_result.end(); cit++, count++){
            if (*cit > max_val_l){
                max_val_l = *cit;
                max_index_l = count;
            }
        }
    }

    // convert to a string
    //std::cout << "predictions: ";
    QStringList confidences_str;
    for (unsigned int i=0; i<detection_result.size(); i++){
        double dr_normalised = detection_result[i]; ///max_val;
        //std::cout << dr_normalised <<" ";
        confidences_str.append(QString::number(dr_normalised));
    }
    //std::cout << std::endl;

    image->SetMetaData<std::string>( mPluginName.toStdString() +"_labels", this->labels.join(",").toStdString() );
    image->SetMetaData<std::string>( mPluginName.toStdString() + "_confidences", confidences_str.join(",").toStdString() );
    image->SetMetaData<std::string>( mPluginName.toStdString() +"_label", this->labels[max_index].toStdString() );
    image->SetMetaData<std::string>( mPluginName.toStdString() + "_confidence", confidences_str[max_index].toStdString() );


    if (this->params.verbose){
        if (max_index == 1){
            std::cout << "[VERBOSE][Worker_FourChDetection::doWork()] Detection result = "<< this->labels[max_index].toStdString() << " ("<<max_val<<"), localised at frame "<< max_index_l<< std::endl;
        } else {
            std::cout << "[VERBOSE][Worker_FourChDetection::doWork()] Detection result = "<< this->labels[max_index].toStdString() << " ("<<max_val<<")"<< std::endl;
        }
    }

    Q_EMIT this->ImageProcessed(image);

}

QString Worker_FourChDetection::getOutput_filename() const
{
    return output_filename;
}

void Worker_FourChDetection::setOutput_filename(const QString &value)
{
    output_filename = value;
}

bool Worker_FourChDetection::absoluteCropBounds() const
{
    return mAbsoluteCropBounds;
}

void Worker_FourChDetection::setAbsoluteCropBounds(bool absoluteCropBounds)
{
    mAbsoluteCropBounds = absoluteCropBounds;
}


template <class T>
void Worker_FourChDetection::array_to_vector(py::array &array, std::vector<T> &vector){
    vector.resize(array.size());
    T *ptr = (T *) array.data();
    for (int i=0 ;i<array.size(); i++){
        vector[i] =  *ptr;
        ptr++;
    }
}


void Worker_FourChDetection::UpdatePixelBuffer(){

    auto imsize = this->image_buffer[0]->GetLargestPossibleRegion().GetSize();
    int n_im_pixels = imsize[0]*imsize[1];
    int total_size = n_im_pixels*this->image_buffer.size();

    if (this->pixel_buffer.size() != total_size ){
        this->pixel_buffer.resize(total_size);
    }

    boost::circular_buffer<GrayImageType2D::Pointer>::const_iterator cit;
    int count = 0;
    for (cit = this->image_buffer.begin(); cit != this->image_buffer.end(); cit++, count++){
        int offset = count*n_im_pixels;
        GrayImageType2D::PixelType* current_pixel_pointer = static_cast<GrayImageType2D::PixelType*>((*cit)->GetBufferPointer());
        std::copy( current_pixel_pointer, current_pixel_pointer+n_im_pixels, std::begin(this->pixel_buffer) + offset );
    }

}


std::vector<int> Worker_FourChDetection::desiredSize() const
{
    return mDesiredSize;
}

void Worker_FourChDetection::setDesiredSize(const std::vector<int> &desiredSize)
{
    mDesiredSize = desiredSize;
}

std::vector<float> Worker_FourChDetection::aratio() const
{
    return mAratio;
}

void Worker_FourChDetection::setAratio(const std::vector<float> &aratio)
{
    mAratio = aratio;
}

std::vector<double> Worker_FourChDetection::cropBounds() const
{
    return mCropBounds;
}

void Worker_FourChDetection::setCropBounds(const std::vector<double> &cropBounds)
{
    mCropBounds = cropBounds;
}
