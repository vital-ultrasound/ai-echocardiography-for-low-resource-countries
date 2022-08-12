#pragma once

#include <Worker.h>

#include <memory>
#include <mutex>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <boost/circular_buffer.hpp>
#include <QAbstractButton>


namespace py = pybind11;

class Worker_FourChDetection : public Worker{
    Q_OBJECT

public:

    typedef Worker_FourChDetection            Self;
    typedef std::shared_ptr<Self>       Pointer;

    /** Constructor */
    static Pointer New(QObject *parent = 0) {
        return Pointer(new Worker_FourChDetection(parent));
    }

    struct Parameters : WorkerParameters
    {
        Parameters() {
            python_folder = "";
        }
        std::string python_folder;
    };
    ~Worker_FourChDetection();

    void Initialize();
    Parameters params;
    std::string model;
    unsigned long buffer_capacity; /// size of the image buffer

    std::vector<double> cropBounds() const;
    void setCropBounds(const std::vector<double> &cropBounds);
    std::vector<float> aratio() const;
    void setAratio(const std::vector<float> &aratio);
    std::vector<int> desiredSize() const;
    void setDesiredSize(const std::vector<int> &desiredSize);

    bool absoluteCropBounds() const;
    void setAbsoluteCropBounds(bool absoluteCropBounds);

    QString getOutput_filename() const;
    void setOutput_filename(const QString &value);



protected:
    Worker_FourChDetection(QObject* parent = 0);

    void doWork(ifind::Image::Pointer image);


    /**
     * @brief mCropBounds
     * x0. y0. width, height
     */
    std::vector<double> mCropBounds;
    std::vector<float> mAratio;
    std::vector<int> mDesiredSize;
    bool mAbsoluteCropBounds;
    QString output_filename;

private:

    /// Python Functions
    py::object PyImageProcessingFunction;
    py::object PyPythonInitializeFunction;
    PyGILState_STATE gstate;
    QStringList labels;

    boost::circular_buffer<GrayImageType2D::Pointer> image_buffer;

    /* This buffer is a memory-contiguous block with all frames, to convert easily to numpy*/
    std::valarray<float> pixel_buffer;

    /**
     * @brief Update the values in pxel_buffer with the
     * current image buffer content
     */
    void UpdatePixelBuffer();

    template <class T>
    void array_to_vector(py::array &array, std::vector<T> &vector);

};


