#pragma once

#include <Plugin.h>
#include "Widget_FourChDetection.h"
#include "Worker_FourChDetection.h"
#include <QtVTKVisualization.h>

class Plugin_FourChDetection : public Plugin {
    Q_OBJECT

public:
    typedef Worker_FourChDetection WorkerType;
    typedef Widget_FourChDetection WidgetType;
    typedef QtVTKVisualization ImageWidgetType;
    Plugin_FourChDetection(QObject* parent = 0);

    QString GetPluginName(void){ return "Four Chamber Detection";}
    QString GetPluginDescription(void) {return "Detection of four chamber view in echo clips.";}
    void SetCommandLineArguments(int argc, char* argv[]);

    void Initialize(void);

protected:
    virtual void SetDefaultArguments();
    template <class T> QString VectorToQString(std::vector<T> vec);
    template <class T> std::vector<T> QStringToVector(QString str);

public Q_SLOTS:
    virtual void slot_configurationReceived(ifind::Image::Pointer image);

private:
    bool mShowAssistantInitially;
};
