#pragma once
#include <QWidget>
#include <ifindImage.h>
#include <QtPluginWidgetBase.h>

class QLabel;
class AspectRatioPixmapLabel;
class QButtonGroup;
class QPushButton;
class QGraphicsView;
class QGraphicsScene;
class QCheckBox;

class Widget_FourChDetection : public QtPluginWidgetBase
{
    Q_OBJECT

public:
    Widget_FourChDetection(QWidget *parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());

    virtual void SendImageToWidgetImpl(ifind::Image::Pointer image);

    bool colorWithLevel() const;
    void setColorWithLevel(bool colorWithLevel);

    QCheckBox *hideCheckbox;

Q_SIGNALS:
    void signal_zone_toggled(int i, int s); // zone, side

protected:
    virtual void showEvent(QShowEvent *event);
    //virtual void resizeEvent(QResizeEvent *event);

protected Q_SLOTS:
    virtual void slot_showEvent();

private:
    // raw pointer to new object which will be deleted by QT hierarchy
    QLabel *mLabel;
    QLabel *infoPanelabel;
    bool mIsBuilt;
    bool mColorWithLevel;
    QGraphicsView *mGv;
    QGraphicsScene *mScene;
    // for the images
    QString mArtPath;



    /**
     * @brief Build the widget
     */
    void Build_AI_View(std::vector<std::string> &labelnames);

};
