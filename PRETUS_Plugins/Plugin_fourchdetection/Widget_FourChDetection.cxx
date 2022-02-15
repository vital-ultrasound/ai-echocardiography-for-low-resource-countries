#include "Widget_FourChDetection.h"
//#include <QSlider>
#include <QLabel>
#include <QPushButton>
#include <QButtonGroup>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QCheckBox>
#include <QtInfoPanelTrafficLightBase.h>
#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QImage>
#include <QTimer>

Widget_FourChDetection::Widget_FourChDetection(
        QWidget *parent, Qt::WindowFlags f)
    : QtPluginWidgetBase(parent, f)
{

    mColorWithLevel = true;
    this->mWidgetLocation = WidgetLocation::top_right;
    mStreamTypes = ifind::InitialiseStreamTypeSetFromString("FourChamberDetection");
    mIsBuilt = false;

    mLabel = new QLabel("Text not set");
    mLabel->setStyleSheet(sQLabelStyle);

    this->hideCheckbox = new QCheckBox("Show Assistant",this);
    this->hideCheckbox->setChecked(true);
    this->hideCheckbox->setStyleSheet(sQCheckBoxStyle);

    //--
    auto vLayout = new QVBoxLayout(this);
    vLayout->setContentsMargins(1, 1, 1, 1);
    vLayout->setSpacing(0);
    this->setLayout(vLayout);

    vLayout->addWidget(mLabel);
    this->AddInputStreamComboboxToLayout(vLayout);
}

void Widget_FourChDetection::Build_AI_View(std::vector<std::string> &labelnames){

    QVBoxLayout * outmost_layout = reinterpret_cast<QVBoxLayout*>(this->layout());
    /// This will have bar graphs of the live scan plane values
    {
        QtInfoPanelTrafficLightBase::Configuration cardiacViewDetectionTrafficLightConfig;

        cardiacViewDetectionTrafficLightConfig.Mode =
                QtInfoPanelTrafficLightBase::Modes::ImmediateBarAbsolute;
        cardiacViewDetectionTrafficLightConfig.LabelNames = labelnames;
        cardiacViewDetectionTrafficLightConfig.NGridColumns = 2;

        cardiacViewDetectionTrafficLightConfig.ValueColorsVector.push_back(
                    QtInfoPanelTrafficLightBase::ValueColors(
                        std::numeric_limits<double>::lowest(), // value
                        QColor("black"), // background colour
                        QColor("silver"))); // text colour

        cardiacViewDetectionTrafficLightConfig.MetadataLabelsKey = "FourChDetection_labels";
        cardiacViewDetectionTrafficLightConfig.MetadataValuesKey = "FourChDetection_confidences";
        cardiacViewDetectionTrafficLightConfig.MetadataSplitCharacter = ',';

        auto infoPanel = new QtInfoPanelTrafficLightBase(cardiacViewDetectionTrafficLightConfig, this);
        infoPanel->setColorWithLevel(this->colorWithLevel());
        infoPanel->SetStreamTypesFromStr("FourChamberDetection");

        // Add a box to enable / disable

        QVBoxLayout *AI_widget_layout = new QVBoxLayout();
        AI_widget_layout->addWidget(this->hideCheckbox);
        AI_widget_layout->addWidget(infoPanel);
        if (this->hideCheckbox->isChecked()){
            infoPanel->setVisible(true);
        } else {
            infoPanel->setVisible(false);
        }


        QObject::connect(this->hideCheckbox, &QCheckBox::toggled,
                         infoPanel, &QWidget::setVisible);

        outmost_layout->addLayout(AI_widget_layout);

        QObject::connect(this, &QtPluginWidgetBase::ImageAvailable,
                         infoPanel, &QtInfoPanelBase::SendImageToWidget);
    }

}

void Widget_FourChDetection::showEvent(QShowEvent *event) {
    QWidget::showEvent( event );
    //mGv->fitInView(mScene->sceneRect(), Qt::KeepAspectRatio);
}

void Widget_FourChDetection::slot_showEvent(){
    this->showEvent(nullptr);
}

void Widget_FourChDetection::SendImageToWidgetImpl(ifind::Image::Pointer image){

    if (mIsBuilt == false){
        mIsBuilt = true;
        std::string labels = image->GetMetaData<std::string>((this->pluginName() +"_labels").toStdString().c_str());

        boost::char_separator<char> sep(",");
        boost::tokenizer< boost::char_separator<char> > tokens(labels, sep);
        std::vector<std::string> labelnames;
        BOOST_FOREACH (const std::string& t, tokens) {
            labelnames.push_back(t);
        }
        this->Build_AI_View(labelnames);

        /// This is required to make sure it resizes properly
        int timeout = 100;
        // send signal after 100 ms
        QTimer::singleShot(timeout, this, SLOT(slot_showEvent()));
    }


    std::stringstream stream;
    stream << "==" << this->mPluginName.toStdString() << "=="<<std::endl;
    stream << "Sending " << ifind::StreamTypeSetToString(this->mStreamTypes);

    mLabel->setText(stream.str().c_str());

    Q_EMIT this->ImageAvailable(image);
}

bool Widget_FourChDetection::colorWithLevel() const
{
    return mColorWithLevel;
}

void Widget_FourChDetection::setColorWithLevel(bool colorWithLevel)
{
    mColorWithLevel = colorWithLevel;
}
