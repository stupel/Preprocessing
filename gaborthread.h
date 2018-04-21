#ifndef GABORTHREAD_H
#define GABORTHREAD_H

#include "preprocessing_config.h"

class GaborThread : public QObject
{
    Q_OBJECT
public:
    explicit GaborThread(QObject *parent = nullptr);
    GaborThread(cv::Mat img, cv::Mat oMap, const GABOR_PARAMS &gaborParams, cv::Rect rect, cv::Mat enhancedImage);

public slots:
    void enhanceFragmentSlot();
    void enhancementDoneSlot();

signals:
    void enhancementDoneSignal();
    void enhanceFragmentSignal(const bool &useFrequencyMap);

private:
    cv::Mat img; // obrazok s odtlackom
    cv::Mat oMap; // smerova mapa odtlacku

    GABOR_PARAMS gabor;

    cv::Rect rect; // oblast odtlacku, ktoru prefiltruje konkretne vlakno
    cv::Mat enhancedImage; // prefiltrovany odtlacok


};

#endif // GABORTHREAD_H
