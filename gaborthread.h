#ifndef GABORTHREAD_H
#define GABORTHREAD_H

#include "preprocessing_config.h"

class GaborThread : public QObject
{
    Q_OBJECT
public:
    explicit GaborThread(QObject *parent = nullptr);
    GaborThread(cv::Mat imgInput, const GABOR_PARAMS &gaborParams, cv::Rect rect, cv::Mat enhancedImage);

public slots:
    void enhanceFragmentSlot();
    void enhancementDoneSlot();

signals:
    void enhancementDoneSignal();
    void enhanceFragmentSignal();

private:
    cv::Mat imgInput; // obrazok s odtlackom

    GABOR_PARAMS gabor;

    cv::Rect rect; // oblast odtlacku, ktoru prefiltruje konkretne vlakno
    cv::Mat enhancedImage; // prefiltrovany odtlacok


};

#endif // GABORTHREAD_H
