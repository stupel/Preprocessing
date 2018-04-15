#ifndef GABORTHREAD_H
#define GABORTHREAD_H

#include "preprocessing_config.h"

class GaborThread : public QObject
{
    Q_OBJECT
public:
    explicit GaborThread(QObject *parent = nullptr);
    GaborThread(cv::Mat img, cv::Mat oMap, cv::Mat fMap, int blockSize, double sigma, double lambda, cv::Rect rect, cv::Mat enhancedImage);

public slots:
    void enhanceFragmentSlot(const bool &useFrequencyMap);
    void enhancementDoneSlot();

signals:
    void enhancementDoneSignal();
    void enhanceFragmentSignal(const bool &useFrequencyMap);

private:
    cv::Mat img; // obrazok s odtlackom
    cv::Mat oMap; // smerova mapa odtlacku
    cv::Mat fMap; // frekvencna mapa odtlacku
    int blockSize; // velkost bloku pre filtrovanie
    double sigma; // sigma
    double lambda; // lambda
    cv::Rect rect; // oblast odtlacku, ktoru prefiltruje konkretne vlakno
    cv::Mat enhancedImage; // prefiltrovany odtlacok


};

#endif // GABORTHREAD_H
