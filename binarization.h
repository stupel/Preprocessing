#ifndef BINARIZATION_H
#define BINARIZATION_H

#include "preprocessing_config.h"

class Binarization : public QObject
{
    Q_OBJECT

public:
    explicit Binarization(QObject *parent = nullptr);

    void setParams(const cv::Mat &imgEnhanced, const BINARIZATION_PARAMS &binarizationParams, const PREPROCESSING_FEATURES &features);
    void binarizeCPU();
    void binarizeGPU();
    void binarizeGaussianBlur();
    void binarizeAdaptive();
    void removeHoles(double holeSize);

    cv::Mat getImgBinarized() const;

private:

    // INPUT
    cv::Mat imgEnhanced;
    BINARIZATION_PARAMS binarization;
    PREPROCESSING_FEATURES features;

    // OUTPUT
    cv::Mat imgBinarized;


    // PRIVATE FUNCTIONS
    void deleteBackground();
};


#endif // BINARIZATION_H
