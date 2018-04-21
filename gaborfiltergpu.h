#ifndef GABORFILTERGPU_H
#define GABORFILTERGPU_H

#include "preprocessing_config.h"

class GaborFilterGPU : public QObject
{
    Q_OBJECT

public:
    GaborFilterGPU();

    void setParams(const cv::Mat &img_, const cv::Mat &orientationMap_, const GABOR_PARAMS &gaborParams);
    void setParams(const cv::Mat &img_, const af::array &orientationMap_, const GABOR_PARAMS &gaborParams);
    void enhanceWithBaseOMap();
    void enhanceWithAdvancedOMAP();

    //getNset
    cv::Mat getImgEnhanced() const;
    float getDuration() const;


private:

    // INPUT
    af::array imgInput;
    af::array oMap;
    af::array fMap;

    GABOR_PARAMS gabor;

    // OUTPUT
    cv::Mat imgEnhanced;
    float duration;

    // PRIVATE FUNCTIONS
    af::array getGaborKernel(const af::array& oMapPixel);
};

#endif // GABORFILTERGPU_H
