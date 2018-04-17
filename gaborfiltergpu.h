#ifndef GABORFILTERGPU_H
#define GABORFILTERGPU_H

#include "preprocessing_config.h"

class GaborFilterGPU : public QObject
{
    Q_OBJECT

public:
    GaborFilterGPU();

    void setParams(const cv::Mat &img_, const cv::Mat &orientationMap_, int blockSize_, double sigma_, double lambda_, bool useFrequencyMap, const cv::Mat &frequencyMap_);
    void enhance();

    cv::Mat getImgEnhanced() const;
    float getDuration() const;

private:
    af::array imgFp;
    af::array oMap;

    af::array fMap;
    bool useFrequencyMap ;

    cv::Mat imgEnhanced;

    int blockSize;
    float sigma;
    float lambda;
    float gamma;
    float psi;

    int origWidth;
    int origHeight;

    float duration;

    af::array getGaborKernel(const af::array& oMap);
    void showImg(const af::array&, const char*);
};

#endif // GABORFILTERGPU_H
