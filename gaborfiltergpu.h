#ifndef GABORFILTERGPU_H
#define GABORFILTERGPU_H

#include <QObject>
#include <QDebug>

#include "helper.h"

class GaborFilterGPU : public QObject
{
    Q_OBJECT

public:
    GaborFilterGPU();

    void setParams(const cv::Mat &img_, const cv::Mat &orientationMap_, const cv::Mat &frequencyMap_, int blockSize_, double sigma_, double lambda_);
    void enhance();

    cv::Mat getImgEnhanced() const;

private:
    af::array imgFp;
    af::array oMap;
    af::array fMap;

    cv::Mat imgEnhanced;

    int blockSize;
    double sigma;
    double lambda;
    double gamma;
    double psi;

    int origWidth;
    int origHeight;

    af::array getGaborKernel(const af::array& oMap);
};

#endif // GABORFILTERGPU_H
