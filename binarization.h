#ifndef BINARIZATION_H
#define BINARIZATION_H

#include "helper.h"

#include <QObject>

class Binarization : public QObject
{
    Q_OBJECT

public:
    explicit Binarization(QObject *parent = nullptr);

    void setParams(const cv::Mat &imgEnhanced, const bool &useMask, const cv::Mat &imgMask, const bool &useQMap, const cv::Mat &imgQMap);
    void binarizeCPU();
    void binarizeGPU();
    void binarizeGaussianBlur();
    void binarizeAdaptive();
    void removeHoles(double holeSize);

    cv::Mat getImgBinarized() const;

private:
    cv::Mat imgEnhanced_;
    cv::Mat imgMask_;
    cv::Mat imgQMap_;
    cv::Mat imgBinarized;

    bool useMask_;
    bool useQMap_;

    void deleteBackground();
};


#endif // BINARIZATION_H
