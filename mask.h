#ifndef MASK_H
#define MASK_H

#include "preprocessing_config.h"
#include "preprocessing_caffenetwork.h"

class Mask : public QObject
{
    Q_OBJECT

public:
    explicit Mask(QObject *parent = nullptr);

    cv::Mat getImgMask() const;

    void loadMaskModel(const CAFFE_FILES &maskFiles);
    void setParams(const cv::Mat &imgOriginal, int blockSize, int exBlockSize, bool useSmooth, bool cpuOnly);
    void generate();

private:
    PreprocessingCaffeNetwork *maskClassifier;

    cv::Mat imgOriginal;
    int blockSize;
    int exBlockSize;
    bool useSmooth;
    bool cpuOnly;

    cv::Mat imgMask;

    bool isMaskModelLoaded;

    void smooth(QImage &smoothedMask, int maskBlockSize);

};

#endif // MASK_H
