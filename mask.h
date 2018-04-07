#ifndef MASK_H
#define MASK_H

#include <QObject>
#include <QImage>
#include <QPainter>

#include "caffenetwork.h"
#include "config.h"

class Mask : public QObject
{
    Q_OBJECT

public:
    explicit Mask(QObject *parent = nullptr);

    cv::Mat getImgMask() const;

    void loadMaskModel(const CAFFE_FILES &maskFiles);
    void generate(cv::Mat imgOriginal, int blockSize, int exBlockSize, bool useSmooth);

private:
    CaffeNetwork *maskClassifier;

    cv::Mat imgMask;

    void smooth(QImage &smoothedMask, int maskBlockSize);

};

#endif // MASK_H
