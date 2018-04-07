#ifndef FREQUENCYMAP_H
#define FREQUENCYMAP_H

#include <QObject>

#include <algorithm>

#include "opencv2/opencv.hpp"

#include "caffenetwork.h"
#include "config.h"

class FrequencyMap : public QObject
{
    Q_OBJECT

public:
    explicit FrequencyMap(QObject *parent = nullptr);

    cv::Mat getFrequencyMap() const;
    cv::Mat getImgFrequencyMap() const;

    void loadFrequencyMapModel(const CAFFE_FILES &freqFiles);
    void generate(const cv::Mat &imgOriginal, const int &blockSize, const int &exBlockSize);

private:
    CaffeNetwork *frequencyClassifier;

    cv::Mat frequencyMap;

};

#endif // FREQUENCYMAP_H
