#ifndef FREQUENCYMAP_H
#define FREQUENCYMAP_H

#include "preprocessing_config.h"
#include "preprocessing_caffenetwork.h"

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
    PreprocessingCaffeNetwork *frequencyClassifier;

    cv::Mat frequencyMap;

    bool isFrequencyModelLoaded;

};

#endif // FREQUENCYMAP_H
