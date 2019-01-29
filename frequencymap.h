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
	void setParams(const cv::Mat &imgOriginal, const FMAP_PARAMS &fmapParams);

	void generate();

private:
	PreprocessingCaffeNetwork *frequencyClassifier;

	cv::Mat imgOriginal;
	FMAP_PARAMS fmap;

	cv::Mat frequencyMap;

	bool isFrequencyModelLoaded;

};

#endif // FREQUENCYMAP_H
