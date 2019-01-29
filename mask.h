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
	void setParams(const cv::Mat &m_imgOriginal, const MASK_PARAMS &maskParams);
	void generate();

private:
	PreprocessingCaffeNetwork *m_maskClassifier;

	cv::Mat m_imgOriginal;
	MASK_PARAMS m_mask;

	cv::Mat m_imgMask;

	bool m_isMaskModelLoaded;

	void smooth(QImage &smoothedMask, int maskBlockSize);

};

#endif // MASK_H
