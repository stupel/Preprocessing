#ifndef BINARIZATION_H
#define BINARIZATION_H

#include "preprocessing_config.h"

class Binarization : public QObject
{
	Q_OBJECT

public:
	explicit Binarization(QObject *parent = nullptr);

	void setParams(const cv::Mat &imgEnhanced, const BINARIZATION_PARAMS &binarizationParams);
	void binarizeCPU();
	void binarizeGPU();
	void binarizeGaussianBlur();
	void binarizeAdaptive();
	void removeHoles(double holeSize);

	cv::Mat getImgBinarized() const;

private:
	void deleteBackground();

private:
	// INPUT
	cv::Mat m_imgEnhanced;
	BINARIZATION_PARAMS m_binarization;

	// OUTPUT
	cv::Mat m_imgBinarized;
};


#endif // BINARIZATION_H
