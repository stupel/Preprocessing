#ifndef GABORFILTERGPU_H
#define GABORFILTERGPU_H

#include "preprocessing_config.h"

class GaborFilterGPU : public QObject
{
	Q_OBJECT

public:
	GaborFilterGPU();

	void setParams(const cv::Mat &m_imgInput, const GABOR_PARAMS &gaborParams);
	void enhanceWithBasicOMap();
	void enhanceWithAdvancedOMap();

	//getNset
	cv::Mat getImgEnhanced() const;
	float getDuration() const;


private:

	// INPUT
	af::array m_imgInput;
	af::array m_oMap;
	af::array m_fMap;

	GABOR_PARAMS m_gabor;

	// OUTPUT
	cv::Mat m_imgEnhanced;
	float m_duration;

	// PRIVATE FUNCTIONS
	af::array getGaborKernel(const af::array &oMapPixel);
};

#endif // GABORFILTERGPU_H
