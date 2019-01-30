#ifndef CONTRASTENHANCEMENT_H
#define CONTRASTENHANCEMENT_H

#include "preprocessing_config.h"

class ContrastEnhancement : public QObject
{
	Q_OBJECT

public:
	explicit ContrastEnhancement(QObject *parent = nullptr);

	void setParams(const cv::Mat &m_imgOriginal, const CONTRAST_PARAMS &contrastParams);
	void enhance();

	cv::Mat getImgContrastEnhanced() const;

private:
	void performSuace();

private:
	cv::Mat m_imgOriginal;
	CONTRAST_PARAMS m_contrast;

	cv::Mat m_imgContrastEnhanced;

};

#endif // CONTRASTENHANCEMENT_H
