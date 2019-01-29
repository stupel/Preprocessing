#ifndef CONTRASTENHANCEMENT_H
#define CONTRASTENHANCEMENT_H

#include "preprocessing_config.h"

class ContrastEnhancement : public QObject
{
	Q_OBJECT

public:
	explicit ContrastEnhancement(QObject *parent = nullptr);

	void setParams(const cv::Mat &imgOriginal, const CONTRAST_PARAMS &contrastParams);
	void enhance();

	cv::Mat getImgContrastEnhanced() const;

private:
	void performSuace();

	cv::Mat imgOriginal;
	CONTRAST_PARAMS contrast;

	cv::Mat imgContrastEnhanced;

};

#endif // CONTRASTENHANCEMENT_H
