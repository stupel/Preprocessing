#include "contrastenhancement.h"

ContrastEnhancement::ContrastEnhancement(QObject *parent) : QObject(parent)
{

}

void ContrastEnhancement::setParams(const cv::Mat &imgOriginal, const CONTRAST_PARAMS &contrastParams)
{
	m_imgOriginal = imgOriginal;
	m_contrast = contrastParams;
}

void ContrastEnhancement::performSuace()
{
	m_contrast.sigma = (m_contrast.sigma + 1) / 8.0;
	CV_Assert(m_imgOriginal.type() == CV_8UC1);
	if (!(m_contrast.distance > 0 && m_contrast.sigma > 0)) {
		CV_Error(cv::Error::StsBadArg, "distance and sigma must be greater 0");
	}
	cv::Mat smoothed;
	int val;
	int a, b;
	int adjuster;
	int half_distance = m_contrast.distance / 2;
	double distance_d = m_contrast.distance;

	cv::GaussianBlur(m_imgOriginal, smoothed, cv::Size(0, 0), m_contrast.sigma);

	for (int x = 0; x < m_imgOriginal.cols; x++)
		for (int y = 0; y < m_imgOriginal.rows; y++) {
			val = m_imgOriginal.at<uchar>(y, x);
			adjuster = smoothed.at<uchar>(y, x);
			if ((val - adjuster) > distance_d) adjuster += (val - adjuster) * 0.5;
			adjuster = adjuster < half_distance ? half_distance : adjuster;
			b = adjuster + half_distance;
			b = b > 255 ? 255 : b;
			a = b - m_contrast.distance;
			a = a < 0 ? 0 : a;

			if (val >= a && val <= b)
			{
				m_imgContrastEnhanced.at<uchar>(y, x) = (int)(((val - a) / distance_d) * 255);
			}
			else if (val < a) {
				m_imgContrastEnhanced.at<uchar>(y, x) = 0;
			}
			else if (val > b) {
				m_imgContrastEnhanced.at<uchar>(y, x) = 255;
			}
		}
}

void ContrastEnhancement::enhance()
{
	m_imgContrastEnhanced = m_imgOriginal.clone();
	cv::GaussianBlur(m_imgContrastEnhanced, m_imgContrastEnhanced, cv::Size(m_contrast.gaussBlock, m_contrast.gaussBlock), m_contrast.gaussSigma);
	performSuace(); // perform SUACE with the parameters
}

cv::Mat ContrastEnhancement::getImgContrastEnhanced() const
{
	return m_imgContrastEnhanced;
}
