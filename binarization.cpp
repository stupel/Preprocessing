#include "binarization.h"

Binarization::Binarization(QObject *parent) : QObject(parent)
{

}

void Binarization::setParams(const cv::Mat &imgEnhanced, const BINARIZATION_PARAMS &binarizationParams)
{
	m_imgEnhanced = imgEnhanced;
	m_binarization = binarizationParams;
}

void Binarization::binarizeGaussianBlur()
{
	m_imgBinarized = cv::Mat(m_imgEnhanced.rows, m_imgEnhanced.cols, m_imgEnhanced.type());
	cv::GaussianBlur(m_imgEnhanced, m_imgBinarized, cv::Size(3,3), 1);
	cv::threshold(m_imgBinarized, m_imgBinarized, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);

	if (*m_binarization.useQualityMap || *m_binarization.useMask) deleteBackground();
}

void Binarization::binarizeAdaptive()
{
	m_imgBinarized = cv::Mat(m_imgEnhanced.rows, m_imgEnhanced.cols, m_imgEnhanced.type());
	cv::adaptiveThreshold(m_imgEnhanced, m_imgBinarized, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 21, 10);

	if (*m_binarization.useQualityMap || *m_binarization.useMask) deleteBackground();
}

void Binarization::deleteBackground()
{
	if (*m_binarization.useQualityMap) {
		for (int x = 0; x < m_imgBinarized.cols; x++) {
			for (int y = 0; y < m_imgBinarized.rows; y++) {
				if (m_binarization.imgQualityMap->at<uchar>(y, x) == 0) m_imgBinarized.at<uchar>(y, x) = 255;
			}
		}
	}
	else if (*m_binarization.useMask) {
		for (int x = 0; x < m_imgBinarized.cols; x++) {
			for (int y = 0; y < m_imgBinarized.rows; y++) {
				if (m_binarization.imgMask->at<uchar>(y, x) == 0) m_imgBinarized.at<uchar>(y, x) = 255;
			}
		}
	}
}

void Binarization::removeHoles(double holeSize)
{
	// invertujem obraz
	cv::bitwise_not(m_imgBinarized, m_imgBinarized);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	// najdem uplne vsetky diery
	cv::findContours(m_imgBinarized, contours, hierarchy,cv::RETR_CCOMP,cv::CHAIN_APPROX_SIMPLE);

	// odstranujem vsetky velke diery, zostanu len male diery - potne pory a nedokonalosti
	for (std::vector<std::vector<cv::Point> >::iterator it = contours.begin(); it!=contours.end(); )
	{
		if (it->size() > holeSize)
			it=contours.erase(it);
		else
			++it;
	}

	// vyplnim diery bielou farbou
	for(int ii=0; ii<contours.size(); ii++)
	{
		cv::drawContours(m_imgBinarized, contours, ii, cv::Scalar(255), -1, cv::LINE_8);
	}

	// invertujem obraz
	cv::bitwise_not(m_imgBinarized, m_imgBinarized);
}

cv::Mat Binarization::getImgBinarized() const
{
	return m_imgBinarized;
}
