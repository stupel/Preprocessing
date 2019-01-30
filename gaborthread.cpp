#include "gaborthread.h"

GaborThread::GaborThread(QObject *parent) : QObject(parent)
{

}

GaborThread::GaborThread(cv::Mat img, const GABOR_PARAMS &gaborParams, cv::Rect rect, cv::Mat enhancedImage)
	: m_imgInput(img)
	, m_gabor(gaborParams)
	, m_rect(rect)
	, m_enhancedImage(enhancedImage)
{

}

void GaborThread::enhanceFragmentSlot()
{
	cv::Mat kernel;
	cv::Mat subMat, sub;
	cv::Scalar s;

	for(int i = m_rect.y; i < m_rect.y + m_rect.height; i++){
		for(int j = m_rect.x; j < m_rect.x + m_rect.width; j++){
			if (*m_gabor.useFrequencyMap) kernel = cv::getGaborKernel(cv::Size(m_gabor.blockSize,m_gabor.blockSize), m_gabor.sigma, m_gabor.oMap->at<float>(i,j), m_gabor.fMap->at<float>(i,j), 1, 0, CV_32F);
			else kernel = cv::getGaborKernel(cv::Size(m_gabor.blockSize,m_gabor.blockSize), m_gabor.sigma, m_gabor.oMap->at<float>(i,j), m_gabor.lambda, 1, 0, CV_32F);
			subMat = m_imgInput(cv::Rect(j-m_gabor.blockSize/2, i-m_gabor.blockSize/2, m_gabor.blockSize, m_gabor.blockSize));
			subMat.convertTo(sub, CV_32F);
			cv::multiply(sub, kernel, sub);
			s = cv::sum(sub);
			m_enhancedImage.at<float>(i,j) = s[0];
		}
	}
	emit enhancementDoneSignal();
}

void GaborThread::enhancementDoneSlot()
{

}
