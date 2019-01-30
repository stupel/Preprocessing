#include "mask.h"

Mask::Mask(QObject *parent)
	: QObject(parent)
	, m_isMaskModelLoaded(false)
{

}

void Mask::loadMaskModel(const CAFFE_FILES &maskFiles)
{
	if (m_isMaskModelLoaded) {
		delete m_maskClassifier;
		m_isMaskModelLoaded = false;
	}

	m_maskClassifier = new PreprocessingCaffeNetwork;
	m_maskClassifier->loadModel(maskFiles.model, maskFiles.trained, maskFiles.imageMean, maskFiles.label);

	m_isMaskModelLoaded = true;
}

void Mask::setParams(const cv::Mat &imgOriginal, const MASK_PARAMS &maskParams)
{
	m_imgOriginal = imgOriginal;
	m_mask = maskParams;
}

void Mask::generate()
{

	if (*m_mask.cpuOnly) Caffe::set_mode(Caffe::CPU);
	else Caffe::set_mode(Caffe::GPU);

	m_imgMask = cv::Mat::zeros(m_imgOriginal.rows + m_mask.blockSize, m_imgOriginal.cols + m_mask.blockSize, CV_8UC1);

	cv::Mat whiteBlock = cv::Mat(m_mask.blockSize, m_mask.blockSize, CV_8UC1);
	whiteBlock.setTo(255);

	cv::Mat borderedOriginal;
	cv::copyMakeBorder(m_imgOriginal, borderedOriginal, m_mask.exBlockSize, m_mask.exBlockSize, m_mask.exBlockSize, m_mask.exBlockSize, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
	std::vector<cv::Mat> blocks;
	int odd = m_mask.exBlockSize % 2;
	for (int x = m_mask.exBlockSize; x < m_imgOriginal.cols + m_mask.exBlockSize; x += m_mask.blockSize) {
		for (int y = m_mask.exBlockSize; y < m_imgOriginal.rows + m_mask.exBlockSize; y += m_mask.blockSize) {
			blocks.push_back(borderedOriginal.colRange(x - m_mask.exBlockSize/2, x + m_mask.exBlockSize/2 + odd).rowRange(y - m_mask.exBlockSize/2, y + m_mask.exBlockSize/2 + odd));

		}
	}

	std::vector<std::vector<Prediction>> predictions;

	//Use Batch
	predictions = m_maskClassifier->classifyBatch(blocks, 2);

	//Without Batch
	/*for (int i = 0; i < blocks.size(); i++) {
		predictions.push_back(maskClassifier->classify(blocks[i]));
	}*/

	std::vector<Prediction> prediction;
	int cnt = 0;
	for (int x = 0; x < m_imgOriginal.cols; x += m_mask.blockSize) {
		for (int y = 0; y < m_imgOriginal.rows; y += m_mask.blockSize) {
			prediction = predictions[cnt];
			if (prediction[0].first[0] == 'f' || prediction[0].first[0] == 'F') {
				whiteBlock.copyTo(m_imgMask(cv::Rect(x, y, m_mask.blockSize, m_mask.blockSize)));
			}
			cnt++;
		}
	}

	m_imgMask = m_imgMask.rowRange(0, m_imgOriginal.rows).colRange(0, m_imgOriginal.cols);

	if (m_mask.useSmooth) {
		QImage smoothedMask(m_imgMask.cols, m_imgMask.rows, QImage::Format_Grayscale8);
		smooth(smoothedMask, m_mask.blockSize);

		m_imgMask = Helper::QImage2Mat(smoothedMask, CV_8UC1);
		cv::erode(m_imgMask, m_imgMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(m_mask.blockSize, m_mask.blockSize)), cv::Point(-1,-1), 1);
	}
}

void Mask::smooth(QImage &smoothedMask, int maskBlockSize)
{
	cv::Mat polygon;
	std::vector<std::vector<cv::Point>> contours;

	polygon = m_imgMask.clone();
	cv::findContours(polygon, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

	// deleting black holes
	for(int main_i=0;main_i<contours.size();main_i++){
		cv::drawContours(polygon,contours,main_i,255,cv::FILLED);
	}

	// deleting white remnants
	cv::morphologyEx(polygon,polygon, cv::MORPH_OPEN,cv::getStructuringElement(cv::MORPH_RECT,cv::Size(19,19)),cv::Point(-1,-1),2);

	cv::findContours(polygon, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

	// drawing the fingerprint mask polygon
	QVector<QPoint> singlePolygon;

	smoothedMask.fill(Qt::black);
	QPainter painter2(&smoothedMask);

	painter2.setPen(QPen(QBrush(QColor(255,255,255)),1));
	painter2.setBrush(QBrush(QColor(255,255,255)));

	for(int main_i=0;main_i<contours.size();main_i++){
		cv::drawContours(polygon,contours,main_i,255,cv::FILLED);
		for(int idx=maskBlockSize/2; idx < contours.at(main_i).size(); idx += maskBlockSize*3)
		{
			singlePolygon.append(QPoint(contours.at(main_i).at(idx).x, contours.at(main_i).at(idx).y));
		}
		painter2.drawPolygon(QPolygon(singlePolygon));
		singlePolygon.clear();
	}
}

cv::Mat Mask::getImgMask() const
{
	return m_imgMask;
}
