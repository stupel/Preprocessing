#include "frequencymap.h"

FrequencyMap::FrequencyMap(QObject *parent) : QObject(parent)
{
	m_isFrequencyModelLoaded = false;
}

void FrequencyMap::loadFrequencyMapModel(const CAFFE_FILES &freqFiles)
{
	if (m_isFrequencyModelLoaded) {
		delete m_frequencyClassifier;
		m_isFrequencyModelLoaded = false;
	}

	m_frequencyClassifier = new PreprocessingCaffeNetwork;
	m_frequencyClassifier->loadModel(freqFiles.model, freqFiles.trained, freqFiles.imageMean, freqFiles.label);

	m_isFrequencyModelLoaded = true;
}

void FrequencyMap::setParams(const cv::Mat &imgOriginal, const FMAP_PARAMS &fmapParams)
{
	m_imgOriginal = imgOriginal;
	m_fmap = fmapParams;
}

void FrequencyMap::generate()
{
	if (*m_fmap.cpuOnly) Caffe::set_mode(Caffe::CPU);
	else Caffe::set_mode(Caffe::GPU);

	m_frequencyMap = cv::Mat(m_imgOriginal.rows + m_fmap.blockSize, m_imgOriginal.cols + m_fmap.blockSize, CV_8UC1);

	cv::Mat lambdaBlock = cv::Mat(m_fmap.blockSize, m_fmap.blockSize, CV_8UC1);
	cv::Mat borderedOriginal;
	cv::copyMakeBorder(m_imgOriginal, borderedOriginal, m_fmap.exBlockSize, m_fmap.exBlockSize, m_fmap.exBlockSize, m_fmap.exBlockSize, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

	std::vector<cv::Mat> blocks;
	int odd = m_fmap.exBlockSize % 2;
	for (int x = m_fmap.exBlockSize; x < m_imgOriginal.cols + m_fmap.exBlockSize; x += m_fmap.blockSize) {
		for (int y = m_fmap.exBlockSize; y < m_imgOriginal.rows + m_fmap.exBlockSize; y += m_fmap.blockSize) {
			blocks.push_back(borderedOriginal.colRange(x - m_fmap.exBlockSize/2, x + m_fmap.exBlockSize/2 + odd).rowRange(y - m_fmap.exBlockSize/2, y + m_fmap.exBlockSize/2 + odd));
		}
	}

	std::vector<std::vector<Prediction>> predictions;
	predictions = m_frequencyClassifier->classifyBatch(blocks, 8);
	/*for (int i = 0; i < blocks.size(); i++) {
		predictions.push_back(frequencyClassifier->classify(blocks.at(i)));
	}*/

	std::vector<Prediction> prediction;
	int cnt = 0;
	for (int x = 0; x < m_imgOriginal.cols; x += m_fmap.blockSize) {
		for (int y = 0; y < m_imgOriginal.rows; y += m_fmap.blockSize) {
			prediction = predictions[cnt];
			lambdaBlock.setTo(QString::fromStdString(prediction[0].first).toInt());
			lambdaBlock.copyTo(m_frequencyMap(cv::Rect(x, y, m_fmap.blockSize, m_fmap.blockSize)));
			cnt++;
		}
	}

	m_frequencyMap = m_frequencyMap.rowRange(0, m_imgOriginal.rows).colRange(0, m_imgOriginal.cols);

	m_frequencyMap.convertTo(m_frequencyMap, CV_32FC1);

	//cv::GaussianBlur(frequencyMap, frequencyMap, cv::Size(121, 121), 10.0, 10.0);
}

cv::Mat FrequencyMap::getFrequencyMap() const
{
	return m_frequencyMap;
}

cv::Mat FrequencyMap::getImgFrequencyMap() const
{
	cv::Mat imgFMap = cv::Mat(m_frequencyMap.rows, m_frequencyMap.cols, CV_8UC1);

	double minOrig;
	double maxOrig;
	cv::Point minLoc;
	cv::Point maxLoc;

	cv::minMaxLoc(m_frequencyMap, &minOrig, &maxOrig, &minLoc, &maxLoc);

	m_frequencyMap.convertTo(imgFMap, CV_8UC1, 255.0 / (maxOrig - minOrig), - 255.0 * minOrig / (maxOrig - minOrig));

	return imgFMap;
}
