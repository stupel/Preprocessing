#include "frequencymap.h"

FrequencyMap::FrequencyMap(QObject *parent) : QObject(parent)
{
    this->isFrequencyModelLoaded = false;
}

void FrequencyMap::loadFrequencyMapModel(const CAFFE_FILES &freqFiles)
{
    if (this->isFrequencyModelLoaded) {
        delete frequencyClassifier;
        this->isFrequencyModelLoaded = false;
    }

    this->frequencyClassifier = new PreprocessingCaffeNetwork;
    this->frequencyClassifier->loadModel(freqFiles.model, freqFiles.trained, freqFiles.imageMean, freqFiles.label);

    this->isFrequencyModelLoaded = true;
}

void FrequencyMap::setParams(const cv::Mat &imgOriginal, int blockSize, int exBlockSize, bool cpuOnly)
{
    this->imgOriginal = imgOriginal;
    this->blockSize = blockSize;
    this->exBlockSize = exBlockSize;
    this->cpuOnly = cpuOnly;
}

void FrequencyMap::generate()
{
    if (this->cpuOnly) Caffe::set_mode(Caffe::CPU);
    else Caffe::set_mode(Caffe::GPU);

    this->frequencyMap = cv::Mat(this->imgOriginal.rows + this->blockSize, this->imgOriginal.cols + this->blockSize, CV_8UC1);

    cv::Mat lambdaBlock = cv::Mat(blockSize, blockSize, CV_8UC1);
    cv::Mat borderedOriginal;
    cv::copyMakeBorder(this->imgOriginal, borderedOriginal, this->exBlockSize, this->exBlockSize, this->exBlockSize, this->exBlockSize, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

    std::vector<cv::Mat> blocks;
    for (int x = this->exBlockSize; x < this->imgOriginal.cols + this->exBlockSize; x += this->blockSize) {
        for (int y = this->exBlockSize; y < this->imgOriginal.rows + exBlockSize; y += this->blockSize) {
            blocks.push_back(borderedOriginal.colRange(x, x + this->blockSize).rowRange(y, y + this->blockSize));
        }
    }

    std::vector<std::vector<Prediction>> predictions;
    predictions = this->frequencyClassifier->classifyBatch(blocks, 4);

    std::vector<Prediction> prediction;
    int cnt = 0;
    for (int x = 0; x < this->imgOriginal.cols; x += this->blockSize) {
        for (int y = 0; y < this->imgOriginal.rows; y += this->blockSize) {
            prediction = predictions[cnt];
            lambdaBlock.setTo(QString::fromStdString(prediction[0].first).toInt());
            lambdaBlock.copyTo(this->frequencyMap(cv::Rect(x, y, this->blockSize, this->blockSize)));
            cnt++;
        }
    }

    this->frequencyMap = this->frequencyMap.rowRange(0, this->imgOriginal.rows).colRange(0, this->imgOriginal.cols);

    this->frequencyMap.convertTo(this->frequencyMap, CV_64F);

    cv::GaussianBlur(this->frequencyMap, this->frequencyMap, cv::Size(121, 121), 10.0, 10.0);
}

cv::Mat FrequencyMap::getFrequencyMap() const
{
    return frequencyMap;
}

cv::Mat FrequencyMap::getImgFrequencyMap() const
{
    cv::Mat imgFMap = cv::Mat(this->frequencyMap.rows, this->frequencyMap.cols, CV_8UC1);

    double minOrig;
    double maxOrig;
    cv::Point minLoc;
    cv::Point maxLoc;

    cv::minMaxLoc(this->frequencyMap, &minOrig, &maxOrig, &minLoc, &maxLoc);

    this->frequencyMap.convertTo(imgFMap, CV_8UC1, 255.0 / (maxOrig - minOrig), - 255.0 * minOrig / (maxOrig - minOrig));

    return imgFMap;
}
