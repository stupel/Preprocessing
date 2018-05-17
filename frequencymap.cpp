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

void FrequencyMap::setParams(const cv::Mat &imgOriginal, const FMAP_PARAMS &fmapParams)
{
    this->imgOriginal = imgOriginal;
    this->fmap = fmapParams;
}

void FrequencyMap::generate()
{
    if (*this->fmap.cpuOnly) Caffe::set_mode(Caffe::CPU);
    else Caffe::set_mode(Caffe::GPU);

    this->frequencyMap = cv::Mat(this->imgOriginal.rows + this->fmap.blockSize, this->imgOriginal.cols + this->fmap.blockSize, CV_8UC1);

    cv::Mat lambdaBlock = cv::Mat(this->fmap.blockSize, this->fmap.blockSize, CV_8UC1);
    cv::Mat borderedOriginal;
    cv::copyMakeBorder(this->imgOriginal, borderedOriginal, this->fmap.exBlockSize, this->fmap.exBlockSize, this->fmap.exBlockSize, this->fmap.exBlockSize, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

    std::vector<cv::Mat> blocks;
    int odd = this->fmap.exBlockSize % 2;
    for (int x = this->fmap.exBlockSize; x < this->imgOriginal.cols + this->fmap.exBlockSize; x += this->fmap.blockSize) {
        for (int y = this->fmap.exBlockSize; y < this->imgOriginal.rows + fmap.exBlockSize; y += this->fmap.blockSize) {
            blocks.push_back(borderedOriginal.colRange(x - this->fmap.exBlockSize/2, x + this->fmap.exBlockSize/2 + odd).rowRange(y - this->fmap.exBlockSize/2, y + this->fmap.exBlockSize/2 + odd));
        }
    }

    std::vector<std::vector<Prediction>> predictions;
    predictions = this->frequencyClassifier->classifyBatch(blocks, 8);
    /*for (int i = 0; i < blocks.size(); i++) {
        predictions.push_back(this->frequencyClassifier->classify(blocks.at(i)));
    }*/

    std::vector<Prediction> prediction;
    int cnt = 0;
    for (int x = 0; x < this->imgOriginal.cols; x += this->fmap.blockSize) {
        for (int y = 0; y < this->imgOriginal.rows; y += this->fmap.blockSize) {
            prediction = predictions[cnt];
            lambdaBlock.setTo(QString::fromStdString(prediction[0].first).toInt());
            lambdaBlock.copyTo(this->frequencyMap(cv::Rect(x, y, this->fmap.blockSize, this->fmap.blockSize)));
            cnt++;
        }
    }

    this->frequencyMap = this->frequencyMap.rowRange(0, this->imgOriginal.rows).colRange(0, this->imgOriginal.cols);

    this->frequencyMap.convertTo(this->frequencyMap, CV_32FC1);

    //cv::GaussianBlur(this->frequencyMap, this->frequencyMap, cv::Size(121, 121), 10.0, 10.0);
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
