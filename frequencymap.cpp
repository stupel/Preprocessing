#include "frequencymap.h"

FrequencyMap::FrequencyMap(QObject *parent) : QObject(parent)
{

}

void FrequencyMap::loadFrequencyMapModel(const CAFFE_FILES &freqFiles)
{
    this->frequencyClassifier = new CaffeNetwork;
    this->frequencyClassifier->moveToThread(this->frequencyClassifier);
    this->frequencyClassifier->start();

    this->frequencyClassifier->loadModel(freqFiles.model, freqFiles.trained, freqFiles.imageMean, freqFiles.label);
}

void FrequencyMap::generate(const cv::Mat &imgOriginal, const int &blockSize, const int &exBlockSize)
{
    this->frequencyMap = cv::Mat(imgOriginal.rows, imgOriginal.cols, CV_8UC1);
    this->frequencyMap.setTo(8);

    cv::Mat lambdaBlock = cv::Mat(blockSize, blockSize, CV_8UC1);
    cv::Mat borderedOriginal;
    cv::copyMakeBorder(imgOriginal, borderedOriginal, exBlockSize, exBlockSize, exBlockSize, exBlockSize, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

    std::vector<std::vector<Prediction>> predictions;
    std::vector<Prediction> prediction;

    std::vector<cv::Mat> blocks;
    for (int x = blockSize/2; x < imgOriginal.cols - blockSize/2-1; x += blockSize) {
        for (int y = blockSize/2; y < imgOriginal.rows - blockSize/2-1; y += blockSize) {
            blocks.push_back(borderedOriginal.rowRange(exBlockSize + y - exBlockSize/2, exBlockSize + y + exBlockSize/2).colRange(exBlockSize + x - exBlockSize/2, exBlockSize + x + exBlockSize/2));
        }
    }

    predictions = this->frequencyClassifier->classifyBatch(blocks, 4);

    int cnt = 0;
    for (int x = blockSize/2; x < imgOriginal.cols - blockSize/2-1; x += blockSize) {
        for (int y = blockSize/2; y < imgOriginal.rows - blockSize/2-1; y += blockSize) {
            prediction = predictions[cnt];
            lambdaBlock.setTo(QString::fromStdString(prediction[0].first).toInt());
            lambdaBlock.copyTo(this->frequencyMap(cv::Rect(x-blockSize/2, y-blockSize/2, blockSize, blockSize)));
            cnt++;
        }
    }

    this->frequencyMap.convertTo(this->frequencyMap, CV_64F);

    /*for (int x = 0; x < this->frequencyMap.cols; x++) {
        for (int y = 0; y < this->frequencyMap.rows; y++) {
             qDebug() << this->frequencyMap.at<uchar>(y, x);
        }
    }*/

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

    //this->frequencyMap.copyTo(imgFMap);
    //this->frequencyMap.convertTo(imgFMap, CV_8UC1, 255.0 / (maxOrig - minOrig), - 255.0 * minOrig / (maxOrig - minOrig));
    this->frequencyMap.convertTo(imgFMap, CV_8UC1 , 255.0/ (20 - 1), - 255.0 * 1 / (20 - 1));

    /*for (int x = 0; x < imgFMap.cols; x++) {
        for (int y = 0; y < imgFMap.rows; y++) {
            qDebug() << imgFMap.at<uchar>(y, x);
        }
    }*/

    return imgFMap;
}
