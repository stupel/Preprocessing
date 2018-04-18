#include "mask.h"

Mask::Mask(QObject *parent) : QObject(parent)
{
    this->isMaskModelLoaded = false;
}

void Mask::loadMaskModel(const CAFFE_FILES &maskFiles)
{
    if (this->isMaskModelLoaded) {
        delete maskClassifier;
        this->isMaskModelLoaded = false;
    }

    this->maskClassifier = new PreprocessingCaffeNetwork;
    this->maskClassifier->loadModel(maskFiles.model, maskFiles.trained, maskFiles.imageMean, maskFiles.label);

    this->isMaskModelLoaded = true;
}

void Mask::setParams(const cv::Mat &imgOriginal, int blockSize, int exBlockSize, bool useSmooth, bool cpuOnly)
{
    this->imgOriginal = imgOriginal;
    this->blockSize = blockSize;
    this->exBlockSize = exBlockSize;
    this->useSmooth = useSmooth;
    this->cpuOnly = cpuOnly;
}

void Mask::generate()
{

    if (this->cpuOnly) Caffe::set_mode(Caffe::CPU);
    else Caffe::set_mode(Caffe::GPU);

    this->imgMask = cv::Mat::zeros(this->imgOriginal.rows + this->blockSize, this->imgOriginal.cols + this->blockSize, CV_8UC1);

    cv::Mat whiteBlock = cv::Mat(this->blockSize, this->blockSize, CV_8UC1);
    whiteBlock.setTo(255);

    cv::Mat borderedOriginal;
    cv::copyMakeBorder(this->imgOriginal, borderedOriginal, this->exBlockSize, this->exBlockSize, this->exBlockSize, this->exBlockSize, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

    std::vector<cv::Mat> blocks;
    for (int x = this->exBlockSize; x < this->imgOriginal.cols + this->exBlockSize; x += this->blockSize) {
        for (int y = this->exBlockSize; y < this->imgOriginal.rows + this->exBlockSize; y += this->blockSize) {
            blocks.push_back(borderedOriginal.colRange(x, x + this->blockSize).rowRange(y, y + this->blockSize));
        }
    }

    std::vector<std::vector<Prediction>> predictions;
    predictions = this->maskClassifier->classifyBatch(blocks, 2);

    std::vector<Prediction> prediction;
    int cnt = 0;
    for (int x = 0; x < this->imgOriginal.cols; x += this->blockSize) {
        for (int y = 0; y < this->imgOriginal.rows; y += this->blockSize) {
            prediction = predictions[cnt];
            if (prediction[0].first[0] == 'f') {
                whiteBlock.copyTo(this->imgMask(cv::Rect(x, y, this->blockSize, this->blockSize)));
            }
            cnt++;
        }
    }

    this->imgMask = this->imgMask.rowRange(0, this->imgOriginal.rows).colRange(0, this->imgOriginal.cols);

    if (useSmooth) {
        QImage smoothedMask(this->imgMask.cols, this->imgMask.rows, QImage::Format_Grayscale8);
        this->smooth(smoothedMask, this->blockSize);

        this->imgMask = Helper::QImage2Mat(smoothedMask, CV_8UC1);
        cv::erode(this->imgMask, this->imgMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(this->blockSize, this->blockSize)), cv::Point(-1,-1), 1);
    }
}

void Mask::smooth(QImage &smoothedMask, int maskBlockSize)
{
    cv::Mat polygon;
    std::vector<std::vector<cv::Point>> contours;

    polygon = this->imgMask.clone();
    cv::findContours(polygon, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

    // deleting black holes
    for(int main_i=0;main_i<contours.size();main_i++){
        cv::drawContours(polygon,contours,main_i,255,cv::FILLED);
    }

    // deleting white remnants
    cv::morphologyEx(polygon,polygon, cv::MORPH_OPEN,cv::getStructuringElement(cv::MORPH_RECT,cv::Size(19,19)),cv::Point(-1,-1),2);

    cv::findContours(polygon, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

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
    return imgMask;
}
