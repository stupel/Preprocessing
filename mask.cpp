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

void Mask::setParams(const cv::Mat &imgOriginal, const MASK_PARAMS &maskParams)
{
    this->imgOriginal = imgOriginal;
    this->mask = maskParams;
}

void Mask::generate()
{

    if (*this->mask.cpuOnly) Caffe::set_mode(Caffe::CPU);
    else Caffe::set_mode(Caffe::GPU);

    this->imgMask = cv::Mat::zeros(this->imgOriginal.rows + this->mask.blockSize, this->imgOriginal.cols + this->mask.blockSize, CV_8UC1);

    cv::Mat whiteBlock = cv::Mat(this->mask.blockSize, this->mask.blockSize, CV_8UC1);
    whiteBlock.setTo(255);

    cv::Mat borderedOriginal;
    cv::copyMakeBorder(this->imgOriginal, borderedOriginal, this->mask.exBlockSize, this->mask.exBlockSize, this->mask.exBlockSize, this->mask.exBlockSize, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    std::vector<cv::Mat> blocks;
    int odd = this->mask.exBlockSize % 2;
    for (int x = this->mask.exBlockSize; x < this->imgOriginal.cols + this->mask.exBlockSize; x += this->mask.blockSize) {
        for (int y = this->mask.exBlockSize; y < this->imgOriginal.rows + this->mask.exBlockSize; y += this->mask.blockSize) {
            blocks.push_back(borderedOriginal.colRange(x - this->mask.exBlockSize/2, x + this->mask.exBlockSize/2 + odd).rowRange(y - this->mask.exBlockSize/2, y + this->mask.exBlockSize/2 + odd));

        }
    }

    std::vector<std::vector<Prediction>> predictions;

    //Use Batch
    predictions = this->maskClassifier->classifyBatch(blocks, 2);

    //Without Batch
    /*for (int i = 0; i < blocks.size(); i++) {
        predictions.push_back(this->maskClassifier->classify(blocks[i]));
    }*/

    std::vector<Prediction> prediction;
    int cnt = 0;
    for (int x = 0; x < this->imgOriginal.cols; x += this->mask.blockSize) {
        for (int y = 0; y < this->imgOriginal.rows; y += this->mask.blockSize) {
            prediction = predictions[cnt];
            if (prediction[0].first[0] == 'f' || prediction[0].first[0] == 'F') {
                whiteBlock.copyTo(this->imgMask(cv::Rect(x, y, this->mask.blockSize, this->mask.blockSize)));
            }
            cnt++;
        }
    }

    this->imgMask = this->imgMask.rowRange(0, this->imgOriginal.rows).colRange(0, this->imgOriginal.cols);

    if (this->mask.useSmooth) {
        QImage smoothedMask(this->imgMask.cols, this->imgMask.rows, QImage::Format_Grayscale8);
        this->smooth(smoothedMask, this->mask.blockSize);

        this->imgMask = Helper::QImage2Mat(smoothedMask, CV_8UC1);
        cv::erode(this->imgMask, this->imgMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(this->mask.blockSize, this->mask.blockSize)), cv::Point(-1,-1), 1);
    }
}

void Mask::smooth(QImage &smoothedMask, int maskBlockSize)
{
    cv::Mat polygon;
    std::vector<std::vector<cv::Point>> contours;

    polygon = this->imgMask.clone();
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
    return imgMask;
}
