#include "mask.h"

Mask::Mask(QObject *parent) : QObject(parent)
{

}

void Mask::loadMaskModel(const CAFFE_FILES &maskFiles)
{
    this->maskClassifier = new CaffeNetwork;
    this->maskClassifier->moveToThread(this->maskClassifier);
    this->maskClassifier->start();

    this->maskClassifier->loadModel(maskFiles.model, maskFiles.trained, maskFiles.imageMean, maskFiles.label);
}

void Mask::generate(cv::Mat imgOriginal, int blockSize, int exBlockSize, bool useSmooth)
{
    this->imgMask = cv::Mat::zeros(imgOriginal.rows, imgOriginal.cols, CV_8UC1);

    cv::Mat whiteBlock = cv::Mat(blockSize, blockSize, CV_8UC1);
    whiteBlock.setTo(255);

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

    predictions = this->maskClassifier->classifyBatch(blocks, 2);

    int cnt = 0;
    for (int x = blockSize/2; x < imgOriginal.cols - blockSize/2-1; x += blockSize) {
        for (int y = blockSize/2; y < imgOriginal.rows - blockSize/2-1; y += blockSize) {
            prediction = predictions[cnt];
            if (prediction[0].first[0] == 'f') {
                whiteBlock.copyTo(this->imgMask(cv::Rect(x-blockSize/2, y-blockSize/2, blockSize, blockSize)));
            }
            cnt++;
        }
    }

    if (useSmooth) {
        QImage smoothedMask(this->imgMask.cols, this->imgMask.rows, QImage::Format_Grayscale8);
        this->smooth(smoothedMask, blockSize);

        this->imgMask = Helper::QImage2Mat(smoothedMask, CV_8UC1);
        cv::erode(this->imgMask, this->imgMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(blockSize, blockSize)), cv::Point(-1,-1), 1);
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
