#include "binarization.h"

Binarization::Binarization(QObject *parent) : QObject(parent)
{

}

void Binarization::setParams(const cv::Mat &imgEnhanced, const BINARIZATION_PARAMS &binarizationParams, const PREPROCESSING_FEATURES &features)
{
    this->imgEnhanced = imgEnhanced;
    this->binarization = binarizationParams;
    this->features = features;
}

void Binarization::binarizeGaussianBlur()
{
    this->imgBinarized = cv::Mat(this->imgEnhanced.rows, this->imgEnhanced.cols, this->imgEnhanced.type());
    cv::GaussianBlur(this->imgEnhanced, this->imgBinarized, cv::Size(3,3), 1);
    cv::threshold(this->imgBinarized, this->imgBinarized, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);

    if (this->features.useQualityMap || this->features.useMask) this->deleteBackground();
}

void Binarization::binarizeAdaptive()
{
    this->imgBinarized = cv::Mat(this->imgEnhanced.rows, this->imgEnhanced.cols, this->imgEnhanced.type());
    cv::adaptiveThreshold(this->imgEnhanced, this->imgBinarized, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 21, 10);

    if (this->features.useQualityMap || this->features.useMask) this->deleteBackground();
}

void Binarization::deleteBackground()
{
    if (this->features.useQualityMap) {
        for (int x = 0; x < this->imgBinarized.cols; x++) {
            for (int y = 0; y < this->imgBinarized.rows; y++) {
                if (this->binarization.imgQualityMap->at<uchar>(y, x) == 0) this->imgBinarized.at<uchar>(y, x) = 255;
            }
        }
    }
    else if (this->features.useMask) {
        for (int x = 0; x < this->imgBinarized.cols; x++) {
            for (int y = 0; y < this->imgBinarized.rows; y++) {
                if (this->binarization.imgMask->at<uchar>(y, x) == 0) this->imgBinarized.at<uchar>(y, x) = 255;
            }
        }
    }
}

void Binarization::removeHoles(double holeSize)
{
    // invertujem obraz
    cv::bitwise_not(this->imgBinarized, this->imgBinarized);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    // najdem uplne vsetky diery
    cv::findContours(this->imgBinarized, contours, hierarchy,cv::RETR_CCOMP,cv::CHAIN_APPROX_SIMPLE);

    // odstranujem vsetky velke diery, zostanu len male diery - potne pory a nedokonalosti
    for (std::vector<std::vector<cv::Point> >::iterator it = contours.begin(); it!=contours.end(); )
    {
        if (it->size() > holeSize)
            it=contours.erase(it);
        else
            ++it;
    }

    // vyplnim diery bielou farbou
    for(int ii=0;ii<contours.size();ii++)
    {
        cv::drawContours(this->imgBinarized, contours, ii, cv::Scalar(255), -1, cv::LINE_8);
    }

    // invertujem obraz
    cv::bitwise_not(this->imgBinarized, this->imgBinarized);
}

cv::Mat Binarization::getImgBinarized() const
{
    return imgBinarized;
}
