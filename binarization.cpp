#include "binarization.h"

Binarization::Binarization(QObject *parent) : QObject(parent)
{

}

void Binarization::setParams(const cv::Mat &imgEnhanced, const bool &useMask, const cv::Mat &imgMask, const bool &useQMap, const cv::Mat &imgQMap)
{
    this->imgEnhanced_ = imgEnhanced;
    this->useMask_ = useMask;
    this->imgMask_ = imgMask;
    this->useQMap_ = useQMap;
    this->imgQMap_ = imgQMap;
}

void Binarization::binarizeGaussianBlur()
{
    this->imgBinarized = cv::Mat(this->imgEnhanced_.rows, this->imgEnhanced_.cols, this->imgEnhanced_.type());
    cv::GaussianBlur(this->imgEnhanced_, this->imgBinarized, cv::Size(3,3), 1);
    cv::threshold(this->imgBinarized, this->imgBinarized, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);

    if (this->useQMap_ || this->useMask_) this->deleteBackground();
}

void Binarization::binarizeAdaptive()
{
    this->imgBinarized = cv::Mat(this->imgEnhanced_.rows, this->imgEnhanced_.cols, this->imgEnhanced_.type());
    cv::adaptiveThreshold(this->imgEnhanced_, this->imgBinarized, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 21, 10);

    if (this->useQMap_ || this->useMask_) this->deleteBackground();
}

void Binarization::deleteBackground()
{
    if (this->useQMap_) {
        for (int x = 0; x < this->imgBinarized.cols; x++) {
            for (int y = 0; y < this->imgBinarized.rows; y++) {
                if (this->imgQMap_.at<uchar>(y, x) == 0) this->imgBinarized.at<uchar>(y, x) = 255;
            }
        }
    }
    else if (this->useMask_) {
        for (int x = 0; x < this->imgBinarized.cols; x++) {
            for (int y = 0; y < this->imgBinarized.rows; y++) {
                if (this->imgMask_.at<uchar>(y, x) == 0) this->imgBinarized.at<uchar>(y, x) = 255;
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
        if (it->size()>holeSize)
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
