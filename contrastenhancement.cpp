#include "contrastenhancement.h"

ContrastEnhancement::ContrastEnhancement(QObject *parent) : QObject(parent)
{

}

void ContrastEnhancement::setParams(const cv::Mat &imgOriginal, const CONTRAST_PARAMS &contrastParams)
{
    this->imgOriginal = imgOriginal;
    this->contrast = contrastParams;
}

void ContrastEnhancement::performSuace()
{
    this->contrast.sigma = (this->contrast.sigma + 1) / 8.0;
    CV_Assert(this->imgOriginal.type() == CV_8UC1);
    if (!(this->contrast.distance > 0 && this->contrast.sigma > 0)) {
        CV_Error(cv::Error::StsBadArg, "distance and sigma must be greater 0");
    }
    cv::Mat smoothed;
    int val;
    int a, b;
    int adjuster;
    int half_distance = this->contrast.distance / 2;
    double distance_d = this->contrast.distance;

    cv::GaussianBlur(this->imgOriginal, smoothed, cv::Size(0, 0), this->contrast.sigma);

    for (int x = 0; x < this->imgOriginal.cols; x++)
        for (int y = 0; y < this->imgOriginal.rows; y++) {
            val = this->imgOriginal.at<uchar>(y, x);
            adjuster = smoothed.at<uchar>(y, x);
            if ((val - adjuster) > distance_d) adjuster += (val - adjuster) * 0.5;
            adjuster = adjuster < half_distance ? half_distance : adjuster;
            b = adjuster + half_distance;
            b = b > 255 ? 255 : b;
            a = b - this->contrast.distance;
            a = a < 0 ? 0 : a;

            if (val >= a && val <= b)
            {
                this->imgContrastEnhanced.at<uchar>(y, x) = (int)(((val - a) / distance_d) * 255);
            }
            else if (val < a) {
                this->imgContrastEnhanced.at<uchar>(y, x) = 0;
            }
            else if (val > b) {
                this->imgContrastEnhanced.at<uchar>(y, x) = 255;
            }
        }
}

void ContrastEnhancement::enhance()
{
    this->imgContrastEnhanced = this->imgOriginal.clone();
    cv::GaussianBlur(this->imgContrastEnhanced, this->imgContrastEnhanced, cv::Size(this->contrast.gaussBlock, this->contrast.gaussBlock), this->contrast.gaussSigma);
    this->performSuace(); // perform SUACE with the parameters
}

cv::Mat ContrastEnhancement::getImgContrastEnhanced() const
{
    return imgContrastEnhanced;
}
