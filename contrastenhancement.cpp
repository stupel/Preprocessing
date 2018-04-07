#include "contrastenhancement.h"

ContrastEnhancement::ContrastEnhancement(QObject *parent) : QObject(parent)
{

}

void ContrastEnhancement::performSuace(const cv::Mat &src, cv::Mat &dst, int distance, double sigma)
{
    CV_Assert(src.type() == CV_8UC1);
    if (!(distance > 0 && sigma > 0)) {
        CV_Error(CV_StsBadArg, "distance and sigma must be greater 0");
    }
    dst = cv::Mat(src.size(), CV_8UC1);
    cv::Mat smoothed;
    int val;
    int a, b;
    int adjuster;
    int half_distance = distance / 2;
    double distance_d = distance;

    cv::GaussianBlur(src, smoothed, cv::Size(0, 0), sigma);

    for (int x = 0;x<src.cols;x++)
        for (int y = 0;y < src.rows;y++) {
            val = src.at<uchar>(y, x);
            adjuster = smoothed.at<uchar>(y, x);
            if ((val - adjuster) > distance_d)adjuster += (val - adjuster)*0.5;
            adjuster = adjuster < half_distance ? half_distance : adjuster;
            b = adjuster + half_distance;
            b = b > 255 ? 255 : b;
            a = b - distance;
            a = a < 0 ? 0 : a;

            if (val >= a && val <= b)
            {
                dst.at<uchar>(y, x) = (int)(((val - a) / distance_d) * 255);
            }
            else if (val < a) {
                dst.at<uchar>(y, x) = 0;
            }
            else if (val > b) {
                dst.at<uchar>(y, x) = 255;
            }
        }
}


void ContrastEnhancement::enhance(const cv::Mat &toEnhance, cv::Mat &enhanced, int distance, double sigma, double gaussBlock, double gaussSigma)
{
    int a = distance;
    double b = sigma;
    enhanced = toEnhance.clone();
    cv::GaussianBlur(enhanced, enhanced, cv::Size(gaussBlock,gaussBlock), gaussSigma);
    performSuace(toEnhance, enhanced, a, (b + 1) / 8.0); // perform SUACE with the parameters
}
