#include "gaborthread.h"

GaborThread::GaborThread(QObject *parent) : QObject(parent)
{

}

GaborThread::GaborThread(cv::Mat img, cv::Mat oMap, const GABOR_PARAMS &gaborParams, cv::Rect rect, cv::Mat enhancedImage):
    img(img), oMap(oMap), gabor(gaborParams), rect(rect), enhancedImage(enhancedImage)
{

}

void GaborThread::enhanceFragmentSlot()
{
    cv::Mat kernel;
    cv::Mat subMat, sub;
    cv::Scalar s;

    for(int i = this->rect.y; i < this->rect.y + this->rect.height; i++){
        for(int j = this->rect.x; j < this->rect.x + this->rect.width; j++){
            if (this->gabor.useFrequencyMap) kernel = cv::getGaborKernel(cv::Size(this->gabor.blockSize,this->gabor.blockSize), this->gabor.sigma, this->oMap.at<float>(i,j), this->gabor.frequencyMap.at<float>(i,j), 1, 0, CV_32F);
            else kernel = cv::getGaborKernel(cv::Size(this->gabor.blockSize,this->gabor.blockSize), this->gabor.sigma, this->oMap.at<float>(i,j), this->gabor.lambda, 1, 0, CV_32F);
            subMat = this->img(cv::Rect(j-this->gabor.blockSize/2, i-this->gabor.blockSize/2, this->gabor.blockSize, this->gabor.blockSize));
            subMat.convertTo(sub, CV_32F);
            cv::multiply(sub, kernel, sub);
            s = cv::sum(sub);
            this->enhancedImage.at<float>(i,j) = s[0];
        }
    }
    emit enhancementDoneSignal();
}

void GaborThread::enhancementDoneSlot()
{

}
