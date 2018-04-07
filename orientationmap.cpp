#include "orientationmap.h"

OrientationMap::OrientationMap(QObject *parent) : QObject(parent)
{

}

void OrientationMap::setParams(const cv::Mat &imgFingerprint_, int &blockSize_, GAUSSIAN_BLUR_SETTINGS &gaussBlurBasic_, GAUSSIAN_BLUR_SETTINGS &gaussBlurAdvanced_)
{
    this->imgFingerprint = imgFingerprint_;
    this->blockSize = blockSize_;
    this->gaussBlurBasic = gaussBlurBasic_;
    this->gaussBlurAdvanced = gaussBlurAdvanced_;
}

void OrientationMap::computeBasicMap()
{
    cv::Mat Gx, Gy;
    int height, width;
    double Vx, Vy;
    height = floor(this->imgFingerprint.rows / this->blockSize);
    width = floor(this->imgFingerprint.cols / this->blockSize);

    int paddingX = this->imgFingerprint.cols - width*this->blockSize;
    int paddingY = this->imgFingerprint.rows - height*this->blockSize;

    // BASIC smerova mapa
    this->oMap_basic = cv::Mat(height, width, CV_64F);

    // vypocet gradientov x a y
    cv::Sobel(this->imgFingerprint,Gx,CV_64FC1, 1, 0);
    cv::Sobel(this->imgFingerprint,Gy,CV_64FC1, 0, 1);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Vx=0.0; Vy=0.0;
            for (int i = y * this->blockSize + paddingY / 2; i < y * this->blockSize + paddingY / 2 + this->blockSize; i++){
                for (int j = x * this->blockSize + paddingX / 2; j < x * this->blockSize + paddingX / 2 + this->blockSize; j++){
                    Vx += (2 * (Gx.at<double>(i,j) * Gy.at<double>(i, j)));
                    Vy += pow(Gx.at<double>(i,j),2) - pow(Gy.at<double>(i,j), 2);
                }
            }
            this->oMap_basic.at<double>(y, x) = 0.5 * atan2(Vx, Vy);
        }
    }

    //vyhladenie smerovej mapy
    cv::Mat sinTheta(this->oMap_basic.size(), CV_64F);
    cv::Mat cosTheta(this->oMap_basic.size(), CV_64F);

    for(int i = 0; i < this->oMap_basic.rows; i++){
        for(int j = 0; j < this->oMap_basic.cols; j++){
            cosTheta.at<double>(i, j) = cos(2* this->oMap_basic.at<double>(i, j));
            sinTheta.at<double>(i, j) = sin(2* this->oMap_basic.at<double>(i, j));
        }
    }

    cv::GaussianBlur(cosTheta, cosTheta, cv::Size(this->gaussBlurBasic.blockSize, this->gaussBlurBasic.blockSize), this->gaussBlurBasic.sigma,this->gaussBlurBasic.sigma);
    cv::GaussianBlur(sinTheta, sinTheta, cv::Size(this->gaussBlurBasic.blockSize, this->gaussBlurBasic.blockSize), this->gaussBlurBasic.sigma,this->gaussBlurBasic.sigma);
    for(int i = 0; i < this->oMap_basic.rows;i++) {
        for(int j = 0; j < this->oMap_basic.cols; j++) {
            this->oMap_basic.at<double>(i, j) = 0.5 * atan2(sinTheta.at<double>(i, j), cosTheta.at<double>(i, j));
        }
    }
}

void OrientationMap::computeAdvancedMap()
{
    this->computeBasicMap();

    // expanzia smerovej mapy
    this->oMap_advanced = cv::Mat(this->imgFingerprint.rows, this->imgFingerprint.cols, this->oMap_basic.type());
    cv::Mat blk;

    for(int i = 0; i < this->oMap_basic.rows; i++) {
        for(int j = 0; j < this->oMap_basic.cols; j++) {
            blk = this->oMap_advanced.rowRange(i * this->blockSize, i * this->blockSize + this->blockSize).colRange(j * this->blockSize, j * this->blockSize + this->blockSize);
            blk.setTo(cv::Scalar(this->oMap_basic.at<double>(i, j)));
        }
    }

    // vyhladenie expandovanej smerovej mapy
    cv::Mat sinTheta_Advanced(this->oMap_advanced.size(), CV_64F);
    cv::Mat cosTheta_Advanced(this->oMap_advanced.size(), CV_64F);

    for(int i = 0; i < this->oMap_advanced.rows; i++) {
        for(int j = 0; j < this->oMap_advanced.cols; j++) {
            cosTheta_Advanced.at<double>(i, j) = cos(2 * this->oMap_advanced.at<double>(i, j));
            sinTheta_Advanced.at<double>(i, j) = sin(2 * this->oMap_advanced.at<double>(i, j));
        }
    }

    cv::GaussianBlur(cosTheta_Advanced, cosTheta_Advanced, cv::Size(this->gaussBlurAdvanced.blockSize, this->gaussBlurAdvanced.blockSize),this->gaussBlurAdvanced.sigma,this->gaussBlurAdvanced.sigma);
    cv::GaussianBlur(sinTheta_Advanced, sinTheta_Advanced, cv::Size(this->gaussBlurAdvanced.blockSize,this->gaussBlurAdvanced.blockSize),this->gaussBlurAdvanced.sigma,this->gaussBlurAdvanced.sigma);
    for(int i = 0; i < this->oMap_advanced.rows; i++) {
        for(int j = 0; j < this->oMap_advanced.cols; j++) {
            this->oMap_advanced.at<double>(i, j) = 0.5 * atan2(sinTheta_Advanced.at<double>(i, j),cosTheta_Advanced.at<double>(i, j));
        }
    }
}


void OrientationMap::drawBasicMap(const cv::Mat &imgOriginal)
{
    // farebny obrazok smerovej mapy po vyhladeni
    cv::cvtColor(imgOriginal, this->imgOMap_basic, CV_GRAY2RGB);


    int height = floor(this->imgFingerprint.rows/this->blockSize);
    int width = floor(this->imgFingerprint.cols/this->blockSize);
    int paddingX = this->imgFingerprint.cols - width*this->blockSize;
    int paddingY = this->imgFingerprint.rows - height*this->blockSize;
    int rowsMat = this->oMap_basic.rows;
    int colsMat = this->oMap_basic.cols;
    double row1, col1, row2, col2, row3, col3, direction;

    for (int y = 0; y<rowsMat; y++){
        for(int x =0; x<colsMat; x++){
            direction = this->oMap_basic.at<double>(y,x)+CV_PI/2;
            row1 = y*this->blockSize+this->blockSize/2+paddingY/2;
            col1 = x*this->blockSize+this->blockSize/2+paddingX/2;
            row2 = row1 - sin(direction)*this->blockSize/2;
            col2 = col1 - cos(direction)*this->blockSize/2;
            row3 = row1 + sin(direction)*this->blockSize/2;
            col3 = col1 + cos(direction)*this->blockSize/2;
            cv::Point endPoint(col2,row2);
            cv::Point endPoint2(col3, row3);
            cv::line(this->imgOMap_basic, endPoint, endPoint2, cv::Scalar(255,0,0),1,4,0);
        }
    }
}

cv::Mat OrientationMap::getImgOMap_basic() const
{
    return imgOMap_basic;
}

cv::Mat OrientationMap::getOMap_advanced() const
{
    return oMap_advanced;
}
