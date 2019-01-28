#include "orientationmap.h"

OrientationMap::OrientationMap(QObject *parent) : QObject(parent)
{
    this->duration = 0;
}

void OrientationMap::setParams(const cv::Mat &imgFingerprint, OMAP_PARAMS omapParams)
{
    this->imgInput = imgFingerprint;
    this->omap = omapParams;
}

void OrientationMap::computeBasicMapCPU()
{
    this->timer.start();

    cv::Mat Gx, Gy;
    int height, width;
    float Vx, Vy;
    height = floor(this->imgInput.rows / this->omap.blockSize);
    width = floor(this->imgInput.cols / this->omap.blockSize);

    int paddingX = this->imgInput.cols - width*this->omap.blockSize;
    int paddingY = this->imgInput.rows - height*this->omap.blockSize;

    // BASIC smerova mapa
    this->oMap_basic = cv::Mat(height, width, CV_32F);

    // vypocet gradientov x a y
    cv::Sobel(this->imgInput, Gx, CV_32FC1, 1, 0);
    cv::Sobel(this->imgInput, Gy, CV_32FC1, 0, 1);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Vx = 0.0; Vy = 0.0;
            for (int i = y * this->omap.blockSize + paddingY / 2; i < y * this->omap.blockSize + paddingY / 2 + this->omap.blockSize; i++){
                for (int j = x * this->omap.blockSize + paddingX / 2; j < x * this->omap.blockSize + paddingX / 2 + this->omap.blockSize; j++){
                    Vx += (2 * (Gx.at<float>(i,j) * Gy.at<float>(i, j)));
                    Vy += pow(Gx.at<float>(i,j),2) - pow(Gy.at<float>(i, j), 2);
                }
            }
            this->oMap_basic.at<float>(y, x) = 0.5 * atan2(Vx, Vy);
        }
    }

    //vyhladenie smerovej mapy
    cv::Mat sinTheta(this->oMap_basic.size(), CV_32F);
    cv::Mat cosTheta(this->oMap_basic.size(), CV_32F);

    for(int i = 0; i < this->oMap_basic.rows; i++){
        for(int j = 0; j < this->oMap_basic.cols; j++){
            cosTheta.at<float>(i, j) = cos(2 * this->oMap_basic.at<float>(i, j));
            sinTheta.at<float>(i, j) = sin(2 * this->oMap_basic.at<float>(i, j));
        }
    }

    cv::GaussianBlur(cosTheta, cosTheta, cv::Size(this->omap.gaussBlurBasic.blockSize, this->omap.gaussBlurBasic.blockSize), this->omap.gaussBlurBasic.sigma,this->omap.gaussBlurBasic.sigma);
    cv::GaussianBlur(sinTheta, sinTheta, cv::Size(this->omap.gaussBlurBasic.blockSize, this->omap.gaussBlurBasic.blockSize), this->omap.gaussBlurBasic.sigma,this->omap.gaussBlurBasic.sigma);
    for(int i = 0; i < this->oMap_basic.rows;i++) {
        for(int j = 0; j < this->oMap_basic.cols; j++) {
            this->oMap_basic.at<float>(i, j) = 0.5 * atan2(sinTheta.at<float>(i, j), cosTheta.at<float>(i, j));
        }
    }

    this->duration = this->timer.elapsed();
}

void OrientationMap::computeBasicMapGPU()
{
    af::timer::start();

    this->imgInputAF = Helper::mat_uchar2array_uchar(this->imgInput);

    af::array Gx, Gy;
    int height, width;
    af::array Vx, Vy;
    height = floor(this->imgInputAF.dims(0) / this->omap.blockSize);
    width = floor(this->imgInputAF.dims(1) / this->omap.blockSize);
    int paddingX = this->imgInputAF.dims(1) - width * this->omap.blockSize;
    int paddingY = this->imgInputAF.dims(0) - height * this->omap.blockSize;

    // vypocet gradientov x a y
    af::sobel(Gx, Gy, this->imgInputAF);

    // vypocet Vx,Vy a Theta
    af::array GxCut = Gx(af::seq(paddingY / 2, height * this->omap.blockSize + paddingY / 2 - 1), af::seq(paddingX / 2, width * this->omap.blockSize+paddingX / 2 - 1));
    af::array GyCut = Gy(af::seq(paddingY / 2, height * this->omap.blockSize + paddingY / 2 - 1), af::seq(paddingX / 2, width * this->omap.blockSize + paddingX / 2 - 1));

    GxCut = af::unwrap(GxCut, this->omap.blockSize, this->omap.blockSize, this->omap.blockSize, this->omap.blockSize);
    GyCut = af::unwrap(GyCut, this->omap.blockSize, this->omap.blockSize, this->omap.blockSize, this->omap.blockSize);

    Vx =  af::sum(2 * GxCut * GyCut);
    Vy =  af::sum(af::pow(GxCut, 2) - af::pow(GyCut, 2));
    this->oMapAF_basic = 0.5* af::atan2(Vx.as(f32), Vy.as(f32));

    this->oMapAF_basic = af::moddims(this->oMapAF_basic, height, width);

    // vyhladenie smerovej mapy
    af::array sinTheta = af::sin(2 * this->oMapAF_basic);
    af::array cosTheta = af::cos(2 * this->oMapAF_basic);

    af::array gk = af::gaussianKernel(this->omap.gaussBlurBasic.blockSize, this->omap.gaussBlurBasic.blockSize, this->omap.gaussBlurBasic.sigma, this->omap.gaussBlurBasic.sigma);

    sinTheta = af::convolve(sinTheta, gk);
    cosTheta = af::convolve(cosTheta, gk);

    this->oMapAF_basic = 0.5* af::atan2(sinTheta, cosTheta);

    this->oMap_basic = Helper::array_float2mat_float(this->oMapAF_basic);

    this->duration = af::timer::stop() * 1000;
}

void OrientationMap::computeAdvancedMapCPU()
{
    this->computeBasicMapCPU();

    this->timer.start();

    // expanzia smerovej mapy
    this->oMap_advanced = cv::Mat(this->imgInput.rows, this->imgInput.cols, this->oMap_basic.type());
    cv::Mat blk;

    for(int i = 0; i < this->oMap_basic.rows; i++) {
        for(int j = 0; j < this->oMap_basic.cols; j++) {
            blk = this->oMap_advanced.rowRange(i * this->omap.blockSize, i * this->omap.blockSize + this->omap.blockSize).colRange(j * this->omap.blockSize, j * this->omap.blockSize + this->omap.blockSize);
            blk.setTo(cv::Scalar(this->oMap_basic.at<float>(i, j)));
        }
    }

    // vyhladenie expandovanej smerovej mapy
    cv::Mat sinTheta_Advanced(this->oMap_advanced.size(), CV_32F);
    cv::Mat cosTheta_Advanced(this->oMap_advanced.size(), CV_32F);

    for(int i = 0; i < this->oMap_advanced.rows; i++) {
        for(int j = 0; j < this->oMap_advanced.cols; j++) {
            cosTheta_Advanced.at<float>(i, j) = cos(2 * this->oMap_advanced.at<float>(i, j));
            sinTheta_Advanced.at<float>(i, j) = sin(2 * this->oMap_advanced.at<float>(i, j));
        }
    }

    cv::GaussianBlur(cosTheta_Advanced, cosTheta_Advanced, cv::Size(this->omap.gaussBlurAdvanced.blockSize, this->omap.gaussBlurAdvanced.blockSize), this->omap.gaussBlurAdvanced.sigma, this->omap.gaussBlurAdvanced.sigma);
    cv::GaussianBlur(sinTheta_Advanced, sinTheta_Advanced, cv::Size(this->omap.gaussBlurAdvanced.blockSize, this->omap.gaussBlurAdvanced.blockSize), this->omap.gaussBlurAdvanced.sigma, this->omap.gaussBlurAdvanced.sigma);
    for(int i = 0; i < this->oMap_advanced.rows; i++) {
        for(int j = 0; j < this->oMap_advanced.cols; j++) {
            this->oMap_advanced.at<float>(i, j) = 0.5 * atan2(sinTheta_Advanced.at<float>(i, j), cosTheta_Advanced.at<float>(i, j));
        }
    }

    this->duration += this->timer.elapsed();
}

void OrientationMap::computeAdvancedMapGPU()
{
    af::timer::start();

    // compute the basic O-Map first
    this->computeBasicMapGPU();

    // basic O-Map expansion
    this->oMapAF_advanced = af::moddims(this->oMapAF_basic, 1, this->oMapAF_basic.dims(0) * this->oMapAF_basic.dims(1));
    this->oMapAF_advanced = af::tile(this->oMapAF_advanced, this->omap.blockSize * this->omap.blockSize);
    this->oMapAF_advanced = af::wrap(this->oMapAF_advanced,
                                     this->oMapAF_basic.dims(0) * this->omap.blockSize,
                                     this->oMapAF_basic.dims(1) * this->omap.blockSize,
                                     this->omap.blockSize,
                                     this->omap.blockSize,
                                     this->omap.blockSize,
                                     this->omap.blockSize);

    // smoothing the expanded O-Map
    af::array sinTheta = af::sin(2 * this->oMapAF_advanced);
    af::array cosTheta = af::cos(2 * this->oMapAF_advanced);
    af::array gk = af::gaussianKernel(this->omap.gaussBlurAdvanced.blockSize,
                                      this->omap.gaussBlurAdvanced.blockSize,
                                      this->omap.gaussBlurAdvanced.sigma,
                                      this->omap.gaussBlurAdvanced.sigma);

    sinTheta = af::convolve(sinTheta, gk);
    cosTheta = af::convolve(cosTheta, gk);

    this->oMapAF_advanced = 0.5* af::atan2(sinTheta, cosTheta);

    this->oMap_advanced = Helper::array_float2mat_float(this->oMapAF_advanced);

    this->duration +=  af::timer::stop();
}

void OrientationMap::drawBasicMap(const cv::Mat &imgOriginal)
{
    // farebny obrazok smerovej mapy po vyhladeni
    this->imgOMap_basic = cv::Mat(imgOriginal.rows, imgOriginal.cols, CV_8UC3);
    cv::cvtColor(imgOriginal, this->imgOMap_basic, cv::COLOR_GRAY2RGB);

    int height = floor(this->imgInput.rows / this->omap.blockSize);
    int width = floor(this->imgInput.cols / this->omap.blockSize);
    int paddingX = this->imgInput.cols - width * this->omap.blockSize;
    int paddingY = this->imgInput.rows - height * this->omap.blockSize;
    int rowsMat = this->oMap_basic.rows;
    int colsMat = this->oMap_basic.cols;
    float row1, col1, row2, col2, row3, col3, direction;

    for (int y = 0; y<rowsMat; y++){
        for(int x =0; x<colsMat; x++){
            direction = this->oMap_basic.at<float>(y,x) + CV_PI / 2;
            row1 = y * this->omap.blockSize + this->omap.blockSize / 2 + paddingY / 2;
            col1 = x * this->omap.blockSize + this->omap.blockSize / 2 + paddingX / 2;
            row2 = row1 - sin(direction) * this->omap.blockSize / 2;
            col2 = col1 - cos(direction) * this->omap.blockSize / 2;
            row3 = row1 + sin(direction) * this->omap.blockSize / 2;
            col3 = col1 + cos(direction) * this->omap.blockSize / 2;
            cv::Point endPoint(col2, row2);
            cv::Point endPoint2(col3, row3);
            cv::line(this->imgOMap_basic, endPoint, endPoint2, cv::Scalar(255,255,0), 1, 4, 0);
        }
    }
}

cv::Mat OrientationMap::getImgOMap_basic() const
{
    return imgOMap_basic;
}

cv::Mat OrientationMap::getOMap_basic() const
{
    return oMap_basic;
}

cv::Mat OrientationMap::getOMap_advanced() const
{
    return oMap_advanced;
}

af::array OrientationMap::getOMapAF_advanced() const
{
    return oMapAF_advanced;
}

af::array OrientationMap::getOMapAF_basic() const
{
    return oMapAF_basic;
}

float OrientationMap::getDuration() const
{
    return duration;
}
