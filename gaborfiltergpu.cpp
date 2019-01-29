#include "gaborfiltergpu.h"

GaborFilterGPU::GaborFilterGPU()
{
	this->duration = 0;
}

void GaborFilterGPU::setParams(const cv::Mat &imgInput, const GABOR_PARAMS &gaborParams)
{
	this->imgInput = Helper::mat_uchar2array_float(imgInput);

	this->gabor = gaborParams;
	this->oMap = *gaborParams.oMapAF;

	if (*gaborParams.useFrequencyMap) this->fMap = Helper::mat_float2array_float(*gaborParams.fMap);
}


af::array GaborFilterGPU::getGaborKernel(const af::array& oMapPixel)
{
	af::array xarr = af::tile(af::flip(af::array(af::seq(- this->gabor.blockSize / 2, this->gabor.blockSize / 2), this->gabor.blockSize, 1), 0), 1, this->gabor.blockSize);
	af::array yarr = af::tile(af::array(af::seq(- this->gabor.blockSize / 2, this->gabor.blockSize / 2), 1, this->gabor.blockSize), this->gabor.blockSize, 1);
	af::array thetaarr = af::tile(oMapPixel + M_PI_2, this->gabor.blockSize, this->gabor.blockSize);
	af::array xciara = xarr * af::cos(thetaarr) + yarr * af::sin(thetaarr);
	af::array yciara = -xarr * af::sin(thetaarr) + yarr * af::cos(thetaarr);
	af::array nom1;
	float denom1;
	nom1 = xciara * xciara + this->gabor.gamma * this->gabor.gamma * yciara * yciara;
	denom1 = 2 * this->gabor.sigma * this->gabor.sigma;
	af::array exp1 = af::exp(-nom1 / denom1);
	af::array cos1 = af::cos(2 * af::Pi * (xciara / this->gabor.lambda) + this->gabor.psi);

	return exp1 * cos1;
}

void GaborFilterGPU::enhanceWithBasicOMap()
{
	// TIMER
	af::timer::start();

	int width = floor(this->imgInput.dims(1) / this->gabor.blockSize);
	int height = floor(this->imgInput.dims(0) / this->gabor.blockSize);
	int paddingWidth = this->imgInput.dims(1) - width * this->gabor.blockSize;
	int paddingHeight = this->imgInput.dims(0) - height * this->gabor.blockSize;

	int origHeight = this->imgInput.dims(0);
	int origWidth = this->imgInput.dims(1);

	this->imgInput = this->imgInput(af::seq(paddingHeight/2, height*this->gabor.blockSize+paddingHeight/2-1), af::seq(paddingWidth/2, width*this->gabor.blockSize+paddingWidth/2-1));

	int cutHeight = this->imgInput.dims(0);
	int cutWidth = this->imgInput.dims(1);

	int tx = this->oMap.dims(1) * this->gabor.blockSize;
	int ty = this->oMap.dims(0) * this->gabor.blockSize;
	int ttx = this->oMap.dims(1);
	int tty = this->oMap.dims(0);

	this->imgInput = af::unwrap(this->imgInput , this->gabor.blockSize, this->gabor.blockSize, 1, 1, this->gabor.blockSize/2, this->gabor.blockSize/2);

	this->oMap = af::flat(this->oMap);

	af::array kernels(this->gabor.blockSize, this->gabor.blockSize, this->oMap.elements());
	gfor(af::seq i, this->oMap.elements()){
		kernels(af::span, af::span, i) = this->getGaborKernel(this->oMap(i));
	}

	kernels = af::moddims(kernels, this->gabor.blockSize * this->gabor.blockSize, this->oMap.elements()); // 3D kernely do 2D unwrapovanych kernelov
	kernels = af::tile(kernels, this->gabor.blockSize);
	kernels = af::moddims(kernels, this->gabor.blockSize * this->gabor.blockSize, this->gabor.blockSize * tty, 1, ttx);

	this->imgInput = af::moddims(this->imgInput, this->gabor.blockSize * this->gabor.blockSize, this->gabor.blockSize * this->gabor.blockSize*tty, 1, ttx);
	this->imgInput = af::moddims(this->imgInput, this->gabor.blockSize * this->gabor.blockSize, this->gabor.blockSize*tty, this->gabor.blockSize, ttx);
	this->imgInput = this->imgInput *  af::tile(kernels, 1, 1, this->gabor.blockSize);
	this->imgInput = af::moddims(this->imgInput, this->gabor.blockSize * this->gabor.blockSize, this->gabor.blockSize * this->gabor.blockSize * tty * ttx);

	this->imgInput = af::sum(this->imgInput.T(), 1);

	// 6. PIXEL REORGANIZATION
	af::array output = af::moddims(this->imgInput, ty, tx);

	// Create and resize Mat to the original size
	Helper::af_normalizeImage(output);
	this->imgEnhanced = cv::Mat(origHeight, origWidth, CV_8UC1);
	Helper::array_uchar2mat_uchar(output).copyTo(this->imgEnhanced(cv::Rect(paddingWidth/2, paddingHeight/2, cutWidth, cutHeight)));

	// Resize the orientation map to the original size
	cv::Mat oMapCV(origHeight, origWidth, CV_32FC1);
	this->gabor.oMap->copyTo(oMapCV(cv::Rect(paddingWidth/2, paddingHeight/2, cutWidth, cutHeight)));
	*this->gabor.oMap = oMapCV.clone();

	this->duration = af::timer::stop() * 1000; // s to ms
}

void GaborFilterGPU::enhanceWithAdvancedOMap(){
	// TIMER
	af::timer::start();

	int width = floor(this->imgInput.dims(1) / this->gabor.blockSize);
	int height = floor(this->imgInput.dims(0) / this->gabor.blockSize);
	int paddingWidth = this->imgInput.dims(1) - width * this->gabor.blockSize;
	int paddingHeight = this->imgInput.dims(0) - height * this->gabor.blockSize;

	int origHeight = this->imgInput.dims(0);
	int origWidth = this->imgInput.dims(1);

	this->imgInput = this->imgInput(af::seq(paddingHeight/2, height*this->gabor.blockSize+paddingHeight/2-1), af::seq(paddingWidth/2, width*this->gabor.blockSize+paddingWidth/2-1));

	int cutHeight = this->imgInput.dims(0);
	int cutWidth = this->imgInput.dims(1);

	this->imgInput = af::unwrap(this->imgInput , this->gabor.blockSize, this->gabor.blockSize, 1, 1, this->gabor.blockSize/2, this->gabor.blockSize/2);

	this->oMap = af::moddims(this->oMap, 1, this->oMap.dims(0) * this->oMap.dims(1));

	af::array kernels(this->gabor.blockSize, this->gabor.blockSize, this->oMap.elements());
	gfor(af::seq i, this->oMap.elements()){
		kernels(af::span, af::span, i) = this->getGaborKernel(this->oMap(i));
	}

	kernels = af::moddims(kernels, this->imgInput.dims(0), this->imgInput.dims(1), this->imgInput.dims(2));

	af::array output = kernels * this->imgInput;
	output = af::sum(output);
	output = af::moddims(output,cutHeight, cutWidth );

	// Create and resize Mat to the original size
	Helper::af_normalizeImage(output);
	this->imgEnhanced = cv::Mat(origHeight, origWidth, CV_8UC1);
	Helper::array_uchar2mat_uchar(output).copyTo(this->imgEnhanced(cv::Rect(paddingWidth/2, paddingHeight/2, cutWidth, cutHeight)));

	// Resize the orientation map to the original size
	cv::Mat oMapCV(origHeight, origWidth, CV_32FC1);
	this->gabor.oMap->copyTo(oMapCV(cv::Rect(paddingWidth/2, paddingHeight/2, cutWidth, cutHeight)));
	*this->gabor.oMap = oMapCV.clone();

	this->duration = af::timer::stop() * 1000; // s to ms
}

cv::Mat GaborFilterGPU::getImgEnhanced() const
{
	return imgEnhanced;
}

float GaborFilterGPU::getDuration() const
{
	return duration;
}
