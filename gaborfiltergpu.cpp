#include "gaborfiltergpu.h"

GaborFilterGPU::GaborFilterGPU()
{
	m_duration = 0;
}

void GaborFilterGPU::setParams(const cv::Mat &imgInput, const GABOR_PARAMS &gaborParams)
{
	m_imgInput = Helper::mat_uchar2array_float(imgInput);

	m_gabor = gaborParams;
	m_oMap = *gaborParams.oMapAF;

	if (*gaborParams.useFrequencyMap) m_fMap = Helper::mat_float2array_float(*gaborParams.fMap);
}


af::array GaborFilterGPU::getGaborKernel(const af::array& oMapPixel)
{
	af::array xarr = af::tile(af::flip(af::array(af::seq(- m_gabor.blockSize / 2, m_gabor.blockSize / 2), m_gabor.blockSize, 1), 0), 1, m_gabor.blockSize);
	af::array yarr = af::tile(af::array(af::seq(- m_gabor.blockSize / 2, m_gabor.blockSize / 2), 1, m_gabor.blockSize), m_gabor.blockSize, 1);
	af::array thetaarr = af::tile(oMapPixel + M_PI_2, m_gabor.blockSize, m_gabor.blockSize);
	af::array xciara = xarr * af::cos(thetaarr) + yarr * af::sin(thetaarr);
	af::array yciara = -xarr * af::sin(thetaarr) + yarr * af::cos(thetaarr);
	af::array nom1;
	float denom1;
	nom1 = xciara * xciara + m_gabor.gamma * m_gabor.gamma * yciara * yciara;
	denom1 = 2 * m_gabor.sigma * m_gabor.sigma;
	af::array exp1 = af::exp(-nom1 / denom1);
	af::array cos1 = af::cos(2 * af::Pi * (xciara / m_gabor.lambda) + m_gabor.psi);

	return exp1 * cos1;
}

void GaborFilterGPU::enhanceWithBasicOMap()
{
	// TIMER
	af::timer::start();

	int width = floor(m_imgInput.dims(1) / m_gabor.blockSize);
	int height = floor(m_imgInput.dims(0) / m_gabor.blockSize);
	int paddingWidth = m_imgInput.dims(1) - width * m_gabor.blockSize;
	int paddingHeight = m_imgInput.dims(0) - height * m_gabor.blockSize;

	int origHeight = m_imgInput.dims(0);
	int origWidth = m_imgInput.dims(1);

	m_imgInput = m_imgInput(af::seq(paddingHeight/2, height*m_gabor.blockSize+paddingHeight/2-1), af::seq(paddingWidth/2, width*m_gabor.blockSize+paddingWidth/2-1));

	int cutHeight = m_imgInput.dims(0);
	int cutWidth = m_imgInput.dims(1);

	int tx = m_oMap.dims(1) * m_gabor.blockSize;
	int ty = m_oMap.dims(0) * m_gabor.blockSize;
	int ttx = m_oMap.dims(1);
	int tty = m_oMap.dims(0);

	m_imgInput = af::unwrap(m_imgInput , m_gabor.blockSize, m_gabor.blockSize, 1, 1, m_gabor.blockSize/2, m_gabor.blockSize/2);

	m_oMap = af::flat(m_oMap);

	af::array kernels(m_gabor.blockSize, m_gabor.blockSize, m_oMap.elements());
	gfor(af::seq i, m_oMap.elements()){
		kernels(af::span, af::span, i) = getGaborKernel(m_oMap(i));
	}

	kernels = af::moddims(kernels, m_gabor.blockSize * m_gabor.blockSize, m_oMap.elements()); // 3D kernely do 2D unwrapovanych kernelov
	kernels = af::tile(kernels, m_gabor.blockSize);
	kernels = af::moddims(kernels, m_gabor.blockSize * m_gabor.blockSize, m_gabor.blockSize * tty, 1, ttx);

	m_imgInput = af::moddims(m_imgInput, m_gabor.blockSize * m_gabor.blockSize, m_gabor.blockSize * m_gabor.blockSize*tty, 1, ttx);
	m_imgInput = af::moddims(m_imgInput, m_gabor.blockSize * m_gabor.blockSize, m_gabor.blockSize*tty, m_gabor.blockSize, ttx);
	m_imgInput = m_imgInput *  af::tile(kernels, 1, 1, m_gabor.blockSize);
	m_imgInput = af::moddims(m_imgInput, m_gabor.blockSize * m_gabor.blockSize, m_gabor.blockSize * m_gabor.blockSize * tty * ttx);

	m_imgInput = af::sum(m_imgInput.T(), 1);

	// 6. PIXEL REORGANIZATION
	af::array output = af::moddims(m_imgInput, ty, tx);

	// Create and resize Mat to the original size
	Helper::af_normalizeImage(output);
	m_imgEnhanced = cv::Mat(origHeight, origWidth, CV_8UC1);
	Helper::array_uchar2mat_uchar(output).copyTo(m_imgEnhanced(cv::Rect(paddingWidth/2, paddingHeight/2, cutWidth, cutHeight)));

	// Resize the orientation map to the original size
	cv::Mat oMapCV(origHeight, origWidth, CV_32FC1);
	m_gabor.oMap->copyTo(oMapCV(cv::Rect(paddingWidth/2, paddingHeight/2, cutWidth, cutHeight)));
	*m_gabor.oMap = oMapCV.clone();

	m_duration = af::timer::stop() * 1000; // s to ms
}

void GaborFilterGPU::enhanceWithAdvancedOMap(){
	// TIMER
	af::timer::start();

	int width = floor(m_imgInput.dims(1) / m_gabor.blockSize);
	int height = floor(m_imgInput.dims(0) / m_gabor.blockSize);
	int paddingWidth = m_imgInput.dims(1) - width * m_gabor.blockSize;
	int paddingHeight = m_imgInput.dims(0) - height * m_gabor.blockSize;

	int origHeight = m_imgInput.dims(0);
	int origWidth = m_imgInput.dims(1);

	m_imgInput = m_imgInput(af::seq(paddingHeight/2, height*m_gabor.blockSize+paddingHeight/2-1), af::seq(paddingWidth/2, width*m_gabor.blockSize+paddingWidth/2-1));

	int cutHeight = m_imgInput.dims(0);
	int cutWidth = m_imgInput.dims(1);

	m_imgInput = af::unwrap(m_imgInput , m_gabor.blockSize, m_gabor.blockSize, 1, 1, m_gabor.blockSize/2, m_gabor.blockSize/2);

	m_oMap = af::moddims(m_oMap, 1, m_oMap.dims(0) * m_oMap.dims(1));

	af::array kernels(m_gabor.blockSize, m_gabor.blockSize, m_oMap.elements());
	gfor(af::seq i, m_oMap.elements()){
		kernels(af::span, af::span, i) = getGaborKernel(m_oMap(i));
	}

	kernels = af::moddims(kernels, m_imgInput.dims(0), m_imgInput.dims(1), m_imgInput.dims(2));

	af::array output = kernels * m_imgInput;
	output = af::sum(output);
	output = af::moddims(output,cutHeight, cutWidth );

	// Create and resize Mat to the original size
	Helper::af_normalizeImage(output);
	m_imgEnhanced = cv::Mat(origHeight, origWidth, CV_8UC1);
	Helper::array_uchar2mat_uchar(output).copyTo(m_imgEnhanced(cv::Rect(paddingWidth/2, paddingHeight/2, cutWidth, cutHeight)));

	// Resize the orientation map to the original size
	cv::Mat oMapCV(origHeight, origWidth, CV_32FC1);
	m_gabor.oMap->copyTo(oMapCV(cv::Rect(paddingWidth/2, paddingHeight/2, cutWidth, cutHeight)));
	*m_gabor.oMap = oMapCV.clone();

	m_duration = af::timer::stop() * 1000; // s to ms
}

cv::Mat GaborFilterGPU::getImgEnhanced() const
{
	return m_imgEnhanced;
}

float GaborFilterGPU::getDuration() const
{
	return m_duration;
}
