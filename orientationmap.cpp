#include "orientationmap.h"

OrientationMap::OrientationMap(QObject *parent)
	: QObject(parent)
	, m_duration(0)
{

}

void OrientationMap::setParams(const cv::Mat &imgFingerprint, OMAP_PARAMS omapParams)
{
	m_imgInput = imgFingerprint;
	m_omap = omapParams;
}

void OrientationMap::computeBasicMapCPU()
{
	m_timer.start();

	cv::Mat Gx, Gy;
	int height, width;
	float Vx, Vy;
	height = floor(m_imgInput.rows / m_omap.blockSize);
	width = floor(m_imgInput.cols / m_omap.blockSize);

	int paddingX = m_imgInput.cols - width*m_omap.blockSize;
	int paddingY = m_imgInput.rows - height*m_omap.blockSize;

	// BASIC smerova mapa
	m_oMap_basic = cv::Mat(height, width, CV_32F);

	// vypocet gradientov x a y
	cv::Sobel(m_imgInput, Gx, CV_32FC1, 1, 0);
	cv::Sobel(m_imgInput, Gy, CV_32FC1, 0, 1);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			Vx = 0.0; Vy = 0.0;
			for (int i = y * m_omap.blockSize + paddingY / 2; i < y * m_omap.blockSize + paddingY / 2 + m_omap.blockSize; i++){
				for (int j = x * m_omap.blockSize + paddingX / 2; j < x * m_omap.blockSize + paddingX / 2 + m_omap.blockSize; j++){
					Vx += (2 * (Gx.at<float>(i,j) * Gy.at<float>(i, j)));
					Vy += pow(Gx.at<float>(i,j),2) - pow(Gy.at<float>(i, j), 2);
				}
			}
			m_oMap_basic.at<float>(y, x) = 0.5 * atan2(Vx, Vy);
		}
	}

	//vyhladenie smerovej mapy
	cv::Mat sinTheta(m_oMap_basic.size(), CV_32F);
	cv::Mat cosTheta(m_oMap_basic.size(), CV_32F);

	for(int i = 0; i < m_oMap_basic.rows; i++){
		for(int j = 0; j < m_oMap_basic.cols; j++){
			cosTheta.at<float>(i, j) = cos(2 * m_oMap_basic.at<float>(i, j));
			sinTheta.at<float>(i, j) = sin(2 * m_oMap_basic.at<float>(i, j));
		}
	}

	cv::GaussianBlur(cosTheta, cosTheta, cv::Size(m_omap.gaussBlurBasic.blockSize, m_omap.gaussBlurBasic.blockSize), m_omap.gaussBlurBasic.sigma,m_omap.gaussBlurBasic.sigma);
	cv::GaussianBlur(sinTheta, sinTheta, cv::Size(m_omap.gaussBlurBasic.blockSize, m_omap.gaussBlurBasic.blockSize), m_omap.gaussBlurBasic.sigma,m_omap.gaussBlurBasic.sigma);
	for(int i = 0; i < m_oMap_basic.rows;i++) {
		for(int j = 0; j < m_oMap_basic.cols; j++) {
			m_oMap_basic.at<float>(i, j) = 0.5 * atan2(sinTheta.at<float>(i, j), cosTheta.at<float>(i, j));
		}
	}

	m_duration = m_timer.elapsed();
}

void OrientationMap::computeBasicMapGPU()
{
	af::timer::start();

	m_imgInputAF = Helper::mat_uchar2array_uchar(m_imgInput);

	af::array Gx, Gy;
	int height, width;
	af::array Vx, Vy;
	height = floor(m_imgInputAF.dims(0) / m_omap.blockSize);
	width = floor(m_imgInputAF.dims(1) / m_omap.blockSize);
	int paddingX = m_imgInputAF.dims(1) - width * m_omap.blockSize;
	int paddingY = m_imgInputAF.dims(0) - height * m_omap.blockSize;

	// vypocet gradientov x a y
	af::sobel(Gx, Gy, m_imgInputAF);

	// vypocet Vx,Vy a Theta
	af::array GxCut = Gx(af::seq(paddingY / 2, height * m_omap.blockSize + paddingY / 2 - 1), af::seq(paddingX / 2, width * m_omap.blockSize+paddingX / 2 - 1));
	af::array GyCut = Gy(af::seq(paddingY / 2, height * m_omap.blockSize + paddingY / 2 - 1), af::seq(paddingX / 2, width * m_omap.blockSize + paddingX / 2 - 1));

	GxCut = af::unwrap(GxCut, m_omap.blockSize, m_omap.blockSize, m_omap.blockSize, m_omap.blockSize);
	GyCut = af::unwrap(GyCut, m_omap.blockSize, m_omap.blockSize, m_omap.blockSize, m_omap.blockSize);

	Vx =  af::sum(2 * GxCut * GyCut);
	Vy =  af::sum(af::pow(GxCut, 2) - af::pow(GyCut, 2));
	m_oMapAF_basic = 0.5* af::atan2(Vx.as(f32), Vy.as(f32));

	m_oMapAF_basic = af::moddims(m_oMapAF_basic, height, width);

	// vyhladenie smerovej mapy
	af::array sinTheta = af::sin(2 * m_oMapAF_basic);
	af::array cosTheta = af::cos(2 * m_oMapAF_basic);

	af::array gk = af::gaussianKernel(m_omap.gaussBlurBasic.blockSize, m_omap.gaussBlurBasic.blockSize, m_omap.gaussBlurBasic.sigma, m_omap.gaussBlurBasic.sigma);

	sinTheta = af::convolve(sinTheta, gk);
	cosTheta = af::convolve(cosTheta, gk);

	m_oMapAF_basic = 0.5* af::atan2(sinTheta, cosTheta);

	m_oMap_basic = Helper::array_float2mat_float(m_oMapAF_basic);

	m_duration = af::timer::stop() * 1000;
}

void OrientationMap::computeAdvancedMapCPU()
{
	computeBasicMapCPU();

	m_timer.start();

	// expanzia smerovej mapy
	m_oMap_advanced = cv::Mat(m_imgInput.rows, m_imgInput.cols, m_oMap_basic.type());
	cv::Mat blk;

	for(int i = 0; i < m_oMap_basic.rows; i++) {
		for(int j = 0; j < m_oMap_basic.cols; j++) {
			blk = m_oMap_advanced.rowRange(i * m_omap.blockSize, i * m_omap.blockSize + m_omap.blockSize).colRange(j * m_omap.blockSize, j * m_omap.blockSize + m_omap.blockSize);
			blk.setTo(cv::Scalar(m_oMap_basic.at<float>(i, j)));
		}
	}

	// vyhladenie expandovanej smerovej mapy
	cv::Mat sinTheta_Advanced(m_oMap_advanced.size(), CV_32F);
	cv::Mat cosTheta_Advanced(m_oMap_advanced.size(), CV_32F);

	for(int i = 0; i < m_oMap_advanced.rows; i++) {
		for(int j = 0; j < m_oMap_advanced.cols; j++) {
			cosTheta_Advanced.at<float>(i, j) = cos(2 * m_oMap_advanced.at<float>(i, j));
			sinTheta_Advanced.at<float>(i, j) = sin(2 * m_oMap_advanced.at<float>(i, j));
		}
	}

	cv::GaussianBlur(cosTheta_Advanced, cosTheta_Advanced, cv::Size(m_omap.gaussBlurAdvanced.blockSize, m_omap.gaussBlurAdvanced.blockSize), m_omap.gaussBlurAdvanced.sigma, m_omap.gaussBlurAdvanced.sigma);
	cv::GaussianBlur(sinTheta_Advanced, sinTheta_Advanced, cv::Size(m_omap.gaussBlurAdvanced.blockSize, m_omap.gaussBlurAdvanced.blockSize), m_omap.gaussBlurAdvanced.sigma, m_omap.gaussBlurAdvanced.sigma);
	for(int i = 0; i < m_oMap_advanced.rows; i++) {
		for(int j = 0; j < m_oMap_advanced.cols; j++) {
			m_oMap_advanced.at<float>(i, j) = 0.5 * atan2(sinTheta_Advanced.at<float>(i, j), cosTheta_Advanced.at<float>(i, j));
		}
	}

	m_duration += m_timer.elapsed();
}

void OrientationMap::computeAdvancedMapGPU()
{
	af::timer::start();

	// compute the basic O-Map first
	computeBasicMapGPU();

	// basic O-Map expansion
	m_oMapAF_advanced = af::moddims(m_oMapAF_basic, 1, m_oMapAF_basic.dims(0) * m_oMapAF_basic.dims(1));
	m_oMapAF_advanced = af::tile(m_oMapAF_advanced, m_omap.blockSize * m_omap.blockSize);
	m_oMapAF_advanced = af::wrap(m_oMapAF_advanced,
									 m_oMapAF_basic.dims(0) * m_omap.blockSize,
									 m_oMapAF_basic.dims(1) * m_omap.blockSize,
									 m_omap.blockSize,
									 m_omap.blockSize,
									 m_omap.blockSize,
									 m_omap.blockSize);

	// smoothing the expanded O-Map
	af::array sinTheta = af::sin(2 * m_oMapAF_advanced);
	af::array cosTheta = af::cos(2 * m_oMapAF_advanced);
	af::array gk = af::gaussianKernel(m_omap.gaussBlurAdvanced.blockSize,
									  m_omap.gaussBlurAdvanced.blockSize,
									  m_omap.gaussBlurAdvanced.sigma,
									  m_omap.gaussBlurAdvanced.sigma);

	sinTheta = af::convolve(sinTheta, gk);
	cosTheta = af::convolve(cosTheta, gk);

	m_oMapAF_advanced = 0.5* af::atan2(sinTheta, cosTheta);

	m_oMap_advanced = Helper::array_float2mat_float(m_oMapAF_advanced);

	m_duration +=  af::timer::stop();
}

void OrientationMap::drawBasicMap(const cv::Mat &imgOriginal)
{
	// farebny obrazok smerovej mapy po vyhladeni
	m_imgOMap_basic = cv::Mat(imgOriginal.rows, imgOriginal.cols, CV_8UC3);
	cv::cvtColor(imgOriginal, m_imgOMap_basic, cv::COLOR_GRAY2RGB);

	int height = floor(m_imgInput.rows / m_omap.blockSize);
	int width = floor(m_imgInput.cols / m_omap.blockSize);
	int paddingX = m_imgInput.cols - width * m_omap.blockSize;
	int paddingY = m_imgInput.rows - height * m_omap.blockSize;
	int rowsMat = m_oMap_basic.rows;
	int colsMat = m_oMap_basic.cols;
	float row1, col1, row2, col2, row3, col3, direction;

	for (int y = 0; y<rowsMat; y++){
		for(int x =0; x<colsMat; x++){
			direction = m_oMap_basic.at<float>(y,x) + CV_PI / 2;
			row1 = y * m_omap.blockSize + m_omap.blockSize / 2 + paddingY / 2;
			col1 = x * m_omap.blockSize + m_omap.blockSize / 2 + paddingX / 2;
			row2 = row1 - sin(direction) * m_omap.blockSize / 2;
			col2 = col1 - cos(direction) * m_omap.blockSize / 2;
			row3 = row1 + sin(direction) * m_omap.blockSize / 2;
			col3 = col1 + cos(direction) * m_omap.blockSize / 2;
			cv::Point endPoint(col2, row2);
			cv::Point endPoint2(col3, row3);
			cv::line(m_imgOMap_basic, endPoint, endPoint2, cv::Scalar(255,255,0), 1, 4, 0);
		}
	}
}

cv::Mat OrientationMap::getImgOMap_basic() const
{
	return m_imgOMap_basic;
}

cv::Mat OrientationMap::getOMap_basic() const
{
	return m_oMap_basic;
}

cv::Mat OrientationMap::getOMap_advanced() const
{
	return m_oMap_advanced;
}

af::array OrientationMap::getOMapAF_advanced() const
{
	return m_oMapAF_advanced;
}

af::array OrientationMap::getOMapAF_basic() const
{
	return m_oMapAF_basic;
}

float OrientationMap::getDuration() const
{
	return m_duration;
}
