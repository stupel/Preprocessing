#ifndef ORIENTATIONMAP_H
#define ORIENTATIONMAP_H

#include "preprocessing_config.h"

class OrientationMap : public QObject
{
	Q_OBJECT

public:
	explicit OrientationMap(QObject *parent = nullptr);

	void setParams(const cv::Mat &m_imgInput, OMAP_PARAMS m_omap);
	void computeBasicMapCPU();
	void computeBasicMapGPU();
	void computeAdvancedMapCPU();
	void computeAdvancedMapGPU();
	void drawBasicMap(const cv::Mat &imgOriginal);

	cv::Mat getOMap_advanced() const;
	cv::Mat getOMap_basic() const;
	cv::Mat getImgOMap_basic() const;
	float getDuration() const;
	af::array getOMapAF_basic() const;
	af::array getOMapAF_advanced() const;

private:
	QTime m_timer;

	cv::Mat m_imgInput; // obrazok odtlacku
	af::array m_imgInputAF;

	OMAP_PARAMS m_omap;

	cv::Mat m_oMap_basic; // BASIC smerova mapa (vyhladena, jeden smer pre cely blok)
	af::array m_oMapAF_basic;
	cv::Mat m_oMap_advanced; // ADVANCED smerova mapa (vyhladena, kazdy pixel ma svoj smer)
	af::array m_oMapAF_advanced;

	cv::Mat m_imgOMap_basic; // obrazok BASIC smerovej mapy (ADVANCED mapa sa neda zobrazit, lebo kazdy pixel ma iny smer)

	float m_duration;

};

#endif // ORIENTATIONMAP_H
