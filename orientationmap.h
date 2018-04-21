#ifndef ORIENTATIONMAP_H
#define ORIENTATIONMAP_H

#include "preprocessing_config.h"

class OrientationMap : public QObject
{
    Q_OBJECT

public:
    explicit OrientationMap(QObject *parent = nullptr);

    void setParams(const cv::Mat &imgInput, OMAP_PARAMS omap);
    void computeBasicMapCPU();
    void computeBasicMapGPU();
    void computeAdvancedMapCPU();
    void computeAdvancedMapGPU();
    void drawBasicMap(const cv::Mat &imgOriginal);

    //getNset
    cv::Mat getOMap_advanced() const;
    cv::Mat getOMap_basic() const;
    cv::Mat getImgOMap_basic() const;
    float getDuration() const;
    af::array getOMapAF_basic() const;
    af::array getOMapAF_advanced() const;


private:
    QTime timer;

    cv::Mat imgInput; // obrazok odtlacku
    af::array imgInputAF;

    OMAP_PARAMS omap;

    cv::Mat oMap_basic; // BASIC smerova mapa (vyhladena, jeden smer pre cely blok)
    af::array oMapAF_basic;
    cv::Mat oMap_advanced; // ADVANCED smerova mapa (vyhladena, kazdy pixel ma svoj smer)
    af::array oMapAF_advanced;

    cv::Mat imgOMap_basic; // obrazok BASIC smerovej mapy (ADVANCED mapa sa neda zobrazit, lebo kazdy pixel ma iny smer)

    float duration;

};

#endif // ORIENTATIONMAP_H
