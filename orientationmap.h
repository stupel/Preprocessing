#ifndef ORIENTATIONMAP_H
#define ORIENTATIONMAP_H

#include "preprocessing_config.h"

// nastavenia pre funkciu GaussianBlur, ktora sa pouziva na vyhladenie smerovej mapy
typedef struct GaussianBlurSettings {
    int blockSize;  // velkost bloku pre vyhladenie smerovej mapy (cez bloky)
    double sigma;   // sigma pre vyhladenie smerovej mapy
} GAUSSIAN_BLUR_SETTINGS;

class OrientationMap : public QObject
{
    Q_OBJECT

public:
    explicit OrientationMap(QObject *parent = nullptr);

    void setParams(const cv::Mat &imgFingerprint_, int &blockSize_, GAUSSIAN_BLUR_SETTINGS &gaussBlurBasic_, GAUSSIAN_BLUR_SETTINGS &gaussBlurAdvanced_);
    void computeAdvancedMapCPU();
    void computeAdvancedMapGPU();
    void drawBasicMap(const cv::Mat &imgOriginal);

    //getNset
    cv::Mat getOMap_advanced() const;
    cv::Mat getOMap_basic() const;
    cv::Mat getImgOMap_basic() const;
    float getDuration() const;

private:

    void computeBasicMapCPU();
    void computeBasicMapGPU();

    QTime timer;

    cv::Mat imgFingerprint; // obrazok odtlacku
    af::array imgFingerprintAF;

    int blockSize; // velkost bloku pre smerovu mapu
    GAUSSIAN_BLUR_SETTINGS gaussBlurBasic, gaussBlurAdvanced;

    cv::Mat oMap_basic; // BASIC smerova mapa (vyhladena, jeden smer pre cely blok)
    af::array oMapAF_basic;
    cv::Mat oMap_advanced; // ADVANCED smerova mapa (vyhladena, kazdy pixel ma svoj smer)
    af::array oMapAF_advanced;

    cv::Mat imgOMap_basic; // obrazok BASIC smerovej mapy (ADVANCED mapa sa neda zobrazit, lebo kazdy pixel ma iny smer)

    float duration;

};

#endif // ORIENTATIONMAP_H
