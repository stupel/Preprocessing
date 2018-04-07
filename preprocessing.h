#ifndef PREPROCESSING_H
#define PREPROCESSING_H
#define _USE_MATH_DEFINES

#include "preprocessing_global.h"

#include "orientationmap.h"
#include "gaborfiltermultithread.h"
#include "gaborfiltergpu.h"
#include "binarization.h"
#include "contrastenhancement.h"
#include "thinning.h"
#include "frequencymap.h"
#include "mask.h"
#include "qualitymap.h"

// OpenCV
#include "opencv2/opencv.hpp"

// C++
#include <iostream>
#include <exception>

// ArrayFire
#include "arrayfire.h"
#include "af/macros.h"

// Qt
#include <QImage>
#include <QtMath>
#include <QDebug>
#include <QColor>
#include <QPainter>
#include <QThread>
#include <QTime>

typedef struct PREPROCESSING_ALL_RESULTS_ {
    cv::Mat imgContrastEnhanced;
    cv::Mat imgEnhanced; // obrazok po prefiltrovani
    cv::Mat imgOrientationMap; // obrazok smerovej mapy
    cv::Mat imgBinarized;
    cv::Mat imgSkeleton;
    cv::Mat imgSkeletonInverted;
    cv::Mat imgMask;
    cv::Mat imgFrequencyMap;
    cv::Mat imgQualityMap;
    cv::Mat frequencyMap;
    cv::Mat orientationMap;
    cv::Mat qualityMap;
} PREPROCESSING_ALL_RESULTS;

typedef struct PREPROCESSING_RESULTS_ {
    cv::Mat imgSkeleton;
    cv::Mat imgSkeletonInverted;
    cv::Mat imgMask;
    cv::Mat qualityMap;
    cv::Mat frequencyMap;
    cv::Mat orientationMap;
} PREPROCESSING_RESULTS;

//durration in ms
typedef struct PREPROCESSING_DURATIONS_ {
    int contrastEnhancement;
    int gaborFilter;
    int binarization;
    int removingHoles;
    int thinning;
    int mask;
    int orientationMap;
    int qualityMap;
    int frequencyMap;
} PREPROCESSING_DURATIONS;

class PREPROCESSINGSHARED_EXPORT Preprocessing : public QThread
{
    Q_OBJECT

public:
    Preprocessing();

    void loadImg(cv::Mat imgInput);
    void setPreprocessingParams(int blockSize, double gaborLambda, double gaborSigma, int gaussBlockBasic, double gaussSigmaBasic, int gaussBlockAdvanced, double gaussSigmaAdvanced, int holeSize);
    void setFeatures(bool advancedMode, int numThreads, bool useGaborFilterGPU = false, bool useContrastEnhancement = true, bool useRemoveHoles = true, bool useFixOrientations = true, bool useMask = false, bool useQualityMap = true, bool useFrequencyMap = false);
    void setFrequencyMapParams(const CAFFE_FILES &freqFiles, const int &blockSize, const int &exBlockSize);
    void setMaskParams(const CAFFE_FILES &maskFiles, const int &blockSize, const int &exBlockSize, const bool &useSmooth);
    void generateMask();

    void run();

private:
    OrientationMap oMap;
    GaborFilterMultiThread gaborMultiThread; // objekt na paralelne filtrovanie odtlacku
    GaborFilterGPU gaborGPU;
    Binarization binarization;
    Thinning thinning;
    ContrastEnhancement contrast;
    FrequencyMap fMap;
    QualityMap qMap;

    QTime timer;

    cv::Mat imgOriginal;

    PREPROCESSING_ALL_RESULTS results;
    PREPROCESSING_DURATIONS durations;

    bool advancedMode;

    GAUSSIAN_BLUR_SETTINGS gaussBlurBasic, gaussBlurAdvanced;
    double gaborLambda, gaborSigma;
    int blockSize;
    int numThreads;
    int holeSize;

    bool useGaborFilterGPU;
    bool useContrastEnhancement;
    bool useRemoveHoles;
    bool useFrequencyMap;
    bool useMask;
    bool useFixOrientations;
    bool useQualityMap;

    bool firstRun;
    bool imgLoaded;
    bool isFreqNetLoaded;
    bool isMaskNetLoaded;

    //Frequency Map
    CAFFE_FILES freqFiles;
    int freqBlockSize;
    int freqExBlockSize;

    //Mask
    CAFFE_FILES maskFiles;
    int maskBlockSize;
    int maskExBlockSize;
    bool maskUseSmooth;

    void continueAfterGabor();
    int preprocessingError(int errorcode);
    void clean();

private slots:
    void allGaborThreadsFinished();

signals:
    void preprocessingAdvancedDoneSignal(PREPROCESSING_ALL_RESULTS results);
    void preprocessingDoneSignal(PREPROCESSING_RESULTS results);
    void preprocessingDurrationSignal(PREPROCESSING_DURATIONS durations);
    void preprocessingErrorSignal(int errorcode);
};

#endif // PREPROCESSING_H
