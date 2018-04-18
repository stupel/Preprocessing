#ifndef PREPROCESSING_H
#define PREPROCESSING_H
#define _USE_MATH_DEFINES

#include "preprocessing_global.h"
#include "preprocessing_config.h"

#include "orientationmap.h"
#include "gaborfiltermultithread.h"
#include "gaborfiltergpu.h"
#include "binarization.h"
#include "contrastenhancement.h"
#include "thinning.h"
#include "frequencymap.h"
#include "mask.h"
#include "qualitymap.h"

typedef struct preprocessing_all_results {
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

typedef struct preprocessing_results {
    cv::Mat imgSkeleton;
    cv::Mat imgSkeletonInverted;
    cv::Mat qualityMap;
    cv::Mat orientationMap;
} PREPROCESSING_RESULTS;

//duration in ms
typedef struct preprocessing_durations {
    int contrastEnhancement;
    float orientationMap;
    float gaborFilter;
    int binarization;
    int removingHoles;
    int thinning;
    int mask;
    int qualityMap;
    int frequencyMap;
} PREPROCESSING_DURATIONS;

class PREPROCESSINGSHARED_EXPORT Preprocessing : public QObject
{
    Q_OBJECT

public:
    Preprocessing();
    void start();

    void loadImg(cv::Mat imgOriginal);
    void setPreprocessingParams(int blockSize = 13, double gaborLambda = 9, double gaborSigma = 3, int gaussBlockBasic = 1, double gaussSigmaBasic = 1.0, int gaussBlockAdvanced = 121, double gaussSigmaAdvanced = 10.0, int holeSize = 20);
    void setFeatures(bool useAdvancedMode, bool useContrastEnhancement = true, bool useHoleRemover = true, bool useOrientationFixer = true, bool useQualityMap = true, bool useMask = false, bool useFrequencyMap = false);
    void setFrequencyMapParams(CAFFE_FILES freqFiles, int blockSize, int exBlockSize);
    void setMaskParams(CAFFE_FILES maskFiles, int blockSize, int exBlockSize, bool useSmooth);
    void setCPUOnly(bool enabled, int numThreads = 0);

    PREPROCESSING_ALL_RESULTS getResults() const;
    PREPROCESSING_DURATIONS getDurations() const;

private:
    ContrastEnhancement contrast;
    OrientationMap oMap;
    QualityMap qMap;
    GaborFilterMultiThread gaborMultiThread; // objekt na paralelne filtrovanie odtlacku
    GaborFilterGPU gaborGPU;
    Binarization binarization;
    Thinning thinning;
    Mask mask;
    FrequencyMap fMap;

    QTime timer;

    cv::Mat imgOriginal;
    af::array advancedOMap;

    PREPROCESSING_ALL_RESULTS results;
    PREPROCESSING_DURATIONS durations;

    bool advancedMode;

    GAUSSIAN_BLUR_SETTINGS gaussBlurBasic, gaussBlurAdvanced;
    double gaborLambda, gaborSigma;
    int blockSize;
    int numThreads;
    int holeSize;

    bool useContrastEnhancement;
    bool useHoleRemover;
    bool useFrequencyMap;
    bool useMask;
    bool useOrientationFixer;
    bool useQualityMap;

    bool cpuOnly;

    bool firstRun;
    bool imgLoaded;

    // Frequency Map
    CAFFE_FILES freqFiles;
    int freqBlockSize;
    int freqExBlockSize;
    bool isFrequencyModelLoaded;

    // Mask
    CAFFE_FILES maskFiles;
    int maskBlockSize;
    int maskExBlockSize;
    bool maskUseSmooth;
    bool isMaskModelLoaded;

    // PRIVATE FUNCTIONS
    void continueAfterGabor();
    int preprocessingError(int errorcode);
    void clean();

private slots:
    void allGaborThreadsFinished();

signals:
    void preprocessingAdvancedDoneSignal(PREPROCESSING_ALL_RESULTS results);
    void preprocessingDoneSignal(PREPROCESSING_RESULTS results);
    void preprocessingDurationSignal(PREPROCESSING_DURATIONS durations);
    void preprocessingErrorSignal(int errorcode);
};

#endif // PREPROCESSING_H
