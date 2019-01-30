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
	cv::Mat imgOriginal;
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

#ifndef PREPROCESSING_RESULTS_DEFINED
typedef struct preprocessing_results {
	cv::Mat imgOriginal;
	cv::Mat imgSkeleton;
	cv::Mat imgSkeletonInverted;
	cv::Mat qualityMap;
	cv::Mat orientationMap;
} PREPROCESSING_RESULTS;
#define PREPROCESSING_RESULTS_DEFINED
#endif

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

	int loadInput(cv::Mat imgOriginal);
	int loadInput(QVector<cv::Mat> imgOriginals);
	int loadInput(QString inputPath);

	int setPreprocessingParams(int blockSize = 13, double gaborLambda = 9, double gaborSigma = 3, int gaussBlockBasic = 1, double gaussSigmaBasic = 1.0, int gaussBlockAdvanced = 121, double gaussSigmaAdvanced = 10.0, int holeSize = 20);
	int setFeatures(bool useAdvancedMode, bool useContrastEnhancement = true, bool useAdvancedOrientationMap = true, bool useHoleRemover = true, bool generateInvertedSkeleton = true, bool useQualityMap = true, bool useMask = false, bool useFrequencyMap = false);
	int setFrequencyMapParams(CAFFE_FILES freqFiles, int blockSize, int exBlockSize);
	int setMaskParams(CAFFE_FILES maskFiles, int blockSize, int exBlockSize, bool useSmooth);
	int setCPUOnly(bool enabled, int threadNum = 0);

private slots:
	void allGaborThreadsFinished();

signals:
	void preprocessingDoneSignal(PREPROCESSING_ALL_RESULTS m_results);
	void preprocessingDoneSignal(PREPROCESSING_RESULTS m_results);

	void preprocessingSequenceDoneSignal(QMap<QString, PREPROCESSING_ALL_RESULTS> m_results);
	void preprocessingSequenceDoneSignal(QMap<QString, PREPROCESSING_RESULTS> m_results);

	void preprocessingDurationSignal(PREPROCESSING_DURATIONS m_durations);
	void preprocessingProgressSignal(int progress);
	void preprocessingErrorSignal(int errorcode);

private:
	void continueAfterGabor();
	void preprocessingError(int errorcode);
	void cleanResults();
	void cleanInput();
	void cleanDurations();
	void startProcess(const cv::Mat &imgOriginal);

private:
	ContrastEnhancement m_contrast;
	OrientationMap m_oMap;
	QualityMap m_qMap;
	GaborFilterMultiThread m_gaborMultiThread; // objekt na paralelne filtrovanie odtlacku
	GaborFilterGPU m_gaborGPU;
	Binarization m_binarization;
	Thinning m_thinning;
	Mask m_mask;
	FrequencyMap m_fMap;

	QTime m_timer;

	bool m_preprocessingIsRunning;

	// INPUT
	INPUT_PARAMS m_inputParams;

	// PARAMS
	OMAP_PARAMS m_omapParams;
	QMAP_PARAMS m_qmapParams;
	FMAP_PARAMS m_fmapParams;
	MASK_PARAMS m_maskParams;
	CONTRAST_PARAMS m_contrastParams;
	GABOR_PARAMS m_gaborParams;
	BINARIZATION_PARAMS m_binarizationParams;

	PREPROCESSING_FEATURES m_features;
	PREPROCESSING_GENERAL m_general;

	// OUTPUT
	af::array m_orientationMapAF;
	PREPROCESSING_ALL_RESULTS m_results;
	QMap<QString, PREPROCESSING_RESULTS> m_resultsMap;
	QMap<QString, PREPROCESSING_ALL_RESULTS> m_allResultsMap;

	PREPROCESSING_DURATIONS m_durations;
};

#endif // PREPROCESSING_H
