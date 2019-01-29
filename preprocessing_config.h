#ifndef PREPROCESSING_CONFIG_H
#define PREPROCESSING_CONFIG_H
#define GLOG_NO_ABBREVIATED_SEVERITIES

//Qt
#include <QObject>
#include <QDebug>
#include <QMetaType>
#include <QtMath>
#include <QThread>
#include <QVector>
#include <QTime>
#include <QString>
#include <QColor>
#include <QPainter>
#include <QDir>
#include <QProcess>
#include <QFileInfo>

//std
#include <iostream>
#include <stdio.h>
#include <vector>
#include <numeric>
#include <fstream>
#include <exception>
#include <algorithm>
#include <math.h>
#include <deque>

// ArrayFire
#include "arrayfire.h"
#include "af/macros.h"

//OpenCV
#include "opencv2/opencv.hpp"

//Helper
#include "helper.h"

#ifndef CAFFE_FILES_DEFINED
typedef struct caffe_files {
	QString model;
	QString trained;
	QString imageMean;
	QString label;
} CAFFE_FILES;
#define CAFFE_FILES_DEFINED
#endif

enum INPUTMODE {image, images, imagePath, imageDirectory};

typedef struct input_params {
	INPUTMODE mode;
	cv::Mat imgOriginal;
	QVector<cv::Mat> imgOriginals;
	QString path;
	QVector<QString> imgNames;
	int cnt;
	bool inputLoaded;
} INPUT_PARAMS;

// nastavenia pre funkciu GaussianBlur, ktora sa pouziva na vyhladenie smerovej mapy
typedef struct gaussian_blur_settings {
	int blockSize;  // velkost bloku pre vyhladenie smerovej mapy (cez bloky)
	float sigma;   // sigma pre vyhladenie smerovej mapy
} GAUSSIAN_BLUR_SETTINGS;

typedef struct omap_params {
	int blockSize;
	GAUSSIAN_BLUR_SETTINGS gaussBlurBasic;
	GAUSSIAN_BLUR_SETTINGS gaussBlurAdvanced;
} OMAP_PARAMS;

typedef struct qmap_params {
	int ppi;
} QMAP_PARAMS;

typedef struct fmap_params {
	int blockSize;
	int exBlockSize;
	bool isModelLoaded;
	bool *cpuOnly;
	CAFFE_FILES caffeFiles;
} FMAP_PARAMS;

typedef struct mask_params {
	int blockSize;
	int exBlockSize;
	bool useSmooth;
	bool isModelLoaded;
	bool *cpuOnly;
	CAFFE_FILES caffeFiles;
} MASK_PARAMS;

typedef struct contrast_params {
	int distance;
	double sigma;
	double gaussBlock;
	double gaussSigma;
} CONTRAST_PARAMS;

typedef struct gabor_params {
	int blockSize;
	float sigma;
	float lambda;
	float gamma;
	float psi;
	cv::Mat *oMap;
	af::array *oMapAF;
	int *threadNum;
	bool *useFrequencyMap;
	cv::Mat *fMap;
} GABOR_PARAMS;

typedef struct binarization_params {
	bool *useQualityMap;
	cv::Mat *imgQualityMap;
	bool *useMask;
	cv::Mat *imgMask;
	int holeSize;
} BINARIZATION_PARAMS;

typedef struct preprocessing_features {
	bool advancedMode;
	bool useContrastEnhancement;
	bool useQualityMap;
	bool useHoleRemover;
	bool useAdvancedOrientationMap;
	bool useMask;
	bool useFrequencyMap;
	bool generateInvertedSceleton;
} PREPROCESSING_FEATURES;

typedef struct preprocessing_general {
	bool cpuOnly;
	int threadNum;
} PREPROCESSING_GENERAL;

#endif // PREPROCESSING_CONFIG_H
