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

//std
#include <iostream>
#include <stdio.h>
#include <vector>
#include <numeric>      // std::accumulate
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

#endif // PREPROCESSING_CONFIG_H
