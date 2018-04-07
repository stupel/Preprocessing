#ifndef CONFIG_H
#define CONFIG_H
#define GLOG_NO_ABBREVIATED_SEVERITIES

#include <QObject>

#include "opencv2/opencv.hpp"
#include "helper.h"

typedef struct caffe_files {
    QString model;
    QString trained;
    QString imageMean;
    QString label;
} CAFFE_FILES;

#endif // CONFIG_H
