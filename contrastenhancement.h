#ifndef CONTRASTENHANCEMENT_H
#define CONTRASTENHANCEMENT_H

#include <QObject>

#include "opencv2/opencv.hpp"

class ContrastEnhancement : public QObject
{
    Q_OBJECT

public:
    explicit ContrastEnhancement(QObject *parent = nullptr);

    void enhance(const cv::Mat &toEnhance, cv::Mat &enhanced, int distance, double sigma, double gaussBlock, double gaussSigma);

private:
    void performSuace(const cv::Mat & src, cv::Mat & dst, int distance, double sigma);

};

#endif // CONTRASTENHANCEMENT_H
