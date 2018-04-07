#ifndef GABORFILTERMULTITHREAD_H
#define GABORFILTERMULTITHREAD_H

#include <QObject>
#include <QMetaType>
#include <QtMath>
#include <QThread>
#include <QVector>
#include <QDebug>
#include <QTime>

#include "gaborthread.h"

// registracia typov, aby som mohol cv::Mat a cv::Rect odosielat v signale
Q_DECLARE_METATYPE(cv::Mat)
Q_DECLARE_METATYPE(cv::Rect)

class GaborFilterMultiThread : public QObject
{
    Q_OBJECT

public:
    explicit GaborFilterMultiThread(QObject *parent = nullptr);

    void setParams(cv::Mat img_, cv::Mat orientationMap_, cv::Mat frequencyMap_, int &blockSize_, double &sigma_, double &lambda_, int &numThreads_);
    void enhance(const bool &useFrequencyMap);

    //getNset
    cv::Mat getImgEnhanced() const;

private:
    int threadsFinished;
    QVector<QThread*> threads; // vektor vlakien, ktore paralelne filtruju odtlacok

    cv::Mat imgFp;
    cv::Mat imgEnhanced;
    cv::Mat oMap;
    cv::Mat frequencyMap;

    int blockSize;
    double sigma;
    double lambda;
    int numThreads;

private slots:
    void oneGaborThreadFinished(); // slot pre ukoncenie jedneho vlakna

signals:
    void gaborThreadsFinished();

};

#endif // GABORFILTERMULTITHREAD_H
