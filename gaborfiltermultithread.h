#ifndef GABORFILTERMULTITHREAD_H
#define GABORFILTERMULTITHREAD_H

#include "preprocessing_config.h"
#include "gaborthread.h"

// registracia typov, aby som mohol cv::Mat a cv::Rect odosielat v signale
Q_DECLARE_METATYPE(cv::Mat)
Q_DECLARE_METATYPE(cv::Rect)

class GaborFilterMultiThread : public QObject
{
    Q_OBJECT

public:
    explicit GaborFilterMultiThread(QObject *parent = nullptr);

    void setParams(const cv::Mat &imgInput, const GABOR_PARAMS &gaborParams);
    void enhance();

    //getNset
    cv::Mat getImgEnhanced() const;

private:
    int threadsFinished;
    QVector<QThread*> threads; // vektor vlakien, ktore paralelne filtruju odtlacok

    // INPUT
    cv::Mat imgInput;

    GABOR_PARAMS gabor;

    // OUTPUT
    cv::Mat imgEnhanced;

private slots:
    void oneGaborThreadFinished(); // slot pre ukoncenie jedneho vlakna

signals:
    void gaborThreadsFinished();

};

#endif // GABORFILTERMULTITHREAD_H
