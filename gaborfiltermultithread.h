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

	void setParams(const cv::Mat &m_imgInput, const GABOR_PARAMS &gaborParams);
	void enhance();

	cv::Mat getImgEnhanced() const;

private slots:
	void oneGaborThreadFinished(); // slot pre ukoncenie jedneho vlakna

signals:
	void gaborThreadsFinished();

private:
	int m_threadsFinished;
	QVector<QThread*> m_threads; // vektor vlakien, ktore paralelne filtruju odtlacok

	// INPUT
	cv::Mat m_imgInput;

	GABOR_PARAMS m_gabor;

	// OUTPUT
	cv::Mat m_imgEnhanced;
};

#endif // GABORFILTERMULTITHREAD_H
