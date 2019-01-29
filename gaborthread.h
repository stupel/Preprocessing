#ifndef GABORTHREAD_H
#define GABORTHREAD_H

#include "preprocessing_config.h"

class GaborThread : public QObject
{
	Q_OBJECT
public:
	explicit GaborThread(QObject *parent = nullptr);
	GaborThread(cv::Mat m_imgInput, const GABOR_PARAMS &gaborParams, cv::Rect m_rect, cv::Mat m_enhancedImage);

public slots:
	void enhanceFragmentSlot();
	void enhancementDoneSlot();

signals:
	void enhancementDoneSignal();
	void enhanceFragmentSignal();

private:
	cv::Mat m_imgInput; // obrazok s odtlackom

	GABOR_PARAMS m_gabor;

	cv::Rect m_rect; // oblast odtlacku, ktoru prefiltruje konkretne vlakno
	cv::Mat m_enhancedImage; // prefiltrovany odtlacok


};

#endif // GABORTHREAD_H
