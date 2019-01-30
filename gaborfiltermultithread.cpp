#include "gaborfiltermultithread.h"

GaborFilterMultiThread::GaborFilterMultiThread(QObject *parent) : QObject(parent)
{

}

void GaborFilterMultiThread::setParams(const cv::Mat &imgInput, const GABOR_PARAMS &gaborParams)
{
	m_imgInput = imgInput;
	m_gabor = gaborParams;
}

void GaborFilterMultiThread::enhance()
{
	m_threadsFinished = 0; // na zaciatku paralelneho filtrovania nastavujem pocet ukoncenych vlakien na 0
	m_imgEnhanced = cv::Mat(m_imgInput.rows,m_imgInput.cols, CV_32F, cv::Scalar::all(255));

	QVector<GaborThread*> gaborThreads; // vlakna na paralelne filtrovanie
	m_threads.clear(); // vymazu sa predchadzajuce vlakna

	for(int i = 0; i < *m_gabor.threadNum; i++){

		int threadBlock_Width = m_imgInput.cols - m_gabor.blockSize;
		int threadBlock_Height = (m_imgInput.rows - m_gabor.blockSize) / *m_gabor.threadNum;
		int topLeftCorner_X = qFloor(m_gabor.blockSize / 2);
		int topLeftCorner_Y = qFloor(m_gabor.blockSize / 2) + i * threadBlock_Height ; // pozor!!!

		m_threads.push_back(new QThread());
		gaborThreads.push_back(new GaborThread(m_imgInput, m_gabor, cv::Rect(topLeftCorner_X, topLeftCorner_Y, threadBlock_Width, threadBlock_Height), m_imgEnhanced));
		gaborThreads.last()->moveToThread(m_threads.last()); // pridanie do samostatneho vlakna

		connect(m_threads.last(), SIGNAL(finished()), gaborThreads.last(), SLOT(deleteLater()));
		connect(m_threads.last(), SIGNAL(finished()), m_threads.last(), SLOT(deleteLater()));
		connect(gaborThreads.last(), SIGNAL(enhanceFragmentSignal()), gaborThreads.last(), SLOT(enhanceFragmentSlot()));
		connect(gaborThreads.last(), SIGNAL(enhancementDoneSignal()), this, SLOT(oneGaborThreadFinished()));
		connect(gaborThreads.last(), SIGNAL(enhancementDoneSignal()), gaborThreads.last(), SLOT(enhancementDoneSlot()));
		m_threads.last()->start(); // vlakno vstupi do event loop-u

		// paralelne filtrovanie odtlacku
		// funguje to tak, ze jednotlive vlakna si zdielaju 1 odtlacok a kazde filtruje len urcitu cast
		emit gaborThreads.last()->enhanceFragmentSignal(); // vyslanie signalu na zacatie filtrovania
	}
}

void GaborFilterMultiThread::oneGaborThreadFinished()
{
	// ak vsetky vlakna ukoncili svoju cinnost
	if(++m_threadsFinished == *m_gabor.threadNum){
		// obrazok sa znormalizuje
		cv::normalize(m_imgEnhanced, m_imgEnhanced, 0.0, 255.0, cv::NORM_MINMAX);
		// konverzia do grayscale formatu
		m_imgEnhanced.convertTo(m_imgEnhanced, CV_8UC1);
		// ukoncenie vlakien
		foreach (QThread* t, m_threads) {
			t->quit();
			//t->wait();
		}
		emit gaborThreadsFinished();
	}
}

cv::Mat GaborFilterMultiThread::getImgEnhanced() const
{

	return m_imgEnhanced;
}
