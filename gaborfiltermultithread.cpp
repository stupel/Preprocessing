#include "gaborfiltermultithread.h"

GaborFilterMultiThread::GaborFilterMultiThread(QObject *parent) : QObject(parent)
{

}

void GaborFilterMultiThread::setParams(cv::Mat img_, cv::Mat orientationMap_, cv::Mat frequencyMap_, int &blockSize_, double &sigma_, double &lambda_, int &numThreads_)
{
    this->imgFp = img_;
    this->oMap = orientationMap_;
    this->frequencyMap = frequencyMap_;
    this->blockSize = blockSize_;
    this->sigma = sigma_;
    this->lambda = lambda_;
    this->numThreads = numThreads_;
}

void GaborFilterMultiThread::enhance(const bool &useFrequencyMap)
{
    this->threadsFinished = 0; // na zaciatku paralelneho filtrovania nastavujem pocet ukoncenych vlakien na 0
    this->imgEnhanced = cv::Mat(this->imgFp.rows,this->imgFp.cols, CV_32F, cv::Scalar::all(255));

    QVector<GaborThread*> gaborThreads; // vlakna na paralelne filtrovanie
    threads.clear(); // vymazu sa predchadzajuce vlakna

    for(int i = 0; i < this->numThreads; i++){

        int threadBlock_Width = this->imgFp.cols - this->blockSize;
        int threadBlock_Height = (this->imgFp.rows - this->blockSize) / this->numThreads;
        int topLeftCorner_X = qFloor(this->blockSize / 2);
        int topLeftCorner_Y = qFloor(this->blockSize / 2) + i * threadBlock_Height ; // pozor!!!

        threads.push_back(new QThread());
        gaborThreads.push_back(new GaborThread(this->imgFp, this->oMap, this->frequencyMap, this->blockSize, this->sigma, this->lambda,
                                               cv::Rect(topLeftCorner_X, topLeftCorner_Y, threadBlock_Width, threadBlock_Height),
                                               this->imgEnhanced));
        gaborThreads.last()->moveToThread(threads.last()); // pridanie do samostatneho vlakna

        connect(threads.last(), SIGNAL(finished()), gaborThreads.last(), SLOT(deleteLater()));
        connect(threads.last(), SIGNAL(finished()), threads.last(), SLOT(deleteLater()));
        connect(gaborThreads.last(), SIGNAL(enhanceFragmentSignal(bool)), gaborThreads.last(), SLOT(enhanceFragmentSlot(bool)));
        connect(gaborThreads.last(), SIGNAL(enhancementDoneSignal()), this, SLOT(oneGaborThreadFinished()));
        connect(gaborThreads.last(), SIGNAL(enhancementDoneSignal()), gaborThreads.last(), SLOT(enhancementDoneSlot()));
        threads.last()->start(); // vlakno vstupi do event loop-u

        // paralelne filtrovanie odtlacku
        // funguje to tak, ze jednotlive vlakna si zdielaju 1 odtlacok a kazde filtruje len urcitu cast
        emit gaborThreads.last()->enhanceFragmentSignal(useFrequencyMap); // vyslanie signalu na zacatie filtrovania
    }
}

void GaborFilterMultiThread::oneGaborThreadFinished()
{
    // ak vsetky vlakna ukoncili svoju cinnost
    if(++this->threadsFinished == this->numThreads){
        // obrazok sa znormalizuje
        cv::normalize(this->imgEnhanced, this->imgEnhanced, 0.0, 255.0,cv::NORM_MINMAX);
        // konverzia do grayscale formatu
        this->imgEnhanced.convertTo(this->imgEnhanced, CV_8UC1);
        // ukoncenie vlakien
        foreach (QThread* t, this->threads) {
            t->quit();
            t->wait();
        }
        emit gaborThreadsFinished();
    }
}

cv::Mat GaborFilterMultiThread::getImgEnhanced() const
{

    return imgEnhanced;
}
