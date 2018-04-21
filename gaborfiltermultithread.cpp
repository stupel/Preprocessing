#include "gaborfiltermultithread.h"

GaborFilterMultiThread::GaborFilterMultiThread(QObject *parent) : QObject(parent)
{

}

void GaborFilterMultiThread::setParams(const cv::Mat &imgInput, const cv::Mat &orientationMap, const GABOR_PARAMS &gaborParams, const int &threadNum)
{
    this->imgInput = imgInput;
    this->oMap = orientationMap;
    this->gabor = gaborParams;
    this->threadNum = threadNum;
}

void GaborFilterMultiThread::enhance()
{
    this->threadsFinished = 0; // na zaciatku paralelneho filtrovania nastavujem pocet ukoncenych vlakien na 0
    this->imgEnhanced = cv::Mat(this->imgInput.rows,this->imgInput.cols, CV_32F, cv::Scalar::all(255));

    QVector<GaborThread*> gaborThreads; // vlakna na paralelne filtrovanie
    threads.clear(); // vymazu sa predchadzajuce vlakna

    for(int i = 0; i < this->threadNum; i++){

        int threadBlock_Width = this->imgInput.cols - this->gabor.blockSize;
        int threadBlock_Height = (this->imgInput.rows - this->gabor.blockSize) / this->threadNum;
        int topLeftCorner_X = qFloor(this->gabor.blockSize / 2);
        int topLeftCorner_Y = qFloor(this->gabor.blockSize / 2) + i * threadBlock_Height ; // pozor!!!

        threads.push_back(new QThread());
        gaborThreads.push_back(new GaborThread(this->imgInput, this->oMap, this->gabor,
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
        emit gaborThreads.last()->enhanceFragmentSignal(this->gabor.useFrequencyMap); // vyslanie signalu na zacatie filtrovania
    }
}

void GaborFilterMultiThread::oneGaborThreadFinished()
{
    // ak vsetky vlakna ukoncili svoju cinnost
    if(++this->threadsFinished == this->threadNum){
        // obrazok sa znormalizuje
        cv::normalize(this->imgEnhanced, this->imgEnhanced, 0.0, 255.0, cv::NORM_MINMAX);
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
