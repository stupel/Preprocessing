#include "preprocessing.h"

Preprocessing::Preprocessing()
{
    this->imgLoaded = false;

    this->isFreqNetLoaded = false;
    this->isMaskNetLoaded = false;
    this->firstRun = true;

    //preprocessingFeatures Default
    this->advancedMode = false;
    this->numThreads = QThread::idealThreadCount();
    this->useContrastEnhancement = true;
    this->useRemoveHoles = true;
    this->useFixOrientations = true;
    this->useMask = false;
    this->useQualityMap = true;
    this->useFrequencyMap = false;

    //preprecessingParams Default
    this->blockSize = 13;
    this->gaborLambda = 9;
    this->gaborSigma = 3;
    this->gaussBlurBasic.blockSize = 1;
    this->gaussBlurBasic.sigma = 1.0;
    this->gaussBlurAdvanced.blockSize = 121;
    this->gaussBlurAdvanced.sigma = 10.0;
    this->holeSize = 20;

    //maskParams Default
    this->maskFiles.model = "./core/config/Caffe/mask_deploy.prototxt";
    this->maskFiles.trained = "./core/config/Caffe/mask.caffemodel";
    this->maskFiles.imageMean = "./core/config/Caffe/mask_imagemean.binaryproto";
    this->maskFiles.label = "./core/config/Caffe/mask_labels.txt";
    this->maskBlockSize = 8;
    this->maskExBlockSize = 19;
    this->maskUseSmooth = false;

    //frequecyMapParams Default
    this->freqFiles.model = "./core/config/Caffe/frequency_deploy.prototxt";
    this->freqFiles.trained = "./core/config/Caffe/frequency.caffemodel";
    this->freqFiles.imageMean = "./core/config/Caffe/frequency_imagemean.binaryproto";
    this->freqFiles.label = "./core/config/Caffe/frequency_labels.txt";
    this->freqBlockSize = 9;           //!!!!!!!??????????
    this->freqExBlockSize = 30;        //!!!!!!!??????????

    connect(&this->gaborMultiThread, SIGNAL(gaborThreadsFinished()), this, SLOT(allGaborThreadsFinished()));
}

void Preprocessing::clean()
{
    this->results.imgBinarized.release();
    this->results.imgContrastEnhanced.release();
    this->results.imgEnhanced.release();
    this->results.imgFrequencyMap.release();
    this->results.imgMask.release();
    this->results.imgOrientationMap.release();
    this->results.imgQualityMap.release();
    this->results.imgSkeleton.release();
    this->results.imgSkeletonInverted.release();
}

void Preprocessing::loadImg(cv::Mat imgInput)
{
    this->imgOriginal = imgInput;

    this->imgLoaded = true;
}

void Preprocessing::setFrequencyMapParams(const CAFFE_FILES &freqFiles, const int &blockSize, const int &exBlockSize)
{
    if (!((this->freqFiles.model == freqFiles.model) && (this->freqFiles.trained == freqFiles.trained) &&
          (this->freqFiles.imageMean == freqFiles.imageMean) && (this->freqFiles.label == freqFiles.label)) )
    {
        this->freqFiles.model = freqFiles.model;
        this->freqFiles.trained = freqFiles.trained;
        this->freqFiles.imageMean = freqFiles.imageMean;
        this->freqFiles.label = freqFiles.label;
    }

    this->freqBlockSize = blockSize;
    this->freqExBlockSize = exBlockSize;
}

void Preprocessing::setMaskParams(const CAFFE_FILES &maskFiles, const int &blockSize, const int &exBlockSize, const bool &useSmooth)
{
    if (!((this->maskFiles.model == maskFiles.model) && (this->maskFiles.trained == maskFiles.trained) &&
          (this->maskFiles.imageMean == maskFiles.imageMean) && (this->maskFiles.label == maskFiles.label)) )
    {
        this->maskFiles.model = maskFiles.model;
        this->maskFiles.trained = maskFiles.trained;
        this->maskFiles.imageMean = maskFiles.imageMean;
        this->maskFiles.label = maskFiles.label;
    }

    this->maskBlockSize = blockSize;
    this->maskExBlockSize = exBlockSize;
    this->maskUseSmooth = useSmooth;
}

void Preprocessing::setPreprocessingParams(int blockSize, double gaborLambda, double gaborSigma, int gaussBlockBasic, double gaussSigmaBasic, int gaussBlockAdvanced, double gaussSigmaAdvanced, int holeSize)
{
    this->blockSize = blockSize;
    this->gaborLambda = gaborLambda;
    this->gaborSigma = gaborSigma;
    this->gaussBlurBasic = GAUSSIAN_BLUR_SETTINGS{gaussBlockBasic, gaussSigmaBasic};
    this->gaussBlurAdvanced = GAUSSIAN_BLUR_SETTINGS{gaussBlockAdvanced, gaussSigmaAdvanced};
    this->holeSize = holeSize;
}

void Preprocessing::setFeatures(bool advancedMode, int numThreads, bool useGaborFilterGPU, bool useContrastEnhancement, bool useRemoveHoles, bool useFixOrientations, bool useMask, bool useQualityMap, bool useFrequencyMap)
{
    this->advancedMode = advancedMode;
    if (numThreads == 0) this->numThreads = QThread::idealThreadCount();
    else this->numThreads = numThreads;
    this->useGaborFilterGPU = useGaborFilterGPU;
    this->useContrastEnhancement = useContrastEnhancement;
    this->useRemoveHoles = useRemoveHoles;
    this->useFixOrientations = useFixOrientations;
    this->useMask = useMask;
    this->useQualityMap = useQualityMap;
    this->useFrequencyMap = useFrequencyMap;
}

void Preprocessing::generateMask()
{
    Mask mask;

    if (!this->isMaskNetLoaded) {
        mask.loadMaskModel(this->maskFiles);
        this->isMaskNetLoaded = true;
    }

    this->timer.start();
    mask.generate(this->imgOriginal, this->maskBlockSize, this->maskExBlockSize, this->maskUseSmooth);
    this->durations.mask = this->timer.elapsed();
    this->results.imgMask = mask.getImgMask();
}

void Preprocessing::run()
{
    if(this->imgLoaded) {

        if (!this->firstRun) this->clean();

        if (this->useMask) {
            this->generateMask();
        }

        if (this->useQualityMap) {
            this->qMap.loadImage(Helper::Mat2QImage(this->imgOriginal, QImage::Format_Grayscale8), 500);

            this->timer.start();
            this->qMap.computeQualityMap();
            this->durations.qualityMap = timer.elapsed();

            this->results.imgQualityMap = this->qMap.getImgQualityMap();
        }

        if (this->useFrequencyMap) {
            //if (!this->isFreqNetLoaded) {
                this->fMap.loadFrequencyMapModel(this->freqFiles);
                this->isFreqNetLoaded = true;
            //}

            this->timer.start();
            this->fMap.generate(this->imgOriginal, this->freqBlockSize, this->freqExBlockSize);
            this->durations.frequencyMap = this->timer.elapsed();

            this->results.frequencyMap = this->fMap.getFrequencyMap();
            if (this->advancedMode) this->results.imgFrequencyMap = this->fMap.getImgFrequencyMap();
        }
        //else if (this->useFrequencyMap && !this->freqParamsSet) this->preprocessingError();

        if (this->useContrastEnhancement) {
            this->timer.start();
            this->contrast.enhance(this->imgOriginal, this->results.imgContrastEnhanced, 10, 30, 3, 2);
            this->durations.contrastEnhancement = this->timer.elapsed();

            this->oMap.setParams(this->results.imgContrastEnhanced, this->blockSize, this->gaussBlurBasic, this->gaussBlurAdvanced);
        }
        else this->oMap.setParams(this->imgOriginal, this->blockSize, this->gaussBlurBasic, this->gaussBlurAdvanced);

        this->timer.start();
        this->oMap.computeAdvancedMap();
        this->durations.orientationMap = this->timer.elapsed();
        this->results.orientationMap = this->oMap.getOMap_advanced();

        if (this->advancedMode) {
            this->oMap.drawBasicMap(this->imgOriginal);
            this->results.imgOrientationMap = this->oMap.getImgOMap_basic();
        }

        if (this->useGaborFilterGPU) {
            if (this->useContrastEnhancement) this->gaborGPU.setParams(this->results.imgContrastEnhanced, this->oMap.getOMap_basic(), this->results.frequencyMap, this->blockSize, this->gaborSigma, this->gaborLambda);
            else this->gaborGPU.setParams(this->imgOriginal, this->oMap.getOMap_basic(), this->results.frequencyMap, this->blockSize, this->gaborSigma, this->gaborLambda);

            this->timer.start();
            this->gaborGPU.enhance();
            this->durations.gaborFilter = this->timer.elapsed();

            this->results.imgEnhanced = this->gaborGPU.getImgEnhanced(); // ziskanie prefiltrovaneho odtlacku
            this->continueAfterGabor();
        }
        else {
            if (this->useContrastEnhancement) this->gaborMultiThread.setParams(this->results.imgContrastEnhanced, this->results.orientationMap, this->results.frequencyMap, this->blockSize, this->gaborSigma, this->gaborLambda, this->numThreads);
            else this->gaborMultiThread.setParams(this->imgOriginal, this->results.orientationMap, this->results.frequencyMap, this->blockSize, this->gaborSigma, this->gaborLambda, this->numThreads);
            this->timer.start();
            this->gaborMultiThread.enhance(this->useFrequencyMap); // filtrovanie so zvolenym typom smerovej mapy
        }
    }
    else this->preprocessingError(10);
}

void Preprocessing::allGaborThreadsFinished()
{
    // ked sa dokoncilo filtrovanie odtlacku
    this->durations.gaborFilter = this->timer.elapsed();
    this->results.imgEnhanced = this->gaborMultiThread.getImgEnhanced(); // ziskanie prefiltrovaneho odtlacku

    this->continueAfterGabor();
}

void Preprocessing::continueAfterGabor()
{
    this->binarization.setParams(this->results.imgEnhanced, this->useMask, this->results.imgMask, this->useQualityMap, this->results.imgQualityMap);

    this->timer.start();
    this->binarization.binarizeAdaptive();
    this->durations.binarization = this->timer.elapsed();
    this->results.imgBinarized = this->binarization.getImgBinarized();

    if (this->useRemoveHoles) {
        this->timer.start();
        this->binarization.removeHoles(this->holeSize);
        this->durations.removingHoles = this->timer.elapsed();
    }

    this->timer.start();
    this->thinning.thinGuoHallFast(this->results.imgBinarized, false);
    if (this->useFixOrientations) this->thinning.thinGuoHallFast(this->results.imgBinarized, true);
    this->durations.thinning = this->timer.elapsed();
    this->results.imgSkeleton = this->thinning.getImgSkeleton();
    if (this->useFixOrientations) this->results.imgSkeletonInverted = this->thinning.getImgSkeletonInverted();

    if (this->advancedMode) {
        emit preprocessingAdvancedDoneSignal(this->results);
    }
    else {
        PREPROCESSING_RESULTS basicResults = {this->results.imgSkeleton, this->results.imgSkeletonInverted, this->results.imgMask,
                                              this->results.qualityMap, this->results.frequencyMap, this->results.orientationMap};
        emit preprocessingDoneSignal(basicResults);
    }

    emit preprocessingDurrationSignal(this->durations);

    this->firstRun = false;
}

int Preprocessing::preprocessingError(int errorcode)
{
    emit preprocessingErrorSignal(errorcode);

    return -1;
}
