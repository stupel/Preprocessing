#include "preprocessing.h"

Preprocessing::Preprocessing()
{
    this->imgLoaded = false;
    this->firstRun = true;

    //preprocessingFeatures Default
    this->advancedMode = false;
    this->numThreads = QThread::idealThreadCount();
    this->useContrastEnhancement = true;
    this->useHoleRemover = true;
    this->useOrientationFixer = true;
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
    this->maskExBlockSize = 30;
    this->maskUseSmooth = false;
    this->isFrequencyModelLoaded = false;

    //frequecyMapParams Default
    this->freqFiles.model = "./core/config/Caffe/frequency_deploy.prototxt";
    this->freqFiles.trained = "./core/config/Caffe/frequency.caffemodel";
    this->freqFiles.imageMean = "./core/config/Caffe/frequency_imagemean.binaryproto";
    this->freqFiles.label = "./core/config/Caffe/frequency_labels.txt";
    this->freqBlockSize = 9;           //!!!!!!!??????????
    this->freqExBlockSize = 30;        //!!!!!!!??????????
    this->isMaskModelLoaded = false;

    //CONNECTS
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

    this->durations.binarization = 0;
    this->durations.contrastEnhancement = 0;
    this->durations.frequencyMap = 0;
    this->durations.gaborFilter = 0;
    this->durations.mask = 0;
    this->durations.orientationMap = 0;
    this->durations.qualityMap = 0;
    this->durations.removingHoles = 0;
    this->durations.thinning = 0;
}

void Preprocessing::loadImg(cv::Mat imgOriginal)
{
    this->imgOriginal = imgOriginal.clone();

    this->imgLoaded = true;
}

void Preprocessing::setFrequencyMapParams(CAFFE_FILES freqFiles, int blockSize, int exBlockSize)
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

    this->isFrequencyModelLoaded = false;
}

void Preprocessing::setMaskParams(CAFFE_FILES maskFiles, int blockSize, int exBlockSize, bool useSmooth)
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

    this->isMaskModelLoaded = false;
}

void Preprocessing::setPreprocessingParams(int numThreads, int blockSize, double gaborLambda, double gaborSigma, int gaussBlockBasic, double gaussSigmaBasic, int gaussBlockAdvanced, double gaussSigmaAdvanced, int holeSize)
{
    if (numThreads == 0) this->numThreads = QThread::idealThreadCount();
    else this->numThreads = numThreads;
    this->blockSize = blockSize;
    this->gaborLambda = gaborLambda;
    this->gaborSigma = gaborSigma;
    this->gaussBlurBasic = GAUSSIAN_BLUR_SETTINGS{gaussBlockBasic, gaussSigmaBasic};
    this->gaussBlurAdvanced = GAUSSIAN_BLUR_SETTINGS{gaussBlockAdvanced, gaussSigmaAdvanced};
    this->holeSize = holeSize;
}

void Preprocessing::setFeatures(bool useAdvancedMode, bool useContrastEnhancement, bool useHoleRemover, bool useOrientationFixer, bool useQualityMap, bool useMask, bool useFrequencyMap)
{
    this->advancedMode = useAdvancedMode;
    this->useContrastEnhancement = useContrastEnhancement;
    this->useHoleRemover = useHoleRemover;
    this->useOrientationFixer = useOrientationFixer;
    this->useMask = useMask;
    this->useQualityMap = useQualityMap;
    this->useFrequencyMap = useFrequencyMap;
}

void Preprocessing::setCPUOnly(bool enabled)
{
    if (enabled) {
        #ifndef CPU_ONLY
            #define CPU_ONLY
        #endif
    }
    else {
        #ifdef CPU_ONLY
            #undef CPU_ONLY
        #endif
    }
}

void Preprocessing::start()
{
    if(this->imgLoaded) {

        if (!this->firstRun) this->clean();

        // MASK WITH NEURAL NETWORK
        if (this->useMask) {
            if (!this->isMaskModelLoaded) mask.loadMaskModel(this->maskFiles);

            this->timer.start();
            mask.generate(this->imgOriginal, this->maskBlockSize, this->maskExBlockSize, this->maskUseSmooth);
            this->durations.mask = this->timer.elapsed();
            this->results.imgMask = mask.getImgMask();
        }

        // QUALITY MAP
        if (this->useQualityMap) {
            this->qMap.loadImage(Helper::Mat2QImage(this->imgOriginal, QImage::Format_Grayscale8), 500);

            this->timer.start();
            this->qMap.computeQualityMap();
            this->durations.qualityMap = timer.elapsed();

            if (this->advancedMode) this->results.imgQualityMap = this->qMap.getImgQualityMap();
            this->results.qualityMap = this->qMap.getQualityMap();
        }

        // FREQUENCY MAP WITH NEURAL NETWORK
        if (this->useFrequencyMap) {
            if (!this->isFrequencyModelLoaded) fMap.loadFrequencyMapModel(this->freqFiles);

            this->timer.start();
            fMap.generate(this->imgOriginal, this->freqBlockSize, this->freqExBlockSize);
            this->durations.frequencyMap = this->timer.elapsed();

            this->results.frequencyMap = fMap.getFrequencyMap();
            if (this->advancedMode) this->results.imgFrequencyMap = fMap.getImgFrequencyMap();
        }

        // CONTRAST ENHANCEMENT
        if (this->useContrastEnhancement) {
            this->timer.start();
            this->contrast.enhance(this->imgOriginal, this->results.imgContrastEnhanced, 10, 30, 3, 2);
            this->durations.contrastEnhancement = this->timer.elapsed();

            // ORIENTATION MAP
            this->oMap.setParams(this->results.imgContrastEnhanced, this->blockSize, this->gaussBlurBasic, this->gaussBlurAdvanced);
        }
        else this->oMap.setParams(this->imgOriginal, this->blockSize, this->gaussBlurBasic, this->gaussBlurAdvanced);

        this->timer.start();

        #ifdef CPU_ONLY
            this->oMap.computeAdvancedMapCPU();
        #else
            this->oMap.computeAdvancedMapGPU();
        #endif

        this->durations.orientationMap = this->timer.elapsed();
        this->results.orientationMap = this->oMap.getOMap_advanced();

        if (this->advancedMode) {
            this->oMap.drawBasicMap(this->imgOriginal);
            this->results.imgOrientationMap = this->oMap.getImgOMap_basic();
        }

        #ifndef CPU_ONLY
        // GABOR FILTER GPU
        if (this->useContrastEnhancement) this->gaborGPU.setParams(this->results.imgContrastEnhanced, this->oMap.getOMap_basic(), this->blockSize, this->gaborSigma, this->gaborLambda, this->useFrequencyMap, this->results.frequencyMap);
        else this->gaborGPU.setParams(this->imgOriginal, this->oMap.getOMap_basic(), this->blockSize, this->gaborSigma, this->gaborLambda, this->useFrequencyMap, this->results.frequencyMap);

        this->gaborGPU.enhance();
        this->durations.gaborFilter = this->gaborGPU.getDuration();

        this->results.imgEnhanced = this->gaborGPU.getImgEnhanced(); // ziskanie prefiltrovaneho odtlacku
        this->continueAfterGabor();

        #else
        // GABOR FILTER CPU MULTITHREAD
        if (this->useContrastEnhancement) this->gaborMultiThread.setParams(this->results.imgContrastEnhanced, this->results.orientationMap, this->results.frequencyMap, this->blockSize, this->gaborSigma, this->gaborLambda, this->numThreads);
        else this->gaborMultiThread.setParams(this->imgOriginal, this->results.orientationMap, this->results.frequencyMap, this->blockSize, this->gaborSigma, this->gaborLambda, this->numThreads);
        this->timer.start();
        this->gaborMultiThread.enhance(this->useFrequencyMap); // filtrovanie so zvolenym typom smerovej mapy
        #endif
    }
    else this->preprocessingError(10);
}

PREPROCESSING_ALL_RESULTS Preprocessing::getResults() const
{
    return results;
}

PREPROCESSING_DURATIONS Preprocessing::getDurations() const
{
    return durations;
}

void Preprocessing::allGaborThreadsFinished()
{
    // GABOR FILTER CPU MULTITHREAD FINISHED
    this->durations.gaborFilter = this->timer.elapsed();
    this->results.imgEnhanced = this->gaborMultiThread.getImgEnhanced(); // ziskanie prefiltrovaneho odtlacku

    this->continueAfterGabor();
}

void Preprocessing::continueAfterGabor()
{
    // BINARIZATION
    this->binarization.setParams(this->results.imgEnhanced, this->useMask, this->results.imgMask, this->useQualityMap, this->results.imgQualityMap);

    this->timer.start();
    this->binarization.binarizeAdaptive();
    this->durations.binarization = this->timer.elapsed();
    this->results.imgBinarized = this->binarization.getImgBinarized();

    // HOLE REMOVER
    if (this->useHoleRemover) {
        this->timer.start();
        this->binarization.removeHoles(this->holeSize);
        this->durations.removingHoles = this->timer.elapsed();
    }

    // THINNING GUO HALL FAST
    this->timer.start();
    this->thinning.thinGuoHallFast(this->results.imgBinarized, false);
    if (this->useOrientationFixer) this->thinning.thinGuoHallFast(this->results.imgBinarized, true);
    this->durations.thinning = this->timer.elapsed();
    this->results.imgSkeleton = this->thinning.getImgSkeleton();
    if (this->useOrientationFixer) this->results.imgSkeletonInverted = this->thinning.getImgSkeletonInverted();

    // EMITS
    if (this->advancedMode) {
        emit preprocessingAdvancedDoneSignal(this->results);
    }
    else {
        PREPROCESSING_RESULTS basicResults = {this->results.imgSkeleton, this->results.imgSkeletonInverted,
                                              this->results.qualityMap, this->results.orientationMap};
        emit preprocessingDoneSignal(basicResults);
    }

    emit preprocessingDurationSignal(this->durations);

    this->firstRun = false;
}

int Preprocessing::preprocessingError(int errorcode)
{
    emit preprocessingErrorSignal(errorcode);

    return -1;
}
