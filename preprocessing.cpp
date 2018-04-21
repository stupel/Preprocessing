#include "preprocessing.h"

Preprocessing::Preprocessing()
{
    // CHECKS
    this->imgLoaded = false;
    this->firstRun = true;
    this->isFrequencyModelLoaded = false;
    this->isMaskModelLoaded = false;

    // PREPROCESSING FEATURES
    this->features.advancedMode = false;
    this->features.useContrastEnhancement = true;
    this->features.useQualityMap = true;
    this->features.useHoleRemover = true;
    this->features.useAdvancedOrientationMap = true;
    this->features.generateInvertedSceleton = true;
    this->features.useMask = false;
    this->features.useFrequencyMap = false;

    // GENERAL PARAMS
    this->general.cpuOnly = false;
    this->general.threadNum = QThread::idealThreadCount();

    // CONTRAST ENHANCEMENT PARAMS
    this->contrastParams.distance = 10;
    this->contrastParams.sigma = 30;
    this->contrastParams.gaussBlock = 3;
    this->contrastParams.gaussSigma = 2;

    // ORIENTATION MAP PARAMS
    this->omapParams.blockSize = 13;
    this->omapParams.gaussBlurBasic.blockSize = 1;
    this->omapParams.gaussBlurBasic.sigma = 1.0;
    this->omapParams.gaussBlurAdvanced.blockSize = 121;
    this->omapParams.gaussBlurAdvanced.sigma = 10.0;

    // GABOR FILTER PARAMS
    this->gaborParams.blockSize = 13;
    this->gaborParams.lambda = 9;
    this->gaborParams.sigma = 3;
    this->gaborParams.gamma = 1;
    this->gaborParams.psi = 0;
    this->gaborParams.oMap = &this->results.orientationMap;
    this->gaborParams.oMapAF = &this->orientationMapAF;
    this->gaborParams.threadNum = &this->general.threadNum;
    this->gaborParams.useFrequencyMap = &this->features.useFrequencyMap;
    this->gaborParams.fMap = &this->results.frequencyMap;

    // BINARIZATION PARAMS
    this->binarizationParams.imgMask = &this->results.imgMask;
    this->binarizationParams.imgQualityMap = &this->results.imgQualityMap;
    this->binarizationParams.holeSize = 20;

    // QUALITY MAP
    this->qmapParams.ppi = 500;

    // MASK PARAMS
    this->maskParams.caffeFiles.model = "./core/config/Caffe/mask_deploy.prototxt";
    this->maskParams.caffeFiles.trained = "./core/config/Caffe/mask.caffemodel";
    this->maskParams.caffeFiles.imageMean = "./core/config/Caffe/mask_imagemean.binaryproto";
    this->maskParams.caffeFiles.label = "./core/config/Caffe/mask_labels.txt";
    this->maskParams.blockSize = 9;
    this->maskParams.exBlockSize = 30;
    this->maskParams.useSmooth = false;
    this->maskParams.cpuOnly = &this->general.cpuOnly;

    // FREQUENCY MAP PARAMS
    this->fmapParams.caffeFiles.model = "./core/config/Caffe/frequency_deploy.prototxt";
    this->fmapParams.caffeFiles.trained = "./core/config/Caffe/frequency.caffemodel";
    this->fmapParams.caffeFiles.imageMean = "./core/config/Caffe/frequency_imagemean.binaryproto";
    this->fmapParams.caffeFiles.label = "./core/config/Caffe/frequency_labels.txt";
    this->fmapParams.blockSize = 9;
    this->fmapParams.exBlockSize = 30;
    this->fmapParams.cpuOnly = &this->general.cpuOnly;

    // RESULTS
    this->results.imgOriginal = this->imgOriginal;

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
    if (!((this->fmapParams.caffeFiles.model == freqFiles.model) && (this->fmapParams.caffeFiles.trained == freqFiles.trained) &&
          (this->fmapParams.caffeFiles.imageMean == freqFiles.imageMean) && (this->fmapParams.caffeFiles.label == freqFiles.label)) )
    {
        this->fmapParams.caffeFiles.model = freqFiles.model;
        this->fmapParams.caffeFiles.trained = freqFiles.trained;
        this->fmapParams.caffeFiles.imageMean = freqFiles.imageMean;
        this->fmapParams.caffeFiles.label = freqFiles.label;
    }

    this->fmapParams.blockSize = blockSize;
    this->fmapParams.exBlockSize = exBlockSize;

    this->isFrequencyModelLoaded = false;
}

void Preprocessing::setMaskParams(CAFFE_FILES maskFiles, int blockSize, int exBlockSize, bool useSmooth)
{
    if (!((this->maskParams.caffeFiles.model == maskFiles.model) && (this->maskParams.caffeFiles.trained == maskFiles.trained) &&
          (this->maskParams.caffeFiles.imageMean == maskFiles.imageMean) && (this->maskParams.caffeFiles.label == maskFiles.label)) )
    {
        this->maskParams.caffeFiles.model = maskFiles.model;
        this->maskParams.caffeFiles.trained = maskFiles.trained;
        this->maskParams.caffeFiles.imageMean = maskFiles.imageMean;
        this->maskParams.caffeFiles.label = maskFiles.label;
    }

    this->maskParams.blockSize = blockSize;
    this->maskParams.exBlockSize = exBlockSize;
    this->maskParams.useSmooth = useSmooth;

    this->isMaskModelLoaded = false;
}

void Preprocessing::setPreprocessingParams(int blockSize, double gaborLambda, double gaborSigma, int gaussBlockBasic, double gaussSigmaBasic, int gaussBlockAdvanced, double gaussSigmaAdvanced, int holeSize)
{
    this->gaborParams.blockSize = blockSize;
    this->omapParams.blockSize = blockSize;
    this->gaborParams.lambda = gaborLambda;
    this->gaborParams.sigma = gaborSigma;
    this->omapParams.gaussBlurBasic = GAUSSIAN_BLUR_SETTINGS{gaussBlockBasic, gaussSigmaBasic};
    this->omapParams.gaussBlurAdvanced = GAUSSIAN_BLUR_SETTINGS{gaussBlockAdvanced, gaussSigmaAdvanced};
    this->binarizationParams.holeSize = holeSize;
}

void Preprocessing::setFeatures(bool useAdvancedMode, bool useContrastEnhancement, bool useAdvancedOrientationMap, bool useHoleRemover, bool generateInvertedSkeleton, bool useQualityMap, bool useMask, bool useFrequencyMap)
{
    this->features.advancedMode = useAdvancedMode;
    this->features.useContrastEnhancement = useContrastEnhancement;
    this->features.useAdvancedOrientationMap = useAdvancedOrientationMap;
    this->features.useQualityMap = useQualityMap;
    this->features.useHoleRemover = useHoleRemover;
    this->features.generateInvertedSceleton = generateInvertedSkeleton;
    this->features.useMask = useMask;
    this->features.useFrequencyMap = useFrequencyMap;
}

void Preprocessing::setCPUOnly(bool enabled, int threadNum)
{
    if (enabled) this->general.cpuOnly = true;
    else this->general.cpuOnly = false;

    if (threadNum == 0) this->general.threadNum = QThread::idealThreadCount();
    else this->general.threadNum = threadNum;
}

void Preprocessing::start()
{
    if(this->imgLoaded) {

        if (!this->firstRun) this->clean();

        // MASK WITH NEURAL NETWORK
        if (this->features.useMask) {
            if (!this->isMaskModelLoaded) mask.loadMaskModel(this->maskParams.caffeFiles);

            this->mask.setParams(this->imgOriginal, this->maskParams);
            this->timer.start();
            this->mask.generate();
            this->durations.mask = this->timer.elapsed();
            this->results.imgMask = mask.getImgMask();
        }

        // QUALITY MAP
        if (this->features.useQualityMap) {
            this->qMap.setParams(this->imgOriginal, this->qmapParams);

            this->timer.start();
            this->qMap.computeQualityMap();
            this->durations.qualityMap = timer.elapsed();

            if (this->features.advancedMode) this->results.imgQualityMap = this->qMap.getImgQualityMap();
            this->results.qualityMap = this->qMap.getQualityMap();
        }

        // FREQUENCY MAP WITH NEURAL NETWORK
        if (this->features.useFrequencyMap) {
            if (!this->isFrequencyModelLoaded) this->fMap.loadFrequencyMapModel(this->fmapParams.caffeFiles);

            this->fMap.setParams(this->imgOriginal, this->fmapParams);
            this->timer.start();
            this->fMap.generate();
            this->durations.frequencyMap = this->timer.elapsed();

            this->results.frequencyMap = fMap.getFrequencyMap();
            if (this->features.advancedMode) this->results.imgFrequencyMap = fMap.getImgFrequencyMap();
        }

        // CONTRAST ENHANCEMENT
        if (this->features.useContrastEnhancement) {
            this->contrast.setParams(this->imgOriginal, this->contrastParams);
            this->timer.start();
            this->contrast.enhance();
            this->durations.contrastEnhancement = this->timer.elapsed();
            this->results.imgContrastEnhanced = this->contrast.getImgContrastEnhanced();

            // ORIENTATION MAP
            this->oMap.setParams(this->results.imgContrastEnhanced, this->omapParams);
        }
        else this->oMap.setParams(this->imgOriginal, this->omapParams);

        if (this->general.cpuOnly) {
            this->oMap.computeAdvancedMapCPU(); // Gabor Filter in CPU mode works only with advanced orientation map
            this->results.orientationMap = this->oMap.getOMap_advanced();
        }
        else {
            this->oMap.computeAdvancedMapGPU();
            if (this->features.useAdvancedOrientationMap) {
                this->orientationMapAF = this->oMap.getOMapAF_advanced();
            }
            else {
                this->orientationMapAF = this->oMap.getOMapAF_basic();
            }
            this->results.orientationMap = this->oMap.getOMap_advanced();
        }

        this->durations.orientationMap = this->oMap.getDuration();

        if (this->features.advancedMode) {
            this->oMap.drawBasicMap(this->imgOriginal);
            this->results.imgOrientationMap = this->oMap.getImgOMap_basic();
        }

        if (!this->general.cpuOnly) {
            // GABOR FILTER GPU
            if (this->features.useContrastEnhancement) this->gaborGPU.setParams(this->results.imgContrastEnhanced, this->gaborParams);
            else this->gaborGPU.setParams(this->imgOriginal, this->gaborParams);

            if (this->features.useAdvancedOrientationMap) this->gaborGPU.enhanceWithAdvancedOMap();
            else this->gaborGPU.enhanceWithBasicOMap();
            this->durations.gaborFilter = this->gaborGPU.getDuration();

            this->results.imgEnhanced = this->gaborGPU.getImgEnhanced(); // ziskanie prefiltrovaneho odtlacku
            this->continueAfterGabor();
        }
        else {
            // GABOR FILTER CPU MULTITHREAD
            if (this->features.useContrastEnhancement) this->gaborMultiThread.setParams(this->results.imgContrastEnhanced, this->gaborParams);
            else this->gaborMultiThread.setParams(this->imgOriginal, this->gaborParams);
            this->timer.start();
            this->gaborMultiThread.enhance(); // filtrovanie so zvolenym typom smerovej mapy
        }
    }
    else this->preprocessingError(10);
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
    this->binarization.setParams(this->results.imgEnhanced, this->binarizationParams, this->features);

    this->timer.start();
    this->binarization.binarizeAdaptive();
    this->durations.binarization = this->timer.elapsed();
    this->results.imgBinarized = this->binarization.getImgBinarized();

    // HOLE REMOVER
    if (this->features.useHoleRemover) {
        this->timer.start();
        this->binarization.removeHoles(this->binarizationParams.holeSize);
        this->durations.removingHoles = this->timer.elapsed();
    }

    // THINNING GUO HALL FAST
    this->timer.start();
    this->thinning.thinGuoHallFast(this->results.imgBinarized, false);
    if (this->features.generateInvertedSceleton) this->thinning.thinGuoHallFast(this->results.imgBinarized, true);
    this->durations.thinning = this->timer.elapsed();
    this->results.imgSkeleton = this->thinning.getImgSkeleton();
    if (this->features.generateInvertedSceleton) this->results.imgSkeletonInverted = this->thinning.getImgSkeletonInverted();

    // EMITS
    if (this->features.advancedMode) {
        emit preprocessingAdvancedDoneSignal(this->results);
    }
    else {
        PREPROCESSING_RESULTS basicResults = {this->imgOriginal, this->results.imgSkeleton, this->results.imgSkeletonInverted,
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

PREPROCESSING_ALL_RESULTS Preprocessing::getResults() const
{
    return results;
}

PREPROCESSING_DURATIONS Preprocessing::getDurations() const
{
    return durations;
}
