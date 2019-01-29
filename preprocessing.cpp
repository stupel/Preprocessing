#include "preprocessing.h"

Preprocessing::Preprocessing()
{  
    this->preprocessingIsRunning = false;

    // INPUT PARAMS
    this->inputParams.mode = image;
    this->inputParams.cnt = 0;
    this->inputParams.inputLoaded = false;

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
    this->binarizationParams.useMask = &this->features.useMask;
    this->binarizationParams.imgMask = &this->results.imgMask;
    this->binarizationParams.useQualityMap = &this->features.useQualityMap;
    this->binarizationParams.imgQualityMap = &this->results.imgQualityMap;
    this->binarizationParams.holeSize = 20;

    // QUALITY MAP
    this->qmapParams.ppi = 500;

    // MASK PARAMS
    this->maskParams.caffeFiles.model = "./core/config/Caffe/mask_deploy.prototxt";
    this->maskParams.caffeFiles.trained = "./core/config/Caffe/mask.caffemodel";
    this->maskParams.caffeFiles.imageMean = "./core/config/Caffe/mask_imagemean.binaryproto";
    this->maskParams.caffeFiles.label = "./core/config/Caffe/mask_labels.txt";
    this->maskParams.blockSize = 8;
    this->maskParams.exBlockSize = 19;
    this->maskParams.useSmooth = false;
    this->maskParams.isModelLoaded = false;
    this->maskParams.cpuOnly = &this->general.cpuOnly;

    // FREQUENCY MAP PARAMS
    this->fmapParams.caffeFiles.model = "./core/config/Caffe/frequency_deploy.prototxt";
    this->fmapParams.caffeFiles.trained = "./core/config/Caffe/frequency.caffemodel";
    this->fmapParams.caffeFiles.imageMean = "./core/config/Caffe/frequency_imagemean.binaryproto";
    this->fmapParams.caffeFiles.label = "./core/config/Caffe/frequency_labels.txt";
    this->fmapParams.blockSize = 8;
    this->fmapParams.exBlockSize = 30;
    this->fmapParams.isModelLoaded = false;
    this->fmapParams.cpuOnly = &this->general.cpuOnly;

    //CONNECTS
    connect(&this->gaborMultiThread, SIGNAL(gaborThreadsFinished()), this, SLOT(allGaborThreadsFinished()));
}

int Preprocessing::setFrequencyMapParams(CAFFE_FILES freqFiles, int blockSize, int exBlockSize)
{
    if (this->preprocessingIsRunning) {
        this->preprocessingError(10);
        return -1;
    }

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

    this->fmapParams.isModelLoaded = false;

    return 1;
}

int Preprocessing::setMaskParams(CAFFE_FILES maskFiles, int blockSize, int exBlockSize, bool useSmooth)
{
    if (this->preprocessingIsRunning) {
        this->preprocessingError(10);
        return -1;
    }

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

    this->maskParams.isModelLoaded = false;

    return 1;
}

int Preprocessing::setPreprocessingParams(int blockSize, double gaborLambda, double gaborSigma, int gaussBlockBasic, double gaussSigmaBasic, int gaussBlockAdvanced, double gaussSigmaAdvanced, int holeSize)
{
    if (this->preprocessingIsRunning) {
        this->preprocessingError(10);
        return -1;
    }

    this->gaborParams.blockSize = blockSize;
    this->omapParams.blockSize = blockSize;
    this->gaborParams.lambda = gaborLambda;
    this->gaborParams.sigma = gaborSigma;
    this->omapParams.gaussBlurBasic = GAUSSIAN_BLUR_SETTINGS{gaussBlockBasic, gaussSigmaBasic};
    this->omapParams.gaussBlurAdvanced = GAUSSIAN_BLUR_SETTINGS{gaussBlockAdvanced, gaussSigmaAdvanced};
    this->binarizationParams.holeSize = holeSize;

    return 1;
}

int Preprocessing::setFeatures(bool useAdvancedMode, bool useContrastEnhancement, bool useAdvancedOrientationMap, bool useHoleRemover, bool generateInvertedSkeleton, bool useQualityMap, bool useMask, bool useFrequencyMap)
{
    if (this->preprocessingIsRunning) {
        this->preprocessingError(10);
        return -1;
    }

    this->features.advancedMode = useAdvancedMode;
    this->features.useContrastEnhancement = useContrastEnhancement;
    this->features.useAdvancedOrientationMap = useAdvancedOrientationMap;
    this->features.useQualityMap = useQualityMap;
    this->features.useHoleRemover = useHoleRemover;
    this->features.generateInvertedSceleton = generateInvertedSkeleton;
    this->features.useMask = useMask;
    this->features.useFrequencyMap = useFrequencyMap;

    return 1;
}

int Preprocessing::setCPUOnly(bool enabled, int threadNum)
{
    if (this->preprocessingIsRunning) {
        this->preprocessingError(10);
        return -1;
    }

    if (enabled) this->general.cpuOnly = true;
    else this->general.cpuOnly = false;

    if (threadNum == 0) this->general.threadNum = QThread::idealThreadCount();
    else this->general.threadNum = threadNum;

    return 1;
}

void Preprocessing::cleanResults()
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
    this->results.frequencyMap.release();
    this->results.orientationMap.release();
    this->results.qualityMap.release();

    this->resultsMap.clear();
    this->allResultsMap.clear();
}

void Preprocessing::cleanInput()
{
    this->inputParams.inputLoaded = false;
    this->inputParams.imgOriginal.release();
    this->inputParams.imgOriginals.clear();
    this->inputParams.imgNames.clear();
    this->inputParams.path = "";
    this->inputParams.cnt = 0;
}

void Preprocessing::cleanDurations()
{
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

// INPUT

int Preprocessing::loadInput(cv::Mat imgOriginal)
{
    if (this->preprocessingIsRunning) {
        this->preprocessingError(10);
        return -1;
    }

    this->cleanInput();
    this->inputParams.imgOriginal = imgOriginal.clone();
    if (this->inputParams.imgOriginal.channels() != 1) cv::cvtColor(this->inputParams.imgOriginal, this->inputParams.imgOriginal, cv::COLOR_BGR2GRAY);
    this->inputParams.inputLoaded = true;
    this->inputParams.mode = image;

    return 1;
}

int Preprocessing::loadInput(QVector<cv::Mat> imgOriginals)
{
    if (this->preprocessingIsRunning) {
        this->preprocessingError(10);
        return -1;
    }

    this->cleanInput();
    this->inputParams.imgOriginals = imgOriginals;
    for (int i = 0; i < imgOriginals.size(); i++) {
        if (this->inputParams.imgOriginals[i].channels() != 1) cv::cvtColor(this->inputParams.imgOriginals[i], this->inputParams.imgOriginals[i], cv::COLOR_BGR2GRAY);
        this->inputParams.imgNames.push_back(QString::number(i+1));
    }
    this->inputParams.inputLoaded = true;
    this->inputParams.mode = images;

    return 1;
}

int Preprocessing::loadInput(QString inputPath)
{
    if (this->preprocessingIsRunning) {
        this->preprocessingError(10);
        return -1;
    }

    this->cleanInput();
    this->inputParams.path = inputPath;
    this->inputParams.inputLoaded = true;

    QFileInfo input(inputPath);

    if (input.isDir()) {
        this->inputParams.mode = imageDirectory;
        QDir inputImages(inputPath, "*.png *.jpg *.bmp *.tif");
        for (auto i:inputImages.entryInfoList()) {
            this->inputParams.imgNames.push_back(i.baseName());
            this->inputParams.imgOriginals.push_back(cv::imread(i.absoluteFilePath().toStdString(), cv::IMREAD_GRAYSCALE));
        }
    }
    else if (input.isFile()) {
        this->inputParams.mode = imagePath;
        QStringList fileTypes;
        fileTypes << "png" << "jpg" << "bmp" << "tif";
        if (fileTypes.contains(input.completeSuffix().toLower())) {
            this->inputParams.imgOriginal = cv::imread(inputPath.toStdString(), cv::IMREAD_GRAYSCALE);
        }
        else {
            this->inputParams.inputLoaded = false;
            this->preprocessingError(22);
        }
    }
    else {
        this->inputParams.inputLoaded = false;
        this->preprocessingError(21);
    }

    return 1;
}

// PREPROCESSING START

void Preprocessing::start()
{
    if (!this->preprocessingIsRunning) {
        if (this->inputParams.inputLoaded) {

            this->cleanResults();
            this->cleanDurations();

            this->preprocessingIsRunning = true;
            if (this->inputParams.mode == image || this->inputParams.mode == imagePath) {
                this->results.imgOriginal = this->inputParams.imgOriginal;
                this->startProcess(this->inputParams.imgOriginal);
            }
            else if (!this->inputParams.imgOriginals.empty()) {
                this->results.imgOriginal = this->inputParams.imgOriginals[0];
                this->startProcess(this->inputParams.imgOriginals[0]);
            }

        }
        else this->preprocessingError(20);
    }
    else this->preprocessingError(10);
}

void Preprocessing::startProcess(const cv::Mat &imgOriginal)
{
    // MASK WITH NEURAL NETWORK
    if (this->features.useMask) {
        if (!this->maskParams.isModelLoaded) {
            mask.loadMaskModel(this->maskParams.caffeFiles);
            this->maskParams.isModelLoaded = true;
        }

        this->mask.setParams(imgOriginal, this->maskParams);
        this->timer.start();
        this->mask.generate();
        this->durations.mask += this->timer.elapsed();
        this->results.imgMask = mask.getImgMask();
    }

    // QUALITY MAP
    if (this->features.useQualityMap) {
        this->qMap.setParams(imgOriginal, this->qmapParams);

        this->timer.start();
        this->qMap.computeQualityMap();
        this->durations.qualityMap += timer.elapsed();

        this->results.imgQualityMap = this->qMap.getImgQualityMap();
        this->results.qualityMap = this->qMap.getQualityMap();
    }

    // FREQUENCY MAP WITH NEURAL NETWORK
    if (this->features.useFrequencyMap) {
        if (!this->fmapParams.isModelLoaded) {
            this->fMap.loadFrequencyMapModel(this->fmapParams.caffeFiles);
            this->fmapParams.isModelLoaded = true;
        }

        this->fMap.setParams(imgOriginal, this->fmapParams);
        this->timer.start();
        this->fMap.generate();
        this->durations.frequencyMap += this->timer.elapsed();

        this->results.frequencyMap = fMap.getFrequencyMap();
        if (this->features.advancedMode) this->results.imgFrequencyMap = fMap.getImgFrequencyMap();
    }

    // CONTRAST ENHANCEMENT
    if (this->features.useContrastEnhancement) {
        this->contrast.setParams(imgOriginal, this->contrastParams);
        this->timer.start();
        this->contrast.enhance();
        this->durations.contrastEnhancement += this->timer.elapsed();
        this->results.imgContrastEnhanced = this->contrast.getImgContrastEnhanced();

        // ORIENTATION MAP
        //this->oMap.setParams(this->results.imgContrastEnhanced, this->omapParams);
    }
    /*else */this->oMap.setParams(imgOriginal, this->omapParams);

    if (this->general.cpuOnly) {
        this->oMap.computeAdvancedMapCPU(); // Gabor Filter in CPU mode works only with advanced orientation map
        this->results.orientationMap = this->oMap.getOMap_advanced();
    }
    else {
        try {
            this->oMap.computeAdvancedMapGPU();
            if (this->features.useAdvancedOrientationMap) {
                this->orientationMapAF = this->oMap.getOMapAF_advanced();
            }
            else {
                this->orientationMapAF = this->oMap.getOMapAF_basic();
            }
            this->results.orientationMap = this->oMap.getOMap_advanced();
        } catch (const af::exception& e) {
            this->preprocessingError(30);
            qDebug() << "ArrayFire exception: " << e.what();
        }
    }

    this->durations.orientationMap += this->oMap.getDuration();

    if (this->features.advancedMode) {
        this->oMap.drawBasicMap(imgOriginal);
        this->results.imgOrientationMap = this->oMap.getImgOMap_basic();
    }

    if (!this->general.cpuOnly) {
        // GABOR FILTER GPU
        try {
            if (this->features.useContrastEnhancement) this->gaborGPU.setParams(this->results.imgContrastEnhanced, this->gaborParams);
            else this->gaborGPU.setParams(imgOriginal, this->gaborParams);

            if (this->features.useAdvancedOrientationMap) this->gaborGPU.enhanceWithAdvancedOMap();
            else this->gaborGPU.enhanceWithBasicOMap();
            this->durations.gaborFilter += this->gaborGPU.getDuration();
        } catch (const af::exception& e) {
            this->preprocessingError(30);
            qDebug() << "ArrayFire exception: " << e.what();
        }

        this->results.imgEnhanced = this->gaborGPU.getImgEnhanced(); // ziskanie prefiltrovaneho odtlacku
        this->continueAfterGabor();
    }
    else {
        // GABOR FILTER CPU MULTITHREAD
        if (this->features.useContrastEnhancement) this->gaborMultiThread.setParams(this->results.imgContrastEnhanced, this->gaborParams);
        else this->gaborMultiThread.setParams(imgOriginal, this->gaborParams);
        this->timer.start();
        this->gaborMultiThread.enhance(); // filtrovanie so zvolenym typom smerovej mapy
    }

}

void Preprocessing::allGaborThreadsFinished()
{
    // GABOR FILTER CPU MULTITHREAD FINISHED
    this->durations.gaborFilter += this->timer.elapsed();
    this->results.imgEnhanced = this->gaborMultiThread.getImgEnhanced(); // ziskanie prefiltrovaneho odtlacku

    this->continueAfterGabor();
}

void Preprocessing::continueAfterGabor()
{
    // BINARIZATION
    this->binarization.setParams(this->results.imgEnhanced, this->binarizationParams);

    this->timer.start();
    this->binarization.binarizeAdaptive();
    this->durations.binarization += this->timer.elapsed();
    this->results.imgBinarized = this->binarization.getImgBinarized();

    // HOLE REMOVER
    if (this->features.useHoleRemover) {
        this->timer.start();
        this->binarization.removeHoles(this->binarizationParams.holeSize);
        this->durations.removingHoles += this->timer.elapsed();
    }

    // THINNING GUO HALL FAST
    this->timer.start();
    this->thinning.thinGuoHallFast(this->results.imgBinarized, false);
    if (this->features.generateInvertedSceleton) this->thinning.thinGuoHallFast(this->results.imgBinarized, true);
    this->durations.thinning += this->timer.elapsed();
    this->results.imgSkeleton = this->thinning.getImgSkeleton();
    if (this->features.generateInvertedSceleton) this->results.imgSkeletonInverted = this->thinning.getImgSkeletonInverted();

    // EMITS
    if (this->inputParams.mode == image || this->inputParams.mode == imagePath) {

        this->preprocessingIsRunning = false;

        if (this->features.advancedMode) {
            emit preprocessingDoneSignal(this->results);
        }
        else {
            PREPROCESSING_RESULTS basicResults = {this->inputParams.imgOriginal, this->results.imgSkeleton, this->results.imgSkeletonInverted,
                                                  this->results.qualityMap, this->results.orientationMap};
            emit preprocessingDoneSignal(basicResults);
        }
        emit preprocessingDurationSignal(this->durations);

        this->cleanResults();
    }
    // IF WE HAVE TO PROCESS MORE FINGERPRINTS
    else {

        // SAVE RESULTS
        if (this->features.advancedMode) this->allResultsMap.insert(this->inputParams.imgNames[this->inputParams.cnt], this->results);
        else {
            PREPROCESSING_RESULTS basicResults = {this->inputParams.imgOriginals[this->inputParams.cnt], this->results.imgSkeleton, this->results.imgSkeletonInverted,
                                                  this->results.qualityMap, this->results.orientationMap};
            this->resultsMap.insert(this->inputParams.imgNames[this->inputParams.cnt], basicResults);
        }

        // SEND PROGRESS VALUE
        emit this->preprocessingProgressSignal((int)(this->inputParams.cnt*1.0/(this->inputParams.imgOriginals.size()-1)*100));

        // IF EVERYTHING DONE
        if (this->inputParams.cnt == this->inputParams.imgOriginals.size()-1) {
            this->preprocessingIsRunning = false;

            if (this->features.advancedMode) emit preprocessingSequenceDoneSignal(this->allResultsMap);
            else emit preprocessingSequenceDoneSignal(this->resultsMap);

            emit preprocessingDurationSignal(this->durations);

            this->cleanResults();
        }
        // IF NOT START PREPROCESSING FOR NEXT IMAGE
        else {
            this->inputParams.cnt++;
            this->results.imgOriginal = this->inputParams.imgOriginals[this->inputParams.cnt];
            this->startProcess(this->inputParams.imgOriginals[this->inputParams.cnt]);
        }
    }
}

void Preprocessing::preprocessingError(int errorcode)
{
    /* ERROR CODES
     * 10 - Preprocessing is alredy running
     * 20 - No valid input file to process
     * 21 - Input path is not valid
     * 22 - Incompatible file type
     * 30 - ArrayFire Exception
     */

    emit preprocessingErrorSignal(errorcode);
}
