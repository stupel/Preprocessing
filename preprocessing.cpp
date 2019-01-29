#include "preprocessing.h"

Preprocessing::Preprocessing()
{
	m_preprocessingIsRunning = false;

	// INPUT PARAMS
	m_inputParams.mode = image;
	m_inputParams.cnt = 0;
	m_inputParams.inputLoaded = false;

	// PREPROCESSING FEATURES
	m_features.advancedMode = false;
	m_features.useContrastEnhancement = true;
	m_features.useQualityMap = true;
	m_features.useHoleRemover = true;
	m_features.useAdvancedOrientationMap = true;
	m_features.generateInvertedSceleton = true;
	m_features.useMask = false;
	m_features.useFrequencyMap = false;

	// GENERAL PARAMS
	m_general.cpuOnly = false;
	m_general.threadNum = QThread::idealThreadCount();

	// CONTRAST ENHANCEMENT PARAMS
	m_contrastParams.distance = 10;
	m_contrastParams.sigma = 30;
	m_contrastParams.gaussBlock = 3;
	m_contrastParams.gaussSigma = 2;

	// ORIENTATION MAP PARAMS
	m_omapParams.blockSize = 13;
	m_omapParams.gaussBlurBasic.blockSize = 1;
	m_omapParams.gaussBlurBasic.sigma = 1.0;
	m_omapParams.gaussBlurAdvanced.blockSize = 121;
	m_omapParams.gaussBlurAdvanced.sigma = 10.0;

	// GABOR FILTER PARAMS
	m_gaborParams.blockSize = 13;
	m_gaborParams.lambda = 9;
	m_gaborParams.sigma = 3;
	m_gaborParams.gamma = 1;
	m_gaborParams.psi = 0;
	m_gaborParams.oMap = &m_results.orientationMap;
	m_gaborParams.oMapAF = &m_orientationMapAF;
	m_gaborParams.threadNum = &m_general.threadNum;
	m_gaborParams.useFrequencyMap = &m_features.useFrequencyMap;
	m_gaborParams.fMap = &m_results.frequencyMap;

	// BINARIZATION PARAMS
	m_binarizationParams.useMask = &m_features.useMask;
	m_binarizationParams.imgMask = &m_results.imgMask;
	m_binarizationParams.useQualityMap = &m_features.useQualityMap;
	m_binarizationParams.imgQualityMap = &m_results.imgQualityMap;
	m_binarizationParams.holeSize = 20;

	// QUALITY MAP
	m_qmapParams.ppi = 500;

	// MASK PARAMS
	m_maskParams.caffeFiles.model = "./core/config/Caffe/mask_deploy.prototxt";
	m_maskParams.caffeFiles.trained = "./core/config/Caffe/mask.caffemodel";
	m_maskParams.caffeFiles.imageMean = "./core/config/Caffe/mask_imagemean.binaryproto";
	m_maskParams.caffeFiles.label = "./core/config/Caffe/mask_labels.txt";
	m_maskParams.blockSize = 8;
	m_maskParams.exBlockSize = 19;
	m_maskParams.useSmooth = false;
	m_maskParams.isModelLoaded = false;
	m_maskParams.cpuOnly = &m_general.cpuOnly;

	// FREQUENCY MAP PARAMS
	m_fmapParams.caffeFiles.model = "./core/config/Caffe/frequency_deploy.prototxt";
	m_fmapParams.caffeFiles.trained = "./core/config/Caffe/frequency.caffemodel";
	m_fmapParams.caffeFiles.imageMean = "./core/config/Caffe/frequency_imagemean.binaryproto";
	m_fmapParams.caffeFiles.label = "./core/config/Caffe/frequency_labels.txt";
	m_fmapParams.blockSize = 8;
	m_fmapParams.exBlockSize = 30;
	m_fmapParams.isModelLoaded = false;
	m_fmapParams.cpuOnly = &m_general.cpuOnly;

	//CONNECTS
	connect(&m_gaborMultiThread, SIGNAL(gaborThreadsFinished()), this, SLOT(allGaborThreadsFinished()));
}

int Preprocessing::setFrequencyMapParams(CAFFE_FILES freqFiles, int blockSize, int exBlockSize)
{
	if (m_preprocessingIsRunning) {
		preprocessingError(10);
		return -1;
	}

	if (!((m_fmapParams.caffeFiles.model == freqFiles.model) && (m_fmapParams.caffeFiles.trained == freqFiles.trained) &&
		  (m_fmapParams.caffeFiles.imageMean == freqFiles.imageMean) && (m_fmapParams.caffeFiles.label == freqFiles.label)) )
	{
		m_fmapParams.caffeFiles.model = freqFiles.model;
		m_fmapParams.caffeFiles.trained = freqFiles.trained;
		m_fmapParams.caffeFiles.imageMean = freqFiles.imageMean;
		m_fmapParams.caffeFiles.label = freqFiles.label;
	}

	m_fmapParams.blockSize = blockSize;
	m_fmapParams.exBlockSize = exBlockSize;

	m_fmapParams.isModelLoaded = false;

	return 1;
}

int Preprocessing::setMaskParams(CAFFE_FILES maskFiles, int blockSize, int exBlockSize, bool useSmooth)
{
	if (m_preprocessingIsRunning) {
		preprocessingError(10);
		return -1;
	}

	if (!((m_maskParams.caffeFiles.model == maskFiles.model) && (m_maskParams.caffeFiles.trained == maskFiles.trained) &&
		  (m_maskParams.caffeFiles.imageMean == maskFiles.imageMean) && (m_maskParams.caffeFiles.label == maskFiles.label)) )
	{
		m_maskParams.caffeFiles.model = maskFiles.model;
		m_maskParams.caffeFiles.trained = maskFiles.trained;
		m_maskParams.caffeFiles.imageMean = maskFiles.imageMean;
		m_maskParams.caffeFiles.label = maskFiles.label;
	}

	m_maskParams.blockSize = blockSize;
	m_maskParams.exBlockSize = exBlockSize;
	m_maskParams.useSmooth = useSmooth;

	m_maskParams.isModelLoaded = false;

	return 1;
}

int Preprocessing::setPreprocessingParams(int blockSize, double gaborLambda, double gaborSigma, int gaussBlockBasic, double gaussSigmaBasic, int gaussBlockAdvanced, double gaussSigmaAdvanced, int holeSize)
{
	if (m_preprocessingIsRunning) {
		preprocessingError(10);
		return -1;
	}

	m_gaborParams.blockSize = blockSize;
	m_omapParams.blockSize = blockSize;
	m_gaborParams.lambda = gaborLambda;
	m_gaborParams.sigma = gaborSigma;
	m_omapParams.gaussBlurBasic = GAUSSIAN_BLUR_SETTINGS{gaussBlockBasic, gaussSigmaBasic};
	m_omapParams.gaussBlurAdvanced = GAUSSIAN_BLUR_SETTINGS{gaussBlockAdvanced, gaussSigmaAdvanced};
	m_binarizationParams.holeSize = holeSize;

	return 1;
}

int Preprocessing::setFeatures(bool useAdvancedMode, bool useContrastEnhancement, bool useAdvancedOrientationMap, bool useHoleRemover, bool generateInvertedSkeleton, bool useQualityMap, bool useMask, bool useFrequencyMap)
{
	if (m_preprocessingIsRunning) {
		preprocessingError(10);
		return -1;
	}

	m_features.advancedMode = useAdvancedMode;
	m_features.useContrastEnhancement = useContrastEnhancement;
	m_features.useAdvancedOrientationMap = useAdvancedOrientationMap;
	m_features.useQualityMap = useQualityMap;
	m_features.useHoleRemover = useHoleRemover;
	m_features.generateInvertedSceleton = generateInvertedSkeleton;
	m_features.useMask = useMask;
	m_features.useFrequencyMap = useFrequencyMap;

	return 1;
}

int Preprocessing::setCPUOnly(bool enabled, int threadNum)
{
	if (m_preprocessingIsRunning) {
		preprocessingError(10);
		return -1;
	}

	if (enabled) m_general.cpuOnly = true;
	else m_general.cpuOnly = false;

	if (threadNum == 0) m_general.threadNum = QThread::idealThreadCount();
	else m_general.threadNum = threadNum;

	return 1;
}

void Preprocessing::cleanResults()
{
	m_results.imgBinarized.release();
	m_results.imgContrastEnhanced.release();
	m_results.imgEnhanced.release();
	m_results.imgFrequencyMap.release();
	m_results.imgMask.release();
	m_results.imgOrientationMap.release();
	m_results.imgQualityMap.release();
	m_results.imgSkeleton.release();
	m_results.imgSkeletonInverted.release();
	m_results.frequencyMap.release();
	m_results.orientationMap.release();
	m_results.qualityMap.release();

	m_resultsMap.clear();
	m_allResultsMap.clear();
}

void Preprocessing::cleanInput()
{
	m_inputParams.inputLoaded = false;
	m_inputParams.imgOriginal.release();
	m_inputParams.imgOriginals.clear();
	m_inputParams.imgNames.clear();
	m_inputParams.path = "";
	m_inputParams.cnt = 0;
}

void Preprocessing::cleanDurations()
{
	m_durations.binarization = 0;
	m_durations.contrastEnhancement = 0;
	m_durations.frequencyMap = 0;
	m_durations.gaborFilter = 0;
	m_durations.mask = 0;
	m_durations.orientationMap = 0;
	m_durations.qualityMap = 0;
	m_durations.removingHoles = 0;
	m_durations.thinning = 0;
}

// INPUT

int Preprocessing::loadInput(cv::Mat imgOriginal)
{
	if (m_preprocessingIsRunning) {
		preprocessingError(10);
		return -1;
	}

	cleanInput();
	m_inputParams.imgOriginal = imgOriginal.clone();
	if (m_inputParams.imgOriginal.channels() != 1) cv::cvtColor(m_inputParams.imgOriginal, m_inputParams.imgOriginal, cv::COLOR_BGR2GRAY);
	m_inputParams.inputLoaded = true;
	m_inputParams.mode = image;

	return 1;
}

int Preprocessing::loadInput(QVector<cv::Mat> imgOriginals)
{
	if (m_preprocessingIsRunning) {
		preprocessingError(10);
		return -1;
	}

	cleanInput();
	m_inputParams.imgOriginals = imgOriginals;
	for (int i = 0; i < imgOriginals.size(); i++) {
		if (m_inputParams.imgOriginals[i].channels() != 1) cv::cvtColor(m_inputParams.imgOriginals[i], m_inputParams.imgOriginals[i], cv::COLOR_BGR2GRAY);
		m_inputParams.imgNames.push_back(QString::number(i+1));
	}
	m_inputParams.inputLoaded = true;
	m_inputParams.mode = images;

	return 1;
}

int Preprocessing::loadInput(QString inputPath)
{
	if (m_preprocessingIsRunning) {
		preprocessingError(10);
		return -1;
	}

	cleanInput();
	m_inputParams.path = inputPath;
	m_inputParams.inputLoaded = true;

	QFileInfo input(inputPath);

	if (input.isDir()) {
		m_inputParams.mode = imageDirectory;
		QDir inputImages(inputPath, "*.png *.jpg *.bmp *.tif");
		for (auto i:inputImages.entryInfoList()) {
			m_inputParams.imgNames.push_back(i.baseName());
			m_inputParams.imgOriginals.push_back(cv::imread(i.absoluteFilePath().toStdString(), cv::IMREAD_GRAYSCALE));
		}
	}
	else if (input.isFile()) {
		m_inputParams.mode = imagePath;
		QStringList fileTypes;
		fileTypes << "png" << "jpg" << "bmp" << "tif";
		if (fileTypes.contains(input.completeSuffix().toLower())) {
			m_inputParams.imgOriginal = cv::imread(inputPath.toStdString(), cv::IMREAD_GRAYSCALE);
		}
		else {
			m_inputParams.inputLoaded = false;
			preprocessingError(22);
		}
	}
	else {
		m_inputParams.inputLoaded = false;
		preprocessingError(21);
	}

	return 1;
}

// PREPROCESSING START

void Preprocessing::start()
{
	if (!m_preprocessingIsRunning) {
		if (m_inputParams.inputLoaded) {

			cleanResults();
			cleanDurations();

			m_preprocessingIsRunning = true;
			if (m_inputParams.mode == image || m_inputParams.mode == imagePath) {
				m_results.imgOriginal = m_inputParams.imgOriginal;
				startProcess(m_inputParams.imgOriginal);
			}
			else if (!m_inputParams.imgOriginals.empty()) {
				m_results.imgOriginal = m_inputParams.imgOriginals[0];
				startProcess(m_inputParams.imgOriginals[0]);
			}

		}
		else preprocessingError(20);
	}
	else preprocessingError(10);
}

void Preprocessing::startProcess(const cv::Mat &imgOriginal)
{
	// MASK WITH NEURAL NETWORK
	if (m_features.useMask) {
		if (!m_maskParams.isModelLoaded) {
			m_mask.loadMaskModel(m_maskParams.caffeFiles);
			m_maskParams.isModelLoaded = true;
		}

		m_mask.setParams(imgOriginal, m_maskParams);
		m_timer.start();
		m_mask.generate();
		m_durations.mask += m_timer.elapsed();
		m_results.imgMask = m_mask.getImgMask();
	}

	// QUALITY MAP
	if (m_features.useQualityMap) {
		m_qMap.setParams(imgOriginal, m_qmapParams);

		m_timer.start();
		m_qMap.computeQualityMap();
		m_durations.qualityMap += m_timer.elapsed();

		m_results.imgQualityMap = m_qMap.getImgQualityMap();
		m_results.qualityMap = m_qMap.getQualityMap();
	}

	// FREQUENCY MAP WITH NEURAL NETWORK
	if (m_features.useFrequencyMap) {
		if (!m_fmapParams.isModelLoaded) {
			m_fMap.loadFrequencyMapModel(m_fmapParams.caffeFiles);
			m_fmapParams.isModelLoaded = true;
		}

		m_fMap.setParams(imgOriginal, m_fmapParams);
		m_timer.start();
		m_fMap.generate();
		m_durations.frequencyMap += m_timer.elapsed();

		m_results.frequencyMap = m_fMap.getFrequencyMap();
		if (m_features.advancedMode) m_results.imgFrequencyMap = m_fMap.getImgFrequencyMap();
	}

	// CONTRAST ENHANCEMENT
	if (m_features.useContrastEnhancement) {
		m_contrast.setParams(imgOriginal, m_contrastParams);
		m_timer.start();
		m_contrast.enhance();
		m_durations.contrastEnhancement += m_timer.elapsed();
		m_results.imgContrastEnhanced = m_contrast.getImgContrastEnhanced();

		// ORIENTATION MAP
		//oMap.setParams(results.imgContrastEnhanced, omapParams);
	}
	/*else */m_oMap.setParams(imgOriginal, m_omapParams);

	if (m_general.cpuOnly) {
		m_oMap.computeAdvancedMapCPU(); // Gabor Filter in CPU mode works only with advanced orientation map
		m_results.orientationMap = m_oMap.getOMap_advanced();
	}
	else {
		try {
			m_oMap.computeAdvancedMapGPU();
			if (m_features.useAdvancedOrientationMap) {
				m_orientationMapAF = m_oMap.getOMapAF_advanced();
			}
			else {
				m_orientationMapAF = m_oMap.getOMapAF_basic();
			}
			m_results.orientationMap = m_oMap.getOMap_advanced();
		} catch (const af::exception& e) {
			preprocessingError(30);
			qDebug() << "ArrayFire exception: " << e.what();
		}
	}

	m_durations.orientationMap += m_oMap.getDuration();

	if (m_features.advancedMode) {
		m_oMap.drawBasicMap(imgOriginal);
		m_results.imgOrientationMap = m_oMap.getImgOMap_basic();
	}

	if (!m_general.cpuOnly) {
		// GABOR FILTER GPU
		try {
			if (m_features.useContrastEnhancement) m_gaborGPU.setParams(m_results.imgContrastEnhanced, m_gaborParams);
			else m_gaborGPU.setParams(imgOriginal, m_gaborParams);

			if (m_features.useAdvancedOrientationMap) m_gaborGPU.enhanceWithAdvancedOMap();
			else m_gaborGPU.enhanceWithBasicOMap();
			m_durations.gaborFilter += m_gaborGPU.getDuration();
		} catch (const af::exception& e) {
			preprocessingError(30);
			qDebug() << "ArrayFire exception: " << e.what();
		}

		m_results.imgEnhanced = m_gaborGPU.getImgEnhanced(); // ziskanie prefiltrovaneho odtlacku
		continueAfterGabor();
	}
	else {
		// GABOR FILTER CPU MULTITHREAD
		if (m_features.useContrastEnhancement) m_gaborMultiThread.setParams(m_results.imgContrastEnhanced, m_gaborParams);
		else m_gaborMultiThread.setParams(imgOriginal, m_gaborParams);
		m_timer.start();
		m_gaborMultiThread.enhance(); // filtrovanie so zvolenym typom smerovej mapy
	}

}

void Preprocessing::allGaborThreadsFinished()
{
	// GABOR FILTER CPU MULTITHREAD FINISHED
	m_durations.gaborFilter += m_timer.elapsed();
	m_results.imgEnhanced = m_gaborMultiThread.getImgEnhanced(); // ziskanie prefiltrovaneho odtlacku

	continueAfterGabor();
}

void Preprocessing::continueAfterGabor()
{
	// BINARIZATION
	m_binarization.setParams(m_results.imgEnhanced, m_binarizationParams);

	m_timer.start();
	m_binarization.binarizeAdaptive();
	m_durations.binarization += m_timer.elapsed();
	m_results.imgBinarized = m_binarization.getImgBinarized();

	// HOLE REMOVER
	if (m_features.useHoleRemover) {
		m_timer.start();
		m_binarization.removeHoles(m_binarizationParams.holeSize);
		m_durations.removingHoles += m_timer.elapsed();
	}

	// THINNING GUO HALL FAST
	m_timer.start();
	m_thinning.thinGuoHallFast(m_results.imgBinarized, false);
	if (m_features.generateInvertedSceleton) m_thinning.thinGuoHallFast(m_results.imgBinarized, true);
	m_durations.thinning += m_timer.elapsed();
	m_results.imgSkeleton = m_thinning.getImgSkeleton();
	if (m_features.generateInvertedSceleton) m_results.imgSkeletonInverted = m_thinning.getImgSkeletonInverted();

	// EMITS
	if (m_inputParams.mode == image || m_inputParams.mode == imagePath) {

		m_preprocessingIsRunning = false;

		if (m_features.advancedMode) {
			emit preprocessingDoneSignal(m_results);
		}
		else {
			PREPROCESSING_RESULTS basicResults = {m_inputParams.imgOriginal, m_results.imgSkeleton, m_results.imgSkeletonInverted,
												  m_results.qualityMap, m_results.orientationMap};
			emit preprocessingDoneSignal(basicResults);
		}
		emit preprocessingDurationSignal(m_durations);

		cleanResults();
	}
	// IF WE HAVE TO PROCESS MORE FINGERPRINTS
	else {

		// SAVE RESULTS
		if (m_features.advancedMode) m_allResultsMap.insert(m_inputParams.imgNames[m_inputParams.cnt], m_results);
		else {
			PREPROCESSING_RESULTS basicResults = {m_inputParams.imgOriginals[m_inputParams.cnt], m_results.imgSkeleton, m_results.imgSkeletonInverted,
												  m_results.qualityMap, m_results.orientationMap};
			m_resultsMap.insert(m_inputParams.imgNames[m_inputParams.cnt], basicResults);
		}

		// SEND PROGRESS VALUE
		emit preprocessingProgressSignal((int)(m_inputParams.cnt*1.0/(m_inputParams.imgOriginals.size()-1)*100));

		// IF EVERYTHING DONE
		if (m_inputParams.cnt == m_inputParams.imgOriginals.size()-1) {
			m_preprocessingIsRunning = false;

			if (m_features.advancedMode) emit preprocessingSequenceDoneSignal(m_allResultsMap);
			else emit preprocessingSequenceDoneSignal(m_resultsMap);

			emit preprocessingDurationSignal(m_durations);

			cleanResults();
		}
		// IF NOT START PREPROCESSING FOR NEXT IMAGE
		else {
			m_inputParams.cnt++;
			m_results.imgOriginal = m_inputParams.imgOriginals[m_inputParams.cnt];
			startProcess(m_inputParams.imgOriginals[m_inputParams.cnt]);
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
