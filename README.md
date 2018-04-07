# Preprocessing
Fingerprint preprocessing module for DBOX

**Getting Started:**  
Build and run the project to generate .so (.dll / .lib) files  
Include the library and header files to your own application  
Copy the 'core' folder to your root project directory  
  
  
**APIs:**  
*void loadImg (cv::Mat img);*  
  
*void run();*  
  
  
Optional:  
*void PsetPreprocessingParams(int blockSize, double gaborLambda, double gaborSigma, int gaussBlockBasic, double gaussSigmaBasic, int gaussBlockAdvanced, double gaussSigmaAdvanced, int holeSize)*  
  
*void setFeatures(bool advancedMode, int numThreads, bool useContrastEnhancement, bool useRemoveHoles, bool useFixOrientations, bool useMask, bool useQualityMap, bool useFrequencyMap)*  
  
*void setMaskParams(CAFFE_FILES maskFiles, int blockSize, int exBlockSize, bool useSmooth)*  
  
*void setFrequencyMapParams(CAFFE_FILES freqFiles, int blockSize, int exBlockSize)*  
  
  
**SIGNALS:**  
*preprocessingAdvancedDoneSignal(PREPROCESSING_ALL_RESULTS results);*  
  
*preprocessingDoneSignal(PREPROCESSING_RESULTS results);*  
  
*preprocessingDurrationSignal(PREPROCESSING_DURATIONS durations);*  
  
*preprocessingErrorSignal(int errorcode);*  
