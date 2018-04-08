# Preprocessing
Fingerprint preprocessing module for DBOX

**Dependencies:**  
- OpenCV 3.4.1 (https://github.com/opencv/opencv)  
- Caffe 1.0 (https://github.com/BVLC/caffe)  
- ArrayFire 3.5.1 (https://arrayfire.com/download/)  
- CUDA 9.1 (https://developer.nvidia.com/cuda-downloads)  

*The mentioned versions are recommended*  
  
**Getting Started:**  
1. Build and run the project to generate .so (.dll / .lib) files  
2. Include the library and header files to your own application  
3. Copy the 'core' folder to your root project directory  
  
  
**APIs:**  
```cpp
void loadImg (cv::Mat img);  
  
void run();  
```
  
Optional:  
```cpp
void setPreprocessingParams(int blockSize, double gaborLambda, double gaborSigma, int gaussBlockBasic, double gaussSigmaBasic, int gaussBlockAdvanced, double gaussSigmaAdvanced, int holeSize);  
  
void setFeatures(bool advancedMode, int numThreads, bool useGaborFilterGPU, bool useContrastEnhancement, bool useRemoveHoles, bool useFixOrientations, bool useMask, bool useQualityMap, bool useFrequencyMap);  
  
void setMaskParams(CAFFE_FILES maskFiles, int blockSize, int exBlockSize, bool useSmooth);  
  
void setFrequencyMapParams(CAFFE_FILES freqFiles, int blockSize, int exBlockSize);  
```
  
**SIGNALS:**
```cpp
preprocessingAdvancedDoneSignal(PREPROCESSING_ALL_RESULTS results);  
  
preprocessingDoneSignal(PREPROCESSING_RESULTS results);  
  
preprocessingDurrationSignal(PREPROCESSING_DURATIONS durations);  
  
preprocessingErrorSignal(int errorcode);  
```
