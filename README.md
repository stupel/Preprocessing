# Preprocessing
Fingerprint preprocessing module for DBOX

**Dependencies:**  
- [OpenCV 3.4.1](https://github.com/opencv/opencv)  
- [Caffe 1.0](https://github.com/BVLC/caffe)  
- [ArrayFire 3.5.1](https://arrayfire.com/download/)  
- [CUDA 9.1 (minimum 8.0)](https://developer.nvidia.com/cuda-downloads) 
- [cuDNN 7.1](https://developer.nvidia.com/rdp/cudnn-download)
- [Qt5 / Qt Creator 4](https://www.qt.io/download)  

*The mentioned or newer versions are recommended*  
  
**Getting Started:**  
1. You need to provide valid paths to these libraries and their header files in ```.pro``` file.
2. Build and run the project to generate .so (.dll / .lib) files  
3. Include the library and header files to your own application  
4. Copy the 'core' folder to your root project directory  
  
<br />  

**APIs:**  
```cpp
void loadImg(cv::Mat imgOriginal);
  
void start();  
```
  
Optional:  
```cpp
void setPreprocessingParams(int blockSize = 13, double gaborLambda = 9, double gaborSigma = 3, int gaussBlockBasic = 1, double gaussSigmaBasic = 1.0, int gaussBlockAdvanced = 121, double gaussSigmaAdvanced = 10.0, int holeSize = 20);  
  
void setFeatures(bool useAdvancedMode, bool useContrastEnhancement = true, bool useAdvancedOrientationMap = true, bool useHoleRemover = true, bool generateInvertedSceleton = true, bool useQualityMap = true, bool useMask = false, bool useFrequencyMap = false);  

void setCPUOnly(bool enabled, int numThreads = 0);  
  
void setMaskParams(CAFFE_FILES maskFiles, int blockSize, int exBlockSize, bool useSmooth);  
  
void setFrequencyMapParams(CAFFE_FILES freqFiles, int blockSize, int exBlockSize);  
```
Important notice:  
```numThreads = 0``` means automatic ideal thread number  
  
<br />  

**SIGNALS:**
```cpp
preprocessingAdvancedDoneSignal(PREPROCESSING_ALL_RESULTS results);  
  
preprocessingDoneSignal(PREPROCESSING_RESULTS results);  
  
preprocessingDurationSignal(PREPROCESSING_DURATIONS durations);  
  
preprocessingErrorSignal(int errorcode);  
```  
Important notice:  
To get ```preprocessingAdvancedDoneSignal``` the *advancedMode* has to be enabled  
To get ```preprocessingDoneSignal``` the *advancedMode* has to be disabled  
