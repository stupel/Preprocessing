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

## API  
**Required**
```cpp
int loadInput(cv::Mat imgOriginal);
int loadInput(QVector<cv::Mat> imgOriginals);
int loadInput(QString inputPath);
  
void start();  
```
Usage:  
1. First you have to load input, which can be:
- cv::Mat image
- QVector of cv::Mat images
- Image input path for a .jpg, .png, .bmp, .tif file
- Image input directory  
2. Just call ```start()```  
  
<br />  
  
**Optional**  
```cpp
void setPreprocessingParams(int blockSize = 13, double gaborLambda = 9, double gaborSigma = 3, int gaussBlockBasic = 1, double gaussSigmaBasic = 1.0, int gaussBlockAdvanced = 121, double gaussSigmaAdvanced = 10.0, int holeSize = 20);  
  
void setFeatures(bool useAdvancedMode, bool useContrastEnhancement = true, bool useAdvancedOrientationMap = true, bool useHoleRemover = true, bool generateInvertedSceleton = true, bool useQualityMap = true, bool useMask = false, bool useFrequencyMap = false);  

void setCPUOnly(bool enabled, int numThreads = 0);  
  
void setMaskParams(CAFFE_FILES maskFiles, int blockSize, int exBlockSize, bool useSmooth);  
  
void setFrequencyMapParams(CAFFE_FILES freqFiles, int blockSize, int exBlockSize);  
```
Usage:  
- With ```setPreprocessingParams(...)``` you can set the must important parameters required for image processing, the default parameters are recommended for 500dpi images
- With ```setFeatures(...)``` you can set some optional features which can affect the output image quality, the *advanceMode* gives you all the intermediate results generated during preprocessing
- With ```setCPUOnly(...)``` you can force the library to use only CPU for image processing (notice: it can be slower than GPU) and set the number of CPU threads you want to use for image processing in defined phases (notice: ```numThreads = 0``` means automatic ideal thread number)
- With ```setMaskParams(...)``` and ```setFrequencyMapParams(...)``` you can set the Caffe model files and parameteres required for classification with neural network to generate mask or frequency map for the input image  
  
<br />  

**SIGNALS:**
```cpp
preprocessingAdvancedDoneSignal(PREPROCESSING_ALL_RESULTS results);  
  
preprocessingDoneSignal(PREPROCESSING_RESULTS results);  
  
preprocessingSequenceAdvancedDoneSignal(QMap<QString, PREPROCESSING_ALL_RESULTS> results);  
  
preprocessingSequenceDoneSignal(QMap<QString, PREPROCESSING_RESULTS> results);
  
preprocessingDurationSignal(PREPROCESSING_DURATIONS durations);  
  
preprocessingErrorSignal(int errorcode);  
```  
Important notice:  
- If you load an image or an image path
  - You get ```preprocessingAdvancedDoneSignal``` if the *advancedMode* is enabled  
  - You get ```preprocessingDoneSignal``` if the *advancedMode* is disabled  
- If you load a vector with images or an input directory  
  - You get ```preprocessingSequenceAdvancedDoneSignal``` if the *advancedMode* is enabled  
  - You get ```preprocessingSequenceDoneSignal``` if *advancedMode* is disabled  
- You get ```preprocessingDurationSignal``` with duration values in ms for each phase during preprocessing if it's finished successfully
- You get ```preprocessingErrorSignal``` with the error code if an error occured during preprocessing
  
<br />  
<br />  
  
**A simple example how to use signals in your application**  
*yourclass.h:*
```cpp  
#include "preprocessing.h"

class YourClass: public QObject
{
    Q_OBJECT  
  
private:  
    Preprocessing p;  
    
private slots:
    void proprocessingResultsSlot(PREPROCESSING_RESULTS result);
}
```

*yourclass.cpp:*
```cpp 
#include "yourclass.h"
YourClass::YourClass()
{
    qRegisterMetaType<PREPROCESSING_RESULTS >("PREPROCESSING_RESULTS");
    connect(&p, SIGNAL(preprocessingDoneSignal(PREPROCESSING_RESULTS)), this, SLOT(proprocessingResultsSlot(PREPROCESSING_RESULTS)));
}

void YourClass::proprocessingResultsSlot(PREPROCESSING_RESULTS result)
{
    cv::imshow("Fingerprint Skeleton", result.imgSkeleton);
}
```
For more please visit [Qt Signals & Slots](http://doc.qt.io/archives/qt-4.8/signalsandslots.html).
