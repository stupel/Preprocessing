#ifndef HELPER_H
#define HELPER_H

// OpenCV
#include "opencv/cv.hpp"

// ArrayFire
#include "arrayfire.h"
#include "af/macros.h"
#undef max
#undef min

// Qt
#include <QImage>

class Helper
{

public:

    static inline QImage Mat2QImage(cv::Mat const &mat,
                             QImage::Format format)
    {
        return QImage(mat.data, mat.cols, mat.rows,
                      mat.step, format).copy();
    }

   static inline  cv::Mat QImage2Mat(QImage const &img, int format)
    {
        return cv::Mat(img.height(), img.width(), format,
                       const_cast<uchar*>(img.bits()),
                       img.bytesPerLine()).clone();
    }

    static inline void mat2array(const cv::Mat& mat, af::array& arr){

        float *imgdata = new float[mat.rows*mat.cols];
        int global=0;
        for(int i=0;i<mat.cols;i++){
            for(int j=0;j<mat.rows;j++){
                imgdata[global++]=mat.at<uchar>(j,i);
            }
        }
        arr = af::array(mat.rows,mat.cols,imgdata);
        delete[] imgdata;
    }
    static inline void mat2arraydouble(const cv::Mat& mat, af::array& arr){

        double *imgdata = new double[mat.rows*mat.cols];
        int global=0;
        for(int i=0;i<mat.cols;i++){
            for(int j=0;j<mat.rows;j++){
                imgdata[global++]=mat.at<double>(j,i);
            }
        }
        arr = af::array(mat.rows,mat.cols,imgdata);
        delete[] imgdata;
    }

    static inline void af_normalizeImage(af::array& in){
            af::array minn =af::tile(af::min(af::min(in)),in.dims(0),in.dims(1));
            af::array maxx =af::tile(af::max(af::max(in)),in.dims(0),in.dims(1));
            in = 255*((in.as(f32)-minn)/(maxx-minn));
    }


    static inline void mat2array_float(const cv::Mat& mat, af::array& arr){

        float *imgdata = new float[mat.rows*mat.cols];
        int global=0;
        for(int i=0;i<mat.cols;i++){
            for(int j=0;j<mat.rows;j++){
                imgdata[global++]=mat.at<uchar>(j,i);
            }
        }
        arr = af::array(mat.rows,mat.cols,imgdata);
        delete[] imgdata;
    }

    static inline void array2mat(af::array& arr, cv::Mat& mat){

        float *imgdata = arr.host<float>();
        int ii=0,jj=0;
        int total = arr.dims(0)*arr.dims(1);

        for(int i=0;i<(arr.dims(0));i++){
            jj=0;
            for(int j=i; j<total;j+=arr.dims(0)){
                mat.at<uchar>(ii,jj) = imgdata[j];
                jj++;
            }
            ii++;
        }
    }

    static inline QByteArray IntToQByteArray(int x) {
        QByteArray ba;
        return ba.setNum(x);
    }

    static inline QByteArray QStringToQByteArray(const QString &x) {
        return x.toUtf8();
    }
};

#endif // HELPER_H
