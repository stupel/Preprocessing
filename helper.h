#ifndef HELPER_H
#define HELPER_H

#include "preprocessing_config.h"

#undef max
#undef min

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

	static inline cv::Mat array_uchar2mat_uchar(const af::array &input)
	{
		uchar* data = input.as(u8).T().host<uchar>();
		cv::Mat output = cv::Mat((int)input.dims(0), (int)input.dims(1), CV_8UC1, data).clone();
		af::freeHost(data);

		return output;
	}

	static inline cv::Mat array_float2mat_float(const af::array &input)
	{
		float* data = input.T().as(f32).host<float>();
		cv::Mat output = cv::Mat((int)input.dims(0), (int)input.dims(1), CV_32FC1, data).clone();
		af::freeHost(data);

		return output;
	}

	static inline af::array mat_uchar2array_uchar(const cv::Mat &input)
	{
		cv::Mat helperMat;
		cv::transpose(input,helperMat);
		return af::array(input.rows, input.cols, helperMat.data).as(u8);
	}

	static inline af::array mat_double2array_double(const cv::Mat& mat){

		double *imgdata = new double[mat.rows*mat.cols];
		int global=0;
		for(int i=0;i<mat.cols;i++){
			for(int j=0;j<mat.rows;j++){
				imgdata[global++]=mat.at<double>(j,i);
			}
		}
		af::array arr = af::array(mat.rows,mat.cols,imgdata);
		delete[] imgdata;

		return arr;
	}

	static inline void af_normalizeImage(af::array& in){
		af::array minn =af::tile(af::min(af::min(in)),in.dims(0),in.dims(1));
		af::array maxx =af::tile(af::max(af::max(in)),in.dims(0),in.dims(1));
		in = 255*((in.as(f32)-minn)/(maxx-minn));
	}

	static inline af::array mat_float2array_float(const cv::Mat& mat){

		float *imgdata = new float[mat.rows*mat.cols];
		int global=0;
		for(int i=0;i<mat.cols;i++){
			for(int j=0;j<mat.rows;j++){
				imgdata[global++]=mat.at<float>(j,i);
			}
		}
		af::array arr = af::array(mat.rows,mat.cols,imgdata);
		delete[] imgdata;

		return arr;
	}

	static inline af::array mat_uchar2array_float(const cv::Mat& mat){

		float *imgdata = new float[mat.rows*mat.cols];
		int global=0;
		for(int i=0;i<mat.cols;i++){
			for(int j=0;j<mat.rows;j++){
				imgdata[global++]=mat.at<uchar>(j,i);
			}
		}
		af::array arr = af::array(mat.rows,mat.cols,imgdata);
		delete[] imgdata;

		return arr;
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
