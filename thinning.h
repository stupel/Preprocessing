#ifndef THINNING_H
#define THINNING_H

#include "preprocessing_config.h"

#include "imagecontour.h"

typedef bool (*VoronoiFn)(uchar*  skeldata, int iter, int col, int row, int cols);

class Thinning : public QObject
{
	Q_OBJECT

public:
	explicit Thinning(QObject *parent = nullptr);

	static const int NOLIMIT = INT_MAX;

	bool thinGuoHallFast(const cv::Mat1b& img, bool inverted, bool crop_img_before = false, int max_iters = NOLIMIT);

	cv::Mat getImgSkeleton() const;
	cv::Mat getImgSkeletonInverted() const;

private:
	bool thin_fast_custom_voronoi_fn(const cv::Mat1b& img, bool inverted, VoronoiFn voronoi_fn, bool crop_img_before = true, int max_iters = NOLIMIT);
	cv::Rect copy_bounding_box_plusone(const cv::Mat1b& img, cv::Mat1b& out, bool crop_img_before = true);
	template<class _T> cv::Rect boundingBox(const cv::Mat_<_T> & img);
	cv::Mat invertColor(const cv::Mat &img);

	cv::Mat thinningGuoHallIteration(cv::Mat imgThin, int iter);

private:
	cv::Mat1b m_imgSkeleton;
	cv::Mat1b m_imgSkeletonInverted;

	cv::Rect m_bbox;
	ImageContour m_skelcontour;

	std::deque<int> m_colsToSet;
	std::deque<int> m_rowsToSet;
	bool m_has_converged;
};

#endif // THINNING_H
