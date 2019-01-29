#include "thinning.h"

Thinning::Thinning(QObject *parent) : QObject(parent)
{

}

static inline bool need_set_guo_hall(uchar*  skeldata, int iter, int col, int row, int cols)
{
	bool p2 = skeldata[(row-1) * cols + col],
			p3 = skeldata[(row-1) * cols + col+1],
			p4 = skeldata[row     * cols + col+1],
			p5 = skeldata[(row+1) * cols + col+1],
			p6 = skeldata[(row+1) * cols + col],
			p7 = skeldata[(row+1) * cols + col-1],
			p8 = skeldata[row     * cols + col-1],
			p9 = skeldata[(row-1) * cols + col-1];

	int C = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
			(!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
	int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
	int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
	int N  = N1 < N2 ? N1 : N2;
	int m  = iter == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

	return (C == 1 && (N >= 2 && N <= 3) && m == 0);
}

cv::Mat Thinning::invertColor(const cv::Mat &img)
{
	cv::Mat imgInverted;

	cv::bitwise_not(img, imgInverted);

	return imgInverted;
}

template<class _T> cv::Rect Thinning::boundingBox(const cv::Mat_<_T> & img) {
	assert(img.isContinuous());
	int xMin = 0, yMin = 0, xMax = 0, yMax = 0;
	bool was_init = false;
	const _T* img_it = img.ptr(0);
	int nrows = img.rows, ncols = img.cols;
	for (int y = 0; y < nrows; ++y) {
		for (int x = 0; x < ncols; ++x) {
			if (*img_it++) {
				if (!was_init) {
					xMin = xMax = x;
					yMin = yMax = y;
					was_init = true;
					continue;
				}
				if (x < xMin)
					xMin = x;
				else if (x > xMax)
					xMax = x;

				if (y < yMin)
					yMin = y;
				else if (y > yMax)
					yMax = y;
			}
		}
	}

	if (!was_init) // no white point found
		return cv::Rect(-1, -1, -1, -1);

	// from http://docs.opencv.org/java/org/opencv/core/Rect.html
	// OpenCV typically assumes that the top and left boundary of the rectangle
	// are inclusive, while the right and bottom boundaries are not.
	// For example, the method Rect_.contains returns true if
	// x <= pt.x < x+width,   y <= pt.y < y+height
	return cv::Rect(xMin, yMin, 1 + xMax - xMin,  1 + yMax - yMin);
}

cv::Rect Thinning::copy_bounding_box_plusone(const cv::Mat1b& img, cv::Mat1b& out, bool crop_img_before)
{
	// get the bounding box of the non-zero pixels of an image + a border of one pixel
	if (!crop_img_before) {
		img.copyTo(out);
		return cv::Rect(0, 0, img.cols, img.rows);
	}

	cv::Rect bbox = boundingBox(img);

	// the top and left boundary of the rectangle are inclusive, while the right and bottom boundaries are not
	if (bbox.x <= 0 || bbox.x + bbox.width >= img.cols || bbox.y <= 0 || bbox.y + bbox.height >= img.rows) {
		out.create(img.size());
		out.setTo(0);
		cv::Mat1b out_roi = out(cv::Rect(1, 1, img.cols-2, img.rows-2));
		cv::resize(img, out_roi, out_roi.size(), 0, 0, cv::INTER_NEAREST);
		return cv::Rect(0, 0, img.cols, img.rows); // full bbox
	}

	// add a border of one pixel
	bbox.x--;
	bbox.y--;
	bbox.width += 2;
	bbox.height += 2;

	img(bbox).copyTo(out);

	return bbox;
}

bool Thinning::thinGuoHallFast(const cv::Mat1b& img, bool inverted, bool crop_img_before, int max_iters)
{
	if (inverted) {
		m_imgSkeletonInverted = cv::Mat(img.rows, img.cols, CV_8UC1);
		return thin_fast_custom_voronoi_fn(img, inverted, need_set_guo_hall, crop_img_before, max_iters);
	}
	else {
		m_imgSkeleton = cv::Mat(img.rows, img.cols, CV_8UC1);
		return thin_fast_custom_voronoi_fn(invertColor(img), inverted, need_set_guo_hall, crop_img_before, max_iters);
	}
}

bool Thinning::thin_fast_custom_voronoi_fn(const cv::Mat1b& img, bool inverted, VoronoiFn voronoi_fn, bool crop_img_before, int max_iters)
{
	if (inverted) {
		m_bbox  = copy_bounding_box_plusone(img, m_imgSkeletonInverted, crop_img_before);
		m_skelcontour.from_image_C4(m_imgSkeletonInverted);
	}
	else {
		m_bbox  = copy_bounding_box_plusone(img, m_imgSkeleton, crop_img_before);
		m_skelcontour.from_image_C4(m_imgSkeleton);
	}

	int cols = m_skelcontour.cols, rows = m_skelcontour.rows;

	// clear queues
	uchar * skelcontour_data = m_skelcontour.data;

	int niters = 0;
	bool change_made = true;
	while (change_made && niters < max_iters) {

		change_made = false;
		for (unsigned short iter = 0; iter < 2; ++iter) {

			uchar *skelcontour_ptr = skelcontour_data;
			m_rowsToSet.clear();
			m_colsToSet.clear();

			// for each point in skelcontour, check if it needs to be changed
			for (int row = 0; row < rows; ++row) {
				for (int col = 0; col < cols; ++col) {
					if (*skelcontour_ptr++ == ImageContour::CONTOUR && voronoi_fn(skelcontour_data, iter, col, row, cols)) {
						m_colsToSet.push_back(col);
						m_rowsToSet.push_back(row);
					}
				}
			}

			// set all points in rows_to_set (of skel)
			unsigned int rows_to_set_size = m_rowsToSet.size();
			for (unsigned int pt_idx = 0; pt_idx < rows_to_set_size; ++pt_idx) {
				if (!change_made) change_made = (m_skelcontour(m_rowsToSet[pt_idx], m_colsToSet[pt_idx]));
				m_skelcontour.set_point_empty_C4(m_rowsToSet[pt_idx], m_colsToSet[pt_idx]);
			} // end for (pt_idx)

			if ((niters++) >= max_iters) // must be at the end of the loop
				break;
		}
	}

	if (inverted) m_imgSkeletonInverted = invertColor(m_skelcontour != ImageContour::EMPTY);
	else m_imgSkeleton = invertColor(m_skelcontour != ImageContour::EMPTY);
	m_has_converged = !change_made;

	return true;
}


cv::Mat Thinning::getImgSkeleton() const
{
	return m_imgSkeleton;
}

cv::Mat Thinning::getImgSkeletonInverted() const
{
	return m_imgSkeletonInverted;
}
