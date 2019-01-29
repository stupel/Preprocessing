#include "qualitymap.h"

QualityMap::QualityMap(QObject *parent) : QObject(parent)
{

}

void QualityMap::setParams(const cv::Mat &imgOriginal, QMAP_PARAMS qmapParams)
{
	m_imgOriginal = imgOriginal;
	idata = imgOriginal.data;
	iw = imgOriginal.cols;
	ih = imgOriginal.rows;
	id = 8 * imgOriginal.elemSize();
	ippi = qmapParams.ppi;
	if(id != 8){
		qDebug() << "Error: Wrong image depth. Image is not grayscale.";
	}
}


void QualityMap::computeQualityMap()
{
	computeImageMaps(); // computes image maps
	gen_quality_map(); // computes overall image quality map
}

int *QualityMap::getQuality_map() const
{
	return quality_map;
}

int QualityMap::getMap_w() const
{
	return map_w;
}

int QualityMap::getMap_h() const
{
	return map_h;
}

cv::Mat QualityMap::getImgQualityMap()
{
	int blockSize = 8;

	cv::Mat imgQualityMap = cv::Mat::zeros(map_h * blockSize, map_w * blockSize, CV_8UC1);
	cv::Mat block = cv::Mat(blockSize, blockSize, CV_8UC1);

	int cnt = 0;
	for (int y = 0; y < map_h * blockSize; y += blockSize) {
		for (int x = 0; x < map_w * blockSize; x += blockSize) {
			if (quality_map[cnt] == 4) block.setTo(255);
			else block.setTo(quality_map[cnt] * 63);
			block.copyTo(imgQualityMap(cv::Rect(x, y, blockSize, blockSize)));
			cnt++;
		}
	}

	return imgQualityMap.rowRange(0, ih).colRange(0, iw);
}

cv::Mat QualityMap::getQualityMap()
{
	int blockSize = 8;

	cv::Mat qualityMap = cv::Mat::zeros(map_h * blockSize, map_w * blockSize, CV_8UC1);
	cv::Mat block = cv::Mat(blockSize, blockSize, CV_8UC1);

	int cnt = 0;
	for (int y = 0; y < map_h * blockSize; y += blockSize) {
		for (int x = 0; x < map_w * blockSize; x += blockSize) {
			if (quality_map[cnt] == 4) block.setTo(100);
			else if (quality_map[cnt] == 3) block.setTo(50);
			else if (quality_map[cnt] == 2) block.setTo(25);
			else if (quality_map[cnt] == 1) block.setTo(10);
			else block.setTo(1);
			block.copyTo(qualityMap(cv::Rect(x, y, blockSize, blockSize)));
			cnt++;
		}
	}

	//cv::GaussianBlur(qualityMap, qualityMap, cv::Size(121, 121), 10.0, 10.0);

	return qualityMap.rowRange(0, ih).colRange(0, iw);
}

void QualityMap::fill_minutiae(MINUTIAE_VECTOR &minutiae)
{
	if(m_minutiae->alloc < minutiae.size()){
		qDebug() << "ERROR: Too many minutiae, exceeding max.";
	}
	int cntr=0;
	for(std::tuple<QPoint,int,int,int> minutia : minutiae){
		m_minutiae->list[cntr] = (MINUTIAQ*)malloc(sizeof(MINUTIAQ));
		m_minutiae->list[cntr]->x = std::get<0>(minutia).x();
		m_minutiae->list[cntr]->y = std::get<0>(minutia).y();
		cntr++;
	}
	m_minutiae->num = cntr;
}


void QualityMap::computeImageMaps()
{
	unsigned char *pdata;
	int pw, ph;
	DIR2RAD *dir2rad;
	DFTWAVES *dftwaves;
	ROTGRIDS *dftgrids;
	int ret, maxpad;

	/******************/
	/* INITIALIZATION */
	/******************/

	/* Determine the maximum amount of image padding required to support */
	/* LFS processes.                                                    */
	maxpad = get_max_padding_V2(lfsparms_V2.windowsize, lfsparms_V2.windowoffset,
								lfsparms_V2.dirbin_grid_w, lfsparms_V2.dirbin_grid_h);

	/* Initialize lookup table for converting integer directions */
	/* to angles in radians.                                     */
	if((ret = init_dir2rad(&dir2rad, lfsparms_V2.num_directions))){
		/* Free memory allocated to this point. */
		qDebug() << "Error: init_dir2rad()";
	}

	/* Initialize wave form lookup tables for DFT analyses. */
	/* used for direction binarization.                             */
	if((ret = init_dftwaves(&dftwaves, dft_coefs, lfsparms_V2.num_dft_waves,
							lfsparms_V2.windowsize))){
		/* Free memory allocated to this point. */
		free_dir2rad(dir2rad);
		qDebug() << "Error: init_dftwaves()";
	}

	/* Initialize lookup table for pixel offsets to rotated grids */
	/* used for DFT analyses.                                     */
	if((ret = init_rotgrids(&dftgrids, iw, ih, maxpad,
							lfsparms_V2.start_dir_angle, lfsparms_V2.num_directions,
							lfsparms_V2.windowsize, lfsparms_V2.windowsize,
							RELATIVE2ORIGIN))){
		/* Free memory allocated to this point. */
		free_dir2rad(dir2rad);
		free_dftwaves(dftwaves);
		qDebug() << "Error: init_rotgrids()";
	}

	/* Pad input image based on max padding. */
	if(maxpad > 0){   /* May not need to pad at all */
		if((ret = pad_uchar_image(&pdata, &pw, &ph, idata, iw, ih,
								  maxpad, lfsparms_V2.pad_value))){
			/* Free memory allocated to this point. */
			free_dir2rad(dir2rad);
			free_dftwaves(dftwaves);
			free_rotgrids(dftgrids);
			qDebug() << "Error: pad_uchar_image()";
		}
	}
	else{
		/* If padding is unnecessary, then copy the input image. */
		pdata = (unsigned char *)malloc(iw*ih);
		if(pdata == (unsigned char *)NULL){
			/* Free memory allocated to this point. */
			free_dir2rad(dir2rad);
			free_dftwaves(dftwaves);
			free_rotgrids(dftgrids);
			fprintf(stderr, "ERROR : lfs_detect_minutiae_V2 : malloc : pdata\n");
			qDebug() << "Error: malloc()";
		}
		memcpy(pdata, idata, iw*ih);
		pw = iw;
		ph = ih;
	}

	/* Scale input image to 6 bits [0..63] */
	/* !!! Would like to remove this dependency eventualy !!!     */
	/* But, the DFT computations will need to be changed, and     */
	/* could not get this work upon first attempt. Also, if not   */
	/* careful, I think accumulated power magnitudes may overflow */
	/* doubles. */

	bits_8to6(pdata, pw, ph);

	/******************/
	/*      MAPS      */
	/******************/

	/* Generate block maps from the input image. */
	gen_image_maps(&(direction_map),
				   &(low_contrast_map),
				   &(low_flow_map),
				   &(high_curve_map),
				   &(map_w),
				   &(map_h),
				   pdata, pw, ph,
				   dir2rad, dftwaves, dftgrids,
				   &(lfsparms_V2));

	/* Deallocate working memories. */
	free_dir2rad(dir2rad);
	free_dftwaves(dftwaves);
	free_rotgrids(dftgrids);

	/******************/
	/*    WRAP-UP     */
	/******************/

	/* Deallocate working memory. */
	free(pdata);
}



/***********************************************************************
************************************************************************
#cat: gen_quality_map - Takes a direction map, low contrast map, low ridge
#cat:              flow map, and high curvature map, and combines them
#cat:              into a single map containing 5 levels of decreasing
#cat:              quality.  This is done through a set of heuristics.

********************************************/

void QualityMap::gen_quality_map()
{
	int *QualMap;
	int thisX, thisY;
	int compX, compY;
	int arrayPos, arrayPos2;
	int QualOffset;

	QualMap = (int *)malloc(map_w * map_h * sizeof(int));
	if(QualMap == (int *)NULL){
		fprintf(stderr, "ERROR : gen_quality_map : malloc : QualMap\n");
		qDebug() << "ERROR : gen_quality_map : malloc";
	}

	/* Foreach row of blocks in maps ... */
	for(thisY=0; thisY<map_h; thisY++){
		/* Foreach block in current row ... */
		for(thisX=0; thisX<map_w; thisX++) {
			/* Compute block index. */
			arrayPos=(thisY*map_w)+thisX;
			/* If current block has low contrast or INVALID direction ... */
			if(low_contrast_map[arrayPos] || direction_map[arrayPos]<0)
				/* Set block's quality to 0/F. */
				QualMap[arrayPos]=0;
			else{
				/* Set baseline quality before looking at neighbors    */
				/*     (will subtract QualOffset below)                */
				/* If current block has low flow or high curvature ... */
				if(low_flow_map[arrayPos] || high_curve_map[arrayPos])
					/* Set block's quality initially to 3/B. */
					QualMap[arrayPos] = 3;  /* offset will be -1..-2 */
				/* Otherwise, block is NOT low flow AND NOT high curvature... */
				else
					/* Set block's quality to 4/A. */
					QualMap[arrayPos]=4;    /* offset will be 0..-2 */

				/* If block within NEIGHBOR_DELTA of edge ... */
				if(thisY < NEIGHBOR_DELTA || thisY > map_h - 1 - NEIGHBOR_DELTA ||
						thisX < NEIGHBOR_DELTA || thisX > map_w - 1 - NEIGHBOR_DELTA)
					/* Set block's quality to 1/E. */
					QualMap[arrayPos]=1;
				/* Otherwise, test neighboring blocks ... */
				else{
					/* Initialize quality adjustment to 0. */
					QualOffset=0;
					/* Foreach row in neighborhood ... */
					for(compY=thisY-NEIGHBOR_DELTA;
						compY<=thisY+NEIGHBOR_DELTA;compY++){
						/* Foreach block in neighborhood */
						/*  (including current block)... */
						for(compX=thisX-NEIGHBOR_DELTA;
							compX<=thisX+NEIGHBOR_DELTA;compX++) {
							/* Compute neighboring block's index. */
							arrayPos2 = (compY*map_w)+compX;
							/* If neighbor block (which might be itself) has */
							/* low contrast or INVALID direction .. */
							if(low_contrast_map[arrayPos2] ||
									direction_map[arrayPos2]<0) {
								/* Set quality adjustment to -2. */
								QualOffset=-2;
								/* Done with neighborhood row. */
								break;
							}
							/* Otherwise, if neighbor block (which might be */
							/* itself) has low flow or high curvature ... */
							else if(low_flow_map[arrayPos2] ||
									high_curve_map[arrayPos2]) {
								/* Set quality to -1 if not already -2. */
								QualOffset=std::min(QualOffset,-1);
							}
						}
					}
					/* Decrement minutia quality by neighborhood adjustment. */
					QualMap[arrayPos]+=QualOffset;
				}
			}
		}
	}

	/* Set output pointer. */
	quality_map = QualMap;
}



/*************************************************************************
**************************************************************************
#cat: free_minutiae - Takes a minutiae list and deallocates all memory
#cat:                 associated with it.

   Input:
	  minutiae - pointer to allocated list of minutia structures
*************************************************************************/
void QualityMap::free_minutiae(MINUTIAE *minutiae)
{
	int i;

	/* Deallocate minutia structures in the list. */
	for(i = 0; i < minutiae->num; i++)
		free_minutia(minutiae->list[i]);
	/* Deallocate list of minutia pointers. */
	free(minutiae->list);

	/* Deallocate the list structure. */
	free(minutiae);
}

/***********************************************************************
************************************************************************
#cat: combined_minutia_quality - Combines quality measures derived from
#cat:              the quality map and neighboring pixel statistics to
#cat:              infer a reliability measure on the scale [0...1].

   Input:
	  minutiae    - structure contining the detected minutia
	  quality_map - map with blocks assigned 1 of 5 quality levels
	  map_w       - width (in blocks) of the map
	  map_h       - height (in blocks) of the map
	  blocksize   - size (in pixels) of each block in the map
	  idata       - 8-bit grayscale fingerprint image
	  iw          - width (in pixels) of the image
	  ih          - height (in pixels) of the image
	  id          - depth (in pixels) of the image
	  ppmm        - scan resolution of the image in pixels/mm
   Output:
	  minutiae    - updated reliability members
************************************************************************/
void QualityMap::combined_minutia_quality(MINUTIAE *minutiae,
										  int *quality_map, const int mw, const int mh, const int blocksize,
										  unsigned char *idata, const int iw, const int ih, const int id,
										  const double ppmm)
{
	int ret, i, index, radius_pix;
	int *pquality_map, qmap_value;
	MINUTIAQ *minutia;
	double gs_reliability, reliability;

	/* If image is not 8-bit grayscale ... */
	if(id != 8){
		fprintf(stderr, "ERROR : combined_miutia_quality : ");
		fprintf(stderr, "image must pixel depth = %d must be 8 ", id);
		fprintf(stderr, "to compute reliability\n");
		qDebug() << "ERROR: image is not 8-bit grayscale";
	}

	/* Compute pixel radius of neighborhood based on image's scan resolution. */
	radius_pix = sround(RADIUS_MM * ppmm);

	/* Expand block map values to pixel map. */
	if((ret = pixelize_map(&pquality_map, iw, ih,
						   quality_map, mw, mh, blocksize))){
		qDebug() << "ERROR: pixelize_map()";
	}


	/* Foreach minutiae detected ... */
	for(i = 0; i < minutiae->num; i++){
		/* Assign minutia pointer. */
		minutia = minutiae->list[i];

		/* Compute reliability from stdev and mean of pixel neighborhood. */
		gs_reliability = grayscale_reliability(minutia, idata, iw, ih, radius_pix);

		/* Lookup quality map value. */
		/* Compute minutia pixel index. */
		index = (minutia->y * iw) + minutia->x;
		/* Switch on pixel's quality value ... */
		qmap_value = pquality_map[index];

		/* Combine grayscale reliability and quality map value. */
		switch(qmap_value){
		/* Quality A : [50..99]% */
		case 4 :
			reliability = 0.50 + (0.49 * gs_reliability);
			break;
			/* Quality B : [25..49]% */
		case 3 :
			reliability = 0.25 + (0.24 * gs_reliability);
			break;
			/* Quality C : [10..24]% */
		case 2 :
			reliability = 0.10 + (0.14 * gs_reliability);
			break;
			/* Quality D : [5..9]% */
		case 1 :
			reliability = 0.05 + (0.04 * gs_reliability);
			break;
			/* Quality E : 1% */
		case 0 :
			reliability = 0.01;
			break;
			/* Error if quality value not in range [0..4]. */
		default:
			fprintf(stderr, "ERROR : combined_miutia_quality : ");
			fprintf(stderr, "unexpected quality map value %d ", qmap_value);
			fprintf(stderr, "not in range [0..4]\n");
			free(pquality_map);
			qDebug() << "ERROR: combined_miutia_quality, unexpected quality map value, not in range [0..4]";
		}
		minutia->reliability = reliability;
	}

	/* NEW 05-08-2002 */
	free(pquality_map);
}


/*************************************************************************
**************************************************************************
#cat: pixelize_map - Takes a block image map and assigns each pixel in the
#cat:            image its corresponding block value.  This allows block
#cat:            values in maps to be directly accessed via pixel addresses.

   Input:
	  iw        - the width (in pixels) of the corresponding image
	  ih        - the height (in pixels) of the corresponding image
	  imap      - input block image map
	  mw        - the width (in blocks) of the map
	  mh        - the height (in blocks) of the map
	  blocksize - the dimension (in pixels) of each block
   Output:
	  omap      - points to the resulting pixelized map
   Return Code:
	  Zero     - successful completion
	  Negative - system error
**************************************************************************/
int QualityMap::pixelize_map(int **omap, const int iw, const int ih,
							 int *imap, const int mw, const int mh, const int blocksize)
{
	int *pmap;
	int ret, x, y;
	int *blkoffs, bw, bh, bi;
	int *spptr, *pptr;

	pmap = (int *)malloc(iw*ih*sizeof(int));
	if(pmap == (int *)NULL){
		fprintf(stderr, "ERROR : pixelize_map : malloc : pmap\n");
		return(-590);
	}

	if((ret = block_offsets(&blkoffs, &bw, &bh, iw, ih, 0, blocksize))){
		return(ret);
	}

	if((bw != mw) || (bh != mh)){
		free(blkoffs);
		fprintf(stderr,
				"ERROR : pixelize_map : block dimensions do not match\n");
		return(-591);
	}

	for(bi = 0; bi < mw*mh; bi++){
		spptr = pmap + blkoffs[bi];
		for(y = 0; y < blocksize; y++){
			pptr = spptr;
			for(x = 0; x < blocksize; x++){
				*pptr++ = imap[bi];
			}
			spptr += iw;
		}
	}

	/* Deallocate working memory. */
	free(blkoffs);
	/* Assign pixelized map to output pointer. */
	*omap = pmap;

	/* Return normally. */
	return(0);
}


/***********************************************************************
************************************************************************
#cat: grayscale_reliability - Given a minutia point, computes a reliability
#cat:              measure from the stdev and mean of its pixel neighborhood.

   Code originally written by Austin Hicklin for FBI ATU
   Modified by Michael D. Garris (NIST) Sept. 25, 2000

   GrayScaleReliability - reasonable reliability heuristic, returns
   0.0 .. 1.0 based on stdev and Mean of a localized histogram where
   "ideal" stdev is >=64; "ideal" Mean is 127.  In a 1 ridge radius
   (11 pixels), if the bytevalue (shade of gray) in the image has a
   stdev of >= 64 & a mean of 127,  returns 1.0 (well defined
   light & dark areas in equal proportions).

   Input:
	  minutia    - structure containing detected minutia
	  idata      - 8-bit grayscale fingerprint image
	  iw         - width (in pixels) of the image
	  ih         - height (in pixels) of the image
	  radius_pix - pixel radius of surrounding neighborhood
   Return Value:
	  reliability - computed reliability measure
************************************************************************/
double QualityMap::grayscale_reliability(MINUTIAQ *minutia, unsigned char *idata,
										 const int iw, const int ih, const int radius_pix)
{
	double mean, stdev;
	double reliability;

	get_neighborhood_stats(&mean, &stdev, minutia, idata, iw, ih, radius_pix);

	reliability = std::min((stdev>IDEALSTDEV ? 1.0 : stdev/(double)IDEALSTDEV),
						   (1.0-(fabs(mean-IDEALMEAN)/(double)IDEALMEAN)));

	return(reliability);
}



/***********************************************************************
************************************************************************
#cat: get_neighborhood_stats - Given a minutia point, computes the mean
#cat:              and stdev of the 8-bit grayscale pixels values in a
#cat:              surrounding neighborhood with specified radius.

   Code originally written by Austin Hicklin for FBI ATU
   Modified by Michael D. Garris (NIST) Sept. 25, 2000

   Input:
	  minutia    - structure containing detected minutia
	  idata      - 8-bit grayscale fingerprint image
	  iw         - width (in pixels) of the image
	  ih         - height (in pixels) of the image
	  radius_pix - pixel radius of surrounding neighborhood
   Output:
	  mean       - mean of neighboring pixels
	  stdev      - standard deviation of neighboring pixels
************************************************************************/
void QualityMap::get_neighborhood_stats(double *mean, double *stdev, MINUTIAQ *minutia,
										unsigned char *idata, const int iw, const int ih,
										const int radius_pix)
{
	int i, x, y, rows, cols;
	int n = 0, sumX = 0, sumXX = 0;
	int histogram[256];

	/* Zero out histogram. */
	memset(histogram, 0, 256 * sizeof(int));

	/* Set minutia's coordinate variables. */
	x = minutia->x;
	y = minutia->y;


	/* If minutiae point is within sampleboxsize distance of image border, */
	/* a value of 0 reliability is returned. */
	if ((x < radius_pix) || (x > iw-radius_pix-1) ||
			(y < radius_pix) || (y > ih-radius_pix-1)) {
		*mean = 0.0;
		*stdev = 0.0;
		return;

	}

	/* Foreach row in neighborhood ... */
	for(rows = y - radius_pix;
		rows <= y + radius_pix;
		rows++){
		/* Foreach column in neighborhood ... */
		for(cols = x - radius_pix;
			cols <= x + radius_pix;
			cols++){
			/* Bump neighbor's pixel value bin in histogram. */
			histogram[*(idata+(rows * iw)+cols)]++;
		}
	}

	/* Foreach grayscale pixel bin ... */
	for(i = 0; i < 256; i++){
		if(histogram[i]){
			/* Accumulate Sum(X[i]) */
			sumX += (i * histogram[i]);
			/* Accumulate Sum(X[i]^2) */
			sumXX += (i * i * histogram[i]);
			/* Accumulate N samples */
			n += histogram[i];
		}
	}

	/* Mean = Sum(X[i])/N */
	*mean = sumX/(double)n;
	/* Stdev = sqrt((Sum(X[i]^2)/N) - Mean^2) */
	*stdev = sqrt((sumXX/(double)n) - ((*mean)*(*mean)));
}


/*************************************************************************
**************************************************************************
#cat: get_max_padding_V2 - Deterines the maximum amount of image pixel padding
#cat:         required by all LFS (Version 2) processes.  Padding is currently
#cat:         required by the rotated grids used in DFT analyses and in
#cat:         directional binarization.  The NIST generalized code enables
#cat:         the parameters governing these processes to be redefined, so a
#cat:         check at runtime is required to determine which process
#cat:         requires the most padding.  By using the maximum as the padding
#cat:         factor, all processes will run safely with a single padding of
#cat:         the input image avoiding the need to repad for further processes.

   Input:
	  map_windowsize  - the size (in pixels) of each window centered about
						each block in the image used in DFT analyses
	  map_windowoffset - the offset (in pixels) from the orgin of the
						surrounding window to the origin of the block
	  dirbin_grid_w   - the width (in pixels) of the rotated grids used in
						directional binarization
	  dirbin_grid_h   - the height (in pixels) of the rotated grids used in
						directional binarization
   Return Code:
	  Non-negative - the maximum padding required for all processes
**************************************************************************/
int QualityMap::get_max_padding_V2(const int map_windowsize, const int map_windowoffset,
								   const int dirbin_grid_w, const int dirbin_grid_h)
{
	int dft_pad, dirbin_pad, max_pad;
	double diag;
	double pad;


	/* 1. Compute pad required for rotated windows used in DFT analyses. */

	/* Explanation of DFT padding:

					B---------------------
					|      window        |
					|                    |
					|                    |
					|      A.......______|__________
					|      :      :      |
					|<-C-->: block:      |
				 <--|--D-->:      :      | image
					|      ........      |
					|      |             |
					|      |             |
					|      |             |
					----------------------
						   |
						   |
						   |

		 Pixel A = Origin of entire fingerprint image
				 = Also origin of first block in image. Each pixel in
				   this block gets the same DFT results computed from
				   the surrounding window.  Note that in general
				   blocks are adjacent and non-overlapping.

		 Pixel B = Origin of surrounding window in which DFT
				   analysis is conducted.  Note that this window is not
				   completely contained in the image but extends to the
				   top and to the right.

		 Distance C = Number of pixels in which the window extends
				   beyond the image (map_windowoffset).

		 Distance D = Amount of padding required to hold the entire
				   rotated window in memory.

   */

	/* Compute pad as difference between the MAP windowsize           */
	/* and the diagonal distance of the window.                       */
	/* (DFT grids are computed with pixel offsets RELATIVE2ORIGIN.)   */
	diag = sqrt((double)(2.0 * map_windowsize * map_windowsize));
	pad = (diag-map_windowsize)/(double)2.0;
	/* Need to truncate precision so that answers are consistent  */
	/* on different computer architectures when rounding doubles. */
	pad = trunc_dbl_precision(pad, TRUNC_SCALE);
	/* Must add the window offset to the rotational padding. */
	dft_pad = sround(pad) + map_windowoffset;

	/* 2. Compute pad required for rotated blocks used in directional  */
	/*    binarization.  Binarization blocks are applied to each pixel */
	/*    in the input image.                                          */
	diag = sqrt((double)((dirbin_grid_w*dirbin_grid_w)+
						 (dirbin_grid_h*dirbin_grid_h)));
	/* Assumption: all grid centers reside in valid/allocated memory. */
	/* (Dirbin grids are computed with pixel offsets RELATIVE2CENTER.) */
	pad = (diag-1)/(double)2.0;
	/* Need to truncate precision so that answers are consistent */
	/* on different computer architectures when rounding doubles. */
	pad = trunc_dbl_precision(pad, TRUNC_SCALE);
	dirbin_pad = sround(pad);

	max_pad = std::max(dft_pad, dirbin_pad);

	/* Return the maximum of the two required paddings.  This padding will */
	/* be sufficiently large for all purposes, so that padding of the      */
	/* input image will only be required once.                             */
	return(max_pad);
}


/*************************************************************************
**************************************************************************
#cat: init_dir2rad - Allocates and initializes a lookup table containing
#cat:                cosine and sine values needed to convert integer IMAP
#cat:                directions to angles in radians.

   Input:
	  ndirs - the number of integer directions to be defined in a
			  semicircle
   Output:
	  optr  - points to the allocated/initialized DIR2RAD structure
   Return Code:
	  Zero     - successful completion
	  Negative - system error
**************************************************************************/
int QualityMap::init_dir2rad(DIR2RAD **optr, const int ndirs)
{
	DIR2RAD *dir2rad;
	int i;
	double theta, pi_factor;
	double cs, sn;

	/* Allocate structure */
	dir2rad = (DIR2RAD *)malloc(sizeof(DIR2RAD));
	if(dir2rad == (DIR2RAD *)NULL){
		fprintf(stderr, "ERROR : init_dir2rad : malloc : dir2rad\n");
		return(-10);
	}

	/* Assign number of directions */
	dir2rad->ndirs = ndirs;

	/* Allocate cosine vector */
	dir2rad->cos = (double *)malloc(ndirs * sizeof(double));
	if(dir2rad->cos == (double *)NULL){
		/* Free memory allocated to this point. */
		free(dir2rad);
		fprintf(stderr, "ERROR : init_dir2rad : malloc : dir2rad->cos\n");
		return(-11);
	}

	/* Allocate sine vector */
	dir2rad->sin = (double *)malloc(ndirs * sizeof(double));
	if(dir2rad->sin == (double *)NULL){
		/* Free memory allocated to this point. */
		free(dir2rad->cos);
		free(dir2rad);
		fprintf(stderr, "ERROR : init_dir2rad : malloc : dir2rad->sin\n");
		return(-12);
	}

	/* Pi_factor sets the period of the trig functions to NDIRS units in x. */
	/* For example, if NDIRS==16, then pi_factor = 2(PI/16) = .3926...      */
	pi_factor = 2.0*M_PI_MINDTCT/(double)ndirs;

	/* Now compute cos and sin values for each direction.    */
	for (i = 0; i < ndirs; ++i) {
		theta = (double)(i * pi_factor);
		cs = cos(theta);
		sn = sin(theta);
		/* Need to truncate precision so that answers are consistent */
		/* on different computer architectures. */
		cs = trunc_dbl_precision(cs, TRUNC_SCALE);
		sn = trunc_dbl_precision(sn, TRUNC_SCALE);
		dir2rad->cos[i] = cs;
		dir2rad->sin[i] = sn;
	}

	*optr = dir2rad;
	return(0);
}



/*************************************************************************
**************************************************************************
#cat: init_dftwaves - Allocates and initializes a set of wave forms needed
#cat:                 to conduct DFT analysis on blocks of the input image

   Input:
	  dft_coefs - array of multipliers used to define the frequency for
				  each wave form to be computed
	  nwaves    - number of wave forms to be computed
	  blocksize - the width and height of each block of image data to
				  be DFT analyzed
   Output:
	  optr     - points to the allocated/initialized DFTWAVES structure
   Return Code:
	  Zero     - successful completion
	  Negative - system error
**************************************************************************/
int QualityMap::init_dftwaves(DFTWAVES **optr, const double *dft_coefs,
							  const int nwaves, const int blocksize)
{
	DFTWAVES *dftwaves;
	int i, j;
	double pi_factor, freq, x;
	double *cptr, *sptr;

	/* Allocate structure */
	dftwaves = (DFTWAVES *)malloc(sizeof(DFTWAVES));
	if(dftwaves == (DFTWAVES *)NULL){
		fprintf(stderr, "ERROR : init_dftwaves : malloc : dftwaves\n");
		return(-20);
	}

	/* Set number of DFT waves */
	dftwaves->nwaves = nwaves;
	/* Set wave length of the DFT waves (they all must be the same length) */
	dftwaves->wavelen = blocksize;

	/* Allocate list of wave pointers */
	dftwaves->waves = (DFTWAVE **)malloc(nwaves * sizeof(DFTWAVE *));
	if(dftwaves == (DFTWAVES *)NULL){
		/* Free memory allocated to this point. */
		free(dftwaves);
		fprintf(stderr, "ERROR : init_dftwaves : malloc : dftwaves->waves\n");
		return(-21);
	}

	/* Pi_factor sets the period of the trig functions to BLOCKSIZE units */
	/* in x.  For example, if BLOCKSIZE==24, then                         */
	/*                         pi_factor = 2(PI/24) = .26179...           */
	pi_factor = 2.0*M_PI_MINDTCT/(double)blocksize;

	/* Foreach of 4 DFT frequency coef ... */
	for (i = 0; i < nwaves; ++i) {
		/* Allocate wave structure */
		dftwaves->waves[i] = (DFTWAVE *)malloc(sizeof(DFTWAVE));
		if(dftwaves->waves[i] == (DFTWAVE *)NULL){
			/* Free memory allocated to this point. */
			{ int _j; for(_j = 0; _j < i; _j++){
					free(dftwaves->waves[_j]->cos);
					free(dftwaves->waves[_j]->sin);
					free(dftwaves->waves[_j]);
				}}
			free(dftwaves->waves);
			free(dftwaves);
			fprintf(stderr,
					"ERROR : init_dftwaves : malloc : dftwaves->waves[i]\n");
			return(-22);
		}
		/* Allocate cosine vector */
		dftwaves->waves[i]->cos = (double *)malloc(blocksize * sizeof(double));
		if(dftwaves->waves[i]->cos == (double *)NULL){
			/* Free memory allocated to this point. */
			{ int _j; for(_j = 0; _j < i; _j++){
					free(dftwaves->waves[_j]->cos);
					free(dftwaves->waves[_j]->sin);
					free(dftwaves->waves[_j]);
				}}
			free(dftwaves->waves[i]);
			free(dftwaves->waves);
			free(dftwaves);
			fprintf(stderr,
					"ERROR : init_dftwaves : malloc : dftwaves->waves[i]->cos\n");
			return(-23);
		}
		/* Allocate sine vector */
		dftwaves->waves[i]->sin = (double *)malloc(blocksize * sizeof(double));
		if(dftwaves->waves[i]->sin == (double *)NULL){
			/* Free memory allocated to this point. */
			{ int _j; for(_j = 0; _j < i; _j++){
					free(dftwaves->waves[_j]->cos);
					free(dftwaves->waves[_j]->sin);
					free(dftwaves->waves[_j]);
				}}
			free(dftwaves->waves[i]->cos);
			free(dftwaves->waves[i]);
			free(dftwaves->waves);
			free(dftwaves);
			fprintf(stderr,
					"ERROR : init_dftwaves : malloc : dftwaves->waves[i]->sin\n");
			return(-24);
		}

		/* Assign pointer nicknames */
		cptr = dftwaves->waves[i]->cos;
		sptr = dftwaves->waves[i]->sin;

		/* Compute actual frequency */
		freq = pi_factor * dft_coefs[i];

		/* Used as a 1D DFT on a 24 long vector of pixel sums */
		for (j = 0; j < blocksize; ++j) {
			/* Compute sample points from frequency */
			x = freq * (double)j;
			/* Store cos and sin components of sample point */
			*cptr++ = cos(x);
			*sptr++ = sin(x);
		}
	}

	*optr = dftwaves;
	return(0);
}


/*************************************************************************
**************************************************************************
#cat: free_dir2rad - Deallocates memory associated with a DIR2RAD structure

   Input:
	  dir2rad - pointer to memory to be freed
*************************************************************************/
void QualityMap::free_dir2rad(DIR2RAD *dir2rad)
{
	free(dir2rad->cos);
	free(dir2rad->sin);
	free(dir2rad);
}

/*************************************************************************
**************************************************************************
#cat: free_dftwaves - Deallocates the memory associated with a DFTWAVES
#cat:                 structure

   Input:
	  dftwaves - pointer to memory to be freed
**************************************************************************/
void QualityMap::free_dftwaves(DFTWAVES *dftwaves)
{
	int i;

	for(i = 0; i < dftwaves->nwaves; i++){
		free(dftwaves->waves[i]->cos);
		free(dftwaves->waves[i]->sin);
		free(dftwaves->waves[i]);
	}
	free(dftwaves->waves);
	free(dftwaves);
}

/*************************************************************************
**************************************************************************
#cat: free_rotgrids - Deallocates the memory associated with a ROTGRIDS
#cat:                 structure

   Input:
	  rotgrids - pointer to memory to be freed
**************************************************************************/
void QualityMap::free_rotgrids(ROTGRIDS *rotgrids)
{
	int i;

	for(i = 0; i < rotgrids->ngrids; i++)
		free(rotgrids->grids[i]);
	free(rotgrids->grids);
	free(rotgrids);
}

/*************************************************************************
**************************************************************************
#cat: free_dir_powers - Deallocate memory associated with DFT power vectors

   Input:
	  powers - vectors of DFT power values (N Waves X M Directions)
	  nwaves - number of DFT wave forms used
**************************************************************************/
void QualityMap::free_dir_powers(double **powers, const int nwaves)
{
	int w;

	for(w = 0; w < nwaves; w++)
		free(powers[w]);

	free(powers);
}


/*************************************************************************
**************************************************************************
#cat: init_rotgrids - Allocates and initializes a set of offsets that address
#cat:                 individual rotated pixels within a grid.
#cat:                 These rotated grids are used to conduct DFT analyses
#cat:                 on blocks of input image data, and they are used
#cat:                 in isotropic binarization.

   Input:
	  iw        - width (in pixels) of the input image
	  ih        - height (in pixels) of the input image
	  pad       - designates the number of pixels to be padded to the perimeter
				  of the input image.  May be passed as UNDEFINED, in which
				  case the specific padding required by the rotated grids
				  will be computed and returned in ROTGRIDS.
	  start_dir_angle - angle from which rotations are to start
	  ndirs     - number of rotations to compute (within a semicircle)
	  grid_w    - width of the grid in pixels to be rotated
	  grid_h    - height of the grid in pixels to be rotated
	  relative2 - designates whether pixel offsets whould be computed
				  relative to the ORIGIN or the CENTER of the grid
   Output:
	  optr      - points to the allcated/initialized ROTGRIDS structure
   Return Code:
	  Zero     - successful completion
	  Negative - system error
**************************************************************************/
int QualityMap::init_rotgrids(ROTGRIDS **optr, const int iw, const int ih, const int ipad,
							  const double start_dir_angle, const int ndirs,
							  const int grid_w, const int grid_h, const int relative2)
{
	ROTGRIDS *rotgrids;
	double pi_offset, pi_incr;
	int dir, ix, iy, grid_size, pw, grid_pad, min_dim;
	int *grid;
	double diag, theta, cs, sn, cx, cy;
	double fxm, fym, fx, fy;
	int ixt, iyt;
	double pad;

	/* Allocate structure */
	rotgrids = (ROTGRIDS *)malloc(sizeof(ROTGRIDS));
	if(rotgrids == (ROTGRIDS *)NULL){
		fprintf(stderr, "ERROR : init_rotgrids : malloc : rotgrids\n");
		return(-30);
	}

	/* Set rotgrid attributes */
	rotgrids->ngrids = ndirs;
	rotgrids->grid_w = grid_w;
	rotgrids->grid_h = grid_h;
	rotgrids->start_angle = start_dir_angle;
	rotgrids->relative2 = relative2;

	/* Compute pad based on diagonal of the grid */
	diag = sqrt((double)((grid_w*grid_w)+(grid_h*grid_h)));
	switch(relative2){
	case RELATIVE2CENTER:
		/* Assumption: all grid centers reside in valid/allocated memory. */
		pad = (diag-1)/(double)2.0;
		/* Need to truncate precision so that answers are consistent */
		/* on different computer architectures when rounding doubles. */
		pad = trunc_dbl_precision(pad, TRUNC_SCALE);
		grid_pad = sround(pad);
		break;
	case RELATIVE2ORIGIN:
		/* Assumption: all grid origins reside in valid/allocated memory. */
		min_dim = std::min(grid_w, grid_h);
		/* Compute pad as difference between the smallest grid dimension */
		/* and the diagonal distance of the grid. */
		pad = (diag-min_dim)/(double)2.0;
		/* Need to truncate precision so that answers are consistent */
		/* on different computer architectures when rounding doubles. */
		pad = trunc_dbl_precision(pad, TRUNC_SCALE);
		grid_pad = sround(pad);
		break;
	default:
		fprintf(stderr,
				"ERROR : init_rotgrids : Illegal relative flag : %d\n",
				relative2);
		free(rotgrids);
		return(-31);
	}

	/* If input padding is UNDEFINED ... */
	if(ipad == UNDEFINED)
		/* Use the padding specifically required by the rotated grids herein. */
		rotgrids->pad = grid_pad;
	else{
		/* Otherwise, input pad was specified, so check to make sure it is */
		/* sufficiently large to handle the rotated grids herein.          */
		if(ipad < grid_pad){
			/* If input pad is NOT large enough, then ERROR. */
			fprintf(stderr, "ERROR : init_rotgrids : Pad passed is too small\n");
			free(rotgrids);
			return(-32);
		}
		/* Otherwise, use the specified input pad in computing grid offsets. */
		rotgrids->pad = ipad;
	}

	/* Total number of points in grid */
	grid_size = grid_w * grid_h;

	/* Compute width of "padded" image */
	pw = iw + (rotgrids->pad<<1);

	/* Center coord of grid (0-oriented). */
	cx = (grid_w-1)/(double)2.0;
	cy = (grid_h-1)/(double)2.0;

	/* Allocate list of rotgrid pointers */
	rotgrids->grids = (int **)malloc(ndirs * sizeof(int *));
	if(rotgrids->grids == (int **)NULL){
		/* Free memory allocated to this point. */
		free(rotgrids);
		fprintf(stderr, "ERROR : init_rotgrids : malloc : rotgrids->grids\n");
		return(-33);
	}

	/* Pi_offset is the offset in radians from which angles are to begin. */
	pi_offset = start_dir_angle;
	pi_incr = M_PI_MINDTCT/(double)ndirs;  /* if ndirs == 16, incr = 11.25 degrees */

	/* For each direction to rotate a grid ... */
	for (dir = 0, theta = pi_offset;
		 dir < ndirs; dir++, theta += pi_incr) {

		/* Allocate a rotgrid */
		rotgrids->grids[dir] = (int *)malloc(grid_size * sizeof(int));
		if(rotgrids->grids[dir] == (int *)NULL){
			/* Free memory allocated to this point. */
			{ int _j; for(_j = 0; _j < dir; _j++){
					free(rotgrids->grids[_j]);
				}}
			free(rotgrids);
			fprintf(stderr,
					"ERROR : init_rotgrids : malloc : rotgrids->grids[dir]\n");
			return(-34);
		}

		/* Set pointer to current grid */
		grid = rotgrids->grids[dir];

		/* Compute cos and sin of current angle */
		cs = cos(theta);
		sn = sin(theta);

		/* This next section of nested FOR loops precomputes a         */
		/* rotated grid.  The rotation is set up to rotate a GRID_W X  */
		/* GRID_H grid on its center point at C=(Cx,Cy). The current   */
		/* pixel being rotated is P=(Ix,Iy).  Therefore, we have a     */
		/* rotation transformation of point P about pivot point C.     */
		/* The rotation transformation about a pivot point in matrix   */
		/* form is:                                                    */
		/*
			+-                                                       -+
			|             cos(T)                   sin(T)           0 |
  [Ix Iy 1] |            -sin(T)                   cos(T)           0 |
			| (1-cos(T))*Cx + Cy*sin(T)  (1-cos(T))*Cy - Cx*sin(T)  1 |
			+-                                                       -+
	  */
		/* Multiplying the 2 matrices and combining terms yeilds the */
		/* equations for rotated coordinates (Rx, Ry):               */
		/*        Rx = Cx + (Ix - Cx)*cos(T) - (Iy - Cy)*sin(T)      */
		/*        Ry = Cy + (Ix - Cx)*sin(T) + (Iy - Cy)*cos(T)      */
		/*                                                           */
		/* Care has been taken to ensure that (for example) when     */
		/* BLOCKSIZE==24 the rotated indices stay within a centered  */
		/* 34X34 area.                                               */
		/* This is important for computing an accurate padding of    */
		/* the input image.  The rotation occurs "in-place" so that  */
		/* outer pixels in the grid are mapped at times from         */
		/* adjoining blocks.  As a result, to keep from accessing    */
		/* "unknown" memory or pixels wrapped from the other side of */
		/* the image, the input image should first be padded by      */
		/* PAD=round((DIAG - BLOCKSIZE)/2.0) where DIAG is the       */
		/* diagonal distance of the grid.                            */
		/* For example, when BLOCKSIZE==24, Dx=34, so PAD=5.         */

		/* Foreach each y coord in block ... */
		for (iy = 0; iy < grid_h; ++iy) {
			/* Compute rotation factors dependent on Iy (include constant) */
			fxm = -1.0 * ((iy - cy) * sn);
			fym = ((iy - cy) * cs);

			/* If offsets are to be relative to the grids origin, then */
			/* we need to subtract CX and CY.                          */
			if(relative2 == RELATIVE2ORIGIN){
				fxm += cx;
				fym += cy;
			}

			/* foreach each x coord in block ... */
			for (ix = 0; ix < grid_w; ++ix) {

				/* Now combine factors dependent on Iy with those of Ix */
				fx = fxm + ((ix - cx) * cs);
				fy = fym + ((ix - cx) * sn);
				/* Need to truncate precision so that answers are consistent */
				/* on different computer architectures when rounding doubles. */
				fx = trunc_dbl_precision(fx, TRUNC_SCALE);
				fy = trunc_dbl_precision(fy, TRUNC_SCALE);
				ixt = sround(fx);
				iyt = sround(fy);

				/* Store the current pixels relative   */
				/* rotated offset.  Make sure to       */
				/* multiply the y-component of the     */
				/* offset by the "padded" image width! */
				*grid++ = ixt + (iyt * pw);
			}/* ix */
		}/* iy */
	}/* dir */

	*optr = rotgrids;
	return(0);
}



/*************************************************************************
**************************************************************************
#cat: pad_uchar_image - Copies an 8-bit grayscale images into a larger
#cat:                   output image centering the input image so as to
#cat:                   add a specified amount of pixel padding along the
#cat:                   entire perimeter of the input image.  The amount of
#cat:                   pixel padding and the intensity of the pixel padding
#cat:                   are specified.  An alternative to padding with a
#cat:                   constant intensity would be to copy the edge pixels
#cat:                   of the centered image into the adjacent pad area.

   Input:
	  idata     - input 8-bit grayscale image
	  iw        - width (in pixels) of the input image
	  ih        - height (in pixels) of the input image
	  pad       - size of padding (in pixels) to be added
	  pad_value - intensity of the padded area
   Output:
	  optr      - points to the newly padded image
	  ow        - width (in pixels) of the padded image
	  oh        - height (in pixels) of the padded image
   Return Code:
	  Zero     - successful completion
	  Negative - system error
**************************************************************************/
int QualityMap::pad_uchar_image(unsigned char **optr, int *ow, int *oh,
								unsigned char *idata, const int iw, const int ih,
								const int pad, const int pad_value)
{
	unsigned char *pdata, *pptr, *iptr;
	int i, pw, ph;
	int pad2, psize;

	/* Account for pad on both sides of image */
	pad2 = pad<<1;

	/* Compute new pad sizes */
	pw = iw + pad2;
	ph = ih + pad2;
	psize = pw * ph;

	/* Allocate padded image */
	pdata = (unsigned char *)malloc(psize * sizeof(unsigned char));
	if(pdata == (unsigned char *)NULL){
		fprintf(stderr, "ERROR : pad_uchar_image : malloc : pdata\n");
		return(-160);
	}

	/* Initialize values to a constant PAD value */
	memset(pdata, pad_value, psize);

	/* Copy input image into padded image one scanline at a time */
	iptr = idata;
	pptr = pdata + (pad * pw) + pad;
	for(i = 0; i < ih; i++){
		memcpy(pptr, iptr, iw);
		iptr += iw;
		pptr += pw;
	}

	*optr = pdata;
	*ow = pw;
	*oh = ph;
	return(0);
}


/*************************************************************************
**************************************************************************
#cat: bits_8to6 - Takes an array of unsigned characters and bitwise shifts
#cat:             each value 2 postitions to the right.  This is equivalent
#cat:             to dividing each value by 4.  This puts original values
#cat:             on the range [0..256) now on the range [0..64).  Another
#cat:             way to say this, is the original 8-bit values now fit in
#cat:             6 bits.  I would really like to make this dependency
#cat:             go away.

   Input:
	  idata - input array of unsigned characters
	  iw    - width (in characters) of the input array
	  ih    - height (in characters) of the input array
   Output:
	  idata - contains the bit-shifted results
**************************************************************************/
void QualityMap::bits_8to6(unsigned char *idata, const int iw, const int ih)
{
	int i, isize;
	unsigned char *iptr;

	isize = iw * ih;
	iptr = idata;
	for(i = 0; i < isize; i++){
		/* Divide every pixel value by 4 so that [0..256) -> [0..64) */
		*iptr++ >>= 2;
	}
}



/*************************************************************************
**************************************************************************
#cat: gray2bin - Takes an 8-bit threshold value and two 8-bit pixel values.
#cat:            Those pixels in the image less than the threhsold are set
#cat:            to the first specified pixel value, whereas those pixels
#cat:            greater than or equal to the threshold are set to the second
#cat:            specified pixel value.  On application for this routine is
#cat:            to convert binary images from 8-bit pixels valued {0,255} to
#cat:            {1,0} and vice versa.

   Input:
	  thresh      - 8-bit pixel threshold
	  less_pix    - pixel value used when image pixel is < threshold
	  greater_pix - pixel value used when image pixel is >= threshold
	  bdata       - 8-bit image data
	  iw          - width (in pixels) of the image
	  ih          - height (in pixels) of the image
   Output:
	  bdata       - altered 8-bit image data
**************************************************************************/
void QualityMap::gray2bin(const int thresh, const int less_pix, const int greater_pix,
						  unsigned char *bdata, const int iw, const int ih)
{
	int i;

	for(i = 0; i < iw*ih; i++){
		if(bdata[i] >= thresh)
			bdata[i] = (unsigned char)greater_pix;
		else
			bdata[i] = (unsigned char)less_pix;
	}
}


/*************************************************************************
**************************************************************************
#cat: alloc_minutiae - Allocates and initializes a minutia list based on the
#cat:            specified maximum number of minutiae to be detected.

   Input:
	  max_minutiae - number of minutia to be allocated in list
   Output:
	  ominutiae    - points to the allocated minutiae list
   Return Code:
	  Zero      - successful completion
	  Negative  - system error
**************************************************************************/
int QualityMap::alloc_minutiae(MINUTIAE **ominutiae, const int max_minutiae)
{
	MINUTIAE *minutiae;

	minutiae = (MINUTIAE *)malloc(sizeof(MINUTIAE));
	if(minutiae == (MINUTIAE *)NULL){
		fprintf(stderr, "ERROR : alloc_minutiae : malloc : minutiae\n");
		exit(-430);
	}
	minutiae->list = (MINUTIAQ **)malloc(max_minutiae * sizeof(MINUTIAQ *));
	if(minutiae->list == (MINUTIAQ **)NULL){
		fprintf(stderr, "ERROR : alloc_minutiae : malloc : minutiae->list\n");
		exit(-431);
	}

	minutiae->alloc = max_minutiae;
	minutiae->num = 0;

	*ominutiae = minutiae;
	return(0);
}


/*************************************************************************
**************************************************************************
#cat: gen_image_maps - Computes a set of image maps based on Version 2
#cat:            of the NIST LFS System.  The first map is a Direction Map
#cat:            which is a 2D vector of integer directions, where each
#cat:            direction represents the dominant ridge flow in a block of
#cat:            the input grayscale image.  The Low Contrast Map flags
#cat:            blocks with insufficient contrast.  The Low Flow Map flags
#cat:            blocks with insufficient ridge flow.  The High Curve Map
#cat:            flags blocks containing high curvature. This routine will
#cat:            generate maps for an arbitrarily sized, non-square, image.

   Input:
	  pdata     - padded input image data (8 bits [0..256) grayscale)
	  pw        - padded width (in pixels) of the input image
	  ph        - padded height (in pixels) of the input image
	  dir2rad   - lookup table for converting integer directions
	  dftwaves  - structure containing the DFT wave forms
	  dftgrids  - structure containing the rotated pixel grid offsets
	  lfsparms  - parameters and thresholds for controlling LFS
   Output:
	  odmap     - points to the created Direction Map
	  olcmap    - points to the created Low Contrast Map
	  olfmap    - points to the Low Ridge Flow Map
	  ohcmap    - points to the High Curvature Map
	  omw       - width (in blocks) of the maps
	  omh       - height (in blocks) of the maps
**************************************************************************/

void QualityMap::gen_image_maps(int **odmap, int **olcmap, int **olfmap, int **ohcmap,
								int *omw, int *omh,
								unsigned char *pdata, const int pw, const int ph,
								const DIR2RAD *dir2rad, const DFTWAVES *dftwaves,
								const ROTGRIDS *dftgrids, const LFSPARMS *lfsparms)
{
	int *direction_map_LOCAL, *low_contrast_map_LOCAL, *low_flow_map_LOCAL, *high_curve_map_LOCAL;
	int mw, mh, iw, ih;
	int *blkoffs;
	int ret; /* return code */

	/* 1. Compute block offsets for the entire image, accounting for pad */
	/* Block_offsets() assumes square block (grid), so ERROR otherwise. */
	if(dftgrids->grid_w != dftgrids->grid_h){
		qDebug() << "ERROR: DFT grids must be square";
	}
	/* Compute unpadded image dimensions. */
	iw = pw - (dftgrids->pad<<1);
	ih = ph - (dftgrids->pad<<1);
	if((ret = block_offsets(&blkoffs, &mw, &mh, iw, ih,
							dftgrids->pad, lfsparms->blocksize))){
		qDebug() << "ERROR: block_offsets()";
	}


	/* 2. Generate initial Direction Map and Low Contrast Map*/
	if((ret = gen_initial_maps(&direction_map_LOCAL, &low_contrast_map_LOCAL,
							   &low_flow_map_LOCAL, blkoffs, mw, mh,
							   pdata, pw, ph, dftwaves, dftgrids, lfsparms))){
		/* Free memory allocated to this point. */
		free(blkoffs);
		qDebug() << "ERROR: gen_initial_maps()";
	}


	if((ret = morph_TF_map(low_flow_map_LOCAL, mw, mh, lfsparms))){
		qDebug() << "ERROR: morph_TF_map()";
	}

	/* 3. Remove directions that are inconsistent with neighbors */
	remove_incon_dirs(direction_map_LOCAL, mw, mh, dir2rad, lfsparms);


	/* 4. Smooth Direction Map values with their neighbors */
	smooth_direction_map(direction_map_LOCAL, low_contrast_map_LOCAL, mw, mh,
						 dir2rad, lfsparms);

	/* 5. Interpolate INVALID direction blocks with their valid neighbors. */
	if((ret = interpolate_direction_map(direction_map_LOCAL, low_contrast_map_LOCAL,
										mw, mh, lfsparms))){
		qDebug() << "ERROR: interpolate_direction_map()";
	}

	/* May be able to skip steps 6 and/or 7 if computation time */
	/* is a critical factor.                                    */

	/* 6. Remove directions that are inconsistent with neighbors */
	remove_incon_dirs(direction_map_LOCAL, mw, mh, dir2rad, lfsparms);

	/* 7. Smooth Direction Map values with their neighbors. */
	smooth_direction_map(direction_map_LOCAL, low_contrast_map_LOCAL, mw, mh,
						 dir2rad, lfsparms);

	/* 8. Set the Direction Map values in the image margin to INVALID. */
	set_margin_blocks(direction_map_LOCAL, mw, mh, INVALID_DIR);

	/* 9. Generate High Curvature Map from interpolated Direction Map. */
	if((ret = gen_high_curve_map(&high_curve_map_LOCAL, direction_map_LOCAL, mw, mh,
								 lfsparms))){
		qDebug() << "ERROR: gen_high_curve_map()";
	}

	/* Deallocate working memory. */
	free(blkoffs);

	*odmap = direction_map_LOCAL;
	*olcmap = low_contrast_map_LOCAL;
	*olfmap = low_flow_map_LOCAL;
	*ohcmap = high_curve_map_LOCAL;
	*omw = mw;
	*omh = mh;
}



/*************************************************************************
**************************************************************************
#cat: binarize_V2 - Takes a padded grayscale input image and its associated
#cat:              Direction Map and produces a binarized version of the
#cat:              image.  It then fills horizontal and vertical "holes" in
#cat:              the binary image results.  Note that the input image must
#cat:              be padded sufficiently to contain in memory rotated
#cat:              directional binarization grids applied to pixels along the
#cat:              perimeter of the input image.

   Input:
	  pdata       - padded input grayscale image
	  pw          - padded width (in pixels) of input image
	  ph          - padded height (in pixels) of input image
	  direction_map - 2-D vector of discrete ridge flow directions
	  mw          - width (in blocks) of the map
	  mh          - height (in blocks) of the map
	  dirbingrids - set of rotated grid offsets used for directional
					binarization
	  lfsparms    - parameters and thresholds for controlling LFS
   Output:
	  odata - points to created (unpadded) binary image
	  ow    - width of binary image
	  oh    - height of binary image
   Return Code:
	  Zero     - successful completion
	  Negative - system error
**************************************************************************/
int QualityMap::binarize_V2(unsigned char **odata, int *ow, int *oh,
							unsigned char *pdata, const int pw, const int ph,
							int *direction_map, const int mw, const int mh,
							const ROTGRIDS *dirbingrids, const LFSPARMS *lfsparms)
{
	unsigned char *bdata;
	int i, bw, bh, ret; /* return code */

	/* 1. Binarize the padded input image using directional block info. */
	if((ret = binarize_image_V2(&bdata, &bw, &bh, pdata, pw, ph,
								direction_map, mw, mh,
								lfsparms->blocksize, dirbingrids))){
		return(ret);
	}

	/* 2. Fill black and white holes in binary image. */
	/* LFS scans the binary image, filling holes, 3 times. */
	for(i = 0; i < lfsparms->num_fill_holes; i++)
		fill_holes(bdata, bw, bh);

	/* Return binarized input image. */
	*odata = bdata;
	*ow = bw;
	*oh = bh;
	return(0);
}


/*************************************************************************
**************************************************************************
#cat: binarize_image_V2 - Takes a grayscale input image and its associated
#cat:              Direction Map and generates a binarized version of the
#cat:              image.  Note that there is no "Isotropic" binarization
#cat:              used in this version.

   Input:
	  pdata       - padded input grayscale image
	  pw          - padded width (in pixels) of input image
	  ph          - padded height (in pixels) of input image
	  direction_map - 2-D vector of discrete ridge flow directions
	  mw          - width (in blocks) of the map
	  mh          - height (in blocks) of the map
	  blocksize   - dimension (in pixels) of each NMAP block
	  dirbingrids - set of rotated grid offsets used for directional
					binarization
   Output:
	  odata  - points to binary image results
	  ow     - points to binary image width
	  oh     - points to binary image height
   Return Code:
	  Zero     - successful completion
	  Negative - system error
**************************************************************************/
int QualityMap::binarize_image_V2(unsigned char **odata, int *ow, int *oh,
								  unsigned char *pdata, const int pw, const int ph,
								  const int *direction_map, const int mw, const int mh,
								  const int blocksize, const ROTGRIDS *dirbingrids)
{
	int ix, iy, bw, bh, bx, by, mapval;
	unsigned char *bdata, *bptr;
	unsigned char *pptr, *spptr;

	/* Compute dimensions of "unpadded" binary image results. */
	bw = pw - (dirbingrids->pad<<1);
	bh = ph - (dirbingrids->pad<<1);

	bdata = (unsigned char *)malloc(bw*bh*sizeof(unsigned char));
	if(bdata == (unsigned char *)NULL){
		fprintf(stderr, "ERROR : binarize_image_V2 : malloc : bdata\n");
		return(-600);
	}

	bptr = bdata;
	spptr = pdata + (dirbingrids->pad * pw) + dirbingrids->pad;
	for(iy = 0; iy < bh; iy++){
		/* Set pixel pointer to start of next row in grid. */
		pptr = spptr;
		for(ix = 0; ix < bw; ix++){

			/* Compute which block the current pixel is in. */
			bx = (int)(ix/blocksize);
			by = (int)(iy/blocksize);
			/* Get corresponding value in Direction Map. */
			mapval = *(direction_map + (by*mw) + bx);
			/* If current block has has INVALID direction ... */
			if(mapval == INVALID_DIR)
				/* Set binary pixel to white (255). */
				*bptr = WHITE_PIXEL;
			/* Otherwise, if block has a valid direction ... */
			else /*if(mapval >= 0)*/
				/* Use directional binarization based on block's direction. */
				*bptr = dirbinarize(pptr, mapval, dirbingrids);

			/* Bump input and output pixel pointers. */
			pptr++;
			bptr++;
		}
		/* Bump pointer to the next row in padded input image. */
		spptr += pw;
	}

	*odata = bdata;
	*ow = bw;
	*oh = bh;
	return(0);
}


/*************************************************************************
**************************************************************************
#cat: free_minutia - Takes a minutia pointer and deallocates all memory
#cat:            associated with it.

   Input:
	  minutia - pointer to allocated minutia structure
*************************************************************************/
void QualityMap::free_minutia(MINUTIAQ *minutia)
{
	/* Deallocate sublists. */
	if(minutia->nbrs != (int *)NULL)
		free(minutia->nbrs);
	if(minutia->ridge_counts != (int *)NULL)
		free(minutia->ridge_counts);

	/* Deallocate the minutia structure. */
	free(minutia);
}



/*************************************************************************
**************************************************************************
#cat: block_offsets - Divides an image into mw X mh equally sized blocks,
#cat:       returning a list of offsets to the top left corner of each block.
#cat:       For images that are even multiples of BLOCKSIZE, blocks do not
#cat:       not overlap and are immediately adjacent to each other.  For image
#cat:       that are NOT even multiples of BLOCKSIZE, blocks continue to be
#cat:       non-overlapping up to the last column and/or last row of blocks.
#cat:       In these cases the blocks are adjacent to the edge of the image and
#cat:       extend inwards BLOCKSIZE units, overlapping the neighboring column
#cat:       or row of blocks.  This routine also accounts for image padding
#cat:       which makes things a little more "messy". This routine is primarily
#cat:       responsible providing the ability to processs arbitrarily-sized
#cat:       images.  The strategy used here is simple, but others are possible.

   Input:
	  iw        - width (in pixels) of the orginal input image
	  ih        - height (in pixels) of the orginal input image
	  pad       - the padding (in pixels) required to support the desired
				  range of block orientations for DFT analysis.  This padding
				  is required along the entire perimeter of the input image.
				  For certain applications, the pad may be zero.
	  blocksize - the width and height (in pixels) of each image block
   Output:
	  optr      - points to the list of pixel offsets to the origin of
				  each block in the "padded" input image
	  ow        - the number of horizontal blocks in the input image
	  oh        - the number of vertical blocks in the input image
   Return Code:
	  Zero     - successful completion
	  Negative - system error
**************************************************************************/
int QualityMap::block_offsets(int **optr, int *ow, int *oh,
							  const int iw, const int ih, const int pad, const int blocksize)
{
	int *blkoffs, bx, by, bw, bh, bi, bsize;
	int blkrow_start, blkrow_size, offset;
	int lastbw, lastbh;
	int pad2, pw, ph;

	/* Test if unpadded image is smaller than a single block */
	if((iw < blocksize) || (ih < blocksize)){
		fprintf(stderr,
				"ERROR : block_offsets : image must be at least %d by %d in size\n",
				blocksize, blocksize);
		return(-80);
	}

	/* Compute padded width and height of image */
	pad2 = pad<<1;
	pw = iw + pad2;
	ph = ih + pad2;

	/* Compute the number of columns and rows of blocks in the image. */
	/* Take the ceiling to account for "leftovers" at the right and   */
	/* bottom of the unpadded image */
	bw = (int)ceil(iw / (double)blocksize);
	bh = (int)ceil(ih / (double)blocksize);

	/* Total number of blocks in the image */
	bsize = bw*bh;

	/* The index of the last column */
	lastbw = bw - 1;
	/* The index of the last row */
	lastbh = bh - 1;

	/* Allocate list of block offsets */
	blkoffs = (int *)malloc(bsize * sizeof(int));
	if(blkoffs == (int *)NULL){
		fprintf(stderr, "ERROR : block_offsets : malloc : blkoffs\n");
		return(-81);
	}

	/* Current block index */
	bi = 0;

	/* Current offset from top of padded image to start of new row of  */
	/* unpadded image blocks. It is initialize to account for the      */
	/* padding and will always be indented the size of the padding     */
	/* from the left edge of the padded image.                         */
	blkrow_start = (pad * pw) + pad;

	/* Number of pixels in a row of blocks in the padded image */
	blkrow_size = pw * blocksize;  /* row width X block height */

	/* Foreach non-overlapping row of blocks in the image */
	for(by = 0; by < lastbh; by++){
		/* Current offset from top of padded image to beginning of */
		/* the next block */
		offset = blkrow_start;
		/* Foreach non-overlapping column of blocks in the image */
		for(bx = 0; bx < lastbw; bx++){
			/* Store current block offset */
			blkoffs[bi++] = offset;
			/* Bump to the beginning of the next block */
			offset += blocksize;
		}

		/* Compute and store "left-over" block in row.    */
		/* This is the block in the last column of row.   */
		/* Start at far right edge of unpadded image data */
		/* and come in BLOCKSIZE pixels.                  */
		blkoffs[bi++] = blkrow_start + iw - blocksize;
		/* Bump to beginning of next row of blocks */
		blkrow_start += blkrow_size;
	}

	/* Compute and store "left-over" row of blocks at bottom of image */
	/* Start at bottom edge of unpadded image data and come up        */
	/* BLOCKSIZE pixels. This too must account for padding.           */
	blkrow_start = ((pad + ih - blocksize) * pw) + pad;
	/* Start the block offset for the last row at this point */
	offset = blkrow_start;
	/* Foreach non-overlapping column of blocks in last row of the image */
	for(bx = 0; bx < lastbw; bx++){
		/* Store current block offset */
		blkoffs[bi++] = offset;
		/* Bump to the beginning of the next block */
		offset += blocksize;
	}

	/* Compute and store last "left-over" block in last row.      */
	/* Start at right edge of unpadded image data and come in     */
	/* BLOCKSIZE pixels.                                          */
	blkoffs[bi++] = blkrow_start + iw - blocksize;

	*optr = blkoffs;
	*ow = bw;
	*oh = bh;
	return(0);
}



/*************************************************************************
**************************************************************************
#cat: gen_initial_maps - Creates an initial Direction Map from the given
#cat:             input image.  It very important that the image be properly
#cat:             padded so that rotated grids along the boundary of the image
#cat:             do not access unkown memory.  The rotated grids are used by a
#cat:             DFT-based analysis to determine the integer directions
#cat:             in the map. Typically this initial vector of directions will
#cat:             subsequently have weak or inconsistent directions removed
#cat:             followed by a smoothing process.  The resulting Direction
#cat:             Map contains valid directions >= 0 and INVALID values = -1.
#cat:             This routine also computes and returns 2 other image maps.
#cat:             The Low Contrast Map flags blocks in the image with
#cat:             insufficient contrast.  Blocks with low contrast have a
#cat:             corresponding direction of INVALID in the Direction Map.
#cat:             The Low Flow Map flags blocks in which the DFT analyses
#cat:             could not determine a significant ridge flow.  Blocks with
#cat:             low ridge flow also have a corresponding direction of
#cat:             INVALID in the Direction Map.

   Input:
	  blkoffs   - offsets to the pixel origin of each block in the padded image
	  mw        - number of blocks horizontally in the padded input image
	  mh        - number of blocks vertically in the padded input image
	  pdata     - padded input image data (8 bits [0..256) grayscale)
	  pw        - width (in pixels) of the padded input image
	  ph        - height (in pixels) of the padded input image
	  dftwaves  - structure containing the DFT wave forms
	  dftgrids  - structure containing the rotated pixel grid offsets
	  lfsparms  - parameters and thresholds for controlling LFS
   Output:
	  odmap     - points to the newly created Direction Map
	  olcmap    - points to the newly created Low Contrast Map
   Return Code:
	  Zero     - successful completion
	  Negative - system error
**************************************************************************/
int QualityMap::gen_initial_maps(int **odmap, int **olcmap, int **olfmap,
								 int *blkoffs, const int mw, const int mh,
								 unsigned char *pdata, const int pw, const int ph,
								 const DFTWAVES *dftwaves, const  ROTGRIDS *dftgrids,
								 const LFSPARMS *lfsparms)
{
	int *direction_map, *low_contrast_map, *low_flow_map;
	int bi, bsize, blkdir;
	int *wis, *powmax_dirs;
	double **powers, *powmaxs, *pownorms;
	int nstats;
	int ret; /* return code */
	int dft_offset;
	int xminlimit, xmaxlimit, yminlimit, ymaxlimit;
	int win_x, win_y, low_contrast_offset;

	/* Compute total number of blocks in map */
	bsize = mw * mh;

	/* Allocate Direction Map memory */
	direction_map = (int *)malloc(bsize * sizeof(int));
	if(direction_map == (int *)NULL){
		fprintf(stderr,
				"ERROR : gen_initial_maps : malloc : direction_map\n");
		return(-550);
	}
	/* Initialize the Direction Map to INVALID (-1). */
	memset(direction_map, INVALID_DIR, bsize * sizeof(int));

	/* Allocate Low Contrast Map memory */
	low_contrast_map = (int *)malloc(bsize * sizeof(int));
	if(low_contrast_map == (int *)NULL){
		free(direction_map);
		fprintf(stderr,
				"ERROR : gen_initial_maps : malloc : low_contrast_map\n");
		return(-551);
	}
	/* Initialize the Low Contrast Map to FALSE (0). */
	memset(low_contrast_map, 0, bsize * sizeof(int));

	/* Allocate Low Ridge Flow Map memory */
	low_flow_map = (int *)malloc(bsize * sizeof(int));
	if(low_flow_map == (int *)NULL){
		free(direction_map);
		free(low_contrast_map);
		fprintf(stderr,
				"ERROR : gen_initial_maps : malloc : low_flow_map\n");
		return(-552);
	}
	/* Initialize the Low Flow Map to FALSE (0). */
	memset(low_flow_map, 0, bsize * sizeof(int));

	/* Allocate DFT directional power vectors */
	if((ret = alloc_dir_powers(&powers, dftwaves->nwaves, dftgrids->ngrids))){
		/* Free memory allocated to this point. */
		free(direction_map);
		free(low_contrast_map);
		free(low_flow_map);
		return(ret);
	}

	/* Allocate DFT power statistic arrays */
	/* Compute length of statistics arrays.  Statistics not needed   */
	/* for the first DFT wave, so the length is number of waves - 1. */
	nstats = dftwaves->nwaves - 1;
	if((ret = alloc_power_stats(&wis, &powmaxs, &powmax_dirs,
								&pownorms, nstats))){
		/* Free memory allocated to this point. */
		free(direction_map);
		free(low_contrast_map);
		free(low_flow_map);
		free_dir_powers(powers, dftwaves->nwaves);
		return(ret);
	}

	/* Compute special window origin limits for determining low contrast.  */
	/* These pixel limits avoid analyzing the padded borders of the image. */
	xminlimit = dftgrids->pad;
	yminlimit = dftgrids->pad;
	xmaxlimit = pw - dftgrids->pad - lfsparms->windowsize - 1;
	ymaxlimit = ph - dftgrids->pad - lfsparms->windowsize - 1;

	/* Foreach block in image ... */
	for(bi = 0; bi < bsize; bi++){
		/* Adjust block offset from pointing to block origin to pointing */
		/* to surrounding window origin.                                 */
		dft_offset = blkoffs[bi] - (lfsparms->windowoffset * pw) -
				lfsparms->windowoffset;

		/* Compute pixel coords of window origin. */
		win_x = dft_offset % pw;
		win_y = (int)(dft_offset / pw);

		/* Make sure the current window does not access padded image pixels */
		/* for analyzing low contrast.                                      */
		win_x = std::max(xminlimit, win_x);
		win_x = std::min(xmaxlimit, win_x);
		win_y = std::max(yminlimit, win_y);
		win_y = std::min(ymaxlimit, win_y);
		low_contrast_offset = (win_y * pw) + win_x;


		/* If block is low contrast ... */
		if((ret = low_contrast_block(low_contrast_offset, lfsparms->windowsize,
									 pdata, pw, ph, lfsparms))){
			/* If system error ... */
			if(ret < 0){
				free(direction_map);
				free(low_contrast_map);
				free(low_flow_map);
				free_dir_powers(powers, dftwaves->nwaves);
				free(wis);
				free(powmaxs);
				free(powmax_dirs);
				free(pownorms);
				return(ret);
			}

			/* Otherwise, block is low contrast ... */

			low_contrast_map[bi] = TRUE;
			/* Direction Map's block is already set to INVALID. */
		}
		/* Otherwise, sufficient contrast for DFT processing ... */
		else {

			/* Compute DFT powers */
			if((ret = dft_dir_powers(powers, pdata, low_contrast_offset, pw, ph,
									 dftwaves, dftgrids))){
				/* Free memory allocated to this point. */
				free(direction_map);
				free(low_contrast_map);
				free(low_flow_map);
				free_dir_powers(powers, dftwaves->nwaves);
				free(wis);
				free(powmaxs);
				free(powmax_dirs);
				free(pownorms);
				return(ret);
			}

			/* Compute DFT power statistics, skipping first applied DFT  */
			/* wave.  This is dependent on how the primary and secondary */
			/* direction tests work below.                               */
			if((ret = dft_power_stats(wis, powmaxs, powmax_dirs, pownorms, powers,
									  1, dftwaves->nwaves, dftgrids->ngrids))){
				/* Free memory allocated to this point. */
				free(direction_map);
				free(low_contrast_map);
				free(low_flow_map);
				free_dir_powers(powers, dftwaves->nwaves);
				free(wis);
				free(powmaxs);
				free(powmax_dirs);
				free(pownorms);
				return(ret);
			}
			/* Conduct primary direction test */
			blkdir = primary_dir_test(powers, wis, powmaxs, powmax_dirs,
									  pownorms, nstats, lfsparms);

			if(blkdir != INVALID_DIR)
				direction_map[bi] = blkdir;
			else{
				/* Conduct secondary (fork) direction test */
				blkdir = secondary_fork_test(powers, wis, powmaxs, powmax_dirs,
											 pownorms, nstats, lfsparms);
				if(blkdir != INVALID_DIR)
					direction_map[bi] = blkdir;
				/* Otherwise current direction in Direction Map remains INVALID */
				else
					/* Flag the block as having LOW RIDGE FLOW. */
					low_flow_map[bi] = TRUE;
			}

		} /* End DFT */
	} /* bi */

	/* Deallocate working memory */
	free_dir_powers(powers, dftwaves->nwaves);
	free(wis);
	free(powmaxs);
	free(powmax_dirs);
	free(pownorms);

	*odmap = direction_map;
	*olcmap = low_contrast_map;
	*olfmap = low_flow_map;
	return(0);
}



/*************************************************************************
**************************************************************************
#cat: morph_tf_map - Takes a 2D vector of TRUE and FALSE values integers
#cat:               and dialates and erodes the map in an attempt to fill
#cat:               in voids in the map.

   Input:
	  tfmap    - vector of integer block values
	  mw       - width (in blocks) of the map
	  mh       - height (in blocks) of the map
	  lfsparms - parameters and thresholds for controlling LFS
   Output:
	  tfmap    - resulting morphed map
**************************************************************************/
int QualityMap::morph_TF_map(int *tfmap, const int mw, const int mh,
							 const LFSPARMS *lfsparms)
{
	unsigned char *cimage, *mimage, *cptr;
	int *mptr;
	int i;


	/* Convert TRUE/FALSE map into a binary byte image. */
	cimage = (unsigned char *)malloc(mw*mh);
	if(cimage == (unsigned char *)NULL){
		fprintf(stderr, "ERROR : morph_TF_map : malloc : cimage\n");
		return(-660);
	}

	mimage = (unsigned char *)malloc(mw*mh);
	if(mimage == (unsigned char *)NULL){
		fprintf(stderr, "ERROR : morph_TF_map : malloc : mimage\n");
		return(-661);
	}

	cptr = cimage;
	mptr = tfmap;
	for(i = 0; i < mw*mh; i++){
		*cptr++ = *mptr++;
	}

	dilate_charimage_2(cimage, mimage, mw, mh);
	dilate_charimage_2(mimage, cimage, mw, mh);
	erode_charimage_2(cimage, mimage, mw, mh);
	erode_charimage_2(mimage, cimage, mw, mh);

	cptr = cimage;
	mptr = tfmap;
	for(i = 0; i < mw*mh; i++){
		*mptr++ = *cptr++;
	}

	free(cimage);
	free(mimage);

	return(0);
}



/*************************************************************************
**************************************************************************
#cat: alloc_dir_powers - Allocates the memory associated with DFT power
#cat:           vectors.  The DFT analysis is conducted block by block in the
#cat:           input image, and within each block, N wave forms are applied
#cat:           at M different directions.

   Input:
	  nwaves - number of DFT wave forms
	  ndirs  - number of orientations (directions) used in DFT analysis
   Output:
	  opowers - pointer to the allcated power vectors
   Return Code:
	  Zero     - successful completion
	  Negative - system error
**************************************************************************/
int QualityMap::alloc_dir_powers(double ***opowers, const int nwaves, const int ndirs)
{
	int w;
	double **powers;

	/* Allocate list of double pointers to hold power vectors */
	powers = (double **)malloc(nwaves * sizeof(double*));
	if(powers == (double **)NULL){
		fprintf(stderr, "ERROR : alloc_dir_powers : malloc : powers\n");
		return(-40);
	}
	/* Foreach DFT wave ... */
	for(w = 0; w < nwaves; w++){
		/* Allocate power vector for all directions */
		powers[w] = (double *)malloc(ndirs * sizeof(double));
		if(powers[w] == (double *)NULL){
			/* Free memory allocated to this point. */
			{ int _j; for(_j = 0; _j < w; _j++){
					free(powers[_j]);
				}}
			free(powers);
			fprintf(stderr, "ERROR : alloc_dir_powers : malloc : powers[w]\n");
			return(-41);
		}
	}

	*opowers = powers;
	return(0);
}



/*************************************************************************
#cat: low_contrast_block - Takes the offset to an image block of specified
#cat:             dimension, and analyzes the pixel intensities in the block
#cat:             to determine if there is sufficient contrast for further
#cat:             processing.

   Input:
	  blkoffset - byte offset into the padded input image to the origin of
				  the block to be analyzed
	  blocksize - dimension (in pixels) of the width and height of the block
				  (passing separate blocksize from LFSPARMS on purpose)
	  pdata     - padded input image data (8 bits [0..256) grayscale)
	  pw        - width (in pixels) of the padded input image
	  ph        - height (in pixels) of the padded input image
	  lfsparms  - parameters and thresholds for controlling LFS
   Return Code:
	  TRUE     - block has sufficiently low contrast
	  FALSE    - block has sufficiently hight contrast
	  Negative - system error
**************************************************************************
**************************************************************************/
int QualityMap::low_contrast_block(const int blkoffset, const int blocksize,
								   unsigned char *pdata, const int pw, const int ph,
								   const LFSPARMS *lfsparms)
{
	int pixtable[IMG_6BIT_PIX_LIMIT], numpix;
	int px, py, pi;
	unsigned char *sptr, *pptr;
	int delta;
	double tdbl;
	int prctmin = 0, prctmax = 0, prctthresh;
	int pixsum, found;

	numpix = blocksize*blocksize;
	memset(pixtable, 0, IMG_6BIT_PIX_LIMIT*sizeof(int));

	tdbl = (lfsparms->percentile_min_max/100.0) * (double)(numpix-1);
	tdbl = trunc_dbl_precision(tdbl, TRUNC_SCALE);
	prctthresh = sround(tdbl);

	sptr = pdata+blkoffset;
	for(py = 0; py < blocksize; py++){
		pptr = sptr;
		for(px = 0; px < blocksize; px++){
			pixtable[*pptr]++;
			pptr++;
		}
		sptr += pw;
	}

	pi = 0;
	pixsum = 0;
	found = FALSE;
	while(pi < IMG_6BIT_PIX_LIMIT){
		pixsum += pixtable[pi];
		if(pixsum >= prctthresh){
			prctmin = pi;
			found = TRUE;
			break;
		}
		pi++;
	}
	if(!found){
		fprintf(stderr,
				"ERROR : low_contrast_block : min percentile pixel not found\n");
		return(-510);
	}

	pi = IMG_6BIT_PIX_LIMIT-1;
	pixsum = 0;
	found = FALSE;
	while(pi >= 0){
		pixsum += pixtable[pi];
		if(pixsum >= prctthresh){
			prctmax = pi;
			found = TRUE;
			break;
		}
		pi--;
	}
	if(!found){
		fprintf(stderr,
				"ERROR : low_contrast_block : max percentile pixel not found\n");
		return(-511);
	}

	delta = prctmax - prctmin;

	if(delta < lfsparms->min_contrast_delta)
		return(TRUE);
	else
		return(FALSE);
}



/*************************************************************************
**************************************************************************
#cat: alloc_power_stats - Allocates memory associated with set of statistics
#cat:             derived from DFT power vectors computed in a block of the
#cat:             input image.  Statistics are not computed for the lowest DFT
#cat:             wave form, so the length of the statistics arrays is 1 less
#cat:             than the number of DFT wave forms used.  The staistics
#cat:             include the Maximum power for each wave form, the direction
#cat:             at which the maximum power occured, and a normalized value
#cat:             for the maximum power.  In addition, the statistics are
#cat:             ranked in descending order based on normalized squared
#cat:             maximum power.

   Input:
	  nstats - the number of waves forms from which statistics are to be
			   derived (N Waves - 1)
   Output:
	  owis      - points to an array to hold the ranked wave form indicies
				  of the corresponding statistics
	  opowmaxs  - points to an array to hold the maximum DFT power for each
				  wave form
	  opowmax_dirs - points to an array to hold the direction corresponding to
				  each maximum power value
	  opownorms - points to an array to hold the normalized maximum power
   Return Code:
	  Zero     - successful completion
	  Negative - system error
**************************************************************************/
int QualityMap::alloc_power_stats(int **owis, double **opowmaxs, int **opowmax_dirs,
								  double **opownorms, const int nstats)
{
	int *wis, *powmax_dirs;
	double *powmaxs, *pownorms;

	/* Allocate DFT wave index vector */
	wis = (int *)malloc(nstats * sizeof(int));
	if(wis == (int *)NULL){
		fprintf(stderr, "ERROR : alloc_power_stats : malloc : wis\n");
		return(-50);
	}

	/* Allocate max power vector */
	powmaxs = (double *)malloc(nstats * sizeof(double));
	if(powmaxs == (double *)NULL){
		/* Free memory allocated to this point. */
		free(wis);
		fprintf(stderr, "ERROR : alloc_power_stats : malloc : powmaxs\n");
		return(-51);
	}

	/* Allocate max power direction vector */
	powmax_dirs = (int *)malloc(nstats * sizeof(int));
	if(powmax_dirs == (int *)NULL){
		/* Free memory allocated to this point. */
		free(wis);
		free(powmaxs);
		fprintf(stderr, "ERROR : alloc_power_stats : malloc : powmax_dirs\n");
		return(-52);
	}

	/* Allocate normalized power vector */
	pownorms = (double *)malloc(nstats * sizeof(double));
	if(pownorms == (double *)NULL){
		/* Free memory allocated to this point. */
		free(wis);
		free(powmaxs);
		free(pownorms);
		fprintf(stderr, "ERROR : alloc_power_stats : malloc : pownorms\n");
		return(-53);
	}

	*owis = wis;
	*opowmaxs = powmaxs;
	*opowmax_dirs = powmax_dirs;
	*opownorms = pownorms;
	return(0);
}



/*************************************************************************
**************************************************************************
#cat: dft_dir_powers - Conducts the DFT analysis on a block of image data.
#cat:         The image block is sampled across a range of orientations
#cat:         (directions) and multiple wave forms of varying frequency are
#cat:         applied at each orientation.  At each orentation, pixels are
#cat:         accumulated along each rotated pixel row, creating a vector
#cat:         of pixel row sums.  Each DFT wave form is then applied
#cat:         individually to this vector of pixel row sums.  A DFT power
#cat:         value is computed for each wave form (frequency0 at each
#cat:         orientaion within the image block.  Therefore, the resulting DFT
#cat:         power vectors are of dimension (N Waves X M Directions).
#cat:         The power signatures derived form this process are used to
#cat:         determine dominant direction flow within the image block.

   Input:
	  pdata     - the padded input image.  It is important that the image
				  be properly padded, or else the sampling at various block
				  orientations may result in accessing unkown memory.
	  blkoffset - the pixel offset form the origin of the padded image to
				  the origin of the current block in the image
	  pw        - the width (in pixels) of the padded input image
	  ph        - the height (in pixels) of the padded input image
	  dftwaves  - structure containing the DFT wave forms
	  dftgrids  - structure containing the rotated pixel grid offsets
   Output:
	  powers    - DFT power computed from each wave form frequencies at each
				  orientation (direction) in the current image block
   Return Code:
	  Zero     - successful completion
	  Negative - system error
**************************************************************************/
int QualityMap::dft_dir_powers(double **powers, unsigned char *pdata,
							   const int blkoffset, const int pw, const int ph,
							   const DFTWAVES *dftwaves, const ROTGRIDS *dftgrids)
{
	int w, dir;
	int *rowsums;
	unsigned char *blkptr;

	/* Allocate line sum vector, and initialize to zeros */
	/* This routine requires square block (grid), so ERROR otherwise. */
	if(dftgrids->grid_w != dftgrids->grid_h){
		fprintf(stderr, "ERROR : dft_dir_powers : DFT grids must be square\n");
		return(-90);
	}
	rowsums = (int *)malloc(dftgrids->grid_w * sizeof(int));
	if(rowsums == (int *)NULL){
		fprintf(stderr, "ERROR : dft_dir_powers : malloc : rowsums\n");
		return(-91);
	}

	/* Foreach direction ... */
	for(dir = 0; dir < dftgrids->ngrids; dir++){
		/* Compute vector of line sums from rotated grid */
		blkptr = pdata + blkoffset;
		sum_rot_block_rows(rowsums, blkptr,
						   dftgrids->grids[dir], dftgrids->grid_w);

		/* Foreach DFT wave ... */
		for(w = 0; w < dftwaves->nwaves; w++){
			dft_power(&(powers[w][dir]), rowsums,
					  dftwaves->waves[w], dftwaves->wavelen);
		}
	}

	/* Deallocate working memory. */
	free(rowsums);

	return(0);
}



/*************************************************************************
**************************************************************************
#cat: primary_dir_test - Applies the primary set of criteria for selecting
#cat:                    an IMAP integer direction from a set of DFT results
#cat:                    computed from a block of image data

   Input:
	  powers      - DFT power computed from each (N) wave frequencies at each
					rotation direction in the current image block
	  wis         - sorted order of the highest N-1 frequency power statistics
	  powmaxs     - maximum power for each of the highest N-1 frequencies
	  powmax_dirs - directions associated with each of the N-1 maximum powers
	  pownorms    - normalized power for each of the highest N-1 frequencies
	  nstats      - N-1 wave frequencies (where N is the length of dft_coefs)
	  lfsparms    - parameters and thresholds for controlling LFS
   Return Code:
	  Zero or Positive - The selected IMAP integer direction
	  INVALID_DIR - IMAP Integer direction could not be determined
**************************************************************************/
int QualityMap::primary_dir_test(double **powers, const int *wis,
								 const double *powmaxs, const int *powmax_dirs,
								 const double *pownorms, const int nstats,
								 const LFSPARMS *lfsparms)
{
	int w;


	/* Look at max power statistics in decreasing order ... */
	for(w = 0; w < nstats; w++){
		/* 1. Test magnitude of current max power (Ex. Thresh==100000)   */
		if((powmaxs[wis[w]] > lfsparms->powmax_min) &&
				/* 2. Test magnitude of normalized max power (Ex. Thresh==3.8)   */
				(pownorms[wis[w]] > lfsparms->pownorm_min) &&
				/* 3. Test magnitude of power of lowest DFT frequency at current */
				/* max power direction and make sure it is not too big.          */
				/* (Ex. Thresh==50000000)                                        */
				(powers[0][powmax_dirs[wis[w]]] <= lfsparms->powmax_max)){

#ifdef LOG_REPORT /*vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv*/
			/* Add 1 to wis[w] to create index to original dft_coefs[] */
			fprintf(logfp,
					"         Selected Wave = %d\n", wis[w]+1);
			fprintf(logfp,
					"         1. Power Magnitude (%12.3f > %12.3f)\n",
					powmaxs[wis[w]], lfsparms->powmax_min);
			fprintf(logfp,
					"         2. Norm Power Magnitude (%9.3f > %9.3f)\n",
					pownorms[wis[w]], lfsparms->pownorm_min);
			fprintf(logfp,
					"         3. Low Freq Wave Magnitude (%12.3f <= %12.3f)\n",
					powers[0][powmax_dirs[wis[w]]], lfsparms->powmax_max);
			fprintf(logfp,
					"         PASSED\n");
			fprintf(logfp,
					"         Selected Direction = %d\n",
					powmax_dirs[wis[w]]);
#endif /*^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*/

			/* If ALL 3 criteria met, return current max power direction. */
			return(powmax_dirs[wis[w]]);

		}
	}

	/* Otherwise test failed. */

#ifdef LOG_REPORT /*vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv*/
	fprintf(logfp, "         1. Power Magnitude ( > %12.3f)\n",
			lfsparms->powmax_min);
	fprintf(logfp, "         2. Norm Power Magnitude ( > %9.3f)\n",
			lfsparms->pownorm_min);
	fprintf(logfp, "         3. Low Freq Wave Magnitude ( <= %12.3f)\n",
			lfsparms->powmax_max);
	fprintf(logfp, "         FAILED\n");
#endif /*^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*/

	return(INVALID_DIR);
}




/*************************************************************************
**************************************************************************
#cat: secondary_fork_test - Applies a secondary set of criteria for selecting
#cat:                    an IMAP integer direction from a set of DFT results
#cat:                    computed from a block of image data.  This test
#cat:                    analyzes the strongest power statistics associated
#cat:                    with a given frequency and direction and analyses
#cat:                    small changes in direction to the left and right to
#cat:                    determine if the block contains a "fork".

   Input:
	  powers      - DFT power computed from each (N) wave frequencies at each
					rotation direction in the current image block
	  wis         - sorted order of the highest N-1 frequency power statistics
	  powmaxs     - maximum power for each of the highest N-1 frequencies
	  powmax_dirs - directions associated with each of the N-1 maximum powers
	  pownorms    - normalized power for each of the highest N-1 frequencies
	  nstats      - N-1 wave frequencies (where N is the length of dft_coefs)
	  lfsparms    - parameters and thresholds for controlling LFS
   Return Code:
	  Zero or Positive - The selected IMAP integer direction
	  INVALID_DIR - IMAP Integer direction could not be determined
**************************************************************************/
int QualityMap::secondary_fork_test(double **powers, const int *wis,
									const double *powmaxs, const int *powmax_dirs,
									const double *pownorms, const int nstats,
									const LFSPARMS *lfsparms)
{
	int ldir, rdir;
	double fork_pownorm_min, fork_pow_thresh;

#ifdef LOG_REPORT
	{  int firstpart = 0; /* Flag to determine if passed 1st part ... */
		fprintf(logfp, "      Secondary\n");
#endif

		/* Relax the normalized power threshold under fork conditions. */
		fork_pownorm_min = lfsparms->fork_pct_pownorm * lfsparms->pownorm_min;

		/* 1. Test magnitude of largest max power (Ex. Thresh==100000)   */
		if((powmaxs[wis[0]] > lfsparms->powmax_min) &&
				/* 2. Test magnitude of corresponding normalized power           */
				/*    (Ex. Thresh==2.85)                                         */
				(pownorms[wis[0]] >= fork_pownorm_min) &&
				/* 3. Test magnitude of power of lowest DFT frequency at largest */
				/* max power direction and make sure it is not too big.          */
				/* (Ex. Thresh==50000000)                                        */
				(powers[0][powmax_dirs[wis[0]]] <= lfsparms->powmax_max)){

#ifdef LOG_REPORT /*vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv*/
			/* First part passed ... */
			firstpart = 1;
			fprintf(logfp,
					"         Selected Wave = %d\n", wis[0]+1);
			fprintf(logfp,
					"         1. Power Magnitude (%12.3f > %12.3f)\n",
					powmaxs[wis[0]], lfsparms->powmax_min);
			fprintf(logfp,
					"         2. Norm Power Magnitude (%9.3f >= %9.3f)\n",
					pownorms[wis[0]], fork_pownorm_min);
			fprintf(logfp,
					"         3. Low Freq Wave Magnitude (%12.3f <= %12.3f)\n",
					powers[0][powmax_dirs[wis[0]]], lfsparms->powmax_max);
#endif /*^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*/

			/* Add FORK_INTERVALs to current direction modulo NDIRS */
			rdir = (powmax_dirs[wis[0]] + lfsparms->fork_interval) %
					lfsparms->num_directions;

			/* Subtract FORK_INTERVALs from direction modulo NDIRS  */
			/* For example, FORK_INTERVAL==2 & NDIRS==16, then      */
			/*            ldir = (dir - (16-2)) % 16                */
			/* which keeps result in proper modulo range.           */
			ldir = (powmax_dirs[wis[0]] + lfsparms->num_directions -
					lfsparms->fork_interval) % lfsparms->num_directions;



			/* Set forked angle threshold to be a % of the max directional */
			/* power. (Ex. thresh==0.7*powmax)                             */
			fork_pow_thresh = powmaxs[wis[0]] * lfsparms->fork_pct_powmax;

			/* Look up and test the computed power for the left and right    */
			/* fork directions.s                                             */
			/* The power stats (and thus wis) are on the range [0..nwaves-1) */
			/* as the statistics for the first DFT wave are not included.    */
			/* The original power vectors exist for ALL DFT waves, therefore */
			/* wis indices must be added by 1 before addressing the original */
			/* powers vector.                                                */
			/* LFS permits one and only one of the fork angles to exceed     */
			/* the relative power threshold.                                 */
			if(((powers[wis[0]+1][ldir] <= fork_pow_thresh) ||
				(powers[wis[0]+1][rdir] <= fork_pow_thresh)) &&
					((powers[wis[0]+1][ldir] > fork_pow_thresh) ||
					 (powers[wis[0]+1][rdir] > fork_pow_thresh))){

#ifdef LOG_REPORT /*vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv*/
				fprintf(logfp,
						"         4. Left Power Magnitude (%12.3f > %12.3f)\n",
						powers[wis[0]+1][ldir], fork_pow_thresh);
				fprintf(logfp,
						"         5. Right Power Magnitude (%12.3f > %12.3f)\n",
						powers[wis[0]+1][rdir], fork_pow_thresh);
				fprintf(logfp, "         PASSED\n");
				fprintf(logfp,
						"         Selected Direction = %d\n", powmax_dirs[wis[0]]);
#endif /*^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*/

				/* If ALL the above criteria hold, then return the direction */
				/* of the largest max power.                                 */
				return(powmax_dirs[wis[0]]);
			}
		}

		/* Otherwise test failed. */

#ifdef LOG_REPORT /*vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv*/
		if(!firstpart){
			fprintf(logfp,
					"         1. Power Magnitude ( > %12.3f)\n",
					lfsparms->powmax_min);
			fprintf(logfp,
					"         2. Norm Power Magnitude ( > %9.3f)\n",
					fork_pownorm_min);
			fprintf(logfp,
					"         3. Low Freq Wave Magnitude ( <= %12.3f)\n",
					lfsparms->powmax_max);
		}
		else{
			fprintf(logfp, "         4. Left Power Magnitude (%12.3f > %12.3f)\n",
					powers[wis[0]+1][ldir], fork_pow_thresh);
			fprintf(logfp, "         5. Right Power Magnitude (%12.3f > %12.3f)\n",
					powers[wis[0]+1][rdir], fork_pow_thresh);
		}
		fprintf(logfp, "         FAILED\n");
	} /* Close scope of firstpart */
#endif /*^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*/

	return(INVALID_DIR);
}



/*************************************************************************
**************************************************************************
#cat: dft_power_stats - Derives statistics from a set of DFT power vectors.
#cat:           Statistics are computed for all but the lowest frequency
#cat:           wave form, including the Maximum power for each wave form,
#cat:           the direction at which the maximum power occured, and a
#cat:           normalized value for the maximum power.  In addition, the
#cat:           statistics are ranked in descending order based on normalized
#cat:           squared maximum power.  These statistics are fundamental
#cat:           to selecting a dominant direction flow for the current
#cat:           input image block.

   Input:
	  powers   - DFT power vectors (N Waves X M Directions) computed for
				 the current image block from which the values in the
				 statistics arrays are derived
	  fw       - the beginning of the range of wave form indices from which
				 the statistcs are to derived
	  tw       - the ending of the range of wave form indices from which
				 the statistcs are to derived (last index is tw-1)
	  ndirs    - number of orientations (directions) at which the DFT
				 analysis was conducted
   Output:
	  wis      - list of ranked wave form indicies of the corresponding
				 statistics based on normalized squared maximum power. These
				 indices will be used as indirect addresses when processing
				 the power statistics in descending order of "dominance"
	  powmaxs  - array holding the maximum DFT power for each wave form
				 (other than the lowest frequecy)
	  powmax_dirs - array to holding the direction corresponding to
				  each maximum power value in powmaxs
	  pownorms - array to holding the normalized maximum powers corresponding
				 to each value in powmaxs
   Return Code:
	  Zero     - successful completion
	  Negative - system error
**************************************************************************/
int QualityMap::dft_power_stats(int *wis, double *powmaxs, int *powmax_dirs,
								double *pownorms, double **powers,
								const int fw, const int tw, const int ndirs)
{
	int w, i;
	int ret; /* return code */

	for(w = fw, i = 0; w < tw; w++, i++){
		get_max_norm(&(powmaxs[i]), &(powmax_dirs[i]),
					 &(pownorms[i]), powers[w], ndirs);
	}

	/* Get sorted order of applied DFT waves based on normalized power */
	if((ret = sort_dft_waves(wis, powmaxs, pownorms, tw-fw)))
		return(ret);

	return(0);
}




/*************************************************************************
**************************************************************************
#cat: erode_charimage_2 - Erodes an 8-bit image by setting true pixels to zero
#cat:             if any of their 4 neighbors is zero.  Allocation of the
#cat:             output image is the responsibility of the caller.  The
#cat:             input image remains unchanged.  This routine will NOT
#cat:             erode pixels indiscriminately along the image border.

   Input:
	  inp       - input 8-bit image to be eroded
	  iw        - width (in pixels) of image
	  ih        - height (in pixels) of image
   Output:
	  out       - contains to the resulting eroded image
**************************************************************************/
void QualityMap::erode_charimage_2(unsigned char *inp, unsigned char *out,
								   const int iw, const int ih)
{
	int row, col;
	unsigned char *itr = inp, *otr = out;

	memcpy(out, inp, iw*ih);

	/* for true pixels. kill pixel if there is at least one false neighbor */
	for ( row = 0 ; row < ih ; row++ )
		for ( col = 0 ; col < iw ; col++ )
		{
			if (*itr)      /* erode only operates on true pixels */
			{
				/* more efficient with C's left to right evaluation of     */
				/* conjuctions. E N S functions not executed if W is false */
				if (!(get_west8_2 ((char *)itr, col        , 1 ) &&
					  get_east8_2 ((char *)itr, col, iw    , 1 ) &&
					  get_north8_2((char *)itr, row, iw    , 1 ) &&
					  get_south8_2((char *)itr, row, iw, ih, 1)))
					*otr = 0;
			}
			itr++ ; otr++;
		}
}

/*************************************************************************
**************************************************************************
#cat: dilate_charimage_2 - Dilates an 8-bit image by setting false pixels to
#cat:             one if any of their 4 neighbors is non-zero.  Allocation
#cat:             of the output image is the responsibility of the caller.
#cat:             The input image remains unchanged.

   Input:
	  inp       - input 8-bit image to be dilated
	  iw        - width (in pixels) of image
	  ih        - height (in pixels) of image
   Output:
	  out       - contains to the resulting dilated image
**************************************************************************/
void QualityMap::dilate_charimage_2(unsigned char *inp, unsigned char *out,
									const int iw, const int ih)
{
	int row, col;
	unsigned char *itr = inp, *otr = out;

	memcpy(out, inp, iw*ih);

	/* for all pixels. set pixel if there is at least one true neighbor */
	for ( row = 0 ; row < ih ; row++ )
		for ( col = 0 ; col < iw ; col++ )
		{
			if (!*itr)     /* pixel is already true, neighbors irrelevant */
			{
				/* more efficient with C's left to right evaluation of     */
				/* conjuctions. E N S functions not executed if W is false */
				if (get_west8_2 ((char *)itr, col        , 0) ||
						get_east8_2 ((char *)itr, col, iw    , 0) ||
						get_north8_2((char *)itr, row, iw    , 0) ||
						get_south8_2((char *)itr, row, iw, ih, 0))
					*otr = 1;
			}
			itr++ ; otr++;
		}
}




/*************************************************************************
**************************************************************************
#cat: sum_rot_block_rows - Computes a vector or pixel row sums by sampling
#cat:               the current image block at a given orientation.  The
#cat:               sampling is conducted using a precomputed set of rotated
#cat:               pixel offsets (called a grid) relative to the orgin of
#cat:               the image block.

   Input:
	  blkptr    - the pixel address of the origin of the current image block
	  grid_offsets - the rotated pixel offsets for a block-sized grid
				  rotated according to a specific orientation
	  blocksize - the width and height of the image block and thus the size
				  of the rotated grid
   Output:
	  rowsums   - the resulting vector of pixel row sums
**************************************************************************/
void QualityMap::sum_rot_block_rows(int *rowsums, const unsigned char *blkptr,
									const int *grid_offsets, const int blocksize)
{
	int ix, iy, gi;

	/* Initialize rotation offset index. */
	gi = 0;

	/* For each row in block ... */
	for(iy = 0; iy < blocksize; iy++){
		/* The sums are accumlated along the rotated rows of the grid, */
		/* so initialize row sum to 0.                                 */
		rowsums[iy] = 0;
		/* Foreach column in block ... */
		for(ix = 0; ix < blocksize; ix++){
			/* Accumulate pixel value at rotated grid position in image */
			rowsums[iy] += *(blkptr + grid_offsets[gi]);
			gi++;
		}
	}
}

/*************************************************************************
**************************************************************************
#cat: dft_power - Computes the DFT power by applying a specific wave form
#cat:             frequency to a vector of pixel row sums computed from a
#cat:             specific orientation of the block image

   Input:
	  rowsums - accumulated rows of pixels from within a rotated grid
				overlaying an input image block
	  wave    - the wave form (cosine and sine components) at a specific
				frequency
	  wavelen - the length of the wave form (must match the height of the
				image block which is the length of the rowsum vector)
   Output:
	  power   - the computed DFT power for the given wave form at the
				given orientation within the image block
**************************************************************************/
void QualityMap::dft_power(double *power, const int *rowsums,
						   const DFTWAVE *wave, const int wavelen)
{
	int i;
	double cospart, sinpart;

	/* Initialize accumulators */
	cospart = 0.0;
	sinpart = 0.0;

	/* Accumulate cos and sin components of DFT. */
	for(i = 0; i < wavelen; i++){
		/* Multiply each rotated row sum by its        */
		/* corresponding cos or sin point in DFT wave. */
		cospart += (rowsums[i] * wave->cos[i]);
		sinpart += (rowsums[i] * wave->sin[i]);
	}

	/* Power is the sum of the squared cos and sin components */
	*power = (cospart * cospart) + (sinpart * sinpart);
}




/*************************************************************************
**************************************************************************
#cat: get_max_norm - Analyses a DFT power vector for a specific wave form
#cat:                applied at different orientations (directions) to the
#cat:                current image block.  The routine retuns the maximum
#cat:                power value in the vector, the direction at which the
#cat:                maximum occurs, and a normalized power value.  The
#cat:                normalized power is computed as the maximum power divided
#cat:                by the average power across all the directions.  These
#cat:                simple statistics are fundamental to the selection of
#cat:                a dominant direction flow for the image block.

   Input:
	  power_vector - the DFT power values derived form a specific wave form
					 applied at different directions
	  ndirs      - the number of directions to which the wave form was applied
   Output:
	  powmax     - the maximum power value in the DFT power vector
	  powmax_dir - the direciton at which the maximum power value occured
	  pownorm    - the normalized power corresponding to the maximum power
**************************************************************************/
void QualityMap::get_max_norm(double *powmax, int *powmax_dir,
							  double *pownorm, const double *power_vector, const int ndirs)
{
	int dir;
	double max_v, powsum;
	int max_i;
	double powmean;

	/* Find max power value and store corresponding direction */
	max_v = power_vector[0];
	max_i = 0;

	/* Sum the total power in a block at a given direction */
	powsum = power_vector[0];

	/* For each direction ... */
	for(dir = 1; dir < ndirs; dir++){
		powsum += power_vector[dir];
		if(power_vector[dir] > max_v){
			max_v = power_vector[dir];
			max_i = dir;
		}
	}

	*powmax = max_v;
	*powmax_dir = max_i;

	/* Powmean is used as denominator for pownorm, so setting  */
	/* a non-zero minimum avoids possible division by zero.    */
	powmean = std::max(powsum, MIN_POWER_SUM)/(double)ndirs;

	*pownorm = *powmax / powmean;
}



/*************************************************************************
**************************************************************************
#cat: sort_dft_waves - Creates a ranked list of DFT wave form statistics
#cat:                  by sorting on the normalized squared maximum power.

   Input:
	  powmaxs  - maximum DFT power for each wave form used to derive
				 statistics
	  pownorms - normalized maximum power corresponding to values in powmaxs
	  nstats   - number of wave forms used to derive statistics (N Wave - 1)
   Output:
	  wis      - sorted list of indices corresponding to the ranked set of
				 wave form statistics.  These indices will be used as
				 indirect addresses when processing the power statistics
				 in descending order of "dominance"
   Return Code:
	  Zero     - successful completion
	  Negative - system error
**************************************************************************/
int QualityMap::sort_dft_waves(int *wis, const double *powmaxs, const double *pownorms,
							   const int nstats)
{
	int i;
	double *pownorms2;

	/* Allocate normalized power^2 array */
	pownorms2 = (double *)malloc(nstats * sizeof(double));
	if(pownorms2 == (double *)NULL){
		fprintf(stderr, "ERROR : sort_dft_waves : malloc : pownorms2\n");
		return(-100);
	}

	for(i = 0; i < nstats; i++){
		/* Wis will hold the sorted statistic indices when all is done. */
		wis[i] = i;
		/* This is normalized squared max power. */
		pownorms2[i] = powmaxs[i] * pownorms[i];
	}

	/* Sort the statistic indices on the normalized squared power. */
	bubble_sort_double_dec_2(pownorms2, wis, nstats);

	/* Deallocate the working memory. */
	free(pownorms2);

	return(0);
}



/*************************************************************************
**************************************************************************
#cat: get_south8_2 - Returns the value of the 8-bit image pixel 1 below the
#cat:                current pixel if defined else it returns  (char)0.

   Input:
	  ptr       - points to current pixel in image
	  row       - y-coord of current pixel
	  iw        - width (in pixels) of image
	  ih        - height (in pixels) of image
	  failcode  - return value if desired pixel does not exist
   Return Code:
	  Zero      - if neighboring pixel is undefined
				  (outside of image boundaries)
	  Pixel     - otherwise, value of neighboring pixel
**************************************************************************/
char QualityMap::get_south8_2(char *ptr, const int row, const int iw, const int ih,
							  const int failcode)
{
	if (row >= ih-1) /* catch case where image is undefined southwards   */
		return failcode; /* use plane geometry and return code.           */

	return *(ptr+iw);
}

/*************************************************************************
**************************************************************************
#cat: get_north8_2 - Returns the value of the 8-bit image pixel 1 above the
#cat:                current pixel if defined else it returns  (char)0.

   Input:
	  ptr       - points to current pixel in image
	  row       - y-coord of current pixel
	  iw        - width (in pixels) of image
	  failcode  - return value if desired pixel does not exist
   Return Code:
	  Zero      - if neighboring pixel is undefined
				  (outside of image boundaries)
	  Pixel     - otherwise, value of neighboring pixel
**************************************************************************/
char QualityMap::get_north8_2(char *ptr, const int row, const int iw,
							  const int failcode)
{
	if (row < 1)     /* catch case where image is undefined northwards   */
		return failcode; /* use plane geometry and return code.           */

	return *(ptr-iw);
}

/*************************************************************************
**************************************************************************
#cat: get_east8_2 - Returns the value of the 8-bit image pixel 1 right of the
#cat:               current pixel if defined else it returns  (char)0.

   Input:
	  ptr       - points to current pixel in image
	  col       - x-coord of current pixel
	  iw        - width (in pixels) of image
	  failcode  - return value if desired pixel does not exist
   Return Code:
	  Zero      - if neighboring pixel is undefined
				  (outside of image boundaries)
	  Pixel     - otherwise, value of neighboring pixel
**************************************************************************/
char QualityMap::get_east8_2(char *ptr, const int col, const int iw,
							 const int failcode)
{
	if (col >= iw-1) /* catch case where image is undefined eastwards    */
		return failcode; /* use plane geometry and return code.           */

	return *(ptr+ 1);
}

/*************************************************************************
**************************************************************************
#cat: get_west8_2 - Returns the value of the 8-bit image pixel 1 left of the
#cat:              current pixel if defined else it returns  (char)0.

   Input:
	  ptr       - points to current pixel in image
	  col       - x-coord of current pixel
	  failcode  - return value if desired pixel does not exist
   Return Code:
	  Zero      - if neighboring pixel is undefined
				  (outside of image boundaries)
	  Pixel     - otherwise, value of neighboring pixel
**************************************************************************/
char QualityMap::get_west8_2(char *ptr, const int col, const int failcode)
{
	if (col < 1)     /* catch case where image is undefined westwards     */
		return failcode; /* use plane geometry and return code.            */

	return *(ptr- 1);
}



/***************************************************************************
**************************************************************************
#cat: bubble_sort_double_dec_2 - Conducts a simple bubble sort returning a list
#cat:        of ranks in decreasing order and their associated items in sorted
#cat:        order as well.

   Input:
	  ranks - list of values to be sorted
	  items - list of items, each corresponding to a particular rank value
	  len   - length of the lists to be sorted
   Output:
	  ranks - list of values sorted in descending order
	  items - list of items in the corresponding sorted order of the ranks.
			  If these items are indices, upon return, they may be used as
			  indirect addresses reflecting the sorted order of the ranks.
****************************************************************************/
void QualityMap::bubble_sort_double_dec_2(double *ranks, int *items,  const int len)
{
	int done = 0;
	int i, p, n, titem;
	double trank;

	n = len;
	while(!done){
		done = 1;
		for (i=1, p = 0;i<n;i++, p++){
			/* If previous rank is < current rank ... */
			if(ranks[p] < ranks[i]){
				/* Swap ranks */
				trank = ranks[i];
				ranks[i] = ranks[p];
				ranks[p] = trank;
				/* Swap corresponding items */
				titem = items[i];
				items[i] = items[p];
				items[p] = titem;
				done = 0;
			}
		}
		n--;
	}
}



/*************************************************************************
**************************************************************************
#cat: remove_incon_dirs - Takes a vector of integer directions and removes
#cat:              individual directions that are too weak or inconsistent.
#cat:              Directions are tested from the center of the IMAP working
#cat:              outward in concentric squares, and the process resets to
#cat:              the center and continues until no changes take place during
#cat:              a complete pass.

   Input:
	  imap      - vector of IMAP integer directions
	  mw        - width (in blocks) of the IMAP
	  mh        - height (in blocks) of the IMAP
	  dir2rad   - lookup table for converting integer directions
	  lfsparms  - parameters and thresholds for controlling LFS
   Output:
	  imap      - vector of pruned input values
**************************************************************************/
void QualityMap::remove_incon_dirs(int *imap, const int mw, const int mh,
								   const DIR2RAD *dir2rad, const LFSPARMS *lfsparms)
{
	int cx, cy;
	int *iptr;
	int nremoved;
	int lbox, rbox, tbox, bbox;

#ifdef LOG_REPORT
	{  int numpass = 0;
		fprintf(logfp, "REMOVE MAP\n");
#endif

		/* Compute center coords of IMAP */
		cx = mw>>1;
		cy = mh>>1;

		/* Do pass, while directions have been removed in a pass ... */
		do{

#ifdef LOG_REPORT
			/* Count number of complete passes through IMAP */
			++numpass;
			fprintf(logfp, "   PASS = %d\n", numpass);
#endif

			/* Reinitialize number of removed directions to 0 */
			nremoved = 0;

			/* Start at center */
			iptr = imap + (cy * mw) + cx;
			/* If valid IMAP direction and test for removal is true ... */
			if((*iptr != INVALID_DIR)&&
					(remove_dir(imap, cx, cy, mw, mh, dir2rad, lfsparms))){

				/* Set to INVALID */
				*iptr = INVALID_DIR;
				/* Bump number of removed IMAP directions */
				nremoved++;
			}

			/* Initialize side indices of concentric boxes */
			lbox = cx-1;
			tbox = cy-1;
			rbox = cx+1;
			bbox = cy+1;

			/* Grow concentric boxes, until ALL edges of imap are exceeded */
			while((lbox >= 0) || (rbox < mw) || (tbox >= 0) || (bbox < mh)){

				/* test top edge of box */
				if(tbox >= 0)
					nremoved += test_top_edge(lbox, tbox, rbox, bbox, imap, mw, mh,
											  dir2rad, lfsparms);

				/* test right edge of box */
				if(rbox < mw)
					nremoved += test_right_edge(lbox, tbox, rbox, bbox, imap, mw, mh,
												dir2rad, lfsparms);

				/* test bottom edge of box */
				if(bbox < mh)
					nremoved += test_bottom_edge(lbox, tbox, rbox, bbox, imap, mw, mh,
												 dir2rad, lfsparms);

				/* test left edge of box */
				if(lbox >=0)
					nremoved += test_left_edge(lbox, tbox, rbox, bbox, imap, mw, mh,
											   dir2rad, lfsparms);

				/* Resize current box */
				lbox--;
				tbox--;
				rbox++;
				bbox++;
			}
		}while(nremoved);

#ifdef LOG_REPORT
	} /* Close scope of numpass */
#endif

}




/*************************************************************************
**************************************************************************
#cat: remove_dir - Determines if an IMAP direction should be removed based
#cat:              on analyzing its adjacent neighbors

   Input:
	  imap      - vector of IMAP integer directions
	  mx        - IMAP X-coord of the current direction being tested
	  my        - IMPA Y-coord of the current direction being tested
	  mw        - width (in blocks) of the IMAP
	  mh        - height (in blocks) of the IMAP
	  dir2rad   - lookup table for converting integer directions
	  lfsparms  - parameters and thresholds for controlling LFS
   Return Code:
	  Positive - direction should be removed from IMAP
	  Zero     - direction should NOT be remove from IMAP
**************************************************************************/
int QualityMap::remove_dir(int *imap, const int mx, const int my,
						   const int mw, const int mh, const DIR2RAD *dir2rad,
						   const LFSPARMS *lfsparms)
{
	int avrdir, nvalid, dist;
	double dir_strength;

	/* Compute average direction from neighbors, returning the */
	/* number of valid neighbors used in the computation, and  */
	/* the "strength" of the average direction.                */
	average_8nbr_dir(&avrdir, &dir_strength, &nvalid, imap, mx, my, mw, mh,
					 dir2rad);

	/* Conduct valid neighbor test (Ex. thresh==3) */
	if(nvalid < lfsparms->rmv_valid_nbr_min){

#ifdef LOG_REPORT /*vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv*/
		fprintf(logfp, "      BLOCK %2d (%2d, %2d)\n",
				mx+(my*mw), mx, my);
		fprintf(logfp, "         Average NBR :   %2d %6.3f %d\n",
				avrdir, dir_strength, nvalid);
		fprintf(logfp, "         1. Valid NBR (%d < %d)\n",
				nvalid, lfsparms->rmv_valid_nbr_min);
#endif /*^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*/

		return(1);
	}

	/* If stregnth of average neighbor direction is large enough to */
	/* put credence in ... (Ex. thresh==0.2)                        */
	if(dir_strength >= lfsparms->dir_strength_min){

		/* Conduct direction distance test (Ex. thresh==3) */
		/* Compute minimum absolute distance between current and       */
		/* average directions accounting for wrapping from 0 to NDIRS. */
		dist = abs(avrdir - *(imap+(my*mw)+mx));
		dist = std::min(dist, dir2rad->ndirs-dist);
		if(dist > lfsparms->dir_distance_max){

#ifdef LOG_REPORT /*vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv*/
			fprintf(logfp, "      BLOCK %2d (%2d, %2d)\n",
					mx+(my*mw), mx, my);
			fprintf(logfp, "         Average NBR :   %2d %6.3f %d\n",
					avrdir, dir_strength, nvalid);
			fprintf(logfp, "         1. Valid NBR (%d < %d)\n",
					nvalid, lfsparms->rmv_valid_nbr_min);
			fprintf(logfp, "         2. Direction Strength (%6.3f >= %6.3f)\n",
					dir_strength, lfsparms->dir_strength_min);
			fprintf(logfp, "         Current Dir =  %d, Average Dir = %d\n",
					*(imap+(my*mw)+mx), avrdir);
			fprintf(logfp, "         3. Direction Distance (%d > %d)\n",
					dist, lfsparms->dir_distance_max);
#endif /*^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*/

			return(2);
		}
	}

	/* Otherwise, the strength of the average direciton is not strong enough */
	/* to put credence in, so leave the current block's directon alone.      */

	return(0);
}



/*************************************************************************
**************************************************************************
#cat: test_top_edge - Walks the top edge of a concentric square in the IMAP,
#cat:                 testing directions along the way to see if they should
#cat:                 be removed due to being too weak or inconsistent with
#cat:                 respect to their adjacent neighbors.

   Input:
	  lbox      - left edge of current concentric square
	  tbox      - top edge of current concentric square
	  rbox      - right edge of current concentric square
	  bbox      - bottom edge of current concentric square
	  imap      - vector of IMAP integer directions
	  mw        - width (in blocks) of the IMAP
	  mh        - height (in blocks) of the IMAP
	  dir2rad   - lookup table for converting integer directions
	  lfsparms  - parameters and thresholds for controlling LFS
   Return Code:
	  Positive - direction should be removed from IMAP
	  Zero     - direction should NOT be remove from IMAP
**************************************************************************/
int QualityMap::test_top_edge(const int lbox, const int tbox, const int rbox,
							  const int bbox, int *imap, const int mw, const int mh,
							  const DIR2RAD *dir2rad, const LFSPARMS *lfsparms)
{
	int bx, by, sx, ex;
	int *iptr, *sptr, *eptr;
	int nremoved;

	/* Initialize number of directions removed on edge to 0 */
	nremoved = 0;

	/* Set start pointer to top-leftmost point of box, or set it to */
	/* the leftmost point in the IMAP row (0), whichever is larger. */
	sx = std::max(lbox, 0);
	sptr = imap + (tbox*mw) + sx;

	/* Set end pointer to either 1 point short of the top-rightmost */
	/* point of box, or set it to the rightmost point in the IMAP   */
	/* row (lastx=mw-1), whichever is smaller.                      */
	ex = std::min(rbox-1, mw-1);
	eptr = imap + (tbox*mw) + ex;

	/* For each point on box's edge ... */
	for(iptr = sptr, bx = sx, by = tbox;
		iptr <= eptr;
		iptr++, bx++){
		/* If valid IMAP direction and test for removal is true ... */
		if((*iptr != INVALID_DIR)&&
				(remove_dir(imap, bx, by, mw, mh, dir2rad, lfsparms))){
			/* Set to INVALID */
			*iptr = INVALID_DIR;
			/* Bump number of removed IMAP directions */
			nremoved++;
		}
	}

	/* Return the number of directions removed on edge */
	return(nremoved);
}

/*************************************************************************
**************************************************************************
#cat: test_right_edge - Walks the right edge of a concentric square in the
#cat:                 IMAP, testing directions along the way to see if they
#cat:                 should be removed due to being too weak or inconsistent
#cat:                 with respect to their adjacent neighbors.

   Input:
	  lbox      - left edge of current concentric square
	  tbox      - top edge of current concentric square
	  rbox      - right edge of current concentric square
	  bbox      - bottom edge of current concentric square
	  imap      - vector of IMAP integer directions
	  mw        - width (in blocks) of the IMAP
	  mh        - height (in blocks) of the IMAP
	  dir2rad   - lookup table for converting integer directions
	  lfsparms  - parameters and thresholds for controlling LFS
   Return Code:
	  Positive - direction should be removed from IMAP
	  Zero     - direction should NOT be remove from IMAP
**************************************************************************/
int QualityMap::test_right_edge(const int lbox, const int tbox, const int rbox,
								const int bbox, int *imap, const int mw, const int mh,
								const DIR2RAD *dir2rad, const LFSPARMS *lfsparms)
{
	int bx, by, sy, ey;
	int *iptr, *sptr, *eptr;
	int nremoved;

	/* Initialize number of directions removed on edge to 0 */
	nremoved = 0;

	/* Set start pointer to top-rightmost point of box, or set it to */
	/* the topmost point in IMAP column (0), whichever is larger.    */
	sy = std::max(tbox, 0);
	sptr = imap + (sy*mw) + rbox;

	/* Set end pointer to either 1 point short of the bottom-    */
	/* rightmost point of box, or set it to the bottommost point */
	/* in the IMAP column (lasty=mh-1), whichever is smaller.    */
	ey = std::min(bbox-1,mh-1);
	eptr = imap + (ey*mw) + rbox;

	/* For each point on box's edge ... */
	for(iptr = sptr, bx = rbox, by = sy;
		iptr <= eptr;
		iptr+=mw, by++){
		/* If valid IMAP direction and test for removal is true ... */
		if((*iptr != INVALID_DIR)&&
				(remove_dir(imap, bx, by, mw, mh, dir2rad, lfsparms))){
			/* Set to INVALID */
			*iptr = INVALID_DIR;
			/* Bump number of removed IMAP directions */
			nremoved++;
		}
	}

	/* Return the number of directions removed on edge */
	return(nremoved);
}

/*************************************************************************
**************************************************************************
#cat: test_bottom_edge - Walks the bottom edge of a concentric square in the
#cat:               IMAP, testing directions along the way to see if they
#cat:               should be removed due to being too weak or inconsistent
#cat:               with respect to their adjacent neighbors.
   Input:
	  lbox      - left edge of current concentric square
	  tbox      - top edge of current concentric square
	  rbox      - right edge of current concentric square
	  bbox      - bottom edge of current concentric square
	  imap      - vector of IMAP integer directions
	  mw        - width (in blocks) of the IMAP
	  mh        - height (in blocks) of the IMAP
	  dir2rad   - lookup table for converting integer directions
	  lfsparms  - parameters and thresholds for controlling LFS
   Return Code:
	  Positive - direction should be removed from IMAP
	  Zero     - direction should NOT be remove from IMAP
**************************************************************************/
int QualityMap::test_bottom_edge(const int lbox, const int tbox, const int rbox,
								 const int bbox, int *imap, const int mw, const int mh,
								 const DIR2RAD *dir2rad, const LFSPARMS *lfsparms)
{
	int bx, by, sx, ex;
	int *iptr, *sptr, *eptr;
	int nremoved;

	/* Initialize number of directions removed on edge to 0 */
	nremoved = 0;

	/* Set start pointer to bottom-rightmost point of box, or set it to the */
	/* rightmost point in the IMAP ROW (lastx=mw-1), whichever is smaller.  */
	sx = std::min(rbox, mw-1);
	sptr = imap + (bbox*mw) + sx;

	/* Set end pointer to either 1 point short of the bottom-    */
	/* lefttmost point of box, or set it to the leftmost point   */
	/* in the IMAP row (x=0), whichever is larger.               */
	ex = std::max(lbox-1, 0);
	eptr = imap + (bbox*mw) + ex;

	/* For each point on box's edge ... */
	for(iptr = sptr, bx = sx, by = bbox;
		iptr >= eptr;
		iptr--, bx--){
		/* If valid IMAP direction and test for removal is true ... */
		if((*iptr != INVALID_DIR)&&
				(remove_dir(imap, bx, by, mw, mh, dir2rad, lfsparms))){
			/* Set to INVALID */
			*iptr = INVALID_DIR;
			/* Bump number of removed IMAP directions */
			nremoved++;
		}
	}

	/* Return the number of directions removed on edge */
	return(nremoved);
}

/*************************************************************************
**************************************************************************
#cat: test_left_edge - Walks the left edge of a concentric square in the IMAP,
#cat:                 testing directions along the way to see if they should
#cat:                 be removed due to being too weak or inconsistent with
#cat:                 respect to their adjacent neighbors.

   Input:
	  lbox      - left edge of current concentric square
	  tbox      - top edge of current concentric square
	  rbox      - right edge of current concentric square
	  bbox      - bottom edge of current concentric square
	  imap      - vector of IMAP integer directions
	  mw        - width (in blocks) of the IMAP
	  mh        - height (in blocks) of the IMAP
	  dir2rad   - lookup table for converting integer directions
	  lfsparms  - parameters and thresholds for controlling LFS
   Return Code:
	  Positive - direction should be removed from IMAP
	  Zero     - direction should NOT be remove from IMAP
**************************************************************************/
int QualityMap::test_left_edge(const int lbox, const int tbox, const int rbox,
							   const int bbox, int *imap, const int mw, const int mh,
							   const DIR2RAD *dir2rad, const LFSPARMS *lfsparms)
{
	int bx, by, sy, ey;
	int *iptr, *sptr, *eptr;
	int nremoved;

	/* Initialize number of directions removed on edge to 0 */
	nremoved = 0;

	/* Set start pointer to bottom-leftmost point of box, or set it to */
	/* the bottommost point in IMAP column (lasty=mh-1), whichever     */
	/* is smaller.                                                     */
	sy = std::min(bbox, mh-1);
	sptr = imap + (sy*mw) + lbox;

	/* Set end pointer to either 1 point short of the top-leftmost */
	/* point of box, or set it to the topmost point in the IMAP    */
	/* column (y=0), whichever is larger.                          */
	ey = std::max(tbox-1, 0);
	eptr = imap + (ey*mw) + lbox;

	/* For each point on box's edge ... */
	for(iptr = sptr, bx = lbox, by = sy;
		iptr >= eptr;
		iptr-=mw, by--){
		/* If valid IMAP direction and test for removal is true ... */
		if((*iptr != INVALID_DIR)&&
				(remove_dir(imap, bx, by, mw, mh, dir2rad, lfsparms))){
			/* Set to INVALID */
			*iptr = INVALID_DIR;
			/* Bump number of removed IMAP directions */
			nremoved++;
		}
	}

	/* Return the number of directions removed on edge */
	return(nremoved);
}




/*************************************************************************
**************************************************************************
#cat: average_8nbr_dir - Given an IMAP direction, computes an average
#cat:                    direction from its adjacent 8 neighbors returning
#cat:                    the average direction, its strength, and the
#cat:                    number of valid direction in the neighborhood.

   Input:
	  imap      - vector of IMAP integer directions
	  mx        - IMAP X-coord of the current direction
	  my        - IMPA Y-coord of the current direction
	  mw        - width (in blocks) of the IMAP
	  mh        - height (in blocks) of the IMAP
	  dir2rad   - lookup table for converting integer directions
   Output:
	  avrdir    - the average direction computed from neighbors
	  dir_strenght - the strength of the average direction
	  nvalid    - the number of valid directions used to compute the
				  average
**************************************************************************/
void QualityMap::average_8nbr_dir(int *avrdir, double *dir_strength, int *nvalid,
								  int *imap, const int mx, const int my,
								  const int mw, const int mh,
								  const DIR2RAD *dir2rad)
{
	int *iptr;
	int e,w,n,s;
	double cospart, sinpart;
	double pi2, pi_factor, theta;
	double avr;

	/* Compute neighbor coordinates to current IMAP direction */
	e = mx+1;  /* East */
	w = mx-1;  /* West */
	n = my-1;  /* North */
	s = my+1;  /* South */

	/* Intialize accumulators */
	*nvalid = 0;
	cospart = 0.0;
	sinpart = 0.0;

	/* 1. Test NW */
	/* If NW point within IMAP boudaries ... */
	if((w >= 0) && (n >= 0)){
		iptr = imap + (n*mw) + w;
		/* If valid direction ... */
		if(*iptr != INVALID_DIR){
			/* Accumulate cosine and sine components of the direction */
			cospart += dir2rad->cos[*iptr];
			sinpart += dir2rad->sin[*iptr];
			/* Bump number of accumulated directions */
			(*nvalid)++;
		}
	}

	/* 2. Test N */
	/* If N point within IMAP boudaries ... */
	if(n >= 0){
		iptr = imap + (n*mw) + mx;
		/* If valid direction ... */
		if(*iptr != INVALID_DIR){
			/* Accumulate cosine and sine components of the direction */
			cospart += dir2rad->cos[*iptr];
			sinpart += dir2rad->sin[*iptr];
			/* Bump number of accumulated directions */
			(*nvalid)++;
		}
	}

	/* 3. Test NE */
	/* If NE point within IMAP boudaries ... */
	if((e < mw) && (n >= 0)){
		iptr = imap + (n*mw) + e;
		/* If valid direction ... */
		if(*iptr != INVALID_DIR){
			/* Accumulate cosine and sine components of the direction */
			cospart += dir2rad->cos[*iptr];
			sinpart += dir2rad->sin[*iptr];
			/* Bump number of accumulated directions */
			(*nvalid)++;
		}
	}

	/* 4. Test E */
	/* If E point within IMAP boudaries ... */
	if(e < mw){
		iptr = imap + (my*mw) + e;
		/* If valid direction ... */
		if(*iptr != INVALID_DIR){
			/* Accumulate cosine and sine components of the direction */
			cospart += dir2rad->cos[*iptr];
			sinpart += dir2rad->sin[*iptr];
			/* Bump number of accumulated directions */
			(*nvalid)++;
		}
	}

	/* 5. Test SE */
	/* If SE point within IMAP boudaries ... */
	if((e < mw) && (s < mh)){
		iptr = imap + (s*mw) + e;
		/* If valid direction ... */
		if(*iptr != INVALID_DIR){
			/* Accumulate cosine and sine components of the direction */
			cospart += dir2rad->cos[*iptr];
			sinpart += dir2rad->sin[*iptr];
			/* Bump number of accumulated directions */
			(*nvalid)++;
		}
	}

	/* 6. Test S */
	/* If S point within IMAP boudaries ... */
	if(s < mh){
		iptr = imap + (s*mw) + mx;
		/* If valid direction ... */
		if(*iptr != INVALID_DIR){
			/* Accumulate cosine and sine components of the direction */
			cospart += dir2rad->cos[*iptr];
			sinpart += dir2rad->sin[*iptr];
			/* Bump number of accumulated directions */
			(*nvalid)++;
		}
	}

	/* 7. Test SW */
	/* If SW point within IMAP boudaries ... */
	if((w >= 0) && (s < mh)){
		iptr = imap + (s*mw) + w;
		/* If valid direction ... */
		if(*iptr != INVALID_DIR){
			/* Accumulate cosine and sine components of the direction */
			cospart += dir2rad->cos[*iptr];
			sinpart += dir2rad->sin[*iptr];
			/* Bump number of accumulated directions */
			(*nvalid)++;
		}
	}

	/* 8. Test W */
	/* If W point within IMAP boudaries ... */
	if(w >= 0){
		iptr = imap + (my*mw) + w;
		/* If valid direction ... */
		if(*iptr != INVALID_DIR){
			/* Accumulate cosine and sine components of the direction */
			cospart += dir2rad->cos[*iptr];
			sinpart += dir2rad->sin[*iptr];
			/* Bump number of accumulated directions */
			(*nvalid)++;
		}
	}

	/* If there were no neighbors found with valid direction ... */
	if(*nvalid == 0){
		/* Return INVALID direction. */
		*dir_strength = 0;
		*avrdir = INVALID_DIR;
		return;
	}

	/* Compute averages of accumulated cosine and sine direction components */
	cospart /= (double)(*nvalid);
	sinpart /= (double)(*nvalid);

	/* Compute directional strength as hypotenuse (without sqrt) of average */
	/* cosine and sine direction components.  Believe this value will be on */
	/* the range of [0 .. 1].                                               */
	*dir_strength = (cospart * cospart) + (sinpart * sinpart);
	/* Need to truncate precision so that answers are consistent   */
	/* on different computer architectures when comparing doubles. */
	*dir_strength = trunc_dbl_precision(*dir_strength, TRUNC_SCALE);

	/* If the direction strength is not sufficiently high ... */
	if(*dir_strength < DIR_STRENGTH_MIN){
		/* Return INVALID direction. */
		*dir_strength = 0;
		*avrdir = INVALID_DIR;
		return;
	}

	/* Compute angle (in radians) from Arctan of avarage         */
	/* cosine and sine direction components.  I think this order */
	/* is necessary because 0 direction is vertical and positive */
	/* direction is clockwise.                                   */
	theta = atan2(sinpart, cospart);

	/* Atan2 returns theta on range [-PI..PI].  Adjust theta so that */
	/* it is on the range [0..2PI].                                  */
	pi2 = 2*M_PI_MINDTCT;
	theta += pi2;
	theta = fmod(theta, pi2);

	/* Pi_factor sets the period of the trig functions to NDIRS units in x. */
	/* For example, if NDIRS==16, then pi_factor = 2(PI/16) = .3926...      */
	/* Dividing theta (in radians) by this factor ((1/pi_factor)==2.546...) */
	/* will produce directions on the range [0..NDIRS].                     */
	pi_factor = pi2/(double)dir2rad->ndirs; /* 2(M_PI/ndirs) */

	/* Round off the direction and return it as an average direction */
	/* for the neighborhood.                                         */
	avr = theta / pi_factor;
	/* Need to truncate precision so that answers are consistent */
	/* on different computer architectures when rounding doubles. */
	avr = trunc_dbl_precision(avr, TRUNC_SCALE);
	*avrdir = sround(avr);

	/* Really do need to map values > NDIRS back onto [0..NDIRS) range. */
	*avrdir %= dir2rad->ndirs;
}




/*************************************************************************
**************************************************************************
#cat: smooth_direction_map - Takes a vector of integer directions and smooths
#cat:               them by analyzing the direction of adjacent neighbors.

   Input:
	  direction_map - vector of integer block values
	  mw        - width (in blocks) of the map
	  mh        - height (in blocks) of the map
	  dir2rad   - lookup table for converting integer directions
	  lfsparms - parameters and thresholds for controlling LFS
   Output:
	  imap      - vector of smoothed input values
**************************************************************************/
void QualityMap::smooth_direction_map(int *direction_map, int *low_contrast_map,
									  const int mw, const int mh,
									  const DIR2RAD *dir2rad, const LFSPARMS *lfsparms)
{
	int mx, my;
	int *dptr, *cptr;
	int avrdir, nvalid;
	double dir_strength;


	/* Assign pointers to beginning of both maps. */
	dptr = direction_map;
	cptr = low_contrast_map;

	/* Foreach block in maps ... */
	for(my = 0; my < mh; my++){
		for(mx = 0; mx < mw; mx++){
			/* If the current block does NOT have LOW CONTRAST ... */
			if(!*cptr){

				/* Compute average direction from neighbors, returning the */
				/* number of valid neighbors used in the computation, and  */
				/* the "strength" of the average direction.                */
				average_8nbr_dir(&avrdir, &dir_strength, &nvalid,
								 direction_map, mx, my, mw, mh, dir2rad);

				/* If average direction strength is strong enough */
				/*    (Ex. thresh==0.2)...                        */
				if(dir_strength >= lfsparms->dir_strength_min){
					/* If Direction Map direction is valid ... */
					if(*dptr != INVALID_DIR){
						/* Conduct valid neighbor test (Ex. thresh==3)... */
						if(nvalid >= lfsparms->rmv_valid_nbr_min){

#ifdef LOG_REPORT /*vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv*/
							fprintf(logfp, "   BLOCK %2d (%2d, %2d)\n",
									mx+(my*mw), mx, my);
							fprintf(logfp, "      Average NBR :   %2d %6.3f %d\n",
									avrdir, dir_strength, nvalid);
							fprintf(logfp, "      1. Valid NBR (%d >= %d)\n",
									nvalid, lfsparms->rmv_valid_nbr_min);
							fprintf(logfp, "      Valid Direction = %d\n", *dptr);
							fprintf(logfp, "      Smoothed Direction = %d\n", avrdir);
#endif /*^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*/

							/* Reassign valid direction with average direction. */
							*dptr = avrdir;
						}
					}
					/* Otherwise direction is invalid ... */
					else{
						/* Even if DIRECTION_MAP value is invalid, if number of */
						/* valid neighbors is big enough (Ex. thresh==7)...     */
						if(nvalid >= lfsparms->smth_valid_nbr_min){

#ifdef LOG_REPORT /*vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv*/
							fprintf(logfp, "   BLOCK %2d (%2d, %2d)\n",
									mx+(my*mw), mx, my);
							fprintf(logfp, "      Average NBR :   %2d %6.3f %d\n",
									avrdir, dir_strength, nvalid);
							fprintf(logfp, "      2. Invalid NBR (%d >= %d)\n",
									nvalid, lfsparms->smth_valid_nbr_min);
							fprintf(logfp, "      Invalid Direction = %d\n", *dptr);
							fprintf(logfp, "      Smoothed Direction = %d\n", avrdir);
#endif /*^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^*/

							/* Assign invalid direction with average direction. */
							*dptr = avrdir;
						}
					}
				}
			}
			/* Otherwise, block has LOW CONTRAST, so keep INVALID direction. */

			/* Bump to next block in maps. */
			dptr++;
			cptr++;
		}
	}
}




/*************************************************************************
**************************************************************************
#cat: interpolate_direction_map - Take a Direction Map and Low Contrast
#cat:             Map and attempts to fill in INVALID directions in the
#cat:             Direction Map based on a blocks valid neighbors.  The
#cat:             valid neighboring directions are combined in a weighted
#cat:             average inversely proportional to their distance from
#cat:             the block being interpolated.  Low Contrast blocks are
#cat:             used to prempt the search for a valid neighbor in a
#cat:             specific direction, which keeps the process from
#cat:             interpolating directions for blocks in the background and
#cat:             and perimeter of the fingerprint in the image.

   Input:
	  direction_map    - map of blocks containing directional ridge flow
	  low_contrast_map - map of blocks flagged as LOW CONTRAST
	  mw        - number of blocks horizontally in the maps
	  mh        - number of blocks vertically in the maps
	  lfsparms  - parameters and thresholds for controlling LFS
   Output:
	  direction_map - contains the newly interpolated results
   Return Code:
	  Zero     - successful completion
	  Negative - system error
**************************************************************************/
int QualityMap::interpolate_direction_map(int *direction_map, int *low_contrast_map,
										  const int mw, const int mh, const LFSPARMS *lfsparms)
{
	int x, y, new_dir;
	int n_dir, e_dir, s_dir, w_dir;
	int n_dist = 0, e_dist = 0, s_dist = 0, w_dist = 0, total_dist;
	int n_found, e_found, s_found, w_found, total_found;
	int n_delta = 0, e_delta = 0, s_delta = 0, w_delta = 0, total_delta;
	int nbr_x, nbr_y;
	int *omap, *dptr, *cptr, *optr;
	double avr_dir;


	/* Allocate output (interpolated) Direction Map. */
	omap = (int *)malloc(mw*mh*sizeof(int));
	if(omap == (int *)NULL){
		fprintf(stderr,
				"ERROR : interpolate_direction_map : malloc : omap\n");
		return(-520);
	}

	/* Set pointers to the first block in the maps. */
	dptr = direction_map;
	cptr = low_contrast_map;
	optr = omap;

	/* Foreach block in the maps ... */
	for(y = 0; y < mh; y++){
		for(x = 0; x < mw; x++){

			/* If image block is NOT LOW CONTRAST and has INVALID direction ... */
			if((!*cptr) && (*dptr == INVALID_DIR)){

				/* Set neighbor accumulators to 0. */
				total_found = 0;
				total_dist = 0;

				/* Find north neighbor. */
				if((n_found = find_valid_block(&n_dir, &nbr_x, &nbr_y,
											   direction_map, low_contrast_map,
											   x, y, mw, mh, 0, -1)) == FOUND){
					/* Compute north distance. */
					n_dist = y - nbr_y;
					/* Accumulate neighbor distance. */
					total_dist += n_dist;
					/* Bump number of neighbors found. */
					total_found++;
				}

				/* Find east neighbor. */
				if((e_found = find_valid_block(&e_dir, &nbr_x, &nbr_y,
											   direction_map, low_contrast_map,
											   x, y, mw, mh, 1, 0)) == FOUND){
					/* Compute east distance. */
					e_dist = nbr_x - x;
					/* Accumulate neighbor distance. */
					total_dist += e_dist;
					/* Bump number of neighbors found. */
					total_found++;
				}

				/* Find south neighbor. */
				if((s_found = find_valid_block(&s_dir, &nbr_x, &nbr_y,
											   direction_map, low_contrast_map,
											   x, y, mw, mh, 0, 1)) == FOUND){
					/* Compute south distance. */
					s_dist = nbr_y - y;
					/* Accumulate neighbor distance. */
					total_dist += s_dist;
					/* Bump number of neighbors found. */
					total_found++;
				}

				/* Find west neighbor. */
				if((w_found = find_valid_block(&w_dir, &nbr_x, &nbr_y,
											   direction_map, low_contrast_map,
											   x, y, mw, mh, -1, 0)) == FOUND){
					/* Compute west distance. */
					w_dist = x - nbr_x;
					/* Accumulate neighbor distance. */
					total_dist += w_dist;
					/* Bump number of neighbors found. */
					total_found++;
				}

				/* If a sufficient number of neighbors found (Ex. 2) ... */
				if(total_found >= lfsparms->min_interpolate_nbrs){

					/* Accumulate weighted sum of neighboring directions     */
					/* inversely related to the distance from current block. */
					total_delta = 0.0;
					/* If neighbor found to the north ... */
					if(n_found){
						n_delta = total_dist - n_dist;
						total_delta += n_delta;
					}
					/* If neighbor found to the east ... */
					if(e_found){
						e_delta = total_dist - e_dist;
						total_delta += e_delta;
					}
					/* If neighbor found to the south ... */
					if(s_found){
						s_delta = total_dist - s_dist;
						total_delta += s_delta;
					}
					/* If neighbor found to the west ... */
					if(w_found){
						w_delta = total_dist - w_dist;
						total_delta += w_delta;
					}

					avr_dir = 0.0;

					if(n_found){
						avr_dir += (n_dir*(n_delta/(double)total_delta));
					}
					if(e_found){
						avr_dir += (e_dir*(e_delta/(double)total_delta));
					}
					if(s_found){
						avr_dir += (s_dir*(s_delta/(double)total_delta));
					}
					if(w_found){
						avr_dir += (w_dir*(w_delta/(double)total_delta));
					}

					/* Need to truncate precision so that answers are consistent  */
					/* on different computer architectures when rounding doubles. */
					avr_dir = trunc_dbl_precision(avr_dir, TRUNC_SCALE);

					/* Assign interpolated direction to output Direction Map. */
					new_dir = sround(avr_dir);



					*optr = new_dir;
				}
				else{
					/* Otherwise, the direction remains INVALID. */
					*optr = *dptr;
				}
			}
			else{
				/* Otherwise, assign the current direction to the output block. */
				*optr = *dptr;
			}

			/* Bump to the next block in the maps ... */
			dptr++;
			cptr++;
			optr++;
		}
	}

	/* Copy the interpolated directions into the input map. */
	memcpy(direction_map, omap, mw*mh*sizeof(int));
	/* Deallocate the working memory. */
	free(omap);

	/* Return normally. */
	return(0);
}




/*************************************************************************
**************************************************************************
#cat: find_valid_block - Take a Direction Map, Low Contrast Map,
#cat:             Starting block address, a direction and searches the
#cat:             maps in the specified direction until either a block valid
#cat:             direction is encountered or a block flagged as LOW CONTRAST
#cat:             is encountered.  If a valid direction is located, it and the
#cat:             address of the corresponding block are returned with a
#cat:             code of FOUND.  Otherwise, a code of NOT_FOUND is returned.

   Input:
	  direction_map    - map of blocks containing directional ridge flows
	  low_contrast_map - map of blocks flagged as LOW CONTRAST
	  sx        - X-block coord where search starts in maps
	  sy        - Y-block coord where search starts in maps
	  mw        - number of blocks horizontally in the maps
	  mh        - number of blocks vertically in the maps
	  x_incr    - X-block increment to direct search
	  y_incr    - Y-block increment to direct search
   Output:
	  nbr_dir   - valid direction found
	  nbr_x     - X-block coord where valid direction found
	  nbr_y     - Y-block coord where valid direction found
   Return Code:
	  FOUND     - neighboring block with valid direction found
	  NOT_FOUND - neighboring block with valid direction NOT found
**************************************************************************/
int QualityMap::find_valid_block(int *nbr_dir, int *nbr_x, int *nbr_y,
								 int *direction_map, int *low_contrast_map,
								 const int sx, const int sy,
								 const int mw, const int mh,
								 const int x_incr, const int y_incr)
{
	int x, y, dir;

	/* Initialize starting block coords. */
	x = sx + x_incr;
	y = sy + y_incr;

	/* While we are not outside the boundaries of the map ... */
	while((x >= 0) && (x < mw) && (y >= 0) && (y < mh)){
		/* Stop unsuccessfully if we encounter a LOW CONTRAST block. */
		if(*(low_contrast_map+(y*mw)+x))
			return(NOT_FOUND);

		/* Stop successfully if we encounter a block with valid direction. */
		if((dir = *(direction_map+(y*mw)+x)) >= 0){
			*nbr_dir = dir;
			*nbr_x = x;
			*nbr_y = y;
			return(FOUND);
		}

		/* Otherwise, advance to the next block in the map. */
		x += x_incr;
		y += y_incr;
	}

	/* If we get here, then we did not find a valid block in the given */
	/* direction in the map.                                           */
	return(NOT_FOUND);
}



/*************************************************************************
**************************************************************************
#cat: set_margin_blocks - Take an image map and sets its perimeter values to
#cat:             the specified value.

   Input:
	  map       - map of blocks to be modified
	  mw        - number of blocks horizontally in the map
	  mh        - number of blocks vertically in the map
	  margin_value - value to be assigned to the perimeter blocks
   Output:
	  map       - resulting map
**************************************************************************/
void QualityMap::set_margin_blocks(int *map, const int mw, const int mh,
								   const int margin_value)
{
	int x, y;
	int *ptr1, *ptr2;

	ptr1 = map;
	ptr2 = map+((mh-1)*mw);
	for(x = 0; x < mw; x++){
		*ptr1++ = margin_value;
		*ptr2++ = margin_value;
	}

	ptr1 = map + mw;
	ptr2 = map + mw + mw - 1;
	for(y = 1; y < mh-1; y++){
		*ptr1 = margin_value;
		*ptr2 = margin_value;
		ptr1 += mw;
		ptr2 += mw;
	}

}




/*************************************************************************
**************************************************************************
#cat: gen_high_curve_map - Takes a Direction Map and generates a new map
#cat:            that flags blocks with HIGH CURVATURE.

   Input:
	  direction_map - map of blocks containing directional ridge flow
	  mw        - the width (in blocks) of the map
	  mh        - the height (in blocks) of the map
	  lfsparms  - parameters and thresholds for controlling LFS
   Output:
	  ohcmap    - points to the created High Curvature Map
   Return Code:
	  Zero     - successful completion
	  Negative - system error
**************************************************************************/
int QualityMap::gen_high_curve_map(int **ohcmap, int *direction_map,
								   const int mw, const int mh, const LFSPARMS *lfsparms)
{
	int *high_curve_map, mapsize;
	int *hptr, *dptr;
	int bx, by;
	int nvalid, cmeasure, vmeasure;

	mapsize = mw*mh;

	/* Allocate High Curvature Map. */
	high_curve_map = (int *)malloc(mapsize * sizeof(int));
	if(high_curve_map == (int *)NULL){
		fprintf(stderr,
				"ERROR: gen_high_curve_map : malloc : high_curve_map\n");
		return(-530);
	}
	/* Initialize High Curvature Map to FALSE (0). */
	memset(high_curve_map, 0, mapsize*sizeof(int));

	hptr = high_curve_map;
	dptr = direction_map;

	/* Foreach row in maps ... */
	for(by = 0; by < mh; by++){
		/* Foreach column in maps ... */
		for(bx = 0; bx < mw; bx++){

			/* Count number of valid neighbors around current block ... */
			nvalid = num_valid_8nbrs(direction_map, bx, by, mw, mh);

			/* If valid neighbors exist ... */
			if(nvalid > 0){
				/* If current block's direction is INVALID ... */
				if(*dptr == INVALID_DIR){
					/* If a sufficient number of VALID neighbors exists ... */
					if(nvalid >= lfsparms->vort_valid_nbr_min){
						/* Measure vorticity of neighbors. */
						vmeasure = vorticity(direction_map, bx, by, mw, mh,
											 lfsparms->num_directions);
						/* If vorticity is sufficiently high ... */
						if(vmeasure >= lfsparms->highcurv_vorticity_min)
							/* Flag block as HIGH CURVATURE. */
							*hptr = TRUE;
					}
				}
				/* Otherwise block has valid direction ... */
				else{
					/* Measure curvature around the valid block. */
					cmeasure = curvature(direction_map, bx, by, mw, mh,
										 lfsparms->num_directions);
					/* If curvature is sufficiently high ... */
					if(cmeasure >= lfsparms->highcurv_curvature_min)
						*hptr = TRUE;
				}
			} /* Else (nvalid <= 0) */

			/* Bump pointers to next block in maps. */
			dptr++;
			hptr++;

		} /* bx */
	} /* by */

	/* Assign High Curvature Map to output pointer. */
	*ohcmap = high_curve_map;

	/* Return normally. */
	return(0);
}



/*************************************************************************
**************************************************************************
#cat: num_valid_8nbrs - Given a block in an IMAP, counts the number of
#cat:                   immediate neighbors that have a valid IMAP direction.

   Input:
	  imap - 2-D vector of directional ridge flows
	  mx   - horizontal coord of current block in IMAP
	  my   - vertical coord of current block in IMAP
	  mw   - width (in blocks) of the IMAP
	  mh   - height (in blocks) of the IMAP
   Return Code:
	  Non-negative - the number of valid IMAP neighbors
**************************************************************************/
int QualityMap::num_valid_8nbrs(int *imap, const int mx, const int my,
								const int mw, const int mh)
{
	int e_ind, w_ind, n_ind, s_ind;
	int nvalid;

	/* Initialize VALID IMAP counter to zero. */
	nvalid = 0;

	/* Compute neighbor coordinates to current IMAP direction */
	e_ind = mx+1;  /* East index */
	w_ind = mx-1;  /* West index */
	n_ind = my-1;  /* North index */
	s_ind = my+1;  /* South index */

	/* 1. Test NW IMAP value.  */
	/* If neighbor indices are within IMAP boundaries and it is VALID ... */
	if((w_ind >= 0) && (n_ind >= 0) && (*(imap + (n_ind*mw) + w_ind) >= 0))
		/* Bump VALID counter. */
		nvalid++;

	/* 2. Test N IMAP value.  */
	if((n_ind >= 0) && (*(imap + (n_ind*mw) + mx) >= 0))
		nvalid++;

	/* 3. Test NE IMAP value. */
	if((n_ind >= 0) && (e_ind < mw) && (*(imap + (n_ind*mw) + e_ind) >= 0))
		nvalid++;

	/* 4. Test E IMAP value. */
	if((e_ind < mw) && (*(imap + (my*mw) + e_ind) >= 0))
		nvalid++;

	/* 5. Test SE IMAP value. */
	if((e_ind < mw) && (s_ind < mh) && (*(imap + (s_ind*mw) + e_ind) >= 0))
		nvalid++;

	/* 6. Test S IMAP value. */
	if((s_ind < mh) && (*(imap + (s_ind*mw) + mx) >= 0))
		nvalid++;

	/* 7. Test SW IMAP value. */
	if((w_ind >= 0) && (s_ind < mh) && (*(imap + (s_ind*mw) + w_ind) >= 0))
		nvalid++;

	/* 8. Test W IMAP value. */
	if((w_ind >= 0) && (*(imap + (my*mw) + w_ind) >= 0))
		nvalid++;

	/* Return number of neighbors with VALID IMAP values. */
	return(nvalid);
}



/*************************************************************************
**************************************************************************
#cat: vorticity - Measures the amount of cummulative curvature incurred
#cat:             among the IMAP neighbors of the given block.

   Input:
	  imap  - 2D vector of ridge flow directions
	  mx    - horizontal coord of current IMAP block
	  my    - vertical coord of current IMAP block
	  mw    - width (in blocks) of the IMAP
	  mh    - height (in blocks) of the IMAP
	  ndirs - number of possible directions in the IMAP
   Return Code:
	  Non-negative - the measured vorticity among the neighbors
**************************************************************************/
int QualityMap::vorticity(int *imap, const int mx, const int my,
						  const int mw, const int mh, const int ndirs)
{
	int e_ind, w_ind, n_ind, s_ind;
	int nw_val, n_val, ne_val, e_val, se_val, s_val, sw_val, w_val;
	int vmeasure;

	/* Compute neighbor coordinates to current IMAP direction */
	e_ind = mx+1;  /* East index */
	w_ind = mx-1;  /* West index */
	n_ind = my-1;  /* North index */
	s_ind = my+1;  /* South index */

	/* 1. Get NW IMAP value.  */
	/* If neighbor indices are within IMAP boundaries ... */
	if((w_ind >= 0) && (n_ind >= 0))
		/* Set neighbor value to IMAP value. */
		nw_val = *(imap + (n_ind*mw) + w_ind);
	else
		/* Otherwise, set the neighbor value to INVALID. */
		nw_val = INVALID_DIR;

	/* 2. Get N IMAP value.  */
	if(n_ind >= 0)
		n_val = *(imap + (n_ind*mw) + mx);
	else
		n_val = INVALID_DIR;

	/* 3. Get NE IMAP value. */
	if((n_ind >= 0) && (e_ind < mw))
		ne_val = *(imap + (n_ind*mw) + e_ind);
	else
		ne_val = INVALID_DIR;

	/* 4. Get E IMAP value. */
	if(e_ind < mw)
		e_val = *(imap + (my*mw) + e_ind);
	else
		e_val = INVALID_DIR;

	/* 5. Get SE IMAP value. */
	if((e_ind < mw) && (s_ind < mh))
		se_val = *(imap + (s_ind*mw) + e_ind);
	else
		se_val = INVALID_DIR;

	/* 6. Get S IMAP value. */
	if(s_ind < mh)
		s_val = *(imap + (s_ind*mw) + mx);
	else
		s_val = INVALID_DIR;

	/* 7. Get SW IMAP value. */
	if((w_ind >= 0) && (s_ind < mh))
		sw_val = *(imap + (s_ind*mw) + w_ind);
	else
		sw_val = INVALID_DIR;

	/* 8. Get W IMAP value. */
	if(w_ind >= 0)
		w_val = *(imap + (my*mw) + w_ind);
	else
		w_val = INVALID_DIR;

	/* Now that we have all IMAP neighbors, accumulate vorticity between */
	/* the neighboring directions.                                       */

	/* Initialize vorticity accumulator to zero. */
	vmeasure = 0;

	/* 1. NW & N */
	accum_nbr_vorticity(&vmeasure, nw_val, n_val, ndirs);

	/* 2. N & NE */
	accum_nbr_vorticity(&vmeasure, n_val, ne_val, ndirs);

	/* 3. NE & E */
	accum_nbr_vorticity(&vmeasure, ne_val, e_val, ndirs);

	/* 4. E & SE */
	accum_nbr_vorticity(&vmeasure, e_val, se_val, ndirs);

	/* 5. SE & S */
	accum_nbr_vorticity(&vmeasure, se_val, s_val, ndirs);

	/* 6. S & SW */
	accum_nbr_vorticity(&vmeasure, s_val, sw_val, ndirs);

	/* 7. SW & W */
	accum_nbr_vorticity(&vmeasure, sw_val, w_val, ndirs);

	/* 8. W & NW */
	accum_nbr_vorticity(&vmeasure, w_val, nw_val, ndirs);

	/* Return the accumulated vorticity measure. */
	return(vmeasure);
}



/*************************************************************************
**************************************************************************
#cat: curvature - Measures the largest change in direction between the
#cat:             current IMAP direction and its immediate neighbors.

   Input:
	  imap  - 2D vector of ridge flow directions
	  mx    - horizontal coord of current IMAP block
	  my    - vertical coord of current IMAP block
	  mw    - width (in blocks) of the IMAP
	  mh    - height (in blocks) of the IMAP
	  ndirs - number of possible directions in the IMAP
   Return Code:
	  Non-negative - maximum change in direction found (curvature)
	  Negative     - No valid neighbor found to measure change in direction
**************************************************************************/
int QualityMap::curvature(int *imap, const int mx, const int my,
						  const int mw, const int mh, const int ndirs)
{
	int *iptr;
	int e_ind, w_ind, n_ind, s_ind;
	int nw_val, n_val, ne_val, e_val, se_val, s_val, sw_val, w_val;
	int cmeasure, dist;

	/* Compute neighbor coordinates to current IMAP direction */
	e_ind = mx+1;  /* East index */
	w_ind = mx-1;  /* West index */
	n_ind = my-1;  /* North index */
	s_ind = my+1;  /* South index */

	/* 1. Get NW IMAP value.  */
	/* If neighbor indices are within IMAP boundaries ... */
	if((w_ind >= 0) && (n_ind >= 0))
		/* Set neighbor value to IMAP value. */
		nw_val = *(imap + (n_ind*mw) + w_ind);
	else
		/* Otherwise, set the neighbor value to INVALID. */
		nw_val = INVALID_DIR;

	/* 2. Get N IMAP value.  */
	if(n_ind >= 0)
		n_val = *(imap + (n_ind*mw) + mx);
	else
		n_val = INVALID_DIR;

	/* 3. Get NE IMAP value. */
	if((n_ind >= 0) && (e_ind < mw))
		ne_val = *(imap + (n_ind*mw) + e_ind);
	else
		ne_val = INVALID_DIR;

	/* 4. Get E IMAP value. */
	if(e_ind < mw)
		e_val = *(imap + (my*mw) + e_ind);
	else
		e_val = INVALID_DIR;

	/* 5. Get SE IMAP value. */
	if((e_ind < mw) && (s_ind < mh))
		se_val = *(imap + (s_ind*mw) + e_ind);
	else
		se_val = INVALID_DIR;

	/* 6. Get S IMAP value. */
	if(s_ind < mh)
		s_val = *(imap + (s_ind*mw) + mx);
	else
		s_val = INVALID_DIR;

	/* 7. Get SW IMAP value. */
	if((w_ind >= 0) && (s_ind < mh))
		sw_val = *(imap + (s_ind*mw) + w_ind);
	else
		sw_val = INVALID_DIR;

	/* 8. Get W IMAP value. */
	if(w_ind >= 0)
		w_val = *(imap + (my*mw) + w_ind);
	else
		w_val = INVALID_DIR;

	/* Now that we have all IMAP neighbors, determine largest change in */
	/* direction from current block to each of its 8 VALID neighbors.   */

	/* Initialize pointer to current IMAP value. */
	iptr = imap + (my*mw) + mx;

	/* Initialize curvature measure to negative as closest_dir_dist() */
	/* always returns -1=INVALID or a positive value.                 */
	cmeasure = -1;

	/* 1. With NW */
	/* Compute closest distance between neighboring directions. */
	dist = closest_dir_dist(*iptr, nw_val, ndirs);
	/* Keep track of maximum. */
	if(dist > cmeasure)
		cmeasure = dist;

	/* 2. With N */
	dist = closest_dir_dist(*iptr, n_val, ndirs);
	if(dist > cmeasure)
		cmeasure = dist;

	/* 3. With NE */
	dist = closest_dir_dist(*iptr, ne_val, ndirs);
	if(dist > cmeasure)
		cmeasure = dist;

	/* 4. With E */
	dist = closest_dir_dist(*iptr, e_val, ndirs);
	if(dist > cmeasure)
		cmeasure = dist;

	/* 5. With SE */
	dist = closest_dir_dist(*iptr, se_val, ndirs);
	if(dist > cmeasure)
		cmeasure = dist;

	/* 6. With S */
	dist = closest_dir_dist(*iptr, s_val, ndirs);
	if(dist > cmeasure)
		cmeasure = dist;

	/* 7. With SW */
	dist = closest_dir_dist(*iptr, sw_val, ndirs);
	if(dist > cmeasure)
		cmeasure = dist;

	/* 8. With W */
	dist = closest_dir_dist(*iptr, w_val, ndirs);
	if(dist > cmeasure)
		cmeasure = dist;

	/* Return maximum difference between current block's IMAP direction */
	/* and the rest of its VALID neighbors.                             */
	return(cmeasure);
}



/*************************************************************************
**************************************************************************
#cat: accum_nbor_vorticity - Accumlates the amount of curvature measures
#cat:                        between neighboring IMAP blocks.

   Input:
	  dir1  - first neighbor's integer IMAP direction
	  dir2  - second neighbor's integer IMAP direction
	  ndirs - number of possible IMAP directions
   Output:
	  vmeasure - accumulated vorticity among neighbors measured so far
**************************************************************************/
void QualityMap::accum_nbr_vorticity(int *vmeasure, const int dir1, const int dir2,
									 const int ndirs)
{
	int dist;

	/* Measure difference in direction between a pair of neighboring */
	/* directions.                                                   */
	/* If both neighbors are not equal and both are VALID ... */
	if((dir1 != dir2) && (dir1 >= 0)&&(dir2 >= 0)){
		/* Measure the clockwise distance from the first to the second */
		/* directions.                                                 */
		dist = dir2 - dir1;
		/* If dist is negative, then clockwise distance must wrap around */
		/* the high end of the direction range. For example:             */
		/*              dir1 = 8                                         */
		/*              dir2 = 3                                         */
		/*       and   ndirs = 16                                        */
		/*             3 - 8 = -5                                        */
		/*        so  16 - 5 = 11  (the clockwise distance from 8 to 3)  */
		if(dist < 0)
			dist += ndirs;
		/* If the change in clockwise direction is larger than 90 degrees as */
		/* in total the total number of directions covers 180 degrees.       */
		if(dist > (ndirs>>1))
			/* Decrement the vorticity measure. */
			(*vmeasure)--;
		else
			/* Otherwise, bump the vorticity measure. */
			(*vmeasure)++;
	}
	/* Otherwise both directions are either equal or  */
	/* one or both directions are INVALID, so ignore. */
}




/*************************************************************************
**************************************************************************
#cat: closest_dir_dist - Takes to integer IMAP directions and determines the
#cat:                    closest distance between them accounting for
#cat:                    wrap-around either at the beginning or ending of
#cat:                    the range of directions.

   Input:
	  dir1  - integer value of the first direction
	  dir2  - integer value of the second direction
	  ndirs - the number of possible directions
   Return Code:
	  Non-negative - distance between the 2 directions
**************************************************************************/
int QualityMap::closest_dir_dist(const int dir1, const int dir2, const int ndirs)
{
	int d1, d2, dist;

	/* Initialize distance to -1 = INVALID. */
	dist = INVALID_DIR;

	/* Measure shortest distance between to directions. */
	/* If both neighbors are VALID ... */
	if((dir1 >= 0)&&(dir2 >= 0)){
		/* Compute inner and outer distances to account for distances */
		/* that wrap around the end of the range of directions, which */
		/* may in fact be closer.                                     */
		d1 = abs(dir2 - dir1);
		d2 = ndirs - d1;
		dist = std::min(d1, d2);
	}
	/* Otherwise one or both directions are INVALID, so ignore */
	/* and return INVALID. */

	/* Return determined closest distance. */
	return(dist);
}



/*************************************************************************
**************************************************************************
#cat: fill_holes - Takes an input image and analyzes triplets of horizontal
#cat:              pixels first and then triplets of vertical pixels, filling
#cat:              in holes of width 1.  A hole is defined as the case where
#cat:              the neighboring 2 pixels are equal, AND the center pixel
#cat:              is different.  Each hole is filled with the value of its
#cat:              immediate neighbors. This routine modifies the input image.

   Input:
	  bdata - binary image data to be processed
	  iw    - width (in pixels) of the binary input image
	  ih    - height (in pixels) of the binary input image
   Output:
	  bdata - points to the results
**************************************************************************/
void QualityMap::fill_holes(unsigned char *bdata, const int iw, const int ih)
{
	int ix, iy, iw2;
	unsigned char *lptr, *mptr, *rptr, *tptr, *bptr, *sptr;

	/* 1. Fill 1-pixel wide holes in horizontal runs first ... */
	sptr = bdata + 1;
	/* Foreach row in image ... */
	for(iy = 0; iy < ih; iy++){
		/* Initialize pointers to start of next line ... */
		lptr = sptr-1;   /* Left pixel   */
		mptr = sptr;     /* Middle pixel */
		rptr = sptr+1;   /* Right pixel  */
		/* Foreach column in image (less far left and right pixels) ... */
		for(ix = 1; ix < iw-1; ix++){
			/* Do we have a horizontal hole of length 1? */
			if((*lptr != *mptr) && (*lptr == *rptr)){
				/* If so, then fill it. */
				*mptr = *lptr;
				/* Bump passed right pixel because we know it will not */
				/* be a hole.                                          */
				lptr+=2;
				mptr+=2;
				rptr+=2;
				/* We bump ix once here and then the FOR bumps it again. */
				ix++;
			}
			else{
				/* Otherwise, bump to the next pixel to the right. */
				lptr++;
				mptr++;
				rptr++;
			}
		}
		/* Bump to start of next row. */
		sptr += iw;
	}

	/* 2. Now, fill 1-pixel wide holes in vertical runs ... */
	iw2 = iw<<1;
	/* Start processing column one row down from the top of the image. */
	sptr = bdata + iw;
	/* Foreach column in image ... */
	for(ix = 0; ix < iw; ix++){
		/* Initialize pointers to start of next column ... */
		tptr = sptr-iw;   /* Top pixel     */
		mptr = sptr;      /* Middle pixel  */
		bptr = sptr+iw;   /* Bottom pixel  */
		/* Foreach row in image (less top and bottom row) ... */
		for(iy = 1; iy < ih-1; iy++){
			/* Do we have a vertical hole of length 1? */
			if((*tptr != *mptr) && (*tptr == *bptr)){
				/* If so, then fill it. */
				*mptr = *tptr;
				/* Bump passed bottom pixel because we know it will not */
				/* be a hole.                                           */
				tptr+=iw2;
				mptr+=iw2;
				bptr+=iw2;
				/* We bump iy once here and then the FOR bumps it again. */
				iy++;
			}
			else{
				/* Otherwise, bump to the next pixel below. */
				tptr+=iw;
				mptr+=iw;
				bptr+=iw;
			}
		}
		/* Bump to start of next column. */
		sptr++;
	}
}



/*************************************************************************
**************************************************************************
#cat: dirbinarize - Determines the binary value of a grayscale pixel based
#cat:               on a VALID IMAP ridge flow direction.

   CAUTION: The image to which the input pixel points must be appropriately
			padded to account for the radius of the rotated grid.  Otherwise,
			this routine may access "unkown" memory.

   Input:
	  pptr        - pointer to current grayscale pixel
	  idir        - IMAP integer direction associated with the block the
					current is in
	  dirbingrids - set of precomputed rotated grid offsets
   Return Code:
	  BLACK_PIXEL - pixel intensity for BLACK
	  WHITE_PIXEL - pixel intensity of WHITE
**************************************************************************/
int QualityMap::dirbinarize(const unsigned char *pptr, const int idir,
							const ROTGRIDS *dirbingrids)
{
	int gx, gy, gi, cy;
	int rsum, gsum, csum = 0;
	int *grid;
	double dcy;

	/* Assign nickname pointer. */
	grid = dirbingrids->grids[idir];
	/* Calculate center (0-oriented) row in grid. */
	dcy = (dirbingrids->grid_h-1)/(double)2.0;
	/* Need to truncate precision so that answers are consistent */
	/* on different computer architectures when rounding doubles. */
	dcy = trunc_dbl_precision(dcy, TRUNC_SCALE);
	cy = sround(dcy);
	/* Initialize grid's pixel offset index to zero. */
	gi = 0;
	/* Initialize grid's pixel accumulator to zero */
	gsum = 0;

	/* Foreach row in grid ... */
	for(gy = 0; gy < dirbingrids->grid_h; gy++){
		/* Initialize row pixel sum to zero. */
		rsum = 0;
		/* Foreach column in grid ... */
		for(gx = 0; gx < dirbingrids->grid_w; gx++){
			/* Accumulate next pixel along rotated row in grid. */
			rsum += *(pptr+grid[gi]);
			/* Bump grid's pixel offset index. */
			gi++;
		}
		/* Accumulate row sum into grid pixel sum. */
		gsum += rsum;
		/* If current row is center row, then save row sum separately. */
		if(gy == cy)
			csum = rsum;
	}

	/* If the center row sum treated as an average is less than the */
	/* total pixel sum in the rotated grid ...                      */
	if((csum * dirbingrids->grid_h) < gsum)
		/* Set the binary pixel to BLACK. */
		return(BLACK_PIXEL);
	else
		/* Otherwise set the binary pixel to WHITE. */
		return(WHITE_PIXEL);
}
