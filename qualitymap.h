#ifndef QUALITYMAP_H
#define QUALITYMAP_H

#include "preprocessing_config.h"

#define MY_IMG 10
#define QMAP_LEVELS  5
#define NUM_DFT_WAVES            4
#define NFEATURES      10
#define BIFURCATION     0
#define RIDGE_ENDING    1
#define DISAPPEARING    0
#define APPEARING       1
#define RELATIVE2ORIGIN          1
#define RELATIVE2CENTER          0
#define MAX_MINUTIAE          1000
#define NEIGHBOR_DELTA 2
#define IDEALSTDEV  64
#define IDEALMEAN    127
#define TRUNC_SCALE          16384.0
#define M_PI_MINDTCT		3.14159265358979323846	/* pi */
#define UNDEFINED               -1
#define INVALID_DIR             -1
#define TRUE                     1
#define FALSE                    0
#define IMG_6BIT_PIX_LIMIT      64
#define MIN_POWER_SUM           10.0
#define DIR_STRENGTH_MIN         0.2
#define WHITE_PIXEL            255
#define BLACK_PIXEL              0
#define FOUND                 TRUE
#define NOT_FOUND            FALSE
#define PAD_VALUE              128
#define JOIN_LINE_RADIUS         1
#define IMAP_BLOCKSIZE          24
#define UNUSED_INT               0
#define UNUSED_DBL               0.0
#define NUM_DIRECTIONS          16
#define START_DIR_ANGLE     (double)(M_PI_MINDTCT/2.0)    /* 90 degrees */
#define RMV_VALID_NBR_MIN        3
#define DIR_DISTANCE_MAX         3
#define SMTH_VALID_NBR_MIN       7
#define VORT_VALID_NBR_MIN       7
#define HIGHCURV_VORTICITY_MIN   5
#define HIGHCURV_CURVATURE_MIN   5
#define POWMAX_MIN          100000.0                 /*     thrhf=1e5f  */
#define POWNORM_MIN              3.8                 /*      disc=3.8f  */
#define POWMAX_MAX        50000000.0                 /*     thrlf=5e7f  */
#define FORK_INTERVAL            2
#define FORK_PCT_POWMAX          0.7
#define FORK_PCT_POWNORM         0.75
#define DIRBIN_GRID_W            7
#define DIRBIN_GRID_H            9
#define ISOBIN_GRID_DIM         11
#define NUM_FILL_HOLES           3
#define MAX_MINUTIA_DELTA       10
#define MAX_HIGH_CURVE_THETA  (double)(M_PI_MINDTCT/3.0)
#define HIGH_CURVE_HALF_CONTOUR 14
#define MIN_LOOP_LEN            20
#define MIN_LOOP_ASPECT_DIST     1.0
#define MIN_LOOP_ASPECT_RATIO    2.25
#define LINK_TABLE_DIM          20
#define MAX_LINK_DIST           20
#define MIN_THETA_DIST           5
#define MAXTRANS                 2
#define SCORE_THETA_NORM        15.0
#define SCORE_DIST_NORM         10.0
#define SCORE_DIST_WEIGHT        4.0
#define SCORE_NUMERATOR      32000.0
#define MAX_RMTEST_DIST          8
#define MAX_RMTEST_DIST_V2      16
#define MAX_HOOK_LEN            15
#define MAX_HOOK_LEN_V2         30
#define MAX_HALF_LOOP           15
#define MAX_HALF_LOOP_V2        30
#define TRANS_DIR_PIX            6
#define SMALL_LOOP_LEN          15
#define SIDE_HALF_CONTOUR        7
#define INV_BLOCK_MARGIN         6
#define RM_VALID_NBR_MIN         7
#define PORES_TRANS_R            3
#define PORES_PERP_STEPS        12
#define PORES_STEPS_FWD         10
#define PORES_STEPS_BWD          8
#define PORES_MIN_DIST2          0.5
#define PORES_MAX_RATIO          2.25
#define MAX_NBRS                 5
#define MAX_RIDGE_STEPS         10
#define MAP_BLOCKSIZE_V2         8 // BLOCK SIZE FOR QUALITY MAP
#define MAP_WINDOWSIZE_V2       24
#define MAP_WINDOWOFFSET_V2      8
#define MIN_INTERPOLATE_NBRS     2
#define PERCENTILE_MIN_MAX      10
#define MIN_CONTRAST_DELTA       5
#define TRANS_DIR_PIX_V2         4
#define INV_BLOCK_MARGIN_V2      4
#define MAX_OVERLAP_DIST         8
#define MAX_OVERLAP_JOIN_DIST    6
#define MALFORMATION_STEPS_1    10
#define MALFORMATION_STEPS_2    20
#define MIN_MALFORMATION_RATIO   2.0
#define MAX_MALFORMATION_DIST   20

#define sround(x) ((int) (((x)<0) ? (x)-0.5 : (x)+0.5))

#define trunc_dbl_precision(x, scale) ((double) (((x)<0.0) \
	? ((int)(((x)*(scale))-0.5))/(scale) \
	: ((int)(((x)*(scale))+0.5))/(scale)))

#define RADIUS_MM  ((double)(11.0 / 19.69))

typedef struct feature_pattern{
	int type;
	int appearing;
	int first[2];
	int second[2];
	int third[2];
} FEATURE_PATTERN;

/* Lookup tables for converting from integer directions */
/* to angles in radians.                                */
typedef struct dir2rad{
	int ndirs;
	double *cos;
	double *sin;
} DIR2RAD;

/* DFT wave form structure containing both cosine and   */
/* sine components for a specific frequency.            */
typedef struct dftwave{
	double *cos;
	double *sin;
} DFTWAVE;

/* DFT wave forms structure containing all wave forms  */
/* to be used in DFT analysis.                         */
typedef struct dftwaves{
	int nwaves;
	int wavelen;
	DFTWAVE **waves;
}DFTWAVES;

/* Rotated pixel offsets for a grid of specified dimensions */
/* rotated at a specified number of different orientations  */
/* (directions).  This structure used by the DFT analysis   */
/* when generating a Direction Map and also for conducting  */
/* isotropic binarization.                                  */
typedef struct rotgrids{
	int pad;
	int relative2;
	double start_angle;
	int ngrids;
	int grid_w;
	int grid_h;
	int **grids;
} ROTGRIDS;

typedef struct minutiaq{
	int x;
	int y;
	int ex;
	int ey;
	int direction;
	double reliability;
	int type;
	int appearing;
	int feature_id;
	int *nbrs;
	int *ridge_counts;
	int num_nbrs;
} MINUTIAQ;

typedef struct minutiae{
	int alloc;
	int num;
	MINUTIAQ **list;
} MINUTIAE;

/* Parameters used by LFS for setting thresholds and  */
/* defining testing criterion.                        */
typedef struct lfsparms{
	/* Image Controls */
	int    pad_value;
	int    join_line_radius;

	/* Map Controls */
	int    blocksize;       /* Pixel dimension image block.                 */
	int    windowsize;      /* Pixel dimension window surrounding block.    */
	int    windowoffset;    /* Offset in X & Y from block to window origin. */
	int    num_directions;
	double start_dir_angle;
	int    rmv_valid_nbr_min;
	double dir_strength_min;
	int    dir_distance_max;
	int    smth_valid_nbr_min;
	int    vort_valid_nbr_min;
	int    highcurv_vorticity_min;
	int    highcurv_curvature_min;
	int    min_interpolate_nbrs;
	int    percentile_min_max;
	int    min_contrast_delta;

	/* DFT Controls */
	int    num_dft_waves;
	double powmax_min;
	double pownorm_min;
	double powmax_max;
	int    fork_interval;
	double fork_pct_powmax;
	double fork_pct_pownorm;

	/* Binarization Controls */
	int    dirbin_grid_w;
	int    dirbin_grid_h;
	int    isobin_grid_dim;
	int    num_fill_holes;

	/* Minutiae Detection Controls */
	int    max_minutia_delta;
	double max_high_curve_theta;
	int    high_curve_half_contour;
	int    min_loop_len;
	double min_loop_aspect_dist;
	double min_loop_aspect_ratio;

	/* Minutiae Link Controls */
	int    link_table_dim;
	int    max_link_dist;
	int    min_theta_dist;
	int    maxtrans;
	double score_theta_norm;
	double score_dist_norm;
	double score_dist_weight;
	double score_numerator;

	/* False Minutiae Removal Controls */
	int    max_rmtest_dist;
	int    max_hook_len;
	int    max_half_loop;
	int    trans_dir_pix;
	int    small_loop_len;
	int    side_half_contour;
	int    inv_block_margin;
	int    rm_valid_nbr_min;
	int    max_overlap_dist;
	int    max_overlap_join_dist;
	int    malformation_steps_1;
	int    malformation_steps_2;
	double min_malformation_ratio;
	int    max_malformation_dist;
	int    pores_trans_r;
	int    pores_perp_steps;
	int    pores_steps_fwd;
	int    pores_steps_bwd;
	double pores_min_dist2;
	double pores_max_ratio;

	/* Ridge Counting Controls */
	int    max_nbrs;
	int    max_ridge_steps;
} LFSPARMS;

typedef QVector<std::tuple<QPoint,int,int,int>> MINUTIAE_VECTOR;

class QualityMap : public QObject
{
	Q_OBJECT
public:
	// *********   interface functions   *********//

	explicit QualityMap(QObject *parent = nullptr);
	void setParams(const cv::Mat &m_imgOriginal, QMAP_PARAMS qmapParams); // loads input image
	void computeQualityMap(); // computes quality map
	void computeQualityMapMinutiae(MINUTIAE_VECTOR& m_minutiae); // computes quality map and assigns quality to minutiae
	int *getQuality_map() const;
	cv::Mat getImgQualityMap();
	cv::Mat getQualityMap();
	int getMap_w() const;
	int getMap_h() const;
	void printMatrix(int *img, int width, int height);
	void printImage(unsigned char *img, int width, int height);

signals:

public slots:

private:
	unsigned char *idata; // image pixel data
	int iw; // image width
	int ih; // image height
	int id; // image color depth
	int ippi; // image resolution
	double pixelsPerMM = 19.685039;

	// pointers to image maps
	int *direction_map;
	int *low_contrast_map;
	int *low_flow_map;
	int *high_curve_map;
	int *quality_map;
	int map_w; // width of all image maps
	int map_h; // height of all image maps

	MINUTIAE *m_minutiae; // struct containing minutiae
	cv::Mat m_imgOriginal; // input image

	/* Allocate and initialize VERSION 2 global LFS parameters structure. */
	LFSPARMS lfsparms_V2 = {
		/* Image Controls */
		PAD_VALUE,
		JOIN_LINE_RADIUS,

		/* Map Controls */
		MAP_BLOCKSIZE_V2,
		MAP_WINDOWSIZE_V2,
		MAP_WINDOWOFFSET_V2,
		NUM_DIRECTIONS,
		START_DIR_ANGLE,
		RMV_VALID_NBR_MIN,
		DIR_STRENGTH_MIN,
		DIR_DISTANCE_MAX,
		SMTH_VALID_NBR_MIN,
		VORT_VALID_NBR_MIN,
		HIGHCURV_VORTICITY_MIN,
		HIGHCURV_CURVATURE_MIN,
		MIN_INTERPOLATE_NBRS,
		PERCENTILE_MIN_MAX,
		MIN_CONTRAST_DELTA,

		/* DFT Controls */
		NUM_DFT_WAVES,
		POWMAX_MIN,
		POWNORM_MIN,
		POWMAX_MAX,
		FORK_INTERVAL,
		FORK_PCT_POWMAX,
		FORK_PCT_POWNORM,

		/* Binarization Controls */
		DIRBIN_GRID_W,
		DIRBIN_GRID_H,
		UNUSED_INT,          /* isobin_grid_dim */
		NUM_FILL_HOLES,

		/* Minutiae Detection Controls */
		MAX_MINUTIA_DELTA,
		MAX_HIGH_CURVE_THETA,
		HIGH_CURVE_HALF_CONTOUR,
		MIN_LOOP_LEN,
		MIN_LOOP_ASPECT_DIST,
		MIN_LOOP_ASPECT_RATIO,

		/* Minutiae Link Controls */
		UNUSED_INT,          /* link_table_dim     */
		UNUSED_INT,          /* max_link_dist      */
		UNUSED_INT,          /* min_theta_dist     */
		MAXTRANS,            /* used for removing overlaps as well */
		UNUSED_DBL,          /* score_theta_norm   */
		UNUSED_DBL,          /* score_dist_norm    */
		UNUSED_DBL,          /* score_dist_weight  */
		UNUSED_DBL,          /* score_numerator    */

		/* False Minutiae Removal Controls */
		MAX_RMTEST_DIST_V2,
		MAX_HOOK_LEN_V2,
		MAX_HALF_LOOP_V2,
		TRANS_DIR_PIX_V2,
		SMALL_LOOP_LEN,
		SIDE_HALF_CONTOUR,
		INV_BLOCK_MARGIN_V2,
		RM_VALID_NBR_MIN,
		MAX_OVERLAP_DIST,
		MAX_OVERLAP_JOIN_DIST,
		MALFORMATION_STEPS_1,
		MALFORMATION_STEPS_2,
		MIN_MALFORMATION_RATIO,
		MAX_MALFORMATION_DIST,
		PORES_TRANS_R,
		PORES_PERP_STEPS,
		PORES_STEPS_FWD,
		PORES_STEPS_BWD,
		PORES_MIN_DIST2,
		PORES_MAX_RATIO,

		/* Ridge Counting Controls */
		MAX_NBRS,
		MAX_RIDGE_STEPS
	};

	// additional variables

	double dft_coefs[NUM_DFT_WAVES]={ 1,2,3,4 };
	int nbr8_dx[8] =          {  0, 1, 1, 1, 0,-1,-1,-1 };
	int nbr8_dy[8] =          { -1,-1, 0, 1, 1, 1, 0,-1 };
	int chaincodes_nbr8[9]={ 3, 2, 1,
							 4,-1, 0,
							 5, 6, 7};
	FEATURE_PATTERN feature_patterns[10]=
	{{RIDGE_ENDING,  /* a. Ridge Ending (appearing) */
	  APPEARING,
	  {0,0},
	  {0,1},
	  {0,0}},

	 {RIDGE_ENDING,  /* b. Ridge Ending (disappearing) */
	  DISAPPEARING,
	  {0,0},
	  {1,0},
	  {0,0}},

	 {BIFURCATION,   /* c. Bifurcation (disappearing) */
	  DISAPPEARING,
	  {1,1},
	  {0,1},
	  {1,1}},

	 {BIFURCATION,   /* d. Bifurcation (appearing) */
	  APPEARING,
	  {1,1},
	  {1,0},
	  {1,1}},

	 {BIFURCATION,   /* e. Bifurcation (disappearing) */
	  DISAPPEARING,
	  {1,0},
	  {0,1},
	  {1,1}},

	 {BIFURCATION,   /* f. Bifurcation (disappearing) */
	  DISAPPEARING,
	  {1,1},
	  {0,1},
	  {1,0}},

	 {BIFURCATION,   /* g. Bifurcation (appearing) */
	  APPEARING,
	  {1,1},
	  {1,0},
	  {0,1}},

	 {BIFURCATION,   /* h. Bifurcation (appearing) */
	  APPEARING,
	  {0,1},
	  {1,0},
	  {1,1}},

	 {BIFURCATION,   /* i. Bifurcation (disappearing) */
	  DISAPPEARING,
	  {1,0},
	  {0,1},
	  {1,0}},

	 {BIFURCATION,   /* j. Bifurcation (appearing) */
	  APPEARING,
	  {0,1},
	  {1,0},
	  {0,1}}};

	// *********   core library functions   *********//
	void fill_minutiae(MINUTIAE_VECTOR& m_minutiae);
	void computeImageMaps();
	void gen_quality_map();
	void free_minutiae(MINUTIAE *m_minutiae);
	void combined_minutia_quality(MINUTIAE *m_minutiae,
								  int *quality_map, const int mw, const int mh, const int blocksize,
								  unsigned char *idata, const int iw, const int ih, const int id,
								  const double ppmm);

	int pixelize_map(int **omap, const int iw, const int ih,
					 int *imap, const int mw, const int mh, const int blocksize);
	double grayscale_reliability(MINUTIAQ *minutia, unsigned char *idata,
								 const int iw, const int ih, const int radius_pix);
	void get_neighborhood_stats(double *mean, double *stdev, MINUTIAQ *minutia,
								unsigned char *idata, const int iw, const int ih,
								const int radius_pix);
	int get_max_padding_V2(const int map_windowsize, const int map_windowoffset,
						   const int dirbin_grid_w, const int dirbin_grid_h);
	int init_dir2rad(DIR2RAD **optr, const int ndirs);
	int init_dftwaves(DFTWAVES **optr, const double *dft_coefs,
					  const int nwaves, const int blocksize);
	void free_dir_powers(double **powers, const int nwaves);
	void free_rotgrids(ROTGRIDS *rotgrids);
	void free_dftwaves(DFTWAVES *dftwaves);
	void free_dir2rad(DIR2RAD *dir2rad);
	int init_rotgrids(ROTGRIDS **optr, const int iw, const int ih, const int ipad,
					  const double start_dir_angle, const int ndirs,
					  const int grid_w, const int grid_h, const int relative2);
	int pad_uchar_image(unsigned char **optr, int *ow, int *oh,
						unsigned char *idata, const int iw, const int ih,
						const int pad, const int pad_value);
	void bits_8to6(unsigned char *idata, const int iw, const int ih);
	void gray2bin(const int thresh, const int less_pix, const int greater_pix,
				  unsigned char *bdata, const int iw, const int ih);
	int alloc_minutiae(MINUTIAE **ominutiae, const int max_minutiae);
	void gen_image_maps(int **odmap, int **olcmap, int **olfmap, int **ohcmap,
						int *omw, int *omh,
						unsigned char *pdata, const int pw, const int ph,
						const DIR2RAD *dir2rad, const DFTWAVES *dftwaves,
						const ROTGRIDS *dftgrids, const LFSPARMS *lfsparms);
	int binarize_V2(unsigned char **odata, int *ow, int *oh,
					unsigned char *pdata, const int pw, const int ph,
					int *direction_map, const int mw, const int mh,
					const ROTGRIDS *dirbingrids, const LFSPARMS *lfsparms);
	int binarize_image_V2(unsigned char **odata, int *ow, int *oh,
						  unsigned char *pdata, const int pw, const int ph,
						  const int *direction_map, const int mw, const int mh,
						  const int blocksize, const ROTGRIDS *dirbingrids);
	void free_minutia(MINUTIAQ *minutia);
	int block_offsets(int **optr, int *ow, int *oh,
					  const int iw, const int ih, const int pad, const int blocksize);
	int gen_initial_maps(int **odmap, int **olcmap, int **olfmap,
						 int *blkoffs, const int mw, const int mh,
						 unsigned char *pdata, const int pw, const int ph,
						 const DFTWAVES *dftwaves, const  ROTGRIDS *dftgrids,
						 const LFSPARMS *lfsparms);
	int morph_TF_map(int *tfmap, const int mw, const int mh,
					 const LFSPARMS *lfsparms);
	int alloc_dir_powers(double ***opowers, const int nwaves, const int ndirs);
	int low_contrast_block(const int blkoffset, const int blocksize,
						   unsigned char *pdata, const int pw, const int ph,
						   const LFSPARMS *lfsparms);
	int alloc_power_stats(int **owis, double **opowmaxs, int **opowmax_dirs,
						  double **opownorms, const int nstats);
	int dft_dir_powers(double **powers, unsigned char *pdata,
					   const int blkoffset, const int pw, const int ph,
					   const DFTWAVES *dftwaves, const ROTGRIDS *dftgrids);
	int primary_dir_test(double **powers, const int *wis,
						 const double *powmaxs, const int *powmax_dirs,
						 const double *pownorms, const int nstats,
						 const LFSPARMS *lfsparms);
	int secondary_fork_test(double **powers, const int *wis,
							const double *powmaxs, const int *powmax_dirs,
							const double *pownorms, const int nstats,
							const LFSPARMS *lfsparms);
	int dft_power_stats(int *wis, double *powmaxs, int *powmax_dirs,
						double *pownorms, double **powers,
						const int fw, const int tw, const int ndirs);
	void erode_charimage_2(unsigned char *inp, unsigned char *out,
						   const int iw, const int ih);
	void dilate_charimage_2(unsigned char *inp, unsigned char *out,
							const int iw, const int ih);
	void sum_rot_block_rows(int *rowsums, const unsigned char *blkptr,
							const int *grid_offsets, const int blocksize);
	void dft_power(double *power, const int *rowsums,
				   const DFTWAVE *wave, const int wavelen);
	void get_max_norm(double *powmax, int *powmax_dir,
					  double *pownorm, const double *power_vector, const int ndirs);
	int sort_dft_waves(int *wis, const double *powmaxs, const double *pownorms,
					   const int nstats);
	char get_south8_2(char *ptr, const int row, const int iw, const int ih,
					  const int failcode);
	char get_north8_2(char *ptr, const int row, const int iw,
					  const int failcode);
	char get_east8_2(char *ptr, const int col, const int iw,
					 const int failcode);
	char get_west8_2(char *ptr, const int col, const int failcode);
	void bubble_sort_double_dec_2(double *ranks, int *items,  const int len);
	void remove_incon_dirs(int *imap, const int mw, const int mh,
						   const DIR2RAD *dir2rad, const LFSPARMS *lfsparms);
	int remove_dir(int *imap, const int mx, const int my,
				   const int mw, const int mh, const DIR2RAD *dir2rad,
				   const LFSPARMS *lfsparms);
	int test_left_edge(const int lbox, const int tbox, const int rbox,
					   const int bbox, int *imap, const int mw, const int mh,
					   const DIR2RAD *dir2rad, const LFSPARMS *lfsparms);
	int test_bottom_edge(const int lbox, const int tbox, const int rbox,
						 const int bbox, int *imap, const int mw, const int mh,
						 const DIR2RAD *dir2rad, const LFSPARMS *lfsparms);
	int test_right_edge(const int lbox, const int tbox, const int rbox,
						const int bbox, int *imap, const int mw, const int mh,
						const DIR2RAD *dir2rad, const LFSPARMS *lfsparms);
	int test_top_edge(const int lbox, const int tbox, const int rbox,
					  const int bbox, int *imap, const int mw, const int mh,
					  const DIR2RAD *dir2rad, const LFSPARMS *lfsparms);
	void average_8nbr_dir(int *avrdir, double *dir_strength, int *nvalid,
						  int *imap, const int mx, const int my,
						  const int mw, const int mh,
						  const DIR2RAD *dir2rad);
	void smooth_direction_map(int *direction_map, int *low_contrast_map,
							  const int mw, const int mh,
							  const DIR2RAD *dir2rad, const LFSPARMS *lfsparms);
	int interpolate_direction_map(int *direction_map, int *low_contrast_map,
								  const int mw, const int mh, const LFSPARMS *lfsparms);
	int find_valid_block(int *nbr_dir, int *nbr_x, int *nbr_y,
						 int *direction_map, int *low_contrast_map,
						 const int sx, const int sy,
						 const int mw, const int mh,
						 const int x_incr, const int y_incr);
	void set_margin_blocks(int *map, const int mw, const int mh,
						   const int margin_value);
	int gen_high_curve_map(int **ohcmap, int *direction_map,
						   const int mw, const int mh, const LFSPARMS *lfsparms);
	int num_valid_8nbrs(int *imap, const int mx, const int my,
						const int mw, const int mh);
	int vorticity(int *imap, const int mx, const int my,
				  const int mw, const int mh, const int ndirs);
	int curvature(int *imap, const int mx, const int my,
				  const int mw, const int mh, const int ndirs);
	void accum_nbr_vorticity(int *vmeasure, const int dir1, const int dir2,
							 const int ndirs);
	int closest_dir_dist(const int dir1, const int dir2, const int ndirs);
	void fill_holes(unsigned char *bdata, const int iw, const int ih);
	int dirbinarize(const unsigned char *pptr, const int idir,
					const ROTGRIDS *dirbingrids);
};

#endif // QUALITYMAP_H
