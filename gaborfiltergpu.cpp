#include "gaborfiltergpu.h"

GaborFilterGPU::GaborFilterGPU()
{
    this->duration = 0;
}

void GaborFilterGPU::setParams(const cv::Mat &img_, const cv::Mat &orientationMap_, const GABOR_PARAMS &gaborParams)
{
    this->imgInput = Helper::mat_uchar2array_float(img_);
    this->oMap = Helper::mat_float2array_float(orientationMap_);

    this->gabor = gaborParams;

    if (gaborParams.useFrequencyMap) this->fMap = Helper::mat_float2array_float(gaborParams.frequencyMap);
}

void GaborFilterGPU::setParams(const cv::Mat &img_, const af::array &orientationMap_, const GABOR_PARAMS &gaborParams)
{
    this->imgInput = Helper::mat_uchar2array_float(img_);
    this->oMap = orientationMap_;

    this->gabor = gaborParams;

    if (gaborParams.useFrequencyMap) this->fMap = Helper::mat_float2array_float(gaborParams.frequencyMap);
}


af::array GaborFilterGPU::getGaborKernel(const af::array& oMapPixel)
{
    af::array xarr = af::tile(af::flip(af::array(af::seq(- this->gabor.blockSize / 2, this->gabor.blockSize / 2), this->gabor.blockSize, 1), 0), 1, this->gabor.blockSize);
    af::array yarr = af::tile(af::array(af::seq(- this->gabor.blockSize / 2, this->gabor.blockSize / 2), 1, this->gabor.blockSize), this->gabor.blockSize, 1);
    af::array thetaarr = af::tile(oMapPixel + M_PI_2, this->gabor.blockSize, this->gabor.blockSize);
    af::array xciara = xarr * af::cos(thetaarr) + yarr * af::sin(thetaarr);
    af::array yciara = -xarr * af::sin(thetaarr) + yarr * af::cos(thetaarr);
    af::array nom1;
    float denom1;
    nom1 = xciara * xciara + this->gabor.gamma * this->gabor.gamma * yciara * yciara;
    denom1 = 2 * this->gabor.sigma * this->gabor.sigma;
    af::array exp1 = af::exp(-nom1 / denom1);
    af::array cos1 = af::cos(2 * af::Pi * (xciara / this->gabor.lambda) + this->gabor.psi);

    return exp1 * cos1;
}

void GaborFilterGPU::enhanceWithBaseOMap()
{
    // TIMER
    af::timer::start();

    af::array kernels; // Gabor kernely
    af::array unwrapped_img = af::constant(255, this->gabor.blockSize * this->oMap.dims(0) + this->gabor.blockSize - 1, this->gabor.blockSize * this->oMap.dims(1) + this->gabor.blockSize - 1); // po blokoch rozdeleny a unwrapovany odtlacok pripraveny na filtrovanie
    af::array unwrapped_img_init; // pomocny orezany obraz
    af::array output; // prefiltrovany odtlacok v rozsahu 0-255
    af::array ThetaFlat; // jednoriadkova smerova mapa

    int ThetaFlatElems; // pocet prvkov smerovej mapy
    int tx = this->oMap.dims(1) * this->gabor.blockSize;
    int ty = this->oMap.dims(0) * this->gabor.blockSize;
    int ttx = this->oMap.dims(1);
    int tty = this->oMap.dims(0);

    // 1. IMAGE CUT + UNWRAP
    unwrapped_img_init = this->imgInput(af::seq(ty) , af::seq(tx)); // orezanie obrazu

    af::copy(unwrapped_img,
             unwrapped_img_init,
             af::seq(this->gabor.blockSize / 2, this->gabor.blockSize / 2 + unwrapped_img_init.dims(0) - 1),
             af::seq(this->gabor.blockSize / 2, this->gabor.blockSize / 2 + unwrapped_img_init.dims(1) - 1));
    unwrapped_img = af::unwrap(unwrapped_img, this->gabor.blockSize, this->gabor.blockSize, 1, 1, 0, 0);

    // 2. GABOR KERNELS
    ThetaFlat = af::flat(this->oMap);
    ThetaFlatElems = ThetaFlat.elements();
    kernels = af::array(this->gabor.blockSize, this->gabor.blockSize, ThetaFlatElems);
    gfor(af::seq i, ThetaFlatElems){
        kernels(af::span, af::span, i) = this->getGaborKernel(ThetaFlat(i));
    }

    // 3. PREPARING GABOR KERNELS
    kernels = af::moddims(kernels, this->gabor.blockSize * this->gabor.blockSize, ThetaFlatElems); // 3D kernely do 2D unwrapovanych kernelov
    kernels = af::tile(kernels, this->gabor.blockSize);
    kernels = af::moddims(kernels, this->gabor.blockSize * this->gabor.blockSize, this->gabor.blockSize*tty, 1, ttx);

    // 4. MULTIPLICATION
    unwrapped_img = af::moddims(unwrapped_img, this->gabor.blockSize * this->gabor.blockSize, this->gabor.blockSize * this->gabor.blockSize*tty, 1, ttx);
    unwrapped_img = af::moddims(unwrapped_img, this->gabor.blockSize * this->gabor.blockSize, this->gabor.blockSize*tty, this->gabor.blockSize, ttx);
    unwrapped_img = unwrapped_img *  af::tile(kernels, 1, 1, this->gabor.blockSize);
    unwrapped_img = af::moddims(unwrapped_img, this->gabor.blockSize * this->gabor.blockSize, this->gabor.blockSize * this->gabor.blockSize * tty * ttx);

    // 5. SUM
    af::array holder = af::sum(unwrapped_img.T(), 1);

    // 6. PIXEL REORGANIZATION
    output = af::moddims(holder, ty, tx);

    // 7. EXPORT
    Helper::af_normalizeImage(output);
    this->imgEnhanced = Helper::array_uchar2mat_uchar(output);

    this->duration = af::timer::stop() * 1000; // s to ms
}

void GaborFilterGPU::enhanceWithAdvancedOMAP(){
    // TIMER
    af::timer::start();

    int height = floor(this->imgInput.dims(0) / this->gabor.blockSize);
    int width = floor(this->imgInput.dims(1) / this->gabor.blockSize);
    int paddingWidth = this->imgInput.dims(1) - width * this->gabor.blockSize;
    int paddingHeight = this->imgInput.dims(0) - height * this->gabor.blockSize;

    this->imgInput = this->imgInput(af::seq(paddingHeight/2, height*this->gabor.blockSize+paddingHeight/2-1), af::seq(paddingWidth/2, width*this->gabor.blockSize+paddingWidth/2-1));

    int origHeight = this->imgInput.dims(0);
    int origWidth = this->imgInput.dims(1);

    this->imgInput = af::unwrap(this->imgInput , this->gabor.blockSize, this->gabor.blockSize, 1, 1, this->gabor.blockSize/2, this->gabor.blockSize/2);

    this->oMap = af::moddims(this->oMap,1, this->oMap.dims(0)*this->oMap.dims(1));

    af::array kernels(this->gabor.blockSize, this->gabor.blockSize, this->oMap.elements());
    gfor(af::seq i, this->oMap.elements()){
        kernels(af::span, af::span, i) = this->getGaborKernel(this->oMap(i));
    }

    kernels = af::moddims(kernels, this->imgInput.dims(0), this->imgInput.dims(1), this->imgInput.dims(2));

    af::array output = kernels * this->imgInput;
    output = af::sum(output);
    output  = af::moddims(output,origHeight, origWidth );


    Helper::af_normalizeImage(output);
    this->imgEnhanced = Helper::array_uchar2mat_uchar(output);

    this->duration = af::timer::stop() * 1000; // s to ms
}

cv::Mat GaborFilterGPU::getImgEnhanced() const
{
    return imgEnhanced;
}

float GaborFilterGPU::getDuration() const
{
    return duration;
}
