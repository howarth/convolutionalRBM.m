#include <math.h>
#include <matrix.h>
#include <mex.h>

// CONVS  Shrinking matrix convolution in CRBM
//   Z = CONVS(X, Y, useCuda)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *a, *b, *t;
    mxArray *c; // Output
    const mwSize *dimsa, *dimsb;
    mwSize *dimsc;
    double *aa, *bb, *cc;
    int H, W, HW, HWC, i, j, ii, jj, ni, ndima, ndimb, channels, channel, Nfilters, nf,
            N, filterH, filterW, filterHW, Wres, Hres;

    a = prhs[0];
    b = prhs[1];
    
    dimsa = mxGetDimensions(a);
    dimsb = mxGetDimensions(b);
    
    ndima = mxGetNumberOfDimensions(a);
    ndimb = mxGetNumberOfDimensions(b);
    
    H = dimsa[0]; // height of A
    W = dimsa[1]; // width of A
    HW = H * W;
    
    // Number of channels
    if (ndima <= 2) 
        channels = 1;
    else 
        channels = dimsa[2];
    
    HWC = HW * channels;
    
    // Number of data points
    if (ndima <= 3) 
        N = 1;
    else 
        N = dimsa[3];
    
    // Height fo filter
    filterH = dimsb[0];
    // Width of filter
    filterW = dimsb[1];
    filterHW = filterH * filterW;
    
    // Number of filters
    if (ndimb <= 3) 
        Nfilters = 1;
    else 
        Nfilters = dimsb[3];
    
    
    Wres = W - filterW + 1;
    Hres = H - filterH + 1;
   
    dimsc = (mwSize*)mxMalloc(sizeof(mwSize)*4);
    dimsc[0] = Hres;
    dimsc[1] = Wres;
    dimsc[2] = Nfilters;
    dimsc[3] = N;
    
    // Create output matrix
    c = plhs[0] = mxCreateNumericArray(4, dimsc, mxDOUBLE_CLASS, mxREAL);
    mxFree(dimsc); 
    
    aa = mxGetPr(a);
    bb = mxGetPr(b);
    cc = mxGetPr(c);
    
    // For each piece of data
    for (ni = 0; ni < N; ni++)
        // For each filter
        for (nf = 0; nf < Nfilters; nf++)
            // For each column
            for (j = 0; j < Wres; j++)
                // For each row
                for (i = 0; i < Hres; i++) {
                    int idxRes = i + Hres * j + Wres * Hres * nf + Nfilters * Wres * Hres * ni;
                    cc[idxRes] = 0; // why necessary?
                
                    // For each channel
                    for (channel = 0; channel < channels; channel++)
                        // For each row in filter????? or is it column
                        for (jj = 0; jj < filterW; jj++)
                            // For each column in filter??? or is it row?
                            for (ii = 0; ii < filterH; ii++)
                                cc[idxRes] += aa[(i+ii) + H * (j+jj) + HW * channel + HWC * ni]
                                    * bb[ii + filterH * jj + filterHW * channel + channels * filterHW * nf];
                }
}
