#include <math.h>
#include <matrix.h>
#include <mex.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *a, *b, *t;
    mxArray *c;
    const mwSize *dimsa, *dimsb;
    mwSize *dimsc;
    double *aa, *bb, *cc;
    int n, m, i, j, ii, jj, ni, ndima, ndimb, N, Nfilters, colors, 
            filtH, filtW, W, H, W1, H1, nf, color;

    a = prhs[0];
    b = prhs[1];
    
    dimsa = mxGetDimensions(a);
    dimsb = mxGetDimensions(b);
    
    ndima = mxGetNumberOfDimensions(a);
    ndimb = mxGetNumberOfDimensions(b);
    
    H = dimsa[0]; W = dimsa[1]; 
    H1 = dimsb[0]; W1 = dimsb[1];
    if (ndima <= 2) colors = 1;
    else colors = dimsa[2];
    if (ndima <= 3) N = 1;
    else N = dimsa[3];
    if (ndimb <= 2) Nfilters = 1;
    else Nfilters = dimsb[2];
    
    filtW = W - W1 + 1;
    filtH = H - H1 + 1;
    
    dimsc = (mwSize*)mxMalloc(sizeof(mwSize) * 4);
    dimsc[0] = filtH; dimsc[1] = filtW; dimsc[2] = colors; dimsc[3] = Nfilters;
    c = plhs[0] = mxCreateNumericArray(4, dimsc, mxDOUBLE_CLASS, mxREAL);
    mxFree(dimsc);
    
    aa = mxGetPr(a);
    bb = mxGetPr(b);
    cc = mxGetPr(c);
    
    for (nf = 0; nf < Nfilters; nf++)
        for (color = 0; color < colors; color++)
            for (i = 0; i < filtH; i++)
                for (j = 0; j < filtW; j++) {
                    int idxRes = i + filtH * j + filtW * filtH * color + colors * filtW * filtH * nf;
                    cc[idxRes] = 0;
                    
                    for (ni = 0; ni < N; ni++) 
                        for (ii = 0; ii < H1; ii++)
                            for (jj = 0; jj < W1; jj++)
                                cc[idxRes] += bb[ii + H1 * jj + W1 * H1 * nf + Nfilters * W1 * H1 * ni]
                                    * aa[(ii + i) + H * (jj + j) + W * H * color + colors * W * H * ni];
                }
}
