// Scattering_inplace.cu
// Scatter-add kernel: atomically accumulates element-wise force contributions
// from uMat back to the global nodal vector Y.
// One thread per element; node indices are stored in register array my_nodes[NPE],
// so no shared memory or intra-block synchronization is required.
//
// Compile: mexcuda -output Scattering_inplace Scattering_inplace.cu
// Usage (MATLAB):
//   % Y has to be gpuArray(double) [numNodes x 3]
//   Scattering_inplace(Y, int32(eNodMat), uMat);  %  gpuArray
//
// Performs (column-major):
//   for comp=1..3
//     Y(:,comp) = accumarray(eNodMat(:), uMat(:, comp:3:24)(:), [numNodes,1]);
//
// Notes:
// - eNodMat is 1-based int32; converted to 0-based in kernel
// - uMat, Y are double (MATLAB default)
// - Requires CC >= 6.0 for native double atomicAdd (Pascal+)

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            mexErrMsgIdAndTxt("Scattering_inplace:cuda", "%s (code %d)",     \
                cudaGetErrorString(err__), static_cast<int>(err__));         \
        }                                                                     \
    } while (0)
#endif

// ---- kernel config ----
constexpr int NPE = 8;   // nodes per element (hex-8). Change if needed.

// Accumulate: one thread per element
// eNodMat: [Ne x NPE] (int32, 1-based) column-major
// uMat   : [Ne x 24 ] (double) column-major
// Y      : [numNodes x 3] (double) column-major (output)
template <typename T>
__global__ void scatter_kernel(const int32_t* __restrict__ eNodMat,
                             const T*  __restrict__ uMat,
                             T*        __restrict__ Y,
                             int Ne, int numNodes)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x; // element row index

    if (e >= Ne) return;

    // Store node IDs in a register array; each thread is independent, no shared memory needed.
    int my_nodes[NPE];

    // Load 8 columns of eNodMat (convert MATLAB 1-based to 0-based).
    #pragma unroll
    for (int j = 0; j < NPE; ++j) {
        my_nodes[j] = eNodMat[e + (size_t)Ne * j] - 1;
    }

    // comp = 0,1,2 refers to MATLAB col 1,2,3
    #pragma unroll
    for (int comp = 0; comp < 3; ++comp) {
        #pragma unroll
        for (int j = 0; j < NPE; ++j) {
            int nid = my_nodes[j];
            if ((unsigned)nid < (unsigned)numNodes) {
                int col0 = comp + 3 * j; // 0-based column in uMat (0..23)
                T val = uMat[e + (size_t)Ne * col0];
                // Y index: nid + numNodes * comp (column-major)
                atomicAdd(&Y[nid + (size_t)numNodes * comp], val);
            }
        }
    }
}

// --------- MEX entry ----------
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs != 3) {
        mexErrMsgIdAndTxt("Scattering_inplace:arity",
            "Expected 3 inputs: (Y, eNodMat, uMat). No outputs.");
    }
    // Init GPU API
    mxInitGPU();

    // Y: gpuArray double [numNodes x 3]
    const mxGPUArray* Y_in = mxGPUCreateFromMxArray(prhs[0]);

    const mxClassID clsY = mxGPUGetClassID(Y_in);
    const mwSize* yDims = mxGPUGetDimensions(Y_in);
    int yND = mxGPUGetNumberOfDimensions(Y_in);
    if (yND != 2 || yDims[1] != 3) {
        mexErrMsgIdAndTxt("Scattering_inplace:shape", "Y must be [numNodes x 3].");
    }
    int numNodes = static_cast<int>(yDims[0]);

    // eNodMat: gpuArray int32 [Ne x NPE]
    const mxGPUArray* EN_in = mxGPUCreateFromMxArray(prhs[1]);
    if (mxGPUGetClassID(EN_in) != mxINT32_CLASS || mxGPUGetComplexity(EN_in) != mxREAL) {
        mexErrMsgIdAndTxt("Scattering_inplace:type", "eNodMat must be gpuArray(int32).");
    }
    const mwSize* enDims = mxGPUGetDimensions(EN_in);
    int enND = mxGPUGetNumberOfDimensions(EN_in);
    if (enND != 2 || enDims[1] != NPE) {
        mexErrMsgIdAndTxt("Scattering_inplace:shape",
            "eNodMat must be [Ne x %d] int32 (1-based).", NPE);
    }
    int Ne = static_cast<int>(enDims[0]);

    // uMat: gpuArray double [Ne x 24]
    const mxGPUArray* U_in = mxGPUCreateFromMxArray(prhs[2]);

    const mwSize* uDims = mxGPUGetDimensions(U_in);
    int uND = mxGPUGetNumberOfDimensions(U_in);
    if (uND != 2 || uDims[0] != (mwSize)Ne || uDims[1] != 24) {
        mexErrMsgIdAndTxt("Scattering_inplace:shape",
            "uMat must be [Ne x 24] double.");
    }

    // Launch with shmem=0; shared memory is no longer used.
    int threads = 256;
    int blocks  = (Ne + threads - 1) / threads;

    if (clsY == mxSINGLE_CLASS) {
        float* dY = reinterpret_cast<float*>(mxGPUGetData((mxGPUArray*)Y_in));
        const int32_t* dEN = reinterpret_cast<const int32_t*>(mxGPUGetDataReadOnly(EN_in));
        const float*  dU  = reinterpret_cast<const float*>(mxGPUGetDataReadOnly(U_in));
        scatter_kernel<float><<<blocks, threads, 0>>>(dEN, dU, dY, Ne, numNodes);
    } else {
        double* dY = reinterpret_cast<double*>(mxGPUGetData((mxGPUArray*)Y_in));
        const int32_t* dEN = reinterpret_cast<const int32_t*>(mxGPUGetDataReadOnly(EN_in));
        const double*  dU  = reinterpret_cast<const double*>(mxGPUGetDataReadOnly(U_in));
        scatter_kernel<double><<<blocks, threads, 0>>>(dEN, dU, dY, Ne, numNodes);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    mxGPUDestroyGPUArray((mxGPUArray*)U_in);
    mxGPUDestroyGPUArray((mxGPUArray*)EN_in);
    mxGPUDestroyGPUArray((mxGPUArray*)Y_in);
}
