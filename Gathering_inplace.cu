// Gathering_inplace.cu
// Gather kernel: reorganizes the global nodal displacement vector uVec into
// the element-wise matrix uMat via indirect indexed reads.
// One thread per element; node indices are kept in registers (local_nodes[8])
// with no shared memory, so no intra-block synchronization is needed.
// No explicit cudaDeviceSynchronize: the MEX call chain provides implicit sync.

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    mexErrMsgIdAndTxt("Gathering_inplace:CUDA", "CUDA error %d: %s", (int)err__, cudaGetErrorString(err__)); \
  } \
} while(0)
#endif

template<typename T, typename IDX>
__global__ void gather_kernel(
    T* __restrict__ d_uMat,           // (nElem x 24), column-major
    const T* __restrict__ d_uVec,     // (numNodes x 3), column-major
    const IDX* __restrict__ d_eNod,   // (nElem x 8), 1-based
    int nElem, int numNodes)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nElem) return;

    const int colSpan = nElem;

    int32_t local_nodes[8];
#pragma unroll
    for (int n = 0; n < 8; ++n) {
        int32_t idx1 = (int32_t)d_eNod[row + n * colSpan];
        local_nodes[n] = idx1 - 1;  // MATLAB 1-based -> 0-based
    }

#pragma unroll
    for (int comp = 0; comp < 3; ++comp) {
        const T* vecComp = d_uVec + (size_t)comp * (size_t)numNodes;

#pragma unroll
        for (int n = 0; n < 8; ++n) {
            int32_t node = local_nodes[n];
            T val = vecComp[node];
            int col = n * 3 + comp; // 0..23
            size_t dst = (size_t)row + (size_t)col * (size_t)nElem; // col-major
            d_uMat[dst] = val;
        }
    }
}

template<typename T>
void launch_kernel(void* d_out, const void* d_u, const void* d_e,
                   bool e_is_uint32, int nElem, int numNodes)
{
    int block = 256;
    int grid  = (nElem + block - 1) / block;
    size_t shmem = 0;

    if (e_is_uint32) {
        gather_kernel<T, uint32_t>
            <<<grid, block, shmem>>>((T*)d_out, (const T*)d_u, (const uint32_t*)d_e, nElem, numNodes);
    } else {
        gather_kernel<T, int32_t>
            <<<grid, block, shmem>>>((T*)d_out, (const T*)d_u, (const int32_t*)d_e, nElem, numNodes);
    }
    CHECK_CUDA(cudaGetLastError());
    // No cudaDeviceSynchronize(): the MATLAB MEX call chain provides implicit sync.
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    // Usage: Gathering_inplace(gUMat, gU, gE)
    // gUMat: gpuArray (nElem x 24), same precision as gU, will be modified in-place
    // gU   : gpuArray (numNodes x 3), single/double
    // gE   : gpuArray (nElem x 8), int32/uint32, 1-based
    if (nrhs != 3) {
        mexErrMsgIdAndTxt("Gathering_inplace:Args",
            "Usage: Gathering_inplace(gUMat, gU, gE)");
    }
    if (nlhs != 0) {
        mexErrMsgIdAndTxt("Gathering_inplace:Args",
            "This function returns no outputs. gUMat is modified in-place.");
    }

    mxInitGPU();

    // Wrap inputs as GPU arrays
    const mxGPUArray *gUMat = mxGPUCreateFromMxArray(prhs[0]);
    const mxGPUArray *gU    = mxGPUCreateFromMxArray(prhs[1]);
    const mxGPUArray *gE    = mxGPUCreateFromMxArray(prhs[2]);

    // Type checks
    mxClassID clsU = mxGPUGetClassID(gU);
    if (!(clsU == mxSINGLE_CLASS || clsU == mxDOUBLE_CLASS)) {
        mxGPUDestroyGPUArray(gUMat);
        mxGPUDestroyGPUArray(gU);
        mxGPUDestroyGPUArray(gE);
        mexErrMsgIdAndTxt("Gathering_inplace:Type", "gU must be single or double gpuArray.");
    }
    if (mxGPUGetClassID(gUMat) != clsU) {
        mxGPUDestroyGPUArray(gUMat);
        mxGPUDestroyGPUArray(gU);
        mxGPUDestroyGPUArray(gE);
        mexErrMsgIdAndTxt("Gathering_inplace:Type", "gUMat precision must match gU.");
    }

    mxClassID clsE = mxGPUGetClassID(gE);
    if (!(clsE == mxINT32_CLASS || clsE == mxUINT32_CLASS)) {
        mxGPUDestroyGPUArray(gUMat);
        mxGPUDestroyGPUArray(gU);
        mxGPUDestroyGPUArray(gE);
        mexErrMsgIdAndTxt("Gathering_inplace:Type", "gE must be int32 or uint32 gpuArray.");
    }

    // Shape checks
    // gU: [numNodes x 3]
    const mwSize *dimsU = mxGPUGetDimensions(gU);
    int ndU = (int)mxGPUGetNumberOfDimensions(gU);
    if (ndU != 2 || dimsU[1] != 3) {
        mxGPUDestroyGPUArray(gUMat);
        mxGPUDestroyGPUArray(gU);
        mxGPUDestroyGPUArray(gE);
        mexErrMsgIdAndTxt("Gathering_inplace:Shape", "gU must be (numNodes x 3).");
    }
    int numNodes = (int)dimsU[0];

    // gE: [nElem x 8]
    const mwSize *dimsE = mxGPUGetDimensions(gE);
    int ndE = (int)mxGPUGetNumberOfDimensions(gE);
    if (ndE != 2 || dimsE[1] != 8) {
        mxGPUDestroyGPUArray(gUMat);
        mxGPUDestroyGPUArray(gU);
        mxGPUDestroyGPUArray(gE);
        mexErrMsgIdAndTxt("Gathering_inplace:Shape", "gE must be (nElem x 8).");
    }
    int nElem = (int)dimsE[0];

    // gUMat: [nElem x 24]
    const mwSize *dimsOut = mxGPUGetDimensions(gUMat);
    int ndOut = (int)mxGPUGetNumberOfDimensions(gUMat);
    if (ndOut != 2 || dimsOut[0] != (mwSize)nElem || dimsOut[1] != 24) {
        mxGPUDestroyGPUArray(gUMat);
        mxGPUDestroyGPUArray(gU);
        mxGPUDestroyGPUArray(gE);
        mexErrMsgIdAndTxt("Gathering_inplace:Shape", "gUMat must be (nElem x 24).");
    }

    // Device pointers (no host copies)
    void*       d_out = mxGPUGetData((mxGPUArray*)gUMat);         // writable
    const void* d_u   = mxGPUGetDataReadOnly(gU);
    const void* d_e   = mxGPUGetDataReadOnly(gE);
    bool e_is_uint32  = (clsE == mxUINT32_CLASS);

    // Dispatch by precision
    if (clsU == mxDOUBLE_CLASS) {
        launch_kernel<double>(d_out, d_u, d_e, e_is_uint32, nElem, numNodes);
    } else {
        launch_kernel<float>(d_out, d_u, d_e, e_is_uint32, nElem, numNodes);
    }

    // Clean up GPU array handles (data lives on device; gUMat modified in-place)
    mxGPUDestroyGPUArray(gUMat);
    mxGPUDestroyGPUArray(gU);
    mxGPUDestroyGPUArray(gE);
}
