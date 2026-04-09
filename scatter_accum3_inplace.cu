
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <stdint.h>

// ---- Error check helper
#define CUDA_OK(stmt) do { \
    cudaError_t err = (stmt); \
    if (err != cudaSuccess) { \
        mexErrMsgIdAndTxt("scatter_accum3:cuda", "CUDA error %s at %s:%d", cudaGetErrorString(err), __FILE__, __LINE__); \
    } \
} while(0)

template<typename T>
__device__ __forceinline__ void atomicAddT(T* addr, T val);

template<>
__device__ __forceinline__ void atomicAddT<float>(float* addr, float val) {
    atomicAdd(addr, val);
}

template<>
__device__ __forceinline__ void atomicAddT<double>(double* addr, double val) {
    // Requires sm_60+ for native double atomics
    atomicAdd(addr, val);
}

// Kernel: one thread per input row i (0..N-1)
// Tmp is column-major (N x 3): col strides are N
// xFiner is column-major (M x 3): col strides are M
template<typename T>
__global__ void scatter_accum3_kernel(const int32_t* __restrict__ keys_1based,
                                      const T* __restrict__ Tmp, int64_t N,
                                      T* __restrict__ xF, int64_t M)
{
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int32_t key = keys_1based[i] - 1;  // MATLAB 1-based -> 0-based
    if (key < 0 || key >= M) return;

    const int64_t strideN = N;
    T v0 = Tmp[i + 0*strideN];
    T v1 = Tmp[i + 1*strideN];
    T v2 = Tmp[i + 2*strideN];

    const int64_t strideM = M;
    T* dst0 = xF + key + 0*strideM;
    T* dst1 = xF + key + 1*strideM;
    T* dst2 = xF + key + 2*strideM;

    atomicAddT(dst0, v0);
    atomicAddT(dst1, v1);
    atomicAddT(dst2, v2);
}

// Launcher selecting float/double
static void launchScatterAccum3(mxGPUArray const* d_keys,
                                mxGPUArray const* d_tmp,
                                mxGPUArray*       d_xf)
{
    const mwSize* tmpDims = mxGPUGetDimensions(d_tmp);
    if (mxGPUGetNumberOfDimensions(d_tmp) != 2 || tmpDims[1] != 3) {
        mexErrMsgIdAndTxt("scatter_accum3:shape", "`Tmp` must be [N x 3] gpuArray.");
    }
    const int64_t N = static_cast<int64_t>(tmpDims[0]);

    const mwSize* xDims = mxGPUGetDimensions(d_xf);
    if (mxGPUGetNumberOfDimensions(d_xf) != 2 || xDims[1] != 3) {
        mexErrMsgIdAndTxt("scatter_accum3:shape", "`xFiner` must be [M x 3] gpuArray.");
    }
    const int64_t M = static_cast<int64_t>(xDims[0]);

    if (mxGPUGetClassID(d_keys) != mxINT32_CLASS) {
        mexErrMsgIdAndTxt("scatter_accum3:type", "`transferMat` must be int32 gpuArray.");
    }
    if (mxGPUGetClassID(d_tmp) != mxGPUGetClassID(d_xf)) {
        mexErrMsgIdAndTxt("scatter_accum3:type", "`Tmp` and `xFiner` must have the same class (single/double).");
    }

    const int32_t* pKeys = static_cast<const int32_t*>(mxGPUGetDataReadOnly(d_keys));
    const void*    pTmp  = mxGPUGetDataReadOnly(d_tmp);
    void*          pXF   = mxGPUGetData(d_xf);  // write in-place

    const int threads = 256;
    const int blocks  = static_cast<int>((N + threads - 1) / threads);

    if (mxGPUGetClassID(d_tmp) == mxSINGLE_CLASS) {
        scatter_accum3_kernel<float><<<blocks, threads>>>(
            pKeys,
            static_cast<const float*>(pTmp),
            N,
            static_cast<float*>(pXF),
            M
        );
    } else if (mxGPUGetClassID(d_tmp) == mxDOUBLE_CLASS) {
        scatter_accum3_kernel<double><<<blocks, threads>>>(
            pKeys,
            static_cast<const double*>(pTmp),
            N,
            static_cast<double*>(pXF),
            M
        );
    } else {
        mexErrMsgIdAndTxt("scatter_accum3:type", "Only single or double supported for `Tmp`/`xFiner`.");
    }
    CUDA_OK(cudaGetLastError());
}

// Gateway:
// scatter_accum3_inplace(transferMatGPU, TmpGPU[Nx3], xFinerGPU[Mx3])
//   - All inputs must be gpuArray
//   - No outputs; xFiner is modified in-place
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
    if (nrhs != 3) {
        mexErrMsgIdAndTxt("scatter_accum3:args",
            "Usage: scatter_accum3_inplace(transferMatGPU[int32], TmpGPU[Nx3], xFinerGPU[Mx3]).");
    }
    if (nlhs != 0) {
        mexErrMsgIdAndTxt("scatter_accum3:nlhs", "No output. xFiner is updated in-place.");
    }

    mxInitGPU();

    // Ensure inputs are gpuArray (mxArray*) before wrapping
    for (int i = 0; i < 3; ++i) {
        if (!mxIsGPUArray(prhs[i])) {
            mexErrMsgIdAndTxt("scatter_accum3:gpu", "All inputs must be gpuArray.");
        }
    }

    // Wrap inputs
    mxGPUArray const* d_keys = mxGPUCreateFromMxArray(prhs[0]);
    mxGPUArray const* d_tmp  = mxGPUCreateFromMxArray(prhs[1]);
    mxGPUArray*       d_xf   = const_cast<mxGPUArray*>(mxGPUCreateFromMxArray(prhs[2])); // will write

    launchScatterAccum3(d_keys, d_tmp, d_xf);

    // Destroy wrappers (device data persists; xFiner updated in place)
    mxGPUDestroyGPUArray(d_keys);
    mxGPUDestroyGPUArray(d_tmp);
    mxGPUDestroyGPUArray(d_xf);
}
