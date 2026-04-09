// assembleKs_higherLevel_inplace.cu
//
// Compile inside MATLAB:
//   mexcuda -output assembleKs_higherLevel_inplace assembleKs_higherLevel_inplace.cu
//
// Purpose: coarse-element stiffness assembly for level 3 and above
//          (non-super-element branch).
//
// Math: Ks[e] = sum_s  B_s^T * KsPrev[upMap[e,s]] * B_s
//
// Difference from assembleKs_level2_inplace:
//   Level 2:      Ksub_s = eleModulus[s] * Ke_unit  (+ boundary correction)
//   This kernel:  Ksub_s = KsPrev[:, :, fineEle]    (direct table lookup from
//                          the previous level; simpler, no modulus multiply)
//   All other logic (KPAD=25 bank-conflict elimination, two matrix multiplies,
//   coalesced I/O) is identical.
//
// Interface:
//   assembleKs_higherLevel_inplace(d_Ks, d_upMap, d_KsPrev, d_Psub)
//     d_Ks    : [24 x 24 x nElem]    double/single gpuArray  (in-place output)
//     d_upMap : [nElem x nSub]        int32 gpuArray, 1-based, 0=void
//     d_KsPrev: [24 x 24 x nFinElem] double/single gpuArray  (previous-level Ks)
//     d_Psub  : [24 x 24 x nSub]     double/single gpuArray  (interpolation sub-matrices)

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        mexErrMsgIdAndTxt("assembleKs_higherLevel_inplace:CUDA", \
            "CUDA error %d: %s", (int)err__, cudaGetErrorString(err__)); \
    } \
} while(0)
#endif

static constexpr int KDIM     = 24;
static constexpr int KMAT     = KDIM * KDIM;   // 576: unpadded size, used for global memory indexing
static constexpr int KPAD     = 25;            // padded stride (coprime with 32, eliminates bank conflicts)
static constexpr int KMAT_PAD = KDIM * KPAD;   // 600: elements per matrix in shared memory (padded)

// ---------------------------------------------------------------------------
// Core kernel: gridDim.x = nElem, blockDim.x = KMAT = 576 (18 full warps).
//
// Thread tx handles matrix entry (row = tx%KDIM, col = tx/KDIM).
// tx   = row + col*KDIM  -> global memory index (unpadded, coalesced access)
// pidx = row + col*KPAD  -> shared memory index (KPAD=25 eliminates bank conflicts)
//
// Ksub is read directly from d_KsPrev[fineEle * 576 + tx] (coalesced);
// no scalar multiply or boundary correction needed — simpler than the level-2 kernel.
//
// Shared memory layout: [B | Ksub | Ttmp | Kout], each 600 elements of type T.
//   double: 4 * 600 * 8 = 19 200 bytes/block
//   float : 4 * 600 * 4 =  9 600 bytes/block
// ---------------------------------------------------------------------------
template<typename T>
__global__ void assemble_higherLevel_kernel(
    T* __restrict__             d_Ks,      // [24 x 24 x nElem], col-major output
    const int32_t* __restrict__ d_upMap,   // [nElem x nSub], 1-based / 0=void
    const T* __restrict__       d_KsPrev,  // [24 x 24 x nFinElem], previous-level Ks
    const T* __restrict__       d_Psub,    // [24 x 24 x nSub], interpolation sub-matrices
    int nElem, int nSub)
{
    const int e = blockIdx.x;
    if (e >= nElem) return;

    const int tx   = threadIdx.x;
    const int row  = tx % KDIM;
    const int col  = tx / KDIM;
    const int pidx = row + col * KPAD;

    extern __shared__ char shmem_raw[];
    T* const B    = reinterpret_cast<T*>(shmem_raw);
    T* const Ksub = B    + KMAT_PAD;
    T* const Ttmp = Ksub + KMAT_PAD;
    T* const Kout = Ttmp + KMAT_PAD;

    Kout[pidx] = (T)0;

    for (int s = 0; s < nSub; ++s) {

        // L1 broadcast: all threads in the block read the same address (e = blockIdx.x is fixed)
        const int32_t fineEle1 = d_upMap[e + (size_t)s * nElem];
        if (fineEle1 <= 0) continue;  // uniform branch across the block; no deadlock risk

        const int32_t fineEle0 = fineEle1 - 1;

        // ---- Load B: coalesced global read (address = s*576+tx), write to padded shared ----
        B[pidx] = __ldg(&d_Psub[(size_t)s * KMAT + tx]);

        // ---- Load Ksub: direct table lookup from the previous level, coalesced (fineEle0*576+tx) ----
        // This is the only difference from the level-2 kernel: no modulus multiply, no boundary branch.
        Ksub[pidx] = __ldg(&d_KsPrev[(size_t)fineEle0 * KMAT + tx]);

        __syncthreads();  // barrier 1: B and Ksub are visible to all threads in the block

        // ---- First matrix multiply: Ttmp = Ksub * B ----
        // Ksub[row, k] = Ksub[row + k*KPAD], stride=25 (coprime with 32) -> zero bank conflict
        // B[k, col]    = B[k + col*KPAD], same col per warp -> L1 broadcast
        {
            T acc = (T)0;
            #pragma unroll
            for (int k = 0; k < KDIM; ++k)
                acc += Ksub[row + k * KPAD] * B[k + col * KPAD];
            Ttmp[pidx] = acc;
        }

        __syncthreads();  // barrier 2: Ttmp is visible to all threads in the block

        // ---- Second matrix multiply: Kout += B^T * Ttmp ----
        // B[k, row]    = B[k + row*KPAD],    stride=25 (coprime with 32) -> zero bank conflict
        // Ttmp[k, col] = Ttmp[k + col*KPAD], same col per warp -> L1 broadcast
        {
            T acc = (T)0;
            #pragma unroll
            for (int k = 0; k < KDIM; ++k)
                acc += B[k + row * KPAD] * Ttmp[k + col * KPAD];
            Kout[pidx] += acc;
        }

        __syncthreads();  // barrier 3: safe to overwrite B/Ksub in the next iteration of s
    }

    // Write back to global: shared uses pidx (padded), global uses tx (unpadded, coalesced)
    d_Ks[(size_t)e * KMAT + tx] = Kout[pidx];
}

// ---------------------------------------------------------------------------
// MEX entry point
// ---------------------------------------------------------------------------
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs != 4)
        mexErrMsgIdAndTxt("assembleKs_higherLevel_inplace:Input",
            "Expected 4 inputs: (d_Ks, d_upMap, d_KsPrev, d_Psub).");

    mxInitGPU();

    const mxGPUArray* gKs     = mxGPUCreateFromMxArray(prhs[0]);
    mxGPUArray const* gUpMap  = mxGPUCreateFromMxArray(prhs[1]);
    mxGPUArray const* gKsPrev = mxGPUCreateFromMxArray(prhs[2]);
    mxGPUArray const* gPsub   = mxGPUCreateFromMxArray(prhs[3]);

    // ---- Type checks ----
    const mxClassID cls = mxGPUGetClassID(gKs);
    if (!(cls == mxSINGLE_CLASS || cls == mxDOUBLE_CLASS))
        mexErrMsgIdAndTxt("assembleKs_higherLevel_inplace:Type",
            "d_Ks must be single/double gpuArray.");
    if (mxGPUGetClassID(gUpMap) != mxINT32_CLASS)
        mexErrMsgIdAndTxt("assembleKs_higherLevel_inplace:Type",
            "d_upMap must be int32 gpuArray.");
    if (mxGPUGetClassID(gKsPrev) != cls || mxGPUGetClassID(gPsub) != cls)
        mexErrMsgIdAndTxt("assembleKs_higherLevel_inplace:Type",
            "d_KsPrev and d_Psub must match d_Ks class.");

    // ---- Shape checks ----
    const mwSize* ksDims = mxGPUGetDimensions(gKs);
    if (mxGPUGetNumberOfDimensions(gKs) != 3 || ksDims[0] != 24 || ksDims[1] != 24)
        mexErrMsgIdAndTxt("assembleKs_higherLevel_inplace:Shape",
            "d_Ks must be [24 x 24 x nElem].");
    const int nElem = (int)ksDims[2];

    const mwSize* upDims = mxGPUGetDimensions(gUpMap);
    if (mxGPUGetNumberOfDimensions(gUpMap) != 2 || (int)upDims[0] != nElem)
        mexErrMsgIdAndTxt("assembleKs_higherLevel_inplace:Shape",
            "d_upMap must be [nElem x nSub].");
    const int nSub = (int)upDims[1];

    const mwSize* prevDims = mxGPUGetDimensions(gKsPrev);
    if (mxGPUGetNumberOfDimensions(gKsPrev) != 3 ||
        prevDims[0] != 24 || prevDims[1] != 24)
        mexErrMsgIdAndTxt("assembleKs_higherLevel_inplace:Shape",
            "d_KsPrev must be [24 x 24 x nFinElem].");

    const mwSize* pDims = mxGPUGetDimensions(gPsub);
    if (mxGPUGetNumberOfDimensions(gPsub) != 3 ||
        pDims[0] != 24 || pDims[1] != 24 || (int)pDims[2] != nSub)
        mexErrMsgIdAndTxt("assembleKs_higherLevel_inplace:Shape",
            "d_Psub must be [24 x 24 x nSub].");

    // ---- Get device pointers ----
    void*          dKs     = mxGPUGetData((mxGPUArray*)gKs);
    const int32_t* dUp     = (const int32_t*)mxGPUGetDataReadOnly(gUpMap);
    const void*    dKsPrev = mxGPUGetDataReadOnly(gKsPrev);
    const void*    dPsub   = mxGPUGetDataReadOnly(gPsub);

    // ---- Launch: 1 block per coarse element, 576 threads (18 full warps) ----
    const int    block = KMAT;   // 576
    const int    grid  = nElem;

    if (cls == mxSINGLE_CLASS) {
        const size_t shmem = 4 * KMAT_PAD * sizeof(float);   // 9 600 bytes
        assemble_higherLevel_kernel<float><<<grid, block, shmem>>>(
            (float*)dKs, dUp,
            (const float*)dKsPrev, (const float*)dPsub,
            nElem, nSub);
    } else {
        const size_t shmem = 4 * KMAT_PAD * sizeof(double);  // 19 200 bytes
        assemble_higherLevel_kernel<double><<<grid, block, shmem>>>(
            (double*)dKs, dUp,
            (const double*)dKsPrev, (const double*)dPsub,
            nElem, nSub);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    if (nlhs > 0) plhs[0] = mxGPUCreateMxArrayOnGPU(gKs);

    mxGPUDestroyGPUArray(gKs);
    mxGPUDestroyGPUArray(gUpMap);
    mxGPUDestroyGPUArray(gKsPrev);
    mxGPUDestroyGPUArray(gPsub);
}
