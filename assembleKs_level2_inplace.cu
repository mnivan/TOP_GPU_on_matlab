// assembleKs_level2_inplace.cu
//
// Compile inside MATLAB:
//   mexcuda -output assembleKs_level2_inplace assembleKs_level2_inplace.cu
//
// Level-2 coarse-element stiffness assembly via the Galerkin relation
//   K_e = sum_s  P_s^T * K_ch * P_s
// where K_ch is the child-element stiffness reconstructed from eleModulus and K_0.
//
// Design: one CUDA block per coarse element, 576 = 24x24 threads per block.
// Each thread owns exactly one scalar entry (row, col) of the output matrix;
// four 24x24 working matrices (B, Kch, Ttmp, Kout) live in shared memory.
//
// Shared memory uses stride KPAD=25 instead of KDIM=24: gcd(24,32)=8 causes
// 6-way bank conflicts at stride 24, while 25 is coprime with 32, eliminating
// read conflicts and reducing write conflicts to at most 2-way.
// Global memory accesses use the unpadded stride (tx = row + col*KDIM) for coalescing.
//
// Index conventions:
//   tx   = row + col * KDIM  (0..575, coalesced global memory index)
//   pidx = row + col * KPAD  (0..598, padded shared memory index)

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        mexErrMsgIdAndTxt("assembleKs_level2_inplace:CUDA", \
            "CUDA error %d: %s", (int)err__, cudaGetErrorString(err__)); \
    } \
} while(0)
#endif

static constexpr int KDIM     = 24;           // matrix dimension
static constexpr int KMAT     = KDIM * KDIM;  // 576: unpadded size, used for global memory indexing
static constexpr int KPAD     = 25;           // padded stride (coprime with 32, eliminates bank conflicts)
static constexpr int KMAT_PAD = KDIM * KPAD;  // 600: elements per padded matrix in shared memory

// ---------------------------------------------------------------------------
// Core kernel: gridDim.x = nElem, blockDim.x = KMAT = 576 (18 full warps).
//
// Thread tx handles matrix entry (row = tx%KDIM, col = tx/KDIM).
// Shared memory index: pidx = row + col*KPAD (padded layout, eliminates bank conflicts).
// Global memory index: tx   = row + col*KDIM (unpadded, guarantees coalesced access).
//
// Shared memory layout: [B | Kch | Ttmp | Kout], each KMAT_PAD = 600 elements of T.
//   double: 4 * 600 * 8 = 19 200 bytes/block
//   float : 4 * 600 * 4 =  9 600 bytes/block
//
// Safety: d_upMap[e + s*nElem] is the same value for all threads in the block
//         (e = blockIdx.x is fixed), so the 'continue' branch is taken uniformly
//         and __syncthreads() will never deadlock.
// ---------------------------------------------------------------------------
template<typename T>
__global__ void assembleKs_kernel(
    T* __restrict__             d_Ks,          // [24 x 24 x nElem], col-major
    const int32_t* __restrict__ d_upMap,       // [nElem x nCh], 1-based / 0=void
    const T* __restrict__       d_eleModulus,  // [nFineElem]
    const T* __restrict__       d_K_0,         // [576]
    const T* __restrict__       d_Pch,         // [576 x nCh], col-major
    const int32_t* __restrict__ d_mapUnique,   // [nFineElem], 0 if none
    const T* __restrict__       d_uniqueFree,  // [576 x nUnique]
    const T* __restrict__       d_uniqueFixed, // [576 x nUnique]
    int nElem, int nCh, int nUnique)
{
    const int e = blockIdx.x;
    if (e >= nElem) return;

    const int tx   = threadIdx.x;      // 0..575: used for coalesced global memory access
    const int row  = tx % KDIM;        // matrix row
    const int col  = tx / KDIM;        // matrix column
    const int pidx = row + col * KPAD; // padded shared memory index (0..598)

    // Shared memory: four padded matrices
    extern __shared__ char shmem_raw[];
    T* const B    = reinterpret_cast<T*>(shmem_raw);
    T* const Kch  = B    + KMAT_PAD;
    T* const Ttmp = Kch  + KMAT_PAD;
    T* const Kout = Ttmp + KMAT_PAD;

    Kout[pidx] = (T)0;

    const bool hasFixed = (nUnique > 0)
                       && (d_mapUnique   != nullptr)
                       && (d_uniqueFree  != nullptr)
                       && (d_uniqueFixed != nullptr);

    for (int s = 0; s < nCh; ++s) {

        // L1 broadcast: all threads in the block read the same address (e = blockIdx.x is fixed)
        const int32_t fineEle1 = d_upMap[e + (size_t)s * nElem];
        if (fineEle1 <= 0) continue;

        const int32_t fineEle0 = fineEle1 - 1;
        const T       mod      = __ldg(&d_eleModulus[fineEle0]);

        // ---- Load B: coalesced global read (address = s*576+tx), write to padded shared ----
        B[pidx] = __ldg(&d_Pch[(size_t)s * KMAT + tx]);

        // ---- Build Kch: coalesced global read (address = tx), write to padded shared ----
        int32_t uid0 = -1;
        if (hasFixed)
            uid0 = __ldg(&d_mapUnique[fineEle0]) - 1;

        if (uid0 >= 0) {
            const size_t off = (size_t)uid0 * KMAT;
            Kch[pidx] = __ldg(&d_uniqueFree [off + tx]) * mod
                      + __ldg(&d_uniqueFixed[off + tx]);
        } else {
            Kch[pidx] = __ldg(&d_K_0[tx]) * mod;
        }

        __syncthreads(); // barrier 1: B and Kch are visible to all threads in the block

        // ---- First matrix multiply: Ttmp = Kch * B ----
        // Ttmp[row, col] = sum_k  Kch[row, k] * B[k, col]
        //
        // Kch[row, k] = Kch[row + k*KPAD]
        //   row is fixed during the k loop -> stride=KPAD=25 (coprime with 32) -> zero bank conflict
        //
        // B[k, col] = B[k + col*KPAD]
        //   threads in the same warp share the same col -> L1 broadcast, zero conflict
        {
            T acc = (T)0;
            #pragma unroll
            for (int k = 0; k < KDIM; ++k)
                acc += Kch[row + k * KPAD] * B[k + col * KPAD];
            Ttmp[pidx] = acc;
        }

        __syncthreads(); // barrier 2: Ttmp is visible to all threads in the block

        // ---- Second matrix multiply: Kout += B^T * Ttmp ----
        // Kout[row, col] += sum_k  B[k, row] * Ttmp[k, col]
        //
        // B[k, row] = B[k + row*KPAD]
        //   row is fixed during the k loop -> stride=KPAD=25 (coprime with 32) -> zero bank conflict
        //   This is exactly the access that caused 6-way conflicts with KDIM=24; KPAD fixes it.
        //
        // Ttmp[k, col] = Ttmp[k + col*KPAD]
        //   same col per warp -> L1 broadcast, zero conflict
        {
            T acc = (T)0;
            #pragma unroll
            for (int k = 0; k < KDIM; ++k)
                acc += B[k + row * KPAD] * Ttmp[k + col * KPAD];
            Kout[pidx] += acc;
        }

        __syncthreads(); // barrier 3: safe to overwrite B/Kch in the next iteration of s
    }

    // ---- Write back to global: shared uses pidx (padded), global uses tx (unpadded, coalesced) ----
    // Output layout [24 x 24 x nElem] col-major: coarse element e occupies d_Ks[e*576 .. e*576+575]
    d_Ks[(size_t)e * KMAT + tx] = Kout[pidx];
}

// ---------------------------------------------------------------------------
// MEX entry point
// ---------------------------------------------------------------------------
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs != 8)
        mexErrMsgIdAndTxt("assembleKs_level2_inplace:Input", "Expected 8 inputs.");

    mxInitGPU();

    const mxGPUArray* gKs          = mxGPUCreateFromMxArray(prhs[0]);
    mxGPUArray const* gUpMap       = mxGPUCreateFromMxArray(prhs[1]);
    mxGPUArray const* gEleMod      = mxGPUCreateFromMxArray(prhs[2]);
    mxGPUArray const* gK0          = mxGPUCreateFromMxArray(prhs[3]);
    mxGPUArray const* gPch         = mxGPUCreateFromMxArray(prhs[4]);
    mxGPUArray const* gMapUnique   = mxGPUCreateFromMxArray(prhs[5]);
    mxGPUArray const* gUniqueFree  = mxGPUCreateFromMxArray(prhs[6]);
    mxGPUArray const* gUniqueFixed = mxGPUCreateFromMxArray(prhs[7]);

    const mxClassID cls = mxGPUGetClassID(gKs);
    if (!(cls == mxSINGLE_CLASS || cls == mxDOUBLE_CLASS))
        mexErrMsgIdAndTxt("assembleKs_level2_inplace:Type",
            "d_Ks must be single/double gpuArray.");
    if (mxGPUGetClassID(gUpMap)     != mxINT32_CLASS ||
        mxGPUGetClassID(gMapUnique) != mxINT32_CLASS)
        mexErrMsgIdAndTxt("assembleKs_level2_inplace:Type",
            "d_upMap and d_mapUnique must be int32 gpuArray.");

    const mwSize* ksDims = mxGPUGetDimensions(gKs);
    if (mxGPUGetNumberOfDimensions(gKs) != 3 || ksDims[0] != 24 || ksDims[1] != 24)
        mexErrMsgIdAndTxt("assembleKs_level2_inplace:Shape",
            "d_Ks must be [24 x 24 x nElem].");
    const int nElem = (int)ksDims[2];

    const mwSize* upDims = mxGPUGetDimensions(gUpMap);
    if (mxGPUGetNumberOfDimensions(gUpMap) != 2 || (int)upDims[0] != nElem)
        mexErrMsgIdAndTxt("assembleKs_level2_inplace:Shape",
            "d_upMap must be [nElem x nCh].");
    const int nCh = (int)upDims[1];

    const mwSize* pDims = mxGPUGetDimensions(gPch);
    if (mxGPUGetNumberOfDimensions(gPch) != 3 ||
        pDims[0] != 24 || pDims[1] != 24 || (int)pDims[2] != nCh)
        mexErrMsgIdAndTxt("assembleKs_level2_inplace:Shape",
            "d_Pch must be [24 x 24 x nCh].");

    const mwSize* iKsDims = mxGPUGetDimensions(gK0);
    if (mxGPUGetNumberOfDimensions(gK0) != 2 || iKsDims[0] * iKsDims[1] != 576)
        mexErrMsgIdAndTxt("assembleKs_level2_inplace:Shape",
            "d_K_0 must contain 576 elements.");

    int nUnique = 0;
    if (mxGPUGetNumberOfElements(gUniqueFree) > 0) {
        const mwSize* ufDims = mxGPUGetDimensions(gUniqueFree);
        if (mxGPUGetNumberOfDimensions(gUniqueFree) != 2 || ufDims[0] != 576)
            mexErrMsgIdAndTxt("assembleKs_level2_inplace:Shape",
                "d_uniqueFree must be [576 x nUnique] or empty.");
        nUnique = (int)ufDims[1];
    }
    if (mxGPUGetNumberOfElements(gUniqueFixed) > 0) {
        const mwSize* uxDims = mxGPUGetDimensions(gUniqueFixed);
        if (mxGPUGetNumberOfDimensions(gUniqueFixed) != 2 ||
            uxDims[0] != 576 || (int)uxDims[1] != nUnique)
            mexErrMsgIdAndTxt("assembleKs_level2_inplace:Shape",
                "d_uniqueFixed must be [576 x nUnique] matching d_uniqueFree.");
    }

    if (mxGPUGetClassID(gEleMod) != cls ||
        mxGPUGetClassID(gK0)     != cls ||
        mxGPUGetClassID(gPch)    != cls)
        mexErrMsgIdAndTxt("assembleKs_level2_inplace:Type",
            "d_eleModulus/d_K_0/d_Pch must match d_Ks class.");
    if (mxGPUGetNumberOfElements(gUniqueFree)  > 0 && mxGPUGetClassID(gUniqueFree)  != cls)
        mexErrMsgIdAndTxt("assembleKs_level2_inplace:Type", "d_uniqueFree class must match d_Ks.");
    if (mxGPUGetNumberOfElements(gUniqueFixed) > 0 && mxGPUGetClassID(gUniqueFixed) != cls)
        mexErrMsgIdAndTxt("assembleKs_level2_inplace:Type", "d_uniqueFixed class must match d_Ks.");

    void*          dKs  = mxGPUGetData((mxGPUArray*)gKs);
    const int32_t* dUp  = (const int32_t*)mxGPUGetDataReadOnly(gUpMap);
    const void*    dEM  = mxGPUGetDataReadOnly(gEleMod);
    const void*    dK0  = mxGPUGetDataReadOnly(gK0);
    const void*    dPch = mxGPUGetDataReadOnly(gPch);
    const int32_t* dMap = (const int32_t*)mxGPUGetDataReadOnly(gMapUnique);

    const void* dUF = nullptr;
    const void* dUX = nullptr;
    if (mxGPUGetNumberOfElements(gUniqueFree)  > 0) dUF = mxGPUGetDataReadOnly(gUniqueFree);
    if (mxGPUGetNumberOfElements(gUniqueFixed) > 0) dUX = mxGPUGetDataReadOnly(gUniqueFixed);

    // 1 block per element, 576 threads (18 full warps)
    const int block = KMAT;  // 576
    const int grid  = nElem;

    if (cls == mxSINGLE_CLASS) {
        const size_t shmem = 4 * KMAT_PAD * sizeof(float);   // 4*600*4 =  9 600 bytes
        assembleKs_kernel<float><<<grid, block, shmem>>>(
            (float*)dKs, dUp,
            (const float*)dEM, (const float*)dK0, (const float*)dPch,
            dMap, (const float*)dUF, (const float*)dUX,
            nElem, nCh, nUnique);
    } else {
        const size_t shmem = 4 * KMAT_PAD * sizeof(double);  // 4*600*8 = 19 200 bytes
        assembleKs_kernel<double><<<grid, block, shmem>>>(
            (double*)dKs, dUp,
            (const double*)dEM, (const double*)dK0, (const double*)dPch,
            dMap, (const double*)dUF, (const double*)dUX,
            nElem, nCh, nUnique);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    if (nlhs > 0) plhs[0] = mxGPUCreateMxArrayOnGPU(gKs);

    mxGPUDestroyGPUArray(gKs);
    mxGPUDestroyGPUArray(gUpMap);
    mxGPUDestroyGPUArray(gEleMod);
    mxGPUDestroyGPUArray(gK0);
    mxGPUDestroyGPUArray(gPch);
    mxGPUDestroyGPUArray(gMapUnique);
    mxGPUDestroyGPUArray(gUniqueFree);
    mxGPUDestroyGPUArray(gUniqueFixed);
}
