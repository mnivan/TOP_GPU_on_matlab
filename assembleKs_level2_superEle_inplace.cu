// assembleKs_level2_superEle_inplace.cu
//
// Compile inside MATLAB:
//   mexcuda -output assembleKs_level2_superEle_inplace assembleKs_level2_superEle_inplace.cu
//
// Super-element variant of the Level-2 coarse-stiffness assembly.
// Differences from the standard version (assembleKs_level2_inplace_modified2):
//
//   Regular sub-element:
//     Ksub = d_Ks0[:, fineEle]          (pre-computed as I_mat * eleModulus, [576 x nFineElem])
//
//   Sub-element with fixed DOFs:
//     Ksub[tx] = sum_{q=0..63}  d_uniqueFree[(uid)*576*64 + q*576 + tx] * d_eleModulus[(fineEle)*64 + q]
//              + d_uniqueFixed[(uid)*576 + tx]
//     i.e. a weighted sum over 64 sub-voxel moduli, replacing the single scalar multiply.
//
// Interface (8 inputs, same count as standard version):
//   prhs[0]  d_Ks          [24x24xnElem]        double/float gpuArray  (in-place output)
//   prhs[1]  d_upMap       [nElem x nSub]        int32 gpuArray  (1-based, 0=void)
//   prhs[2]  d_Ks0         [576 x nFineElem]     double/float gpuArray
//   prhs[3]  d_Psub        [24 x 24 x nSub]      double/float gpuArray
//   prhs[4]  d_mapUnique   [nFineElem]            int32 gpuArray  (1-based uid, 0=none)
//   prhs[5]  d_eleModulus  [64 x nFineElem]       double/float gpuArray
//   prhs[6]  d_uniqueFree  [576 x 64 x nUnique]  double/float gpuArray  (may be empty)
//   prhs[7]  d_uniqueFixed [576 x nUnique]        double/float gpuArray  (may be empty)
//
// Launch config  : gridDim.x = nElem,  blockDim.x = 576 (18 warps)
// Shared memory  : [B | Ksub | Ttmp | Kout], each KMAT_PAD=600 elements
//                  double: 4*600*8 = 19200 bytes/block
//                  float : 4*600*4 =  9600 bytes/block

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        mexErrMsgIdAndTxt("assembleKs_level2_superEle_inplace:CUDA", \
            "CUDA error %d: %s", (int)err__, cudaGetErrorString(err__)); \
    } \
} while(0)
#endif

static const int KDIM     = 24;
static const int KMAT     = KDIM * KDIM;   // 576
static const int KPAD     = 25;            // stride coprime with 32 -> zero bank conflict
static const int KMAT_PAD = KDIM * KPAD;  // 600
static const int NSUB_MOD = 64;           // sub-voxels per super-element

// ---------------------------------------------------------------------------
// Kernel: one block per coarse element, 576 threads (18 full warps).
//
// Thread tx handles matrix entry (row = tx%24, col = tx/24).
// Shared memory index: pidx = row + col*KPAD  (padded, zero bank conflict).
// Global memory index: tx   = row + col*KDIM  (unpadded, coalesced).
//
// Per sub-element s:
//   1. Load B = Psub[:, :, s]
//   2. Compute Ksub:
//        if uid > 0  ->  Ksub = sum_{q} uniqueFree[:,q,uid]*eleModulus[q,fineEle]
//                                + uniqueFixed[:,uid]
//        else        ->  Ksub = Ks0[:, fineEle]
//   3. Ttmp = Ksub * B
//   4. Kout += B^T * Ttmp
// Write d_Ks[e, :, :] = Kout
// ---------------------------------------------------------------------------
template<typename T>
__global__ void assemble_level2_superEle_kernel(
    T* __restrict__             d_Ks,
    const int32_t* __restrict__ d_upMap,
    const T* __restrict__       d_Ks0,
    const T* __restrict__       d_Psub,
    const int32_t* __restrict__ d_mapUnique,
    const T* __restrict__       d_eleModulus,
    const T* __restrict__       d_uniqueFree,
    const T* __restrict__       d_uniqueFixed,
    int nElem, int nSub, int nUnique)
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

    const bool hasFixed = (nUnique > 0)
                       && (d_mapUnique   != 0)
                       && (d_uniqueFree  != 0)
                       && (d_uniqueFixed != 0);

    for (int s = 0; s < nSub; ++s) {

        const int32_t fineEle1 = d_upMap[e + (size_t)s * nElem];
        if (fineEle1 <= 0) continue;

        const int32_t fineEle0 = fineEle1 - 1;

        // Load B (coalesced global -> padded shared)
        B[pidx] = __ldg(&d_Psub[(size_t)s * KMAT + tx]);

        // Build Ksub
        int32_t uid1 = 0;
        if (hasFixed)
            uid1 = __ldg(&d_mapUnique[fineEle0]);

        if (uid1 > 0) {
            // Fixed-DOF sub-element: weighted sum over 64 sub-voxels
            // uniqueFree layout [576 x 64 x nUnique] col-major:
            //   element (tx, q, uid0) at  uid0*576*64 + q*576 + tx
            // eleModulus layout [64 x nFineElem] col-major:
            //   element (q, fineEle0) at  fineEle0*64 + q
            const int32_t uid0      = uid1 - 1;
            const size_t  freeBase  = (size_t)uid0 * KMAT * NSUB_MOD;
            const size_t  fixedBase = (size_t)uid0 * KMAT;
            const size_t  modBase   = (size_t)fineEle0 * NSUB_MOD;

            T acc = (T)0;
            #pragma unroll 8
            for (int q = 0; q < NSUB_MOD; ++q)
                acc += __ldg(&d_uniqueFree[freeBase + (size_t)q * KMAT + tx])
                     * __ldg(&d_eleModulus[modBase + q]);

            Ksub[pidx] = acc + __ldg(&d_uniqueFixed[fixedBase + tx]);

        } else {
            // Regular sub-element: use pre-computed Ks0 (already summed over sub-voxels)
            Ksub[pidx] = __ldg(&d_Ks0[(size_t)fineEle0 * KMAT + tx]);
        }

        __syncthreads();

        // First multiply: Ttmp = Ksub * B
        // Ksub[row,k] = Ksub[row + k*KPAD]: stride KPAD=25 coprime with 32 -> zero conflict
        // B[k,col]    = B[k + col*KPAD]:    same col per warp -> L1 broadcast
        {
            T acc = (T)0;
            #pragma unroll
            for (int k = 0; k < KDIM; ++k)
                acc += Ksub[row + k * KPAD] * B[k + col * KPAD];
            Ttmp[pidx] = acc;
        }

        __syncthreads();

        // Second multiply: Kout += B^T * Ttmp
        // B[k,row]    = B[k + row*KPAD]:    stride KPAD -> zero conflict
        // Ttmp[k,col] = Ttmp[k + col*KPAD]: same col per warp -> broadcast
        {
            T acc = (T)0;
            #pragma unroll
            for (int k = 0; k < KDIM; ++k)
                acc += B[k + row * KPAD] * Ttmp[k + col * KPAD];
            Kout[pidx] += acc;
        }

        __syncthreads();
    }

    // Write back: shared uses pidx (padded), global uses tx (unpadded, coalesced)
    d_Ks[(size_t)e * KMAT + tx] = Kout[pidx];
}

// ---------------------------------------------------------------------------
// MEX entry point
// ---------------------------------------------------------------------------
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs != 8)
        mexErrMsgIdAndTxt("assembleKs_level2_superEle_inplace:Input",
            "Expected 8 inputs: d_Ks, d_upMap, d_Ks0, d_Psub, "
            "d_mapUnique, d_eleModulus, d_uniqueFree, d_uniqueFixed.");

    mxInitGPU();

    const mxGPUArray* gKs         = mxGPUCreateFromMxArray(prhs[0]);
    const mxGPUArray* gUpMap      = mxGPUCreateFromMxArray(prhs[1]);
    const mxGPUArray* gKs0        = mxGPUCreateFromMxArray(prhs[2]);
    const mxGPUArray* gPsub       = mxGPUCreateFromMxArray(prhs[3]);
    const mxGPUArray* gMapUnique  = mxGPUCreateFromMxArray(prhs[4]);
    const mxGPUArray* gEleMod     = mxGPUCreateFromMxArray(prhs[5]);
    const mxGPUArray* gUniqueFree = mxGPUCreateFromMxArray(prhs[6]);
    const mxGPUArray* gUniqueFixed= mxGPUCreateFromMxArray(prhs[7]);

    // Type checks
    const mxClassID cls = mxGPUGetClassID(gKs);
    if (!(cls == mxSINGLE_CLASS || cls == mxDOUBLE_CLASS))
        mexErrMsgIdAndTxt("assembleKs_level2_superEle_inplace:Type",
            "d_Ks must be single/double gpuArray.");
    if (mxGPUGetClassID(gUpMap)    != mxINT32_CLASS ||
        mxGPUGetClassID(gMapUnique)!= mxINT32_CLASS)
        mexErrMsgIdAndTxt("assembleKs_level2_superEle_inplace:Type",
            "d_upMap and d_mapUnique must be int32 gpuArray.");
    if (mxGPUGetClassID(gKs0)   != cls ||
        mxGPUGetClassID(gPsub)  != cls ||
        mxGPUGetClassID(gEleMod)!= cls)
        mexErrMsgIdAndTxt("assembleKs_level2_superEle_inplace:Type",
            "d_Ks0, d_Psub, d_eleModulus must match d_Ks class.");
    if (mxGPUGetNumberOfElements(gUniqueFree)  > 0 &&
        mxGPUGetClassID(gUniqueFree)  != cls)
        mexErrMsgIdAndTxt("assembleKs_level2_superEle_inplace:Type",
            "d_uniqueFree class must match d_Ks.");
    if (mxGPUGetNumberOfElements(gUniqueFixed) > 0 &&
        mxGPUGetClassID(gUniqueFixed) != cls)
        mexErrMsgIdAndTxt("assembleKs_level2_superEle_inplace:Type",
            "d_uniqueFixed class must match d_Ks.");

    // Shape checks
    const mwSize* ksDims = mxGPUGetDimensions(gKs);
    if (mxGPUGetNumberOfDimensions(gKs) != 3 ||
        ksDims[0] != 24 || ksDims[1] != 24)
        mexErrMsgIdAndTxt("assembleKs_level2_superEle_inplace:Shape",
            "d_Ks must be [24 x 24 x nElem].");
    const int nElem = (int)ksDims[2];

    const mwSize* upDims = mxGPUGetDimensions(gUpMap);
    if (mxGPUGetNumberOfDimensions(gUpMap) != 2 || (int)upDims[0] != nElem)
        mexErrMsgIdAndTxt("assembleKs_level2_superEle_inplace:Shape",
            "d_upMap must be [nElem x nSub].");
    const int nSub = (int)upDims[1];

    if (mxGPUGetNumberOfElements(gKs0) % KMAT != 0)
        mexErrMsgIdAndTxt("assembleKs_level2_superEle_inplace:Shape",
            "d_Ks0 total elements must be a multiple of 576.");

    const mwSize* pDims = mxGPUGetDimensions(gPsub);
    if (mxGPUGetNumberOfDimensions(gPsub) != 3 ||
        pDims[0] != 24 || pDims[1] != 24 || (int)pDims[2] != nSub)
        mexErrMsgIdAndTxt("assembleKs_level2_superEle_inplace:Shape",
            "d_Psub must be [24 x 24 x nSub].");

    if (mxGPUGetNumberOfElements(gEleMod) % NSUB_MOD != 0)
        mexErrMsgIdAndTxt("assembleKs_level2_superEle_inplace:Shape",
            "d_eleModulus total elements must be a multiple of 64.");

    int nUnique = 0;
    if (mxGPUGetNumberOfElements(gUniqueFree) > 0) {
        const mwSize* ufDims = mxGPUGetDimensions(gUniqueFree);
        mwSize        ufNdim = mxGPUGetNumberOfDimensions(gUniqueFree);
        if (ufNdim == 3) {
            if (ufDims[0] != 576 || (int)ufDims[1] != NSUB_MOD)
                mexErrMsgIdAndTxt("assembleKs_level2_superEle_inplace:Shape",
                    "d_uniqueFree must be [576 x 64 x nUnique].");
            nUnique = (int)ufDims[2];
        } else if (ufNdim == 2) {
            if ((int)ufDims[0] != KMAT * NSUB_MOD)
                mexErrMsgIdAndTxt("assembleKs_level2_superEle_inplace:Shape",
                    "d_uniqueFree must be [576*64 x nUnique] or [576 x 64 x nUnique].");
            nUnique = (int)ufDims[1];
        } else {
            mexErrMsgIdAndTxt("assembleKs_level2_superEle_inplace:Shape",
                "d_uniqueFree must be 2-D or 3-D.");
        }
    }
    if (mxGPUGetNumberOfElements(gUniqueFixed) > 0) {
        const mwSize* uxDims = mxGPUGetDimensions(gUniqueFixed);
        if (mxGPUGetNumberOfDimensions(gUniqueFixed) != 2 ||
            (int)uxDims[0] != 576 || (int)uxDims[1] != nUnique)
            mexErrMsgIdAndTxt("assembleKs_level2_superEle_inplace:Shape",
                "d_uniqueFixed must be [576 x nUnique] matching d_uniqueFree.");
    }

    // Raw pointers
    // gKs is const mxGPUArray* from mxGPUCreateFromMxArray, but we need a writable pointer.
    // mxGPUGetData requires a non-const handle; we cast away const here (inplace output).
    void*          dKs  = mxGPUGetData((mxGPUArray*)gKs);
    const int32_t* dUp  = (const int32_t*)mxGPUGetDataReadOnly(gUpMap);
    const void*    dKs0 = mxGPUGetDataReadOnly(gKs0);
    const void*    dPS  = mxGPUGetDataReadOnly(gPsub);
    const int32_t* dMap = (const int32_t*)mxGPUGetDataReadOnly(gMapUnique);
    const void*    dEM  = mxGPUGetDataReadOnly(gEleMod);

    const void* dUF = 0;
    const void* dUX = 0;
    if (mxGPUGetNumberOfElements(gUniqueFree)  > 0) dUF = mxGPUGetDataReadOnly(gUniqueFree);
    if (mxGPUGetNumberOfElements(gUniqueFixed) > 0) dUX = mxGPUGetDataReadOnly(gUniqueFixed);

    const int block = KMAT;   // 576
    const int grid  = nElem;

    if (cls == mxSINGLE_CLASS) {
        const size_t shmem = 4 * KMAT_PAD * sizeof(float);
        assemble_level2_superEle_kernel<float><<<grid, block, shmem>>>(
            (float*)dKs, dUp,
            (const float*)dKs0, (const float*)dPS,
            dMap,
            (const float*)dEM,
            (const float*)dUF, (const float*)dUX,
            nElem, nSub, nUnique);
    } else {
        const size_t shmem = 4 * KMAT_PAD * sizeof(double);
        assemble_level2_superEle_kernel<double><<<grid, block, shmem>>>(
            (double*)dKs, dUp,
            (const double*)dKs0, (const double*)dPS,
            dMap,
            (const double*)dEM,
            (const double*)dUF, (const double*)dUX,
            nElem, nSub, nUnique);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    if (nlhs > 0) plhs[0] = mxGPUCreateMxArrayOnGPU(gKs);

    mxGPUDestroyGPUArray(gKs);
    mxGPUDestroyGPUArray(gUpMap);
    mxGPUDestroyGPUArray(gKs0);
    mxGPUDestroyGPUArray(gPsub);
    mxGPUDestroyGPUArray(gMapUnique);
    mxGPUDestroyGPUArray(gEleMod);
    mxGPUDestroyGPUArray(gUniqueFree);
    mxGPUDestroyGPUArray(gUniqueFixed);
}
