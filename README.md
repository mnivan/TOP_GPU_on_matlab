# TOP_GPU: GPU-Accelerated 3D Topology Optimization

A MATLAB/CUDA implementation of large-scale 3D structural topology optimization on NVIDIA GPUs. The solver uses a voxel-based FEA formulation with a multi-grid preconditioned conjugate gradient (PCG) method to efficiently handle fine-resolution designs.

## Requirements

- MATLAB (R2020a or later recommended)
- CUDA Toolkit 11+
- NVIDIA GPU with compute capability ≥ 6.0 (Pascal or newer)
- MATLAB Parallel Computing Toolbox (for `gpuArray`)

## Compilation

Before the first run, compile all CUDA MEX kernels from within MATLAB:

```matlab
mexcuda -output assembleKs_level2_inplace      assembleKs_level2_inplace_modified2.cu
mexcuda -output assembleKs_higherLevel_inplace  assembleKs_higherLevel_inplace.cu
mexcuda -output assembleKs_level2_superEle_inplace assembleKs_level2_superEle_inplace.cu
mexcuda -output buildUMat_inplace               buildUMat_inplace.cu
mexcuda -output accumarrayY_inplace             accumarrayY_inplace.cu
mexcuda -output scatter_accum3_inplace          scatter_accum3_inplace.cu
```

This only needs to be done once (or after modifying `.cu` files).

## Usage

### Rectangular domain

```matlab
TOP_GPU(true(nely, nelx, nelz), 'optCase', 1, 'V0', 0.12)
```

The first argument is a 3D logical array defining the design domain. All `true` voxels are included in the optimization.

### File-based voxel model

```matlab
TOP_GPU('./data/model.TopVoxel', 'V0', 0.12, 'ft', 2)
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `optCase` | `1` | Load/BC case for rectangular domains. `1` = cantilever beam, `2` = MBB beam |
| `V0` | `0.12` | Target volume fraction |
| `rMin` | `sqrt(3)` | Filter radius in voxels |
| `nLoop` | `50` | Maximum optimization iterations |
| `ft` | `1` | Filter type: `1` = sensitivity filter, `2` = PDE density filter |
| `mixed_Precision` | `0` | `0` = double precision, `1` = mixed (single inside V-cycle) |
| `super_element` | `0` | `0` = standard voxel, `1` = 4×4×4 super-element mode |

### Examples

```matlab
% Small cantilever beam, 30x60x10 voxels
TOP_GPU(true(30, 60, 10), 'optCase', 1, 'V0', 0.3)

% Larger domain with PDE filter and mixed precision
TOP_GPU(true(60, 120, 20), 'optCase', 1, 'V0', 0.2, 'ft', 2, 'mixed_Precision', 1)

% File-based model with super-element mode
TOP_GPU('./data/femur.TopVoxel', 'V0', 0.12, 'super_element', 1)
```

## Output

Results are saved to `./out/<run_name>/`:

| File | Content |
|---|---|
| `RunLog.log` | Iteration history (compliance, volume, convergence) |
| `*.nii` | Optimized density field (NIfTI volume format) |
| `*.stl` | Isosurface mesh of the optimized structure (STL) |

The run name is auto-generated from the input parameters (e.g., `case1_60x30x10_V0.2_r1.732_ft1_n50_mp0_se0`).
