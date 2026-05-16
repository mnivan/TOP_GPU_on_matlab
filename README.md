# TOP_GPU: GPU-Accelerated 3D Topology Optimization

A MATLAB/CUDA implementation of large-scale 3D structural topology optimization on NVIDIA GPUs. The solver uses a voxel-based FEA formulation with a multi-grid preconditioned conjugate gradient (PCG) method to efficiently handle fine-resolution designs.

## Requirements

- MATLAB R2024b or later
- CUDA Toolkit 12.8
- NVIDIA GPU with compute capability ≥ 6.0 (Pascal or newer); tested on sm_89 (Ada Lovelace)
- MATLAB Parallel Computing Toolbox (for `gpuArray`)

## External Datasets

Pre-built voxel models (Femur, Molar, GEbracket) are hosted separately:

**Download:** https://syncandshare.lrz.de/getlink/fiW6M69m5HoTUcH4T7wLKZ/ (available until 2026-11-25)

Place the downloaded `.TopVoxel` files in the `./data/` directory before use.

## Compilation

Before the first run, compile all CUDA MEX kernels from within MATLAB. A convenience script is provided:

```matlab
run('compile_all.m')
```

Or compile individually:

```matlab
mexcuda -output assembleKs_level2_inplace      assembleKs_level2_inplace.cu
mexcuda -output assembleKs_higherLevel_inplace  assembleKs_higherLevel_inplace.cu
mexcuda -output assembleKs_level2_superEle_inplace assembleKs_level2_superEle_inplace.cu
mexcuda -output Gathering_inplace               Gathering_inplace.cu
mexcuda -output Scattering_inplace              Scattering_inplace.cu
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

> **Note:** Super-element mode (`super_element=1`) currently supports box-shaped domains only. The input must be a full `true(nely, nelx, nelz)` array; arbitrary non-cuboid geometries loaded from `.TopVoxel` files are not yet supported in this mode. In addition, `ft=2` (PDE density filter) is required when using super-element mode.

### Examples

```matlab
% Cantilever beam, 24x48x24 voxels
TOP_GPU(true(24, 48, 24), 'optCase', 1, 'V0', 0.12)

% MBB beam with super-element mode (ft=2 required)
TOP_GPU(true(10, 60, 10), 'optCase', 2, 'V0', 0.2, 'ft', 2, 'super_element', 1)

% File-based model (standard voxel mode)
TOP_GPU('./data/femur.TopVoxel', 'V0', 0.4, 'ft', 2)
```

## Output

Results are saved to `./out/<run_name>/`:

| File | Content |
|---|---|
| `RunLog.log` | Iteration history (compliance, volume, convergence) |
| `*.nii` | Optimized density field (NIfTI volume format) |
| `*.stl` | Isosurface mesh of the optimized structure (STL) |

The run name is auto-generated from the input parameters (e.g., `case1_60x30x10_V0.2_r1.732_ft1_n50_mp0_se0`).

## Acknowledgements

This code was developed with reference to the implementation accompanying the paper:

> Wang, J., Aage, N., Wu, J., Sigmund, O., & Westermann, R. *Efficient large-scale 3D topology optimization with matrix-free MATLAB code.* Structural and Multidisciplinary Optimization, 2025.

GitHub repository: https://github.com/PSLer/TOP3D_XL
