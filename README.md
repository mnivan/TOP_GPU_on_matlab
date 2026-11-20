# TOP_GPU: GPU-Accelerated 3D Topology Optimization

A MATLAB/CUDA implementation of large-scale 3D structural topology optimization on NVIDIA GPUs. The solver uses a voxel-based FEA formulation with a multi-grid preconditioned conjugate gradient (PCG) method to efficiently handle fine-resolution designs.

## Requirements

- MATLAB R2024b or later
- CUDA Toolkit 12.8
- NVIDIA GPU supported by the selected CUDA compiler architecture target
- MATLAB Parallel Computing Toolbox (for `gpuArray`)
- MATLAB Image Processing Toolbox (for `imfilter` in the distance-filter path)


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
`compile_all.m` defaults to `sm_89`. Set `TOP_GPU_CUDA_ARCH` to the target
supported by your CUDA compiler before running it, for example:

```matlab
setenv('TOP_GPU_CUDA_ARCH','sm_120');
run('compile_all.m')
```

The default `sm_89` build does not establish native-code support for every
GPU architecture.

## Model data

The repository includes the full `data/femur.TopVoxel` and
`data/Molar.TopVoxel` files. Their SHA-256 hashes are in
`data/SHA256SUMS`; on a system with `sha256sum`, run
`cd data && sha256sum -c SHA256SUMS` to verify a downloaded archive.

## Usage

### Global-volume topology optimization (TO)

```matlab
TOP_GPU(true(nely, nelx, nelz), 'consType', 'GLOBAL', ...
    'optCase', 1, 'V0', 0.12, 'ft', 2, 'filter_method', 'pde')
```

The first argument is a 3D logical array defining the design domain. All `true` voxels are included in the optimization.

### Local-volume porous infill optimization (PIO)

```matlab
TOP_GPU('./data/model.TopVoxel', 'consType', 'LOCAL', ...
    'V0', 0.5, 'rMin', sqrt(3), 'rHat', 6, 'nLoop', 300)
```

PIO has its own optimization path and always uses standard voxels with PDE
filters. Do not pass `ft`, `filter_method`, `mixed_Precision`, or
`super_element` for PIO.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `consType` | `'GLOBAL'` | `'GLOBAL'` = TO, `'LOCAL'` = PIO |
| `optCase` | `1` | Load/BC case for rectangular domains. `1` = cantilever beam, `2` = MBB beam |
| `V0` | TO: `0.12`, PIO: `0.5` | Global target volume fraction or PIO local-volume limit |
| `rMin` | `sqrt(3)` | Filter radius in voxels |
| `rHat` | `6` | PIO local-volume influence radius |
| `nLoop` | TO: `50`, PIO: `300` | Maximum optimization iterations |
| `numLevels` | `[]` | Final multigrid level count for TO or PIO; `[]` selects automatically, otherwise use an integer of at least `3` |
| `tol` | `1e-3` | PCG relative-residual tolerance for TO and PIO |
| `ft` | `1` | TO only: `1` = sensitivity filter, `2` = density filter |
| `filter_method` | `'pde'` | TO only: `'pde'` or distance-based `'distance'` |
| `mixed_Precision` | `0` | TO only: `0` = double precision, `1` = mixed (single inside V-cycle) |
| `super_element` | `0` | TO only: `0` = standard voxel, `1` = super-element mode |
| `Num_` | `4` | TO only: number of fine elements per super-element axis (integer at least `2`) |

> **Note:** Super-element mode (`super_element=1`) currently supports box-shaped domains only. The input must be a full `true(nely, nelx, nelz)` array; arbitrary non-cuboid geometries loaded from `.TopVoxel` files are not yet supported. Super-element mode requires `ft=2` and `filter_method='distance'`.

### Examples

```matlab
% Cantilever TO, using the default sensitivity/PDE filter combination
TOP_GPU(true(24, 48, 24), 'consType', 'GLOBAL', ...
    'optCase', 1, 'V0', 0.12)

% MBB TO with super-elements: density + distance filter is mandatory
TOP_GPU(true(10, 60, 10), 'consType', 'GLOBAL', ...
    'optCase', 2, 'V0', 0.2, 'ft', 2, ...
    'filter_method', 'distance', 'super_element', 1, 'Num_', 4)

% File-based PIO; no ft or filter_method argument
TOP_GPU('./data/femur.TopVoxel', 'consType', 'LOCAL', ...
    'V0', 0.5, 'rMin', sqrt(3), 'rHat', 6, 'nLoop', 300)
```

## Periodic Microstructure Optimization

`topX3D_GPU` performs matrix-free periodic material optimization with six
unit macro-strain cases. This path always runs on the GPU and always uses the
repository's `Gathering_inplace` and `Scattering_inplace` CUDA MEX kernels;
there is no CPU or MATLAB gather/scatter fallback.

```matlab
topX3D_GPU(40, 40, 40, 0.3, 3, 1.5, 2, ...
    'maxDesignIter', 50, 'pcgTol', 1e-12, 'pcgMaxIter', 800)
```

Run `compile_all.m` before this function if the platform-specific MEX files
have not yet been built.

## Output

Results are saved to `./out/<category>/<run_name>/`. The categories are `verification`, `benchmark`, `mtop`, `pio`, `validation`, and `memory`:

| File | Content |
|---|---|
| `RunLog.log` | Iteration history (compliance, volume, convergence) |
| `*.nii` | Optimized density field (NIfTI volume format) |
| `*.stl` | Isosurface mesh of the optimized structure (STL) |

The run name is auto-generated from the input parameters (for example,
`case1_60x30x10_TO_V0.2_r1.732_ft1_fmpde_n50_mp0_se0`).

## Acknowledgements

This code was developed with reference to the implementation accompanying the paper:

> Wang, J., Aage, N., Wu, J., Sigmund, O., & Westermann, R. (2025). *Efficient large-scale 3D topology optimization with matrix-free MATLAB code.* Structural and Multidisciplinary Optimization, 68(9), Article 174. https://doi.org/10.1007/s00158-025-04127-3

GitHub repository: https://github.com/PSLer/TOP3D_XL

The `topX3D_GPU.m` implementation is adapted from the MATLAB code presented
in the following paper:

> Xia, L., & Breitkopf, P. (2015). *Design of materials using topology optimization and energy-based homogenization approach in Matlab.* Structural and Multidisciplinary Optimization, 52(6), 1229-1241. https://doi.org/10.1007/s00158-015-1294-0

The original MATLAB implementation is provided in the paper's appendix. No
official GitHub repository is associated with that paper.
