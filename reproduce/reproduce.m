% Paper reproduction commands. Run one MATLAB section at a time.
% Compile CUDA MEX files with compile_all.m before GPU runs.
% After cloning, run "python reproduce/restore_large_results.py" in a shell
% to restore result files stored as .gz due to GitHub's 100 MiB file limit.
repoRoot = fileparts(fileparts(mfilename('fullpath')));
cd(repoRoot);
addpath(fullfile(repoRoot,'reproduce'));

%% Section 5.1: PCG tolerance and irregular models
TOP_GPU(true(48,96,48),'optCase',1,'V0',0.12,'rMin',sqrt(3), ...
    'ft',2,'filter_method','pde','nLoop',200,'tol',1e-3,'numLevels',3);
TOP_GPU(true(48,96,48),'optCase',1,'V0',0.12,'rMin',sqrt(3), ...
    'ft',2,'filter_method','pde','nLoop',200,'tol',1e-12,'numLevels',3);
TOP_GPU('./data/femur.TopVoxel','V0',0.4,'rMin',sqrt(3), ...
    'ft',2,'filter_method','pde','nLoop',200,'tol',1e-3,'numLevels',3);
TOP_GPU('./data/Molar.TopVoxel','V0',0.4,'rMin',sqrt(3), ...
    'ft',2,'filter_method','pde','nLoop',200,'tol',1e-3,'numLevels',3);

%% Section 5.1: Femur and Molar CPU baseline comparisons
TOP3D_XL('./data/femur.TopVoxel','GLOBAL',0.4,200,sqrt(3), ...
    'ft',2,'tol',1e-3,'numLevels',3,'vcycle','Standard');
TOP3D_XL('./data/Molar.TopVoxel','GLOBAL',0.4,200,sqrt(3), ...
    'ft',2,'tol',1e-3,'numLevels',3,'vcycle','Standard');

%% Small-grid operator and sensitivity validation
% Produces two ValidationReport.txt files for standard B8 and B8/64 paths.
validate_TOP_GPU_small();

%% Section 5.3, runtime summary: five GPU timing runs per mesh
% Read RunLog.log and iteration history for 50-iteration time, compliance,
% volume, and the total number of PCG iterations.
for nx = [48 96 192 384]
    levels = 3 + (nx == 384);
    for repeat = 1:5
        TOP_GPU(true(nx/2,nx,nx/2),'optCase',1,'V0',0.12, ...
            'rMin',sqrt(3),'ft',2,'filter_method','pde', ...
            'nLoop',50,'tol',1e-3,'mixed_Precision',0,'numLevels',levels);
    end
end

%% Section 5.3, runtime summary: five CPU timing runs per mesh
% TOP3D_XL uses its default thread-based pool configuration.
for nx = [48 96 192 384]
    levels = 3 + (nx == 384);
    for repeat = 1:5
        TOP3D_XL(true(nx/2,nx,nx/2),'GLOBAL',0.12,50,sqrt(3), ...
            'ft',2,'tol',1e-3,'numLevels',levels,'vcycle','Adapted');
    end
end

%% Section 5.4, MBB beam: standard TO versus B8/64 MTOP
% Both use a 60x10x10 displacement mesh and a distance density filter.
% Both runs use 50 optimization iterations.
TOP_GPU(true(10,60,10),'optCase',2,'V0',0.2,'rMin',sqrt(3), ...
    'ft',2,'filter_method','distance','nLoop',50,'tol',1e-3, ...
    'super_element',0);
TOP_GPU(true(10,60,10),'optCase',2,'V0',0.2,'rMin',4*sqrt(3), ...
    'ft',2,'filter_method','distance','nLoop',50,'tol',1e-3, ...
    'super_element',1,'Num_',4);

%% Section 5.4, MTOP comparison: standard versus MTOP filter-radius sweep
% rMin is in fine density-element edge lengths for both methods.
% Num_=4 maps the 96x48x48 MTOP mesh to 384x192x192 density resolution.
% The smallest-radius MTOP design is the iteration-50 result.
for radiusFactor = [4 2 1]
    radius = radiusFactor*sqrt(3);
    TOP_GPU(true(192,384,192),'optCase',1,'V0',0.12, ...
        'rMin',radius,'ft',2,'filter_method','distance', ...
        'nLoop',200,'tol',1e-3,'super_element',0);
    mtopIterations = 200;
    finePCGMaxIterations = 800;
    if radiusFactor == 1, mtopIterations = 50; end
    TOP_GPU(true(48,96,48),'optCase',1,'V0',0.12, ...
        'rMin',radius,'ft',2,'filter_method','distance', ...
        'nLoop',mtopIterations,'tol',1e-3,'super_element',1,'Num_',4);
    runPrefix = sprintf( ...
        'case1_96x48x48_TO_V0.12_r%.4g_ft2_fmdistance_n%d_mp0_se1', ...
        radius,mtopIterations);
    mtopDir = latestCompletedRun(runPrefix,'xPhys_sorted.dat');
    fineCompliance = Compute_Compliance(true(192,384,192), ...
        fullfile(mtopDir,'xPhys_sorted.dat'),'optCase',1,'mixed_Precision',0,'maxIT',finePCGMaxIterations);
    fprintf('MTOP fine-mesh compliance at rMin=%.6g: %.8g\n', ...
        radius,fineCompliance);
end

%% Section 5.5: PIO final constraint validation
TOP_GPU(true(48,96,48),'optCase',1,'consType','LOCAL', ...
    'V0',0.5,'rMin',sqrt(3),'rHat',6,'nLoop',200,'tol',1e-12);
pioDir = latestCompletedRun( ...
    'case1_96x48x48_PIO_V0.5_r1.732_rh6_n200','DesignVolume.nii');
pioMetrics = pio_final_constraint(pioDir);

%% Section 5.5: periodic design and uniform-material validation
topX3D_GPU(50,50,50,0.5,5,2.5,1,'maxDesignIter',200, ...
    'changeTol',0.01,'pcgTol',1e-12,'pcgMaxIter',800);
topX3D_GPU(8,8,8,0.5,5,2.5,1, ...
    'uniformValidation',true,'pcgTol',1e-12,'pcgMaxIter',800);

%% Section 5.2, operator timing and stage breakdown
% Run time.m separately; it clears MATLAB variables:
%   run('reproduce/time.m')
% It performs one warm-up and ten measured runs per operator and mesh.

%% Section 5.3, optimization memory sampling (PowerShell, not MATLAB)
% In pwsh, from this repository's directory. Host and GPU peaks use separate
% MATLAB processes with a requested 20 ms sampling interval:
%   pwsh -NoProfile -File .\reproduce\run_external_memory.ps1
% The manuscript also reports the 384x192x192 mesh with a four-level
% hierarchy. Pass both options: -MGLevels 4 alone would rerun every mesh
% with four levels.
%   pwsh -NoProfile -File .\reproduce\run_external_memory.ps1 -Sizes 384 -MGLevels 4
% Dry run, printing MATLAB calls without executing:
%   pwsh -NoProfile -File .\reproduce\run_external_memory.ps1 -DryRun
