repoRoot = fileparts(fileparts(mfilename('fullpath')));
cd(repoRoot); addpath(fullfile(repoRoot,'reproduce'));
for radiusFactor = [4 2 1]
    radius = radiusFactor*sqrt(3); n = 200; pcgMaxIterations = 800;
    if radiusFactor == 1, n = 50; end
    prefix = sprintf('case1_96x48x48_TO_V0.12_r%.4g_ft2_fmdistance_n%d_mp0_se1',radius,n);
    runDir = latestCompletedRun(prefix,'xPhys_sorted.dat');
    Compute_Compliance(true(192,384,192), fullfile(runDir,'xPhys_sorted.dat'), ...
        'optCase',1,'mixed_Precision',0,'maxIT',pcgMaxIterations,'showFigure',false);
end
