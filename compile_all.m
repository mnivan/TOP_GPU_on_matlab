% compile_all.m - compile all CUDA MEX kernels
cd(fileparts(mfilename('fullpath')));
arch = getenv('TOP_GPU_CUDA_ARCH');
if isempty(arch), arch = 'sm_89'; end
assert(~isempty(regexp(arch, '^sm_[0-9]+$', 'once')), ...
    'TOP_GPU_CUDA_ARCH must have the form sm_89 or sm_120.');
flags = sprintf('NVCCFLAGS="--allow-unsupported-compiler -arch=%s"', arch);
fprintf('CUDA architecture target: %s\n', arch);

files = {
    'assembleKs_level2_inplace',        'assembleKs_level2_inplace.cu';
    'assembleKs_higherLevel_inplace',    'assembleKs_higherLevel_inplace.cu';
    'assembleKs_level2_superEle_inplace','assembleKs_level2_superEle_inplace.cu';
    'Gathering_inplace',                 'Gathering_inplace.cu';
    'Scattering_inplace',                'Scattering_inplace.cu';
    'scatter_accum3_inplace',            'scatter_accum3_inplace.cu';
};

for k = 1:size(files, 1)
    outName = files{k, 1};
    srcFile = files{k, 2};
    fprintf('Compiling %s -> %s ...\n', srcFile, outName);
    cmd = sprintf('mexcuda -output %s %s %s', outName, srcFile, flags);
    eval(cmd);
    fprintf('  Done.\n');
end
fprintf('All kernels compiled successfully.\n');
