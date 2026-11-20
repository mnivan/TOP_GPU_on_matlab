clear;
clc;

% Batch benchmark for standard double-precision, full rectangular domains.
% For each nx, ny=nz=nx/2. Each timed operation has one untimed warm-up and
% nRuns measured executions. Results are saved beside this script.

xValues = [48 96 192 384];
nRuns = 10;
scriptFolder = fileparts(mfilename('fullpath'));
repoRoot = fileparts(scriptFolder);
addpath(repoRoot);
resultFileMat = fullfile(scriptFolder,'time_results_all_sizes.mat');
resultFileCsv = fullfile(scriptFolder,'time_results_all_sizes.csv');

fprintf('Gathering MEX : %s\n',which('Gathering_inplace'));
fprintf('Scattering MEX: %s\n',which('Scattering_inplace'));
if ~strcmpi(which('Gathering_inplace'),fullfile(repoRoot,['Gathering_inplace.' mexext])) || ...
        ~strcmpi(which('Scattering_inplace'),fullfile(repoRoot,['Scattering_inplace.' mexext]))
    error('time:MissingMEX', ...
        'Compile Gathering_inplace and Scattering_inplace in the repository root with compile_all.m first.');
end

results = repmat(empty_result(),numel(xValues),1);
for caseIndex = 1:numel(xValues)
    nx = xValues(caseIndex);
    ny = nx/2;
    nz = nx/2;
    fprintf('\n============================================================\n');
    fprintf('Case %d/%d: %d x %d x %d elements\n', ...
        caseIndex,numel(xValues),nx,ny,nz);
    fprintf('============================================================\n');
    try
        results(caseIndex) = benchmark_case(nx,ny,nz,nRuns);
    catch ME
        results(caseIndex) = empty_result();
        results(caseIndex).nx = nx;
        results(caseIndex).ny = ny;
        results(caseIndex).nz = nz;
        results(caseIndex).numElements = nx*ny*nz;
        results(caseIndex).numNodes = (nx+1)*(ny+1)*(nz+1);
        results(caseIndex).nRuns = nRuns;
        results(caseIndex).status = "ERROR: " + string(ME.message);
        fprintf(2,'Case failed: %s\n',ME.message);
    end
    resultTable = struct2table(results);
    save(resultFileMat,'results','resultTable');
    writetable(resultTable,resultFileCsv);
end

resultTable = struct2table(results);
fprintf('\n==================== ALL RESULTS ====================\n');
disp(resultTable);
save(resultFileMat,'results','resultTable');
writetable(resultTable,resultFileCsv);
fprintf('MAT results: %s\n',resultFileMat);
fprintf('CSV results: %s\n',resultFileCsv);

% Release benchmark variables and unload the two CUDA MEX modules.
clear Gathering_inplace Scattering_inplace
clearvars

function result = benchmark_case(nx,ny,nz,nRuns)
    result = empty_result();
    result.nx = nx;
    result.ny = ny;
    result.nz = nz;
    result.numElements = nx*ny*nz;
    result.numNodes = (nx+1)*(ny+1)*(nz+1);
    result.nRuns = nRuns;
    result.status = "RUNNING";

    numElements = result.numElements;
    numNodes = result.numNodes;
    eNodMat = build_eNodMat_match(nx,ny,nz);

    rng(0);
    uVec = rand(numNodes,3);
    Ke = rand(24,24);
    Ke = 0.5*(Ke+Ke');
    Ee = rand(numElements,1);

    cpuMatrixFreePeak = ...
        numel(uVec)*8 + numel(eNodMat)*4 + numel(Ee)*8 + numel(Ke)*8 + ...
        numElements*24*8*2 + numNodes*3*8;
    gpuMatrixFreePeak = cpuMatrixFreePeak;
    result.cpuMatrixFreeGiB = cpuMatrixFreePeak/2^30;
    result.gpuMatrixFreeGiB = gpuMatrixFreePeak/2^30;

    try
        [memoryUser,~] = memory;
        cpuMemoryAvailable = memoryUser.MemAvailableAllArrays;
    catch
        cpuMemoryAvailable = inf;
    end
    if cpuMatrixFreePeak > 1.0*cpuMemoryAvailable
        error('CPU matrix-free estimate %.2f GiB exceeds safe available memory %.2f GiB.', ...
            cpuMatrixFreePeak/2^30,cpuMemoryAvailable/2^30);
    end

    % ---------------- CPU matrix-free K*u ----------------
    uMat = zeros(numElements,24);
    Ycpu = zeros(numNodes,3);
    uMat = gather_cpu(uMat,uVec,eNodMat);
    a = uMat*Ke.*Ee;
    Ycpu = scatter_cpu(Ycpu,eNodMat,a);

    tCpuGather = zeros(nRuns,1);
    tCpuMatmul = zeros(nRuns,1);
    tCpuScatter = zeros(nRuns,1);
    for k = 1:nRuns
        tic;
        uMat = gather_cpu(uMat,uVec,eNodMat);
        tCpuGather(k) = toc;

        tic;
        a = uMat*Ke.*Ee;
        tCpuMatmul(k) = toc;

        Ycpu(:) = 0;
        tic;
        Ycpu = scatter_cpu(Ycpu,eNodMat,a);
        tCpuScatter(k) = toc;
    end
    result.cpuGatherMean = mean(tCpuGather);
    result.cpuGatherStd = std(tCpuGather);
    result.cpuMatmulMean = mean(tCpuMatmul);
    result.cpuMatmulStd = std(tCpuMatmul);
    result.cpuScatterMean = mean(tCpuScatter);
    result.cpuScatterStd = std(tCpuScatter);
    result.cpuMatrixFreeTotal = result.cpuGatherMean + ...
        result.cpuMatmulMean + result.cpuScatterMean;

    % ---------------- GPU matrix-free K*u ----------------
    dev = gpuDevice;
    gpuCleanup = onCleanup(@() reset_gpu_safely(dev));
    gpuMatrixFreeCompleted = gpuMatrixFreePeak <= 1.0*dev.AvailableMemory;
    if gpuMatrixFreeCompleted
        dKe = gpuArray(Ke);
        dEe = gpuArray(Ee);
        gU = gpuArray(uVec);
        gE = gpuArray(eNodMat);
        guMat = zeros(numElements,24,'gpuArray');
        Ygpu = zeros(numNodes,3,'gpuArray');
        wait(dev);

        Gathering_inplace(guMat,gU,gE);
        a1 = guMat*dKe.*dEe;
        Ygpu(:) = 0;
        wait(dev);
        Scattering_inplace(Ygpu,gE,a1);
        wait(dev);

        YgpuHost = gather(Ygpu);
        result.matrixFreeError = norm(Ycpu(:)-YgpuHost(:))/max(norm(Ycpu(:)),eps);
        clear YgpuHost

        tGpuGather = zeros(nRuns,1);
        tGpuMatmul = zeros(nRuns,1);
        tGpuScatter = zeros(nRuns,1);
        for k = 1:nRuns
            tic;
            Gathering_inplace(guMat,gU,gE);
            wait(dev);
            tGpuGather(k) = toc;

            tic;
            a1 = guMat*dKe.*dEe;
            wait(dev);
            tGpuMatmul(k) = toc;

            Ygpu(:) = 0;
            wait(dev);
            tic;
            Scattering_inplace(Ygpu,gE,a1);
            wait(dev);
            tGpuScatter(k) = toc;
        end
        result.gpuGatherMean = mean(tGpuGather);
        result.gpuGatherStd = std(tGpuGather);
        result.gpuMatmulMean = mean(tGpuMatmul);
        result.gpuMatmulStd = std(tGpuMatmul);
        result.gpuScatterMean = mean(tGpuScatter);
        result.gpuScatterStd = std(tGpuScatter);
        result.gpuMatrixFreeTotal = result.gpuGatherMean + ...
            result.gpuMatmulMean + result.gpuScatterMean;
        result.matrixFreeSpeedup = result.cpuMatrixFreeTotal/result.gpuMatrixFreeTotal;
        result.gpuMatrixFreeStatus = "COMPLETED";
    else
        result.gpuMatrixFreeStatus = string(sprintf( ...
            'SKIPPED: need %.2f GiB, available %.2f GiB', ...
            gpuMatrixFreePeak/2^30,dev.AvailableMemory/2^30));
    end

    % ---------------- Explicit sparse assembly ----------------
    numTriplets = numElements*24^2;
    result.numTriplets = numTriplets;
    estimatedNnz = 9*(3*(nx+1)-2)*(3*(ny+1)-2)*(3*(nz+1)-2);
    numDofs = 3*numNodes;
    compactCpuKBytes = estimatedNnz*16 + (numDofs+1)*8;
    compactGpuKBytes = estimatedNnz*12 + (numDofs+1)*4;
    findTripletBytes = estimatedNnz*3*8;
    assemblyBlockSize = min(100000,numElements);
    numAssemblyBlocks = ceil(numElements/assemblyBlockSize);
    blockTriplets = assemblyBlockSize*24^2;
    blockTemporaryUpperBound = ...
        assemblyBlockSize*24*8 + blockTriplets*(3*8+16);
    result.estimatedNnzK = estimatedNnz;
    result.assemblyBlockSize = assemblyBlockSize;
    result.assemblyNumBlocks = numAssemblyBlocks;
    result.cpuExplicitEstimateGiB = ...
        (compactCpuKBytes+2*numDofs*8)/2^30;
    result.gpuExplicitEstimateGiB = ...
        (compactGpuKBytes+2*numDofs*8)/2^30;
    blockAssemblyEstimate = cpuMatrixFreePeak + ...
        2*compactCpuKBytes + blockTemporaryUpperBound;
    compressionEstimate = cpuMatrixFreePeak + ...
        compactCpuKBytes + findTripletBytes;
    result.cpuAssemblyEstimateGiB = blockAssemblyEstimate/2^30;
    result.cpuCompressionEstimateGiB = compressionEstimate/2^30;
    assemblySafetyEstimate = 1.2*max(blockAssemblyEstimate,compressionEstimate);
    result.explicitSafeEstimateGiB = assemblySafetyEstimate/2^30;

    try
        [memoryUser,~] = memory;
        cpuMemoryAvailable = memoryUser.MemAvailableAllArrays;
    catch
        cpuMemoryAvailable = inf;
    end

    if assemblySafetyEstimate <= 1.0*cpuMemoryAvailable
        fprintf(['Block-assembling explicit K from %.0f total triplets ' ...
            '(%d blocks, at most %d elements/block)...\n'], ...
            numTriplets,numAssemblyBlocks,assemblyBlockSize);
        K = sparse(numDofs,numDofs);
        assemblyClock = tic;
        memoryProbeTime = 0;
        measuredAssemblyPeak = cpuMatrixFreePeak;
        for blockId = 1:numAssemblyBlocks
            firstElement = (blockId-1)*assemblyBlockSize+1;
            lastElement = min(blockId*assemblyBlockSize,numElements);
            range = firstElement:lastElement;
            numBlockElements = numel(range);
            numBlockTriplets = numBlockElements*24^2;

            eNodBlock = eNodMat(range,:);
            eDofBlock = zeros(numBlockElements,24);
            eDofBlock(:,1:3:24) = 3*double(eNodBlock)-2;
            eDofBlock(:,2:3:24) = 3*double(eNodBlock)-1;
            eDofBlock(:,3:3:24) = 3*double(eNodBlock);
            iKblock = reshape(kron(eDofBlock,ones(24,1))',numBlockTriplets,1);
            jKblock = reshape(kron(eDofBlock,ones(1,24))',numBlockTriplets,1);
            sKblock = reshape(Ke(:)*Ee(range)',numBlockTriplets,1);
            Kblock = sparse(iKblock,jKblock,sKblock,numDofs,numDofs);
            Knew = K+Kblock;

            probeClock = tic;
            sparseInfo = whos('K','Kblock','Knew');
            currentVisible = cpuMatrixFreePeak + sum([sparseInfo.bytes]) + ...
                numel(eNodBlock)*4 + numel(eDofBlock)*8 + ...
                numel(iKblock)*8 + numel(jKblock)*8 + numel(sKblock)*8;
            measuredAssemblyPeak = max(measuredAssemblyPeak,currentVisible);
            memoryProbeTime = memoryProbeTime+toc(probeClock);

            clear K Kblock eNodBlock eDofBlock iKblock jKblock sKblock
            K = Knew;
            clear Knew
            fprintf('  assembled block %d/%d\n',blockId,numAssemblyBlocks);
        end
        result.cpuAssemblyTime = toc(assemblyClock)-memoryProbeTime;
        result.cpuAssemblyPeakGiB = measuredAssemblyPeak/2^30;

        KUncompressedInfo = whos('K');
        actualNnz = nnz(K);
        result.nnzK = actualNnz;
        result.nzmaxBefore = nzmax(K);
        result.KStorageBeforeGiB = KUncompressedInfo.bytes/2^30;
        if nzmax(K)>actualNnz
            compressionClock = tic;
            [numKRows,numKCols] = size(K);
            [kRow,kCol,kValue] = find(K);
            compressionPeak1 = KUncompressedInfo.bytes + 3*actualNnz*8;
            clear K
            K = sparse(kRow,kCol,kValue,numKRows,numKCols,actualNnz);
            Kinfo = whos('K');
            compressionPeak2 = Kinfo.bytes + 3*actualNnz*8;
            clear kRow kCol kValue
            result.cpuCompressionTime = toc(compressionClock);
            result.cpuCompressionPeakGiB = ...
                (cpuMatrixFreePeak+max(compressionPeak1,compressionPeak2))/2^30;
        else
            result.cpuCompressionTime = 0;
            Kinfo = KUncompressedInfo;
            result.cpuCompressionPeakGiB = ...
                (cpuMatrixFreePeak+Kinfo.bytes)/2^30;
        end
        result.nzmaxAfter = nzmax(K);
        result.KStorageAfterGiB = Kinfo.bytes/2^30;

        uDof = reshape(uVec',3*numNodes,1);
        yCpuSpmv = K*uDof;
        tCpuSpmv = zeros(nRuns,1);
        for k = 1:nRuns
            tic;
            yCpuSpmv = K*uDof;
            tCpuSpmv(k) = toc;
        end
        result.cpuSpmvMean = mean(tCpuSpmv);
        result.cpuSpmvStd = std(tCpuSpmv);
        result.cpuExplicitSpmvGiB = ...
            (Kinfo.bytes+numel(uDof)*8+numel(yCpuSpmv)*8)/2^30;
        yMatrixFree = reshape(Ycpu',3*numNodes,1);
        result.explicitCpuError = ...
            norm(yCpuSpmv-yMatrixFree)/max(norm(yMatrixFree),eps);
        result.cpuExplicitVsMatrixFree = ...
            result.cpuMatrixFreeTotal/result.cpuSpmvMean;
        result.cpuExplicitStatus = "COMPLETED";

        % GPU sparse upload and steady-state SpMV, if device memory allows it.
        expectedGpuExplicit = actualNnz*12 + ...
            (3*numNodes+1)*4 + 2*(3*numNodes)*8;
        if expectedGpuExplicit <= 0.8*dev.AvailableMemory
            gpuAvailableBeforeExplicit = dev.AvailableMemory;
            uploadClock = tic;
            dK = gpuArray(K);
            duDof = gpuArray(uDof);
            wait(dev);
            result.gpuUploadTime = toc(uploadClock);

            yGpuSpmv = dK*duDof;
            wait(dev);
            tGpuSpmv = zeros(nRuns,1);
            for k = 1:nRuns
                tic;
                yGpuSpmv = dK*duDof;
                wait(dev);
                tGpuSpmv(k) = toc;
            end
            result.gpuSpmvMean = mean(tGpuSpmv);
            result.gpuSpmvStd = std(tGpuSpmv);
            result.gpuExplicitSpmvGiB = max(0, ...
                gpuAvailableBeforeExplicit-dev.AvailableMemory)/2^30;
            yGpuSpmvHost = gather(yGpuSpmv);
            result.explicitGpuError = ...
                norm(yGpuSpmvHost-yCpuSpmv)/max(norm(yCpuSpmv),eps);
            result.spmvSpeedup = result.cpuSpmvMean/result.gpuSpmvMean;
            if gpuMatrixFreeCompleted
                result.gpuExplicitVsMatrixFree = ...
                    result.gpuMatrixFreeTotal/result.gpuSpmvMean;
            end
            result.gpuExplicitStatus = "COMPLETED";
        else
            result.gpuExplicitStatus = string(sprintf( ...
                'SKIPPED: need about %.2f GiB, available %.2f GiB', ...
                expectedGpuExplicit/2^30,dev.AvailableMemory/2^30));
        end
    else
        result.cpuExplicitStatus = string(sprintf( ...
            'SKIPPED: safe estimate %.2f GiB, available %.2f GiB', ...
            assemblySafetyEstimate/2^30,cpuMemoryAvailable/2^30));
        result.gpuExplicitStatus = "SKIPPED: CPU K was not assembled";
    end

    result.status = "COMPLETED";
    print_case_result(result);
end

function result = empty_result()
    nanValue = NaN;
    result = struct( ...
        'nx',nanValue,'ny',nanValue,'nz',nanValue, ...
        'numElements',nanValue,'numNodes',nanValue,'nRuns',nanValue, ...
        'status',"NOT_RUN", ...
        'gpuMatrixFreeStatus',"NOT_RUN", ...
        'cpuExplicitStatus',"NOT_RUN", ...
        'gpuExplicitStatus',"NOT_RUN", ...
        'matrixFreeError',nanValue, ...
        'cpuGatherMean',nanValue,'cpuGatherStd',nanValue, ...
        'cpuMatmulMean',nanValue,'cpuMatmulStd',nanValue, ...
        'cpuScatterMean',nanValue,'cpuScatterStd',nanValue, ...
        'cpuMatrixFreeTotal',nanValue, ...
        'gpuGatherMean',nanValue,'gpuGatherStd',nanValue, ...
        'gpuMatmulMean',nanValue,'gpuMatmulStd',nanValue, ...
        'gpuScatterMean',nanValue,'gpuScatterStd',nanValue, ...
        'gpuMatrixFreeTotal',nanValue,'matrixFreeSpeedup',nanValue, ...
        'cpuMatrixFreeGiB',nanValue,'gpuMatrixFreeGiB',nanValue, ...
        'numTriplets',nanValue,'explicitSafeEstimateGiB',nanValue, ...
        'estimatedNnzK',nanValue, ...
        'assemblyBlockSize',nanValue,'assemblyNumBlocks',nanValue, ...
        'cpuExplicitEstimateGiB',nanValue,'gpuExplicitEstimateGiB',nanValue, ...
        'cpuAssemblyEstimateGiB',nanValue,'cpuCompressionEstimateGiB',nanValue, ...
        'cpuAssemblyTime',nanValue,'cpuCompressionTime',nanValue, ...
        'gpuUploadTime',nanValue, ...
        'cpuSpmvMean',nanValue,'cpuSpmvStd',nanValue, ...
        'gpuSpmvMean',nanValue,'gpuSpmvStd',nanValue, ...
        'spmvSpeedup',nanValue, ...
        'cpuExplicitVsMatrixFree',nanValue, ...
        'gpuExplicitVsMatrixFree',nanValue, ...
        'explicitCpuError',nanValue,'explicitGpuError',nanValue, ...
        'nnzK',nanValue,'nzmaxBefore',nanValue,'nzmaxAfter',nanValue, ...
        'KStorageBeforeGiB',nanValue,'KStorageAfterGiB',nanValue, ...
        'cpuExplicitSpmvGiB',nanValue,'gpuExplicitSpmvGiB',nanValue, ...
        'cpuAssemblyPeakGiB',nanValue,'cpuCompressionPeakGiB',nanValue);
end

function print_case_result(r)
    fprintf('\n--- Matrix-free K*u ---\n');
    fprintf('CPU gather: %.6f +/- %.6f s\n',r.cpuGatherMean,r.cpuGatherStd);
    fprintf('CPU matmul : %.6f +/- %.6f s\n',r.cpuMatmulMean,r.cpuMatmulStd);
    fprintf('CPU scatter : %.6f +/- %.6f s\n',r.cpuScatterMean,r.cpuScatterStd);
    fprintf('CPU total  : %.6f s\n',r.cpuMatrixFreeTotal);
    fprintf('GPU status : %s\n',r.gpuMatrixFreeStatus);
    if isfinite(r.gpuMatrixFreeTotal)
        fprintf('GPU gather: %.6f +/- %.6f s\n',r.gpuGatherMean,r.gpuGatherStd);
        fprintf('GPU matmul : %.6f +/- %.6f s\n',r.gpuMatmulMean,r.gpuMatmulStd);
        fprintf('GPU scatter : %.6f +/- %.6f s\n',r.gpuScatterMean,r.gpuScatterStd);
        fprintf('GPU total  : %.6f s; speedup %.3fx; error %.3e\n', ...
            r.gpuMatrixFreeTotal,r.matrixFreeSpeedup,r.matrixFreeError);
    end
    fprintf('CPU/GPU matrix-free memory: %.3f / %.3f GiB\n', ...
        r.cpuMatrixFreeGiB,r.gpuMatrixFreeGiB);

    fprintf('\n--- Explicit sparse matrix ---\n');
    fprintf('CPU status : %s\n',r.cpuExplicitStatus);
    fprintf('GPU status : %s\n',r.gpuExplicitStatus);
    fprintf(['Estimated nnz: %.0f; CPU/GPU SpMV memory: %.3f / %.3f GiB; ' ...
        'assembly/compress: %.3f / %.3f GiB\n'], ...
        r.estimatedNnzK,r.cpuExplicitEstimateGiB,r.gpuExplicitEstimateGiB, ...
        r.cpuAssemblyEstimateGiB,r.cpuCompressionEstimateGiB);
    fprintf('Assembly blocks: %.0f, elements/block: %.0f\n', ...
        r.assemblyNumBlocks,r.assemblyBlockSize);
    if isfinite(r.cpuSpmvMean)
        fprintf('Assembly/compression: %.6f / %.6f s\n', ...
            r.cpuAssemblyTime,r.cpuCompressionTime);
        fprintf('CPU SpMV: %.6f +/- %.6f s\n',r.cpuSpmvMean,r.cpuSpmvStd);
        fprintf('K nnz: %.0f; nzmax before/after: %.0f / %.0f\n', ...
            r.nnzK,r.nzmaxBefore,r.nzmaxAfter);
        fprintf('K storage before/after: %.3f / %.3f GiB\n', ...
            r.KStorageBeforeGiB,r.KStorageAfterGiB);
        fprintf('CPU SpMV/assembly/compress memory: %.3f / %.3f / %.3f GiB\n', ...
            r.cpuExplicitSpmvGiB,r.cpuAssemblyPeakGiB,r.cpuCompressionPeakGiB);
    end
    if isfinite(r.gpuSpmvMean)
        fprintf('Upload: %.6f s; GPU SpMV: %.6f +/- %.6f s\n', ...
            r.gpuUploadTime,r.gpuSpmvMean,r.gpuSpmvStd);
        fprintf('GPU SpMV memory: %.3f GiB; CPU/GPU SpMV speedup: %.3fx\n', ...
            r.gpuExplicitSpmvGiB,r.spmvSpeedup);
    end
end

function uMat = gather_cpu(uMat,uVec,eNodMat)
    tmp = uVec(:,1); uMat(:,1:3:24) = tmp(eNodMat);
    tmp = uVec(:,2); uMat(:,2:3:24) = tmp(eNodMat);
    tmp = uVec(:,3); uMat(:,3:3:24) = tmp(eNodMat);
end

function Y = scatter_cpu(Y,eNodMat,uMat)
    numNodes = size(Y,1);
    tmp = uMat(:,1:3:24);
    Y(:,1) = accumarray(eNodMat(:),tmp(:),[numNodes,1]);
    tmp = uMat(:,2:3:24);
    Y(:,2) = accumarray(eNodMat(:),tmp(:),[numNodes,1]);
    tmp = uMat(:,3:3:24);
    Y(:,3) = accumarray(eNodMat(:),tmp(:),[numNodes,1]);
end

function eNodMat = build_eNodMat_match(nx,ny,nz)
    nodenrs = reshape(1:(nx+1)*(ny+1)*(nz+1),ny+1,nx+1,nz+1);
    eNodVec = reshape(nodenrs(1:end-1,1:end-1,1:end-1)+1,nx*ny*nz,1);
    offsets = [0,ny+1,ny,-1, ...
        (ny+1)*(nx+1), ...
        (ny+1)*(nx+1)+ny+1, ...
        (ny+1)*(nx+1)+ny, ...
        (ny+1)*(nx+1)-1];
    eNodMat = int32(eNodVec+offsets);
end

function reset_gpu_safely(dev)
    try
        wait(dev);
        reset(dev);
    catch
    end
end
