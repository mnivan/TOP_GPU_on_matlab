function d_Ks = assembleKs_level2_gpu(interpolatingKe, eDofMat, elementUpwardMap, eleModulus, iKeCol, d_mapUniqueKes, d_uniqueKesFree, d_uniqueKesFixed)
% Assemble level-2 coarse-element stiffness on GPU with CUDA MEX.
% This matches: tmpK = P' * Kproj * P, but computes sum(B_t' * K_t * B_t) directly.
%
% Inputs are CPU arrays from local_pcg.m context.


numElements = size(elementUpwardMap, 1);
nCh = size(elementUpwardMap, 2);
Pch = zeros(24, 24, nCh);
interpolatingKe=full(interpolatingKe);
for s = 1:nCh
    Pch(:, :, s) = interpolatingKe(eDofMat(s, :), :);
end
    d_Ks = gpuArray.zeros(24, 24, numElements, 'double');
    d_eleModulus = gpuArray(double(eleModulus(:)));
    d_K_0 = gpuArray(double(iKeCol(:)));
    d_Pch = gpuArray(double(Pch));

d_upMap = gpuArray(int32(elementUpwardMap));
assembleKs_level2_inplace(d_Ks, d_upMap, d_eleModulus, d_K_0, d_Pch, d_mapUniqueKes, d_uniqueKesFree, d_uniqueKesFixed);
end
