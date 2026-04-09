% ---- assembleKs_higherLevel_gpu.m ----------------------------------------
function d_Ks = assembleKs_higherLevel_gpu(interpolatingKe, eDofMat, elementUpwardMap, d_KsPrev)
% % GPU stiffness assembly wrapper for level 3 and above.
% % Computes: Ks[e] = sum_s B_s' * KsPrev[upMap[e,s]] * B_s
% %
% % Inputs (CPU):
% %   interpolatingKe  - interpolation operator [numProjectDOFs x 24]
% %   eDofMat          - sub-element DOF matrix [nSub x 24]
% %   elementUpwardMap - coarse-to-fine map [nElem x nSub], 1-based, 0=void
% % Input (GPU):
% %   d_KsPrev         - previous-level Ks [24 x 24 x nFinElem] gpuArray
% % Output (GPU):
% %   d_Ks             - current-level Ks  [24 x 24 x nElem]    gpuArray
%
numElements = size(elementUpwardMap, 1);
nSub        = size(elementUpwardMap, 2);

% Build sub-element interpolation matrices B_s: same logic as assembleKs_level2_gpu.m
Psub = zeros(24, 24, nSub);
for s = 1:nSub
    Psub(:, :, s) = full(interpolatingKe(eDofMat(s, :), :));
end

d_Ks    = gpuArray.zeros(24, 24, numElements, 'double');
d_upMap = gpuArray(int32(elementUpwardMap));
d_Psub  = gpuArray(double(Psub));

assembleKs_higherLevel_inplace(d_Ks, d_upMap, d_KsPrev, d_Psub);
end
% --------------------------------------------------------------------------
