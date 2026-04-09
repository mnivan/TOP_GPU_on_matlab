function d_Ks = assembleKs_level2_superEle_gpu( ...
    interpolatingKe, eDofMat, elementUpwardMap, ...
    d_Ks0, d_eleModulus, d_mapUniqueKes, d_uniqueKesFree, d_uniqueKesFixed)
% Assemble level-2 coarse-element stiffness for super-element mode on GPU.
%
% This replaces the CPU for-loop (Super_element_==1 branch) in
% Solving_AssembleFEAstencil with a single CUDA MEX call.
%
% Inputs
%   interpolatingKe  : [numProjDOFs x 24] sparse/full restriction operator
%   eDofMat          : [nSub x 24] DOF index matrix of sub-elements
%   elementUpwardMap : [nElem x nSub] fine-element index map (1-based, 0=void)
%   d_Ks0            : [576 x nFineElem] gpuArray, pre-computed I_mat*eleModulus
%   d_eleModulus     : [64 x nFineElem]  gpuArray, sub-voxel moduli per fine element
%   d_mapUniqueKes   : [nFineElem x 1]   int32 gpuArray (1-based uid, 0 = no fixed DOFs)
%   d_uniqueKesFree  : [576 x 64 x nUnique] gpuArray (or empty)
%   d_uniqueKesFixed : [576 x nUnique]   gpuArray (or empty)
%
% Output
%   d_Ks : [24 x 24 x nElem] gpuArray

numElements = size(elementUpwardMap, 1);
nSub        = size(elementUpwardMap, 2);

% Build sub-element restriction matrices Psub(:,:,s) = interpolatingKe(eDofMat(s,:), :)
interpolatingKe = full(interpolatingKe);
Psub = zeros(24, 24, nSub);
for s = 1:nSub
    Psub(:,:,s) = interpolatingKe(eDofMat(s,:), :);
end

% Allocate output and transfer remaining CPU arrays to GPU
d_Ks    = gpuArray.zeros(24, 24, numElements, 'double');
d_Psub  = gpuArray(double(Psub));
d_upMap = gpuArray(int32(elementUpwardMap));

% Call CUDA MEX (inplace: d_Ks is modified in-place on GPU)
assembleKs_level2_superEle_inplace( ...
    d_Ks, d_upMap, d_Ks0, d_Psub, ...
    d_mapUniqueKes, d_eleModulus, ...
    d_uniqueKesFree, d_uniqueKesFixed);
end
