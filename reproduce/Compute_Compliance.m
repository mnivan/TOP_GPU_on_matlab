function compliance = Compute_Compliance(inputModel, densityFile, varargin)
% Compute_Compliance  Compute structural compliance for a given density field.
%
% Usage (rectangular domain):
%   compliance = Compute_Compliance(true(nely,nelx,nelz), 'xPhys_sorted.dat')
%   compliance = Compute_Compliance(true(nely,nelx,nelz), 'xPhys_sorted.dat', 'optCase',1, 'mixed_Precision',0)
%
% Usage (file-based model):
%   compliance = Compute_Compliance('./data/femur.TopVoxel', 'xPhys_sorted.dat')
%
% The density file must contain one rho value per line for each active fine-grid
% element (numElements values), in the standard element ordering used by
% meshHierarchy_(1).eleModulus. Rho is converted to E by SIMP before FEA.
%
% Optional name-value parameters:
%   'optCase'          1     Load/BC case (rectangular domain only)
%   'mixed_Precision'  0     0=double, 1=mixed precision
%   'maxIT'           800   Maximum PCG iterations

p = inputParser;
addRequired(p, 'inputModel');
addRequired(p, 'densityFile');
addParameter(p, 'optCase',        1);
addParameter(p, 'resultFile', fullfile(fileparts(char(densityFile)), 'FineMeshCompliance.csv'));
addParameter(p, 'showFigure', true, @(v)islogical(v) && isscalar(v));
addParameter(p, 'mixed_Precision',0);
addParameter(p, 'maxIT', 800, @(v) isnumeric(v) && isreal(v) && isscalar(v) && isfinite(v) && v >= 1 && v == fix(v));
parse(p, inputModel, densityFile, varargin{:});

mixed_Precision = p.Results.mixed_Precision;
optCase_in      = p.Results.optCase;

reset(gpuDevice);
global mixed_Precision_; mixed_Precision_ = mixed_Precision;
global Super_element_;   Super_element_   = 0;   % non-super-element only
global optCase;          optCase          = optCase_in;
global meshHierarchy_ specifyPassiveRegions_ passiveElements_;
global modulus_ modulusMin_ SIMPpenalty_ tol_ maxIT_;
global densityLayout_ F_ d_F_;
global cellSize_ typeVcycle_ nonDyadic_ poissonRatio_;
global Num_;

if islogical(inputModel)
	[~, nelx_sz, ~] = size(inputModel);
	InitialSettings(nelx_sz);
else
	InitialSettings(1);
end
maxIT_ = p.Results.maxIT;

CreateVoxelFEAmodel(inputModel);
global nelx_ nely_ nelz_;

FEA_ApplyBoundaryCondition();
FEA_SetupVoxelBased();

TopOpti_SetPassiveElements(specifyPassiveRegions_(1), specifyPassiveRegions_(2), specifyPassiveRegions_(3));
numElements = meshHierarchy_(1).numElements;

% Load density values from file (numElements x 1 or 1 x numElements)
rho = load(densityFile);
rho = rho(:);
if numel(rho) ~= numElements
	error('Density file has %d values but model has %d elements.', numel(rho), numElements);
end

densityLayout_ = rho;
E = modulusMin_ + rho'.^SIMPpenalty_ .* (modulus_ - modulusMin_);
meshHierarchy_(1).eleModulus   = E;
meshHierarchy_(1).d_eleModulus = gpuArray(meshHierarchy_(1).eleModulus);

Solving_AssembleFEAstencil();

d_F_ = gpuArray(F_);
U    = zeros(size(F_), 'gpuArray');
solveIterations = zeros(size(F_,2),1);
trueResiduals = zeros(size(F_,2),1);
for ii = 1:size(F_, 2)
	[U(:,ii), solveIterations(ii)] = Solving_PCG(@Solving_KbyU_MatrixFree, @Solving_Vcycle, ...
		d_F_(:,ii), tol_, maxIT_, [0 1]);
    trueResiduals(ii) = gather(norm(Solving_KbyU_MatrixFree(U(:,ii))-d_F_(:,ii))/norm(d_F_(:,ii)));
end

ceList     = TopOpti_ComputeUnitCompliance(U);
compliance = gather(meshHierarchy_(1).eleModulus * ceList);

fprintf('Compliance: %16.8e\n', compliance);
result = table(string(densityFile), nelx_, nely_, nelz_, double(compliance), tol_, maxIT_, ...
    max(solveIterations), max(trueResiduals), all(trueResiduals < tol_), ...
    'VariableNames', {'DensityFile','Nelx','Nely','Nelz','Compliance','PCGTolerance', ...
    'PCGMaxIterations','PCGIterations','TrueRelativeResidual','Converged'});
writetable(result, p.Results.resultFile);
fprintf('Fine-mesh result saved: %s\n', p.Results.resultFile);
if ~p.Results.showFigure, return; end
allVoxels = zeros(size(meshHierarchy_(1).eleMapForward), 'like', densityLayout_);
	allVoxels(meshHierarchy_(1).eleMapBack,1) = densityLayout_;
	isovals = reshape(allVoxels,meshHierarchy_(1).resY,meshHierarchy_(1).resX,meshHierarchy_(1).resZ);
    isovals = flip(isovals,1);
    isovals = smooth3(isovals,'box',1);    
	figure;
	facesIsosurface = isosurface(isovals,0.5);
	facesIsocap = isocaps(isovals,0.5);
     patch(facesIsosurface,'FaceColor',[0 127 0]/255,'EdgeColor','none');
    patch(facesIsocap,'FaceColor',[0 127 0]/255,'EdgeColor','none');
     view(55,25); axis equal tight on; axis off; xlabel('X'); ylabel('Y'); zlabel('Z');
     lighting('gouraud');
     material('dull');
     camlight('headlight','infinite');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function InitialSettings(nelx)
	global modulus_; modulus_ = 1.0;
	global poissonRatio_; poissonRatio_ = 0.3;
	global modulusMin_; modulusMin_ = 1.0e-6 * modulus_;
	global SIMPpenalty_; SIMPpenalty_ = 3;
	global cellSize_; cellSize_ = 1/nelx;
	global Num_; Num_ = 4;
	global tol_; tol_ = 1.0e-3;
	global maxIT_; maxIT_ = 800;
	global weightFactorJacobi_; weightFactorJacobi_ = 0.6;
	global coarsestResolutionControl_; coarsestResolutionControl_ = 50000;
	global typeVcycle_; typeVcycle_ = 'Standard';
	global nonDyadic_; nonDyadic_ = 1;
	global specifyPassiveRegions_; specifyPassiveRegions_ = [0 0 0];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function CreateVoxelFEAmodel(inputModel)
	global voxelizedVolume_ nelx_ nely_ nelz_;
	global fixingCond_ loadingCond_ objWeightingList_;
	global meshHierarchy_ passiveElements_ densityLayout_;
	global optCase;
	loadingCond_ = cell(1,1);
	if ischar(inputModel) || isstring(inputModel)
		fid = fopen(inputModel, 'r');
		fscanf(fid, '%s %s %s %s', 4);
		fscanf(fid, '%s', 1);
		versionID = fscanf(fid, '%f', 1);
		switch versionID
			case 1.0
				if ~strcmp(fscanf(fid,'%s',1),'Resolution:'), error('Incompatible Mesh Data Format!'); end
				resolutions = fscanf(fid, '%d %d %d', 3);
				nelx_ = resolutions(1); nely_ = resolutions(2); nelz_ = resolutions(3);
				fscanf(fid, '%s %s', 2);
				checkDensityValuesIncluded = fscanf(fid, '%d', 1);
				fscanf(fid, '%s %s', 2);
				numSolidVoxels = fscanf(fid, '%d', 1);
				if checkDensityValuesIncluded
					voxelsState = fscanf(fid, '%d %e', [2 numSolidVoxels])';
					solidVoxels = voxelsState(:,1);
					densityLayout_ = voxelsState(:,2);
				else
					solidVoxels = fscanf(fid, '%d', [1 numSolidVoxels])';
				end
				if ~strcmp(fscanf(fid,'%s',1),'Passive'), error('Incompatible Mesh Data Format!'); end
				if ~strcmp(fscanf(fid,'%s',1),'elements:'), error('Incompatible Mesh Data Format!'); end
				numPassiveElements = fscanf(fid, '%d', 1);
				passiveElements_ = fscanf(fid, '%d', [1 numPassiveElements])';
				if ~strcmp(fscanf(fid,'%s',1),'Fixations:'), error('Incompatible Mesh Data Format!'); end
				numFixedNodes = fscanf(fid, '%d', 1);
				if numFixedNodes>0
					fixingCond_ = fscanf(fid, '%d %d %d %d', [4 numFixedNodes])';
				end
				if ~strcmp(fscanf(fid,'%s',1),'Loads:'), error('Incompatible Mesh Data Format!'); end
				numLoadedNodes = fscanf(fid, '%d', 1);
				if numLoadedNodes>0
					loadingCond_{1} = fscanf(fid, '%d %e %e %e', [4 numLoadedNodes])';
				end
				AdditionalLoads = fscanf(fid, '%s %s', 2);
				numAdditionalLoads = fscanf(fid, '%d', 1);
				if numAdditionalLoads>0
					for ii=1:numAdditionalLoads
						if ~strcmp(fscanf(fid,'%s',1),'Loads:'), error('Incompatible Mesh Data Format!'); end
						numLoadedNodes = fscanf(fid, '%d', 1);
						idxLoad = fscanf(fid, '%d', 1);
						loadingCond_{idxLoad,1} = fscanf(fid, '%d %e %e %e', [4 numLoadedNodes])';
					end
				end
				objWeightingList_ = ones(1,numel(loadingCond_))/numel(loadingCond_);
			otherwise
				warning('Unsupported Data!'); fclose(fid); return;
		end
		fclose(fid);
		voxelizedVolume_ = false(nelx_*nely_*nelz_,1);
		voxelizedVolume_(solidVoxels) = true;
		voxelizedVolume_ = reshape(voxelizedVolume_, nely_, nelx_, nelz_);
		FEA_VoxelBasedDiscretization();
		if ~isempty(fixingCond_)
			[~,uniqueFixedNodes] = unique(fixingCond_(:,1));
			fixingCond_ = fixingCond_(uniqueFixedNodes,:);
			[~,sortMap] = sort(fixingCond_(:,1),'ascend');
			fixingCond_ = fixingCond_(sortMap,:);
			fixingCond_ = AdaptBCExternalMdl(fixingCond_, [meshHierarchy_(1).resX+1 meshHierarchy_(1).resY+1 meshHierarchy_(1).resZ+1]);
			fixingCond_(:,1) = double(meshHierarchy_(1).nodMapForward(fixingCond_(:,1)));
		end
		if ~isempty(loadingCond_{1})
			for ii=1:numel(loadingCond_)
				iLoad = loadingCond_{ii};
				[~,uniqueLoadedNodes] = unique(iLoad(:,1));
				iLoad = iLoad(uniqueLoadedNodes,:);
				[~,sortMap] = sort(iLoad(:,1),'ascend');
				iLoad = iLoad(sortMap,:);
				iLoad = AdaptBCExternalMdl(iLoad, [meshHierarchy_(1).resX+1 meshHierarchy_(1).resY+1 meshHierarchy_(1).resZ+1]);
				iLoad(:,1) = double(meshHierarchy_(1).nodMapForward(iLoad(:,1)));
				loadingCond_{ii} = iLoad;
			end
		end
		if ~isempty(passiveElements_)
			passiveElements_ = sort(passiveElements_, 'ascend');
			passiveElements_ = AdaptPassiveElementsExternalMdl(passiveElements_, [meshHierarchy_(1).resX meshHierarchy_(1).resY meshHierarchy_(1).resZ]);
			passiveElements_ = meshHierarchy_(1).eleMapForward(passiveElements_);
		end
	elseif islogical(inputModel)
		voxelizedVolume_ = inputModel;
		[nely_, nelx_, nelz_] = size(inputModel);
		if nelx_<3 || nely_<3 || nelz_<3
			error('Inappropriate Input Model!');
		end
		FEA_VoxelBasedDiscretization();
		nodeVolume4ApplyingBC = zeros(meshHierarchy_(1).resY+1, meshHierarchy_(1).resX+1, meshHierarchy_(1).resZ+1);
		switch optCase
			case 1
				nodeVolume4ApplyingBC(1:nely_+1,1,1:nelz_+1) = 1;
				fixingCond_ = find(1==nodeVolume4ApplyingBC);
				fixingCond_ = double(meshHierarchy_(1).nodMapForward(fixingCond_));
				fixingCond_ = [fixingCond_ ones(numel(fixingCond_),3)];
				nodeVolume4ApplyingBC = zeros(meshHierarchy_(1).resY+1, meshHierarchy_(1).resX+1, meshHierarchy_(1).resZ+1);
				nodeVolume4ApplyingBC(1:nely_+1,nelx_+1,1:round(nelz_/6+1)) = 1;
				iLoad = find(1==nodeVolume4ApplyingBC);
				iLoad = double(meshHierarchy_(1).nodMapForward(iLoad));
				iLoad = [iLoad repmat([0 0 -1]/numel(iLoad), numel(iLoad), 1)];
			case 2
				nodeVolume4ApplyingBC(1,1,1) = 1;
				nodeVolume4ApplyingBC(1,nelx_+1,1) = 1;
				nodeVolume4ApplyingBC(nely_+1,1,1) = 1;
				nodeVolume4ApplyingBC(nely_+1,nelx_+1,1) = 1;
				fixingCond_ = find(1==nodeVolume4ApplyingBC);
				fixingCond_ = double(meshHierarchy_(1).nodMapForward(fixingCond_));
				fixingCond_ = [fixingCond_ ones(numel(fixingCond_),3)];
				nodeVolume4ApplyingBC = zeros(meshHierarchy_(1).resY+1, meshHierarchy_(1).resX+1, meshHierarchy_(1).resZ+1);
				nodeVolume4ApplyingBC(nely_/2+1,nelx_/2+1,nelz_+1) = 1;
				iLoad = find(1==nodeVolume4ApplyingBC);
				iLoad = double(meshHierarchy_(1).nodMapForward(iLoad));
				iLoad = [iLoad [0,0,-1]];
		end
		loadingCond_{1} = iLoad;
		objWeightingList_ = 1;
		passiveElements_ = [];
	end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function FEA_VoxelBasedDiscretization()
	global nelx_ nely_ nelz_ boundingBox_ voxelizedVolume_;
	global meshHierarchy_ coarsestResolutionControl_ eNodMatHalfTemp_ numLevels_;
	[nely_, nelx_, nelz_] = size(voxelizedVolume_);
	numVoxels = nnz(voxelizedVolume_);
	numLevels_ = 0;
	while numVoxels>=coarsestResolutionControl_
		numLevels_ = numLevels_+1;
		numVoxels = round(numVoxels/8);
	end
	numLevels_ = max(3,numLevels_);
	adjustedNelx = ceil(nelx_/2^numLevels_)*2^numLevels_;
	adjustedNely = ceil(nely_/2^numLevels_)*2^numLevels_;
	adjustedNelz = ceil(nelz_/2^numLevels_)*2^numLevels_;
	numLevels_ = numLevels_ + 1;
	if adjustedNelx>nelx_, voxelizedVolume_(:,end+1:adjustedNelx,:) = false(nely_,adjustedNelx-nelx_,nelz_); end
	if adjustedNely>nely_, voxelizedVolume_(end+1:adjustedNely,:,:) = false(adjustedNely-nely_,adjustedNelx,nelz_); end
	if adjustedNelz>nelz_, voxelizedVolume_(:,:,end+1:adjustedNelz) = false(adjustedNely,adjustedNelx,adjustedNelz-nelz_); end
	boundingBox_ = [0 0 0; adjustedNelx adjustedNely adjustedNelz];
	meshHierarchy_ = Data_CartesianMeshStruct();
	meshHierarchy_.resX = adjustedNelx; nx = meshHierarchy_.resX;
	meshHierarchy_.resY = adjustedNely; ny = meshHierarchy_.resY;
	meshHierarchy_.resZ = adjustedNelz; nz = meshHierarchy_.resZ;
	meshHierarchy_.eleSize = (boundingBox_(2,:) - boundingBox_(1,:)) ./ [nx ny nz];
	voxelizedVolume_ = voxelizedVolume_(:);
	meshHierarchy_.eleMapBack = int32(find(voxelizedVolume_));
	meshHierarchy_.numElements = numel(meshHierarchy_.eleMapBack);
	meshHierarchy_.eleMapForward = zeros(nx*ny*nz,1,'int32');
	meshHierarchy_.eleMapForward(meshHierarchy_.eleMapBack) = (1:meshHierarchy_.numElements)';
	nodenrs = reshape(1:(nx+1)*(ny+1)*(nz+1), 1+ny, 1+nx, 1+nz); nodenrs = int32(nodenrs);
	eNodVec = reshape(nodenrs(1:end-1,1:end-1,1:end-1)+1, nx*ny*nz, 1);
	eNodMatHalfTemp_ = repmat(eNodVec,1,8);
	eNodMat = int32(repmat(eNodVec(meshHierarchy_.eleMapBack),1,8));
	tmp = int32([0 ny+[1 0] -1 (ny+1)*(nx+1)+[0 ny+[1 0] -1]]);
	for ii=1:8
		eNodMat(:,ii) = eNodMat(:,ii) + repmat(tmp(ii), meshHierarchy_.numElements,1);
		eNodMatHalfTemp_(:,ii) = eNodMatHalfTemp_(:,ii) + repmat(tmp(ii), nx*ny*nz,1);
	end
	eNodMatHalfTemp_ = eNodMatHalfTemp_(:,[3 4 7 8]);
	meshHierarchy_.nodMapBack = unique(eNodMat);
	meshHierarchy_.numNodes = length(meshHierarchy_.nodMapBack);
	meshHierarchy_.numDOFs = meshHierarchy_.numNodes*3;
	meshHierarchy_.nodMapForward = zeros((nx+1)*(ny+1)*(nz+1),1,'int32');
	meshHierarchy_.nodMapForward(meshHierarchy_.nodMapBack) = (1:meshHierarchy_.numNodes)';
	for ii=1:8, eNodMat(:,ii) = meshHierarchy_.nodMapForward(eNodMat(:,ii)); end
	meshHierarchy_.numNod2ElesVec = zeros(meshHierarchy_.numNodes,1);
	for jj=1:8
		iNodes = eNodMat(:,jj);
		meshHierarchy_.numNod2ElesVec(iNodes) = meshHierarchy_.numNod2ElesVec(iNodes) + 1;
	end
	meshHierarchy_.nodesOnBoundary = int32(find(meshHierarchy_.numNod2ElesVec<8));
	allNodes = zeros(meshHierarchy_.numNodes,1,'int32');
	allNodes(meshHierarchy_.nodesOnBoundary) = 1;
	tmp2 = zeros(meshHierarchy_.numElements,1,'int32');
	for ii=1:8, tmp2 = tmp2 + allNodes(eNodMat(:,ii)); end
	meshHierarchy_.elementsOnBoundary = int32(find(tmp2>0));
	meshHierarchy_.eNodMat = eNodMat;
	meshHierarchy_.d_eNodMat = gpuArray(eNodMat);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function FEA_ApplyBoundaryCondition()
	global meshHierarchy_ F_ loadingCond_ fixingCond_;
	for ii=1:numel(loadingCond_)
		iLoad = loadingCond_{ii};
		[~, nodesLoadedFixed] = setdiff(fixingCond_(:,1), iLoad(:,1));
		fixingCond_ = fixingCond_(nodesLoadedFixed,:);
		[~,uniqueFixedNodes] = unique(fixingCond_(:,1));
		fixingCond_ = fixingCond_(uniqueFixedNodes,:);
		[~,uniqueLoadedNodes] = unique(iLoad(:,1));
		iLoad = iLoad(uniqueLoadedNodes,:);
		if isempty(iLoad), warning('No Loads!'); return; end
		if isempty(fixingCond_), warning('No Fixations!'); return; end
		loadingCond_{ii} = iLoad;
	end
	F_ = sparse(meshHierarchy_(1).numNodes*3, numel(loadingCond_));
	for ii=1:numel(loadingCond_)
		iLoad = loadingCond_{ii};
		iFarr = sparse(meshHierarchy_(1).numNodes, 3);
		iFarr(iLoad(:,1),:) = iLoad(:,2:end);
		F_(:,ii) = reshape(iFarr',meshHierarchy_(1).numDOFs,1);
	end
	meshHierarchy_(1).fixedDOFs = [];
	meshHierarchy_(1).freeDOFs = [];
	fixedDOFs = 3*fixingCond_(:,1);
	fixedDOFs = fixedDOFs - [2 1 0];
	fixedDOFs = reshape(fixedDOFs', numel(fixedDOFs), 1);
	fixingState = fixingCond_(:,2:end)';
	fixedDOFs = fixedDOFs(1==fixingState(:));
	freeDOFs = true(meshHierarchy_(1).numDOFs,1);
	freeDOFs(fixedDOFs) = false;
	meshHierarchy_(1).freeDOFs = freeDOFs;
	meshHierarchy_(1).fixedDOFs = false(meshHierarchy_(1).numDOFs,1);
	meshHierarchy_(1).fixedDOFs(fixedDOFs) = true;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function FEA_SetupVoxelBased()
	global meshHierarchy_ mixed_Precision_;
	meshHierarchy_(1).Ke = FEA_VoxelBasedElementStiffnessMatrix();
	Solving_BuildingMeshHierarchy();
	meshHierarchy_(1).Ks = meshHierarchy_(1).Ke;
	if mixed_Precision_==1
		meshHierarchy_(1).d_Ks = single(gpuArray(meshHierarchy_(1).Ks));
	else
		meshHierarchy_(1).d_Ks = gpuArray(meshHierarchy_(1).Ks);
	end
	Solving_SetupKeWithFixedDOFs();
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function KE = FEA_VoxelBasedElementStiffnessMatrix()
	global poissonRatio_ cellSize_;
	nu = poissonRatio_;
	C = [2/9 1/18 1/24 1/36 1/48 5/72 1/3 1/6 1/12];
	A11 = [-C(1) -C(3) -C(3) C(2) C(3) C(3); -C(3) -C(1) -C(3) -C(3) -C(4) -C(5);...
		-C(3) -C(3) -C(1) -C(3) -C(5) -C(4); C(2) -C(3) -C(3) -C(1) C(3) C(3);...
		C(3) -C(4) -C(5) C(3) -C(1) -C(3); C(3) -C(5) -C(4) C(3) -C(3) -C(1)];
	B11 = [C(7) 0 0 0 -C(8) -C(8); 0 C(7) 0 C(8) 0 0; 0 0 C(7) C(8) 0 0;...
		0 C(8) C(8) C(7) 0 0; -C(8) 0 0 0 C(7) 0; -C(8) 0 0 0 0 C(7)];
	A22 = [-C(1) -C(3) C(3) C(2) C(3) -C(3); -C(3) -C(1) C(3) -C(3) -C(4) C(5);...
		C(3) C(3) -C(1) C(3) C(5) -C(4); C(2) -C(3) C(3) -C(1) C(3) -C(3);...
		C(3) -C(4) C(5) C(3) -C(1) C(3); -C(3) C(5) -C(4) -C(3) C(3) -C(1)];
	B22 = [C(7) 0 0 0 -C(8) C(8); 0 C(7) 0 C(8) 0 0; 0 0 C(7) -C(8) 0 0;...
		0 C(8) -C(8) C(7) 0 0; -C(8) 0 0 0 C(7) 0; C(8) 0 0 0 0 C(7)];
	A12 = [C(6) C(3) C(5) -C(4) -C(3) -C(5); C(3) C(6) C(5) C(3) C(2) C(3);...
		-C(5) -C(5) C(4) -C(5) -C(3) -C(4); -C(4) C(3) C(5) C(6) -C(3) -C(5);...
		-C(3) C(2) C(3) -C(3) C(6) C(5); C(5) -C(3) -C(4) C(5) -C(5) C(4)];
	B12 = [-C(9) 0 -C(9) 0 C(8) 0; 0 -C(9) -C(9) -C(8) 0 -C(8); C(9) C(9) -C(9) 0 C(8) 0;...
		0 -C(8) 0 -C(9) 0 C(9); C(8) 0 -C(8) 0 -C(9) -C(9); 0 C(8) 0 -C(9) C(9) -C(9)];
	A13 = [-C(4) -C(5) -C(3) C(6) C(5) C(3); -C(5) -C(4) -C(3) -C(5) C(4) -C(5);...
		C(3) C(3) C(2) C(3) C(5) C(6); C(6) -C(5) -C(3) -C(4) C(5) C(3);...
		C(5) C(4) -C(5) C(5) -C(4) -C(3); -C(3) C(5) C(6) -C(3) C(3) C(2)];
	B13 = [0 0 C(8) -C(9) -C(9) 0; 0 0 C(8) C(9) -C(9) C(9); -C(8) -C(8) 0 0 -C(9) -C(9);...
		-C(9) C(9) 0 0 0 -C(8); -C(9) -C(9) C(9) 0 0 C(8); 0 -C(9) -C(9) C(8) -C(8) 0];
	A14 = [C(2) C(5) C(5) C(4) -C(5) -C(5); C(5) C(2) C(5) C(5) C(6) C(3);...
		C(5) C(5) C(2) C(5) C(3) C(6); C(4) C(5) C(5) C(2) -C(5) -C(5);...
		-C(5) C(6) C(3) -C(5) C(2) C(5); -C(5) C(3) C(6) -C(5) C(5) C(2)];
	B14 = [-C(9) 0 0 -C(9) C(9) C(9); 0 -C(9) 0 -C(9) -C(9) 0; 0 0 -C(9) -C(9) 0 -C(9);...
		-C(9) -C(9) -C(9) -C(9) 0 0; C(9) -C(9) 0 0 -C(9) 0; C(9) 0 -C(9) 0 0 -C(9)];
	A23 = [C(2) C(5) -C(5) C(4) -C(5) C(5); C(5) C(2) -C(5) C(5) C(6) -C(3);...
		-C(5) -C(5) C(2) -C(5) -C(3) C(6); C(4) C(5) -C(5) C(2) -C(5) C(5);...
		-C(5) C(6) -C(3) -C(5) C(2) -C(5); C(5) -C(3) C(6) C(5) -C(5) C(2)];
	B23 = [-C(9) 0 0 -C(9) C(9) -C(9); 0 -C(9) 0 -C(9) -C(9) 0; 0 0 -C(9) C(9) 0 -C(9);...
		-C(9) -C(9) C(9) -C(9) 0 0; C(9) -C(9) 0 0 -C(9) 0; -C(9) 0 -C(9) 0 0 -C(9)];
	KE = 1/(1+nu)/(2*nu-1)*([A11 A12 A13 A14; A12' A22 A23 A13'; A13' A23' A22 A12'; A14' A13 A12 A11] +...
		nu*[B11 B12 B13 B14; B12' B22 B23 B13'; B13' B23' B22 B12'; B14' B13 B12 B11]);
	KE = KE * cellSize_;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Solving_BuildingMeshHierarchy()
	global meshHierarchy_ numLevels_ eNodMatHalfTemp_ nonDyadic_;
	if numel(meshHierarchy_)>1, return; end
	nodeVolume = 1:(int32(meshHierarchy_(1).resX)+1)*(int32(meshHierarchy_(1).resY)+1)*(int32(meshHierarchy_(1).resZ)+1);
	nodeVolume = reshape(nodeVolume(:), meshHierarchy_(1).resY+1, meshHierarchy_(1).resX+1, meshHierarchy_(1).resZ+1);
	if 1==nonDyadic_ && numLevels_>=4
		numLevels_ = numLevels_ - 1;
	else
		nonDyadic_ = 0;
	end
	for ii=2:numLevels_
		if ii==2 && 1==nonDyadic_, spanWidth = 4; else, spanWidth = 2; end
		nx = meshHierarchy_(ii-1).resX/spanWidth;
		ny = meshHierarchy_(ii-1).resY/spanWidth;
		nz = meshHierarchy_(ii-1).resZ/spanWidth;
		meshHierarchy_(ii) = Data_CartesianMeshStruct();
		meshHierarchy_(ii).resX = nx;
		meshHierarchy_(ii).resY = ny;
		meshHierarchy_(ii).resZ = nz;
		meshHierarchy_(ii).eleSize = meshHierarchy_(ii-1).eleSize*spanWidth;
		meshHierarchy_(ii).spanWidth = spanWidth;
		iEleVolume = reshape(meshHierarchy_(ii-1).eleMapForward, spanWidth*ny, spanWidth*nx, spanWidth*nz);
		iEleVolumeTemp = reshape((1:int32(spanWidth^3*nx*ny*nz))', spanWidth*ny, spanWidth*nx, spanWidth*nz);
		iFineNodVolumeTemp = reshape((1:int32((spanWidth*nx+1)*(spanWidth*ny+1)*(spanWidth*nz+1)))', spanWidth*ny+1, spanWidth*nx+1, spanWidth*nz+1);
		elementUpwardMap = zeros(nx*ny*nz,spanWidth^3,'int32');
		elementUpwardMapTemp = zeros(nx*ny*nz,spanWidth^3,'int32');
		transferMatTemp = zeros((spanWidth+1)^3,nx*ny*nz,'int32');
		for jj=1:nz
			iFineEleGroup = iEleVolume(:,:,spanWidth*(jj-1)+1:spanWidth*(jj-1)+spanWidth);
			iFineEleGroupTemp = iEleVolumeTemp(:,:,spanWidth*(jj-1)+1:spanWidth*(jj-1)+spanWidth);
			iFineNodGroupTemp = iFineNodVolumeTemp(:,:,spanWidth*(jj-1)+1:spanWidth*jj+1);
			for kk=1:nx
				iFineEleSubGroup = iFineEleGroup(:,spanWidth*(kk-1)+1:spanWidth*(kk-1)+spanWidth,:);
				iFineEleSubGroupTemp = iFineEleGroupTemp(:,spanWidth*(kk-1)+1:spanWidth*(kk-1)+spanWidth,:);
				iFineNodSubGroupTemp = iFineNodGroupTemp(:,spanWidth*(kk-1)+1:spanWidth*kk+1,:);
				for gg=1:ny
					iFineEles = iFineEleSubGroup(spanWidth*(gg-1)+1:spanWidth*(gg-1)+spanWidth,:,:);
					iFineEles = reshape(iFineEles, spanWidth^3, 1)';
					iFineElesTemp = iFineEleSubGroupTemp(spanWidth*(gg-1)+1:spanWidth*(gg-1)+spanWidth,:,:);
					iFineElesTemp = reshape(iFineElesTemp, spanWidth^3, 1)';
					iFineNodsTemp = iFineNodSubGroupTemp(spanWidth*(gg-1)+1:spanWidth*gg+1,:,:);
					iFineNodsTemp = reshape(iFineNodsTemp, (spanWidth+1)^3, 1)';
					eleIndex = (jj-1)*ny*nx + (kk-1)*ny + gg;
					elementUpwardMap(eleIndex,:) = iFineEles;
					elementUpwardMapTemp(eleIndex,:) = iFineElesTemp;
					transferMatTemp(:,eleIndex) = iFineNodsTemp;
				end
			end
		end
		unemptyElements = find(sum(elementUpwardMap,2)>0);
		elementUpwardMapTemp = elementUpwardMapTemp(unemptyElements,:);
		elementsIncVoidLastLevelGlobalOrdering = reshape(elementUpwardMapTemp, numel(elementUpwardMapTemp), 1);
		nodesIncVoidLastLevelGlobalOrdering = eNodMatHalfTemp_(elementsIncVoidLastLevelGlobalOrdering,:);
		nodesIncVoidLastLevelGlobalOrdering = Common_RecoverHalfeNodMat(nodesIncVoidLastLevelGlobalOrdering);
		nodesIncVoidLastLevelGlobalOrdering = unique(nodesIncVoidLastLevelGlobalOrdering);
		meshHierarchy_(ii).intermediateNumNodes = length(nodesIncVoidLastLevelGlobalOrdering);
		transferMatTemp = transferMatTemp(:,unemptyElements);
		temp = zeros((spanWidth*nx+1)*(spanWidth*ny+1)*(spanWidth*nz+1),1,'int32');
		temp(nodesIncVoidLastLevelGlobalOrdering) = (1:meshHierarchy_(ii).intermediateNumNodes)';
		meshHierarchy_(ii).transferMat = temp(transferMatTemp);
		meshHierarchy_(ii).d_transferMat = gpuArray(meshHierarchy_(ii).transferMat);
		meshHierarchy_(ii).transferMatCoeffi = zeros(meshHierarchy_(ii).intermediateNumNodes,1);
		for kk=1:1:(spanWidth+1)^3
			solidNodesLastLevel = meshHierarchy_(ii).transferMat(kk,:);
			meshHierarchy_(ii).transferMatCoeffi(solidNodesLastLevel,1) = meshHierarchy_(ii).transferMatCoeffi(solidNodesLastLevel,1) + 1;
		end
		meshHierarchy_(ii).d_transferMatCoeffi = gpuArray(meshHierarchy_(ii).transferMatCoeffi);
		elementsLastLevelGlobalOrdering = meshHierarchy_(ii-1).eleMapBack;
		nodesLastLevelGlobalOrdering = eNodMatHalfTemp_(elementsLastLevelGlobalOrdering,:);
		nodesLastLevelGlobalOrdering = Common_RecoverHalfeNodMat(nodesLastLevelGlobalOrdering);
		nodesLastLevelGlobalOrdering = unique(nodesLastLevelGlobalOrdering);
		[~,meshHierarchy_(ii).solidNodeMapCoarser2Finer] = intersect(nodesIncVoidLastLevelGlobalOrdering, nodesLastLevelGlobalOrdering);
		meshHierarchy_(ii).solidNodeMapCoarser2Finer = int32(meshHierarchy_(ii).solidNodeMapCoarser2Finer);
		meshHierarchy_(ii).d_solidNodeMapCoarser2Finer = gpuArray(meshHierarchy_(ii).solidNodeMapCoarser2Finer);
		meshHierarchy_(ii).eleMapForward = zeros(nx*ny*nz,1,'int32');
		meshHierarchy_(ii).eleMapBack = int32(unemptyElements);
		meshHierarchy_(ii).numElements = length(unemptyElements);
		meshHierarchy_(ii).eleMapForward(unemptyElements) = (1:meshHierarchy_(ii).numElements)';
		elementUpwardMap = elementUpwardMap(unemptyElements,:);
		meshHierarchy_(ii).elementUpwardMap = elementUpwardMap; clear elementUpwardMap
		nodenrs = reshape(1:int32((nx+1)*(ny+1)*(nz+1)), 1+meshHierarchy_(ii).resY, 1+meshHierarchy_(ii).resX, 1+meshHierarchy_(ii).resZ);
		eNodVec = reshape(nodenrs(1:end-1,1:end-1,1:end-1)+1,nx*ny*nz, 1);
		eNodMat = repmat(eNodVec(meshHierarchy_(ii).eleMapBack),1,8);
		eNodMatHalfTemp_ = repmat(eNodVec,1,8);
		tmp = [0 ny+[1 0] -1 (ny+1)*(nx+1)+[0 ny+[1 0] -1]]; tmp = int32(tmp);
		for jj=1:8
			eNodMat(:,jj) = eNodMat(:,jj) + repmat(tmp(jj), meshHierarchy_(ii).numElements,1);
			eNodMatHalfTemp_(:,jj) = eNodMatHalfTemp_(:,jj) + repmat(tmp(jj), nx*ny*nz,1);
		end
		eNodMatHalfTemp_ = eNodMatHalfTemp_(:,[3 4 7 8]);
		meshHierarchy_(ii).nodMapBack = unique(eNodMat);
		meshHierarchy_(ii).numNodes = length(meshHierarchy_(ii).nodMapBack);
		meshHierarchy_(ii).numDOFs = meshHierarchy_(ii).numNodes*3;
		meshHierarchy_(ii).nodMapForward = zeros((nx+1)*(ny+1)*(nz+1),1,'int32');
		meshHierarchy_(ii).nodMapForward(meshHierarchy_(ii).nodMapBack) = (1:meshHierarchy_(ii).numNodes)';
		for jj=1:8, eNodMat(:,jj) = meshHierarchy_(ii).nodMapForward(eNodMat(:,jj)); end
		if 1==nonDyadic_, kk = ii; else, kk = ii-1; end
		tmp = nodeVolume(1:2^kk:meshHierarchy_(1).resY+1, 1:2^kk:meshHierarchy_(1).resX+1, 1:2^kk:meshHierarchy_(1).resZ+1);
		tmp = reshape(tmp,numel(tmp),1);
		meshHierarchy_(ii).nodMapBack = tmp(meshHierarchy_(ii).nodMapBack);
		meshHierarchy_(ii).multiGridOperatorRI = Solving_Operator4MultiGridRestrictionAndInterpolation('inNODE', spanWidth);
		meshHierarchy_(ii).d_multiGridOperatorRI = gpuArray(full(meshHierarchy_(ii).multiGridOperatorRI));
		numElesAroundNode = zeros(meshHierarchy_(ii).numNodes,1,'int32');
		for jj=1:8
			iNodes = eNodMat(:,jj);
			numElesAroundNode(iNodes,:) = numElesAroundNode(iNodes) + 1;
		end
		meshHierarchy_(ii).nodesOnBoundary = int32(find(numElesAroundNode<8));
		meshHierarchy_(ii).eNodMat = eNodMat;
		meshHierarchy_(ii).d_eNodMat = gpuArray(eNodMat);
	end
	clear -global eNodMatHalfTemp_
	disp('Mesh Hierarchy...');
	disp('             #Resolutions         #Elements   #DOFs');
	for ii=1:numel(meshHierarchy_)
		disp([sprintf('...Level %i', ii), sprintf(': %4i x %4i x %4i', [meshHierarchy_(ii).resX meshHierarchy_(ii).resY ...
			meshHierarchy_(ii).resZ]), sprintf(' %11i', meshHierarchy_(ii).numElements), sprintf(' %11i', meshHierarchy_(ii).numDOFs)]);
	end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = Data_CartesianMeshStruct()
	val = struct(...
		'resX',0,'resY',0,'resZ',0,'spanWidth',0,'eleSize',[],'numElements',0,...
		'numNodes',0,'numDOFs',0,'eNodMat',[],'d_eNodMat',[],'numNod2ElesVec',0,...
		'freeDOFs',0,'fixedDOFs',0,'Ke',[],'eleModulus',[],'d_eleModulus',[],...
		'Ks',[],'d_Ks',[],'storingState',0,'diagK',[],'d_diagK',[],...
		'eleMapBack',[],'eleMapForward',[],'nodMapBack',[],'nodMapForward',[],...
		'solidNodeMapCoarser2Finer',[],'d_solidNodeMapCoarser2Finer',[],...
		'intermediateNumNodes',0,'nodesOnBoundary',[],'boundaryNodeCoords',[],...
		'elementsOnBoundary',[],'boundaryEleFaces',[],'elementUpwardMap',[],...
		'multiGridOperatorRI',[],'d_multiGridOperatorRI',[],...
		'transferMat',[],'d_transferMat',[],'transferMatCoeffi',[],'d_transferMatCoeffi',[]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function eNodMat = Common_RecoverHalfeNodMat(eNodMatHalf)
	if 4~=size(eNodMatHalf,2), eNodMat = []; return; end
	numEles = size(eNodMatHalf,1);
	eNodMat = zeros(numEles,8,'int32');
	eNodMat(:,[3 4 7 8]) = eNodMatHalf;
	eNodMat(:,[2 1 6 5]) = eNodMatHalf + 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Solving_SetupKeWithFixedDOFs()
	global meshHierarchy_;
	global isThisEle2ndLevelIncludingFixedDOFsOn1stLevel_;
	global uniqueKesFixed_ d_uniqueKesFixed_ uniqueKesFree_ d_uniqueKesFree_;
	global sonElesWithFixedDOFs_ mapUniqueKes_ d_mapUniqueKes_;
	global fixingCond_ modulus_ d_eleWithFixedDOFs_;
	allElements = zeros(meshHierarchy_(1).numElements,1,'int32');
	allElements(meshHierarchy_(1).elementsOnBoundary) = 1;
	allNodes = zeros(meshHierarchy_(1).numNodes,1,'int32');
	allNodes(meshHierarchy_(1).nodesOnBoundary) = (1:numel(meshHierarchy_(1).nodesOnBoundary))';
	nodeBoundaryStruct = struct('arr', []);
	nodeBoundaryStruct = repmat(nodeBoundaryStruct, numel(meshHierarchy_(1).nodesOnBoundary), 1);
	for ii=1:meshHierarchy_(1).numElements
		if allElements(ii)
			iNodes = meshHierarchy_(1).eNodMat(ii,:);
			for jj=1:8
				jiNode = allNodes(iNodes(jj));
				if jiNode, nodeBoundaryStruct(jiNode).arr(1,end+1) = ii; end
			end
		end
	end
	[~,fixedNodeIndices] = intersect(meshHierarchy_(1).nodesOnBoundary, fixingCond_(:,1));
	fixedNodesStruct = nodeBoundaryStruct(fixedNodeIndices,1);
	allElementsWithFixedDOFs = unique([fixedNodesStruct.arr]);
	numElementsWithFixedDOFs = numel(allElementsWithFixedDOFs);
	mapUniqueKes_ = zeros(meshHierarchy_(1).numElements,1,'int32');
	mapUniqueKes_(allElementsWithFixedDOFs,1) = (1:numElementsWithFixedDOFs)';
	KeCol = meshHierarchy_(1).Ke(:);
	uniqueKesFree_ = repmat(KeCol,1,numElementsWithFixedDOFs);
	uniqueKesFixed_ = zeros(size(uniqueKesFree_));
	for ii=1:numel(fixedNodesStruct)
		iFixation = fixingCond_(ii,:);
		iNode = iFixation(1);
		iNodeFixationState = iFixation(2:end);
		iNodEles = fixedNodesStruct(ii).arr;
		numElesOfThisNode = numel(iNodEles);
		for jj=1:numElesOfThisNode
			jEleNodes = meshHierarchy_(1).eNodMat(iNodEles(jj),:);
			fixedNodeLocally = find(iNode==jEleNodes);
			jEleLocally = mapUniqueKes_(iNodEles(jj));
			fixedDOFsLocally = 3*fixedNodeLocally - [2 1 0];
			fixedDOFsLocally = fixedDOFsLocally(1==iNodeFixationState);
			[uniqueKesFree_(:,jEleLocally), uniqueKesFixed_(:,jEleLocally)] = ...
				Solving_ApplyBConEleStiffMat_B(uniqueKesFree_(:,jEleLocally), uniqueKesFixed_(:,jEleLocally), fixedDOFsLocally, numElesOfThisNode);
		end
	end
	uniqueKesFixed_ = uniqueKesFixed_ * modulus_;
	isThisEle2ndLevelIncludingFixedDOFsOn1stLevel_ = zeros(meshHierarchy_(2).numElements,1,'int32');
	allElements = zeros(size(allElements),'int32');
	allElements(allElementsWithFixedDOFs,1) = 1;
	elementUpwardMap = meshHierarchy_(2).elementUpwardMap;
	safeIdx = max(elementUpwardMap, 1);
	hasFixed = any(allElements(safeIdx) & (elementUpwardMap > 0), 2);
	isThisEle2ndLevelIncludingFixedDOFsOn1stLevel_(hasFixed) = 1;
	eles2ndFinestIncludingElesFinestWithFixedDOFs = find(isThisEle2ndLevelIncludingFixedDOFsOn1stLevel_>0);
	isThisEle2ndLevelIncludingFixedDOFsOn1stLevel_(eles2ndFinestIncludingElesFinestWithFixedDOFs) = ...
		1:numel(eles2ndFinestIncludingElesFinestWithFixedDOFs);
	sonElesWithFixedDOFs_ = struct('arr', []);
	sonElesWithFixedDOFs_ = repmat(sonElesWithFixedDOFs_, numel(eles2ndFinestIncludingElesFinestWithFixedDOFs), 1);
	for jj=1:meshHierarchy_(2).numElements
		idx = isThisEle2ndLevelIncludingFixedDOFsOn1stLevel_(jj);
		if idx
			sonEles = elementUpwardMap(jj,:);
			[~,sonElesWithFixedDOFs_(idx).arr] = intersect(sonEles, allElementsWithFixedDOFs);
		end
	end
	d_uniqueKesFixed_ = gpuArray(uniqueKesFixed_);
	d_uniqueKesFree_  = gpuArray(uniqueKesFree_);
	d_mapUniqueKes_   = gpuArray(int32(mapUniqueKes_));
	d_eleWithFixedDOFs_ = int32(find(d_mapUniqueKes_>0));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [oKeFreeDOFs, oKeFixedDOFs] = Solving_ApplyBConEleStiffMat_B(iKeFreeDOFs, iKeFixedDOFs, fixedDOFsLocally, nAdjEles)
	iKeFreeDOFs = reshape(iKeFreeDOFs,24,24);
	iKeFixedDOFs = reshape(iKeFixedDOFs,24,24);
	oKeFreeDOFs = iKeFreeDOFs;
	oKeFixedDOFs = iKeFixedDOFs;
	oKeFreeDOFs(fixedDOFsLocally,:) = 0;
	oKeFreeDOFs(:,fixedDOFsLocally) = 0;
	for ii=1:numel(fixedDOFsLocally)
		oKeFixedDOFs(fixedDOFsLocally(ii), fixedDOFsLocally(ii)) = 1/nAdjEles;
	end
	oKeFreeDOFs = oKeFreeDOFs(:);
	oKeFixedDOFs = oKeFixedDOFs(:);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Solving_AssembleFEAstencil()
	global meshHierarchy_ numLevels_ cholFac_ cholPermut_;
	global isThisEle2ndLevelIncludingFixedDOFsOn1stLevel_;
	global uniqueKesFixed_ uniqueKesFree_ d_uniqueKesFixed_ d_uniqueKesFree_;
	global sonElesWithFixedDOFs_ mapUniqueKes_ d_mapUniqueKes_;
	global mixed_Precision_;
	reOrdering = [1 9 17 2 10 18 3 11 19 4 12 20 5 13 21 6 14 22 7 15 23 8 16 24];
	t_asm  = zeros(1, numLevels_-1);
	t_diag = zeros(1, numLevels_-1);
	for ii=2:numLevels_
		spanWidth = meshHierarchy_(ii).spanWidth;
		interpolatingKe = Solving_Operator4MultiGridRestrictionAndInterpolation('inDOF',spanWidth);
		eNodMat4Finer2Coarser = Solving_SubEleNodMat(spanWidth);
		[rowIndice, colIndice, ~] = find(ones(24));
		eDofMat4Finer2Coarser = [3*eNodMat4Finer2Coarser-2 3*eNodMat4Finer2Coarser-1 3*eNodMat4Finer2Coarser];
		eDofMat4Finer2Coarser = eDofMat4Finer2Coarser(:, reOrdering);
		iK = eDofMat4Finer2Coarser(:,rowIndice)';
		jK = eDofMat4Finer2Coarser(:,colIndice)';
		numProjectNodes = (spanWidth+1)^3;
		numProjectDOFs = numProjectNodes*3;
		meshHierarchy_(ii).storingState = 1;
		meshHierarchy_(ii).Ke = meshHierarchy_(ii-1).Ke*spanWidth;
		numElements = meshHierarchy_(ii).numElements;
		diagK = zeros(meshHierarchy_(ii).numNodes,3);
		finerKes = zeros(24*24,spanWidth^3);
		elementUpwardMap = meshHierarchy_(ii).elementUpwardMap;
		t1 = tic;
		nSub = size(elementUpwardMap, 2);
		Psub = zeros(24, 24, nSub);
		interpolatingKe = full(interpolatingKe);
		for s = 1:nSub
			Psub(:, :, s) = interpolatingKe(eDofMat4Finer2Coarser(s, :), :);
		end
		d_Ks = gpuArray.zeros(24, 24, numElements, 'double');
		d_upMap = gpuArray(int32(elementUpwardMap));
		d_Psub = gpuArray(double(Psub));
		if 2==ii
			d_eleModulus = gpuArray(double(meshHierarchy_(1).eleModulus(:)));
			d_K_0 = gpuArray(double(meshHierarchy_(1).Ke(:)));
			assembleKs_level2_inplace(d_Ks, d_upMap, d_eleModulus, d_K_0, ...
				d_Psub, d_mapUniqueKes_, d_uniqueKesFree_, d_uniqueKesFixed_);
			if ii==numLevels_, Ks = gather(d_Ks); end
		else
			assembleKs_higherLevel_inplace(d_Ks, d_upMap, ...
				meshHierarchy_(ii-1).d_Ks, d_Psub);
			if ii==numLevels_, Ks = gather(d_Ks); end
		end
		t_asm(ii-1) = toc(t1);
		meshHierarchy_(ii).d_Ks = d_Ks;
		clear d_Ks;
		if ii<numLevels_
			t2 = tic;
			d_Ks_flat     = reshape(meshHierarchy_(ii).d_Ks, 576, []);
			d_diagKeBlock = d_Ks_flat(1:25:end, :)';
			d_diagK_out   = zeros(meshHierarchy_(ii).numNodes, 3, 'gpuArray');
			Scattering_inplace(d_diagK_out, meshHierarchy_(ii).d_eNodMat, d_diagKeBlock);
			d_diagK_vec = reshape(d_diagK_out.', meshHierarchy_(ii).numDOFs, 1);
			meshHierarchy_(ii).diagK = gather(d_diagK_vec);
			if mixed_Precision_==1
				meshHierarchy_(ii).d_diagK = single(d_diagK_vec);
			else
				meshHierarchy_(ii).d_diagK = d_diagK_vec;
			end
			t_diag(ii-1) = toc(t2);
		end
	end
	t_diagL1 = tic;
	numElements = meshHierarchy_(1).numElements;
	diagKe  = diag(meshHierarchy_(1).Ke);
	bEleIdx = find(mapUniqueKes_ > 0);
	bKIdx   = mapUniqueKes_(bEleIdx);
	d_diagKe      = gpuArray(diagKe(:)');
	d_diagKeBlock = meshHierarchy_(1).d_eleModulus(:) .* d_diagKe;
	if ~isempty(bEleIdx)
		d_diagFree_bd  = d_uniqueKesFree_(1:25:end, bKIdx);
		d_diagFixed_bd = d_uniqueKesFixed_(1:25:end, bKIdx);
		d_eleM_bd      = meshHierarchy_(1).d_eleModulus(bEleIdx);
		d_diagKeBlock(bEleIdx, :) = (d_diagFree_bd .* d_eleM_bd(:)' + d_diagFixed_bd)';
	end
	d_diagK_out = zeros(meshHierarchy_(1).numNodes, 3, 'gpuArray');
	Scattering_inplace(d_diagK_out, meshHierarchy_(1).d_eNodMat, d_diagKeBlock);
	d_diagK_vec = reshape(d_diagK_out.', meshHierarchy_(1).numDOFs, 1);
	meshHierarchy_(1).diagK = gather(d_diagK_vec);
	if mixed_Precision_==1
		meshHierarchy_(1).d_diagK = single(d_diagK_vec);
	else
		meshHierarchy_(1).d_diagK = d_diagK_vec;
	end
	t_diagL1 = toc(t_diagL1);
	[rowIndice, colIndice, ~] = find(ones(24));
	t_coarsest = tic;
	sK = zeros(24^2, meshHierarchy_(end).numElements);
	for ii=1:meshHierarchy_(end).numElements
		iKe = Ks(:,:,ii); sK(:,ii) = iKe(:);
	end
	eNodMat = meshHierarchy_(end).eNodMat;
	eDofMat = [3*eNodMat-2 3*eNodMat-1 3*eNodMat];
	eDofMat = eDofMat(:,reOrdering);
	iK = eDofMat(:,rowIndice);
	jK = eDofMat(:,colIndice);
	KcoarsestLevel = sparse(iK, jK, sK');
	[cholFac_, ~, cholPermut_] = chol(KcoarsestLevel,'lower');
	t_coarsest = toc(t_coarsest);
	msg = '  [AssembleStencil]';
	for ii = 2:numLevels_
		msg = [msg sprintf(' L%d:%.3fs', ii, t_asm(ii-1))];
	end
	msg = [msg sprintf(' L1-diagK:%.3fs Chol:%.3fs', t_diagL1, t_coarsest)];
	fprintf('%s\n', msg);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function blockIndex = Solving_MissionPartition(totalSize, blockSize)
	numBlocks = ceil(totalSize/blockSize);
	blockIndex = ones(numBlocks,2);
	blockIndex(1:numBlocks-1,2) = (1:1:numBlocks-1)' * blockSize;
	blockIndex(2:numBlocks,1) = blockIndex(2:numBlocks,1) + blockIndex(1:numBlocks-1,2);
	blockIndex(numBlocks,2) = totalSize;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [y, varargout] = Solving_PCG(AtX, PtV, b, tol, maxIT, printP, varargin)
	global mixed_Precision_;
	normB = gather(norm(b));
	b = full(b);
	if 7==nargin, y = varargin{1}; else, y = zeros(size(b),'gpuArray'); end
	rVec1 = b - AtX(y);
	zVec = PtV(rVec1);
	pVec = zVec;
	x1Val = dot(zVec, rVec1);
	for its=1:maxIT
		vVec = AtX(pVec);
		lambda = x1Val / dot(pVec, vVec);
		y = y + double(lambda * pVec);
		rVec1 = rVec1 - lambda*vVec;
		resnorm = gather(norm(rVec1))/normB;
		if printP(1), disp([' It.: ' sprintf('%4i',its) ' Res.: ' sprintf('%16.6e',resnorm)]); end
		if resnorm<tol
			if printP(2)
				disp(['CG converged at iteration' sprintf('%5i',its) ' residual' sprintf('%16.6e',resnorm)]);
			end
			break;
		end
		zVec = PtV(rVec1);
		x2Val = dot(zVec, rVec1);
		pVec = zVec + x2Val / x1Val * pVec;
		x1Val = x2Val;
	end
	if its==maxIT, warning('Exceed the maximum iterate numbers'); end
	if 2==nargout, varargout{1} = its; end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Y = Solving_KbyU_MatrixFree(uVec, varargin)
	global meshHierarchy_ uniqueKesFixed_ d_uniqueKesFixed_ uniqueKesFree_ d_uniqueKesFree_;
	global mapUniqueKes_ d_mapUniqueKes_ d_eleWithFixedDOFs_ mixed_Precision_;
	if 1==nargin, iLevel = 1; end
	if 2==nargin, iLevel = varargin{1}; end
	if mixed_Precision_==1
		uVec = single(reshape(uVec,3,meshHierarchy_(iLevel).numNodes)');
	else
		uVec = reshape(uVec,3,meshHierarchy_(iLevel).numNodes)';
	end
	if 1==iLevel
		Ke = meshHierarchy_(iLevel).d_Ks;
		numNodes = meshHierarchy_(1).numNodes;
		eNodMat = meshHierarchy_(1).d_eNodMat;
		Ee = meshHierarchy_(1).d_eleModulus;
		if mixed_Precision_==1
			Y = zeros(numNodes,3,'single','gpuArray');
			uMat = zeros(size(eNodMat,1),24,'single','gpuArray');
		else
			Y = zeros(numNodes,3,'gpuArray');
			uMat = zeros(size(eNodMat,1),24,'gpuArray');
		end
		Gathering_inplace(uMat, uVec, eNodMat);
		nEle = numel(d_eleWithFixedDOFs_);
		K_free  = reshape(d_uniqueKesFree_,  24, 24, nEle);
		K_fixed = reshape(d_uniqueKesFixed_, 24, 24, nEle);
		Ee_pages = reshape(Ee(d_eleWithFixedDOFs_), 1, 1, nEle);
		if mixed_Precision_==1
			K_combined = single(K_free .* Ee_pages + K_fixed);
		else
			K_combined = K_free .* Ee_pages + K_fixed;
		end
		sub = uMat(d_eleWithFixedDOFs_, :);
		U_pages = permute(sub, [2 3 1]);
		R = pagemtimes(K_combined, U_pages);
		subDisVecUnique = reshape(permute(R, [3 1 2]), nEle, 24);
		uMat = uMat*Ke.*Ee(:);
		uMat(d_eleWithFixedDOFs_,:) = subDisVecUnique;
		Scattering_inplace(Y, eNodMat, uMat);
		Y = Y'; Y = double(Y(:));
		wait(gpuDevice);
	else
		numNodes = meshHierarchy_(iLevel).numNodes;
		eNodMat = meshHierarchy_(iLevel).d_eNodMat;
		if mixed_Precision_==1
			Y = zeros(numNodes,3,'single','gpuArray');
			uMat = zeros(size(eNodMat,1),24,'single','gpuArray');
		else
			Y = zeros(numNodes,3,'gpuArray');
			uMat = zeros(size(eNodMat,1),24,'gpuArray');
		end
		Gathering_inplace(uMat, uVec, eNodMat);
		U3D = permute(uMat, [3 2 1]);
		Uout = pagemtimes(U3D, meshHierarchy_(iLevel).d_Ks);
		uMat = permute(Uout, [3 2 1]);
		Scattering_inplace(Y, eNodMat, uMat);
		Y = Y'; Y = double(Y(:));
		wait(gpuDevice);
	end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = Solving_Vcycle(r)
	global meshHierarchy_ weightFactorJacobi_ numLevels_;
	global cholFac_ cholPermut_ mixed_Precision_ typeVcycle_;
	if mixed_Precision_==1, r = single(r); end
	x_cell = cell(numLevels_, 1);
	r_cell = cell(numLevels_, 1);
	r_cell{1} = r;
	d_weightFactorJacobi = gpuArray(weightFactorJacobi_);
	for ii = 1:numLevels_-1
		x_cell{ii} = d_weightFactorJacobi * (r_cell{ii} ./ meshHierarchy_(ii).d_diagK);
		switch typeVcycle_
			case 'Adapted'
				r_cell{ii+1} = Solving_RestrictResidual(r_cell{ii}, ii+1);
			case 'Standard'
				d = r_cell{ii} - Solving_KbyU_MatrixFree(x_cell{ii}, ii);
				r_cell{ii+1} = Solving_RestrictResidual(d, ii+1);
		end
	end
	c_r = gather(r_cell{numLevels_});
	if mixed_Precision_==1
		c_x = single(cholPermut_ * (cholFac_'\(cholFac_\(cholPermut_'*double(c_r)))));
	else
		c_x = cholPermut_ * (cholFac_'\(cholFac_\(cholPermut_'*c_r)));
	end
	x_cell{numLevels_} = gpuArray(c_x);
	for ii = numLevels_-1:-1:1
		x_cell{ii} = x_cell{ii} + Solving_InterpolationDeviation(x_cell{ii+1}, ii+1);
		switch typeVcycle_
			case 'Adapted'
				x_cell{ii} = x_cell{ii} + d_weightFactorJacobi*r_cell{ii}./meshHierarchy_(ii).d_diagK;
			case 'Standard'
				x_cell{ii} = x_cell{ii} + d_weightFactorJacobi*(r_cell{ii} - ...
					Solving_KbyU_MatrixFree(x_cell{ii}, ii))./meshHierarchy_(ii).d_diagK;
		end
	end
	x = double(x_cell{1});
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rCoaser = Solving_RestrictResidual(rFiner, ii)
	global meshHierarchy_ mixed_Precision_;
	rFiner = reshape(rFiner,3,meshHierarchy_(ii-1).numNodes)';
	if mixed_Precision_==1
		rFiner1 = zeros(meshHierarchy_(ii).intermediateNumNodes,3,'single','gpuArray');
		rFiner1(meshHierarchy_(ii).d_solidNodeMapCoarser2Finer,:) = rFiner;
		rFiner1 = rFiner1./meshHierarchy_(ii).d_transferMatCoeffi;
		rCoaser = zeros(meshHierarchy_(ii).numNodes,3,'single','gpuArray');
	else
		rFiner1 = zeros(meshHierarchy_(ii).intermediateNumNodes,3,'gpuArray');
		rFiner1(meshHierarchy_(ii).d_solidNodeMapCoarser2Finer,:) = rFiner;
		rFiner1 = rFiner1./meshHierarchy_(ii).d_transferMatCoeffi;
		rCoaser = zeros(meshHierarchy_(ii).numNodes,3,'gpuArray');
	end
	RI   = meshHierarchy_(ii).d_multiGridOperatorRI;
	Ne   = size(meshHierarchy_(ii).d_eNodMat, 1);
	NPE  = size(meshHierarchy_(ii).d_eNodMat, 2);
	nInt = size(RI, 1);
	keys = meshHierarchy_(ii).d_eNodMat(:);
	F_evals = rFiner1(meshHierarchy_(ii).d_transferMat, :);
	F_pages = reshape(F_evals, [nInt, Ne, 3]);
	P3D     = pagemtimes(RI,'transpose', F_pages,'none');
	P_ne_npe_3 = permute(P3D, [2 1 3]);
	P2D        = reshape(P_ne_npe_3, [Ne*NPE, 3]);
	scatter_accum3_inplace(keys, P2D, rCoaser);
	rCoaser = reshape(rCoaser', 3*meshHierarchy_(ii).numNodes, 1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function xFiner = Solving_InterpolationDeviation(xCoarser, ii)
	global meshHierarchy_ mixed_Precision_;
	xCoarser = reshape(xCoarser,3,meshHierarchy_(ii).numNodes)';
	if mixed_Precision_==1
		xFiner = zeros(meshHierarchy_(ii).intermediateNumNodes,3,'single','gpuArray');
	else
		xFiner = zeros(meshHierarchy_(ii).intermediateNumNodes,3,'gpuArray');
	end
	transferMat = meshHierarchy_(ii).d_transferMat(:);
	RI   = meshHierarchy_(ii).d_multiGridOperatorRI;
	Ne   = size(meshHierarchy_(ii).d_eNodMat,1);
	NPE  = size(meshHierarchy_(ii).d_eNodMat,2);
	nInt = size(RI,1);
	Evals  = xCoarser(meshHierarchy_(ii).d_eNodMat, :);
	Epages = reshape(Evals, [Ne, NPE, 3]);
	Epages = permute(Epages, [2 1 3]);
	Tmp    = pagemtimes(RI, Epages);
	Tmp2D  = reshape(Tmp, [nInt*Ne, 3]);
	scatter_accum3_inplace(transferMat, Tmp2D, xFiner);
	xFiner = xFiner ./ meshHierarchy_(ii).d_transferMatCoeffi;
	xFiner = xFiner(meshHierarchy_(ii).d_solidNodeMapCoarser2Finer,:);
	xFiner = reshape(xFiner', 3*meshHierarchy_(ii-1).numNodes, 1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ceList = TopOpti_ComputeUnitCompliance(U)
	global meshHierarchy_ objWeightingList_;
	Ne       = meshHierarchy_(1).numElements;
	numNodes = meshHierarchy_(1).numNodes;
	d_eNodMat = meshHierarchy_(1).d_eNodMat;
	d_Ke     = gpuArray(meshHierarchy_(1).Ke);
	nRHS     = size(U, 2);
	ceList_gpu = gpuArray.zeros(Ne, nRHS);
	for jj = 1:nRHS
		ithU = U(:, jj);
		if ~isa(ithU, 'gpuArray'), ithU = gpuArray(ithU); end
		uVec = reshape(ithU, 3, numNodes)';
		iReshapedU = zeros(Ne, 24, 'gpuArray');
		Gathering_inplace(iReshapedU, uVec, d_eNodMat);
		ceList_gpu(:, jj) = sum((iReshapedU * d_Ke) .* iReshapedU, 2);
	end
	ceList = ceList_gpu * objWeightingList_(:);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [passiveElementsOnBoundary, passiveElementsNearLoads, passiveElementsNearFixation] = TopOpti_SetPassiveElements(numLayerboundary, numLayerLoads, numLayerFixation)
	global meshHierarchy_ loadingCond_ fixingCond_ passiveElements_;
	numLayerboundary  = round(numLayerboundary);
	numLayerLoads     = round(numLayerLoads);
	numLayerFixation  = round(numLayerFixation);
	existingPassiveElements = passiveElements_;
	passiveElementsOnBoundary = [];
	if numLayerboundary>0
		index = 1;
		while index<=numLayerboundary
			if 1==index
				passiveElementsOnBoundary = double(meshHierarchy_(1).elementsOnBoundary);
			else
				passiveElementsOnBoundary = Common_IncludeAdjacentElements(passiveElementsOnBoundary);
			end
			index = index + 1;
		end
	end
	passiveElementsNearLoads = [];
	passiveElementsNearFixation = [];
	if numLayerLoads>0 || numLayerFixation>0
		allElements = zeros(meshHierarchy_(1).numElements,1);
		allElements(meshHierarchy_(1).elementsOnBoundary) = 1;
		nodeStruct_ = struct('arr', []);
		nodeStruct_ = repmat(nodeStruct_, meshHierarchy_(1).numNodes, 1);
		for ii=1:meshHierarchy_(1).numElements
			if allElements(ii)
				iNodes = meshHierarchy_(1).eNodMat(ii,:);
				for jj=1:8, nodeStruct_(iNodes(jj)).arr(1,end+1) = ii; end
			end
		end
		passiveElementsNearLoads = [];
		if numLayerLoads>0
			for ii=1:numel(loadingCond_)
				iLoad = loadingCond_{ii};
				loadedNodes = meshHierarchy_(1).nodesOnBoundary(iLoad(:,1));
				allLoadedNodes = nodeStruct_(loadedNodes);
				passiveElementsNearLoads = [passiveElementsNearLoads, unique([allLoadedNodes.arr])];
			end
			index = 2;
			while index<=numLayerLoads
				passiveElementsNearLoads = Common_IncludeAdjacentElements(passiveElementsNearLoads);
				index = index + 1;
			end
		end
		passiveElementsNearFixation = [];
		if numLayerFixation>0
			fixedNodes = meshHierarchy_(1).nodesOnBoundary(fixingCond_(:,1));
			allFixedNodes = nodeStruct_(fixedNodes);
			passiveElementsNearFixation = unique([allFixedNodes.arr]);
			index = 2;
			while index<=numLayerFixation
				passiveElementsNearFixation = Common_IncludeAdjacentElements(passiveElementsNearFixation);
				index = index + 1;
			end
		end
		passiveElements_ = unique([existingPassiveElements; passiveElementsOnBoundary(:); ...
			passiveElementsNearLoads(:); passiveElementsNearFixation(:)]);
	else
		passiveElements_ = unique([existingPassiveElements; passiveElementsOnBoundary(:)]);
	end
	passiveElements_ = unique([existingPassiveElements; passiveElements_]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function oEleList = Common_IncludeAdjacentElements(iEleList)
	global meshHierarchy_;
	iEleListMapBack = meshHierarchy_(1).eleMapBack(iEleList);
	resX = meshHierarchy_(1).resX;
	resY = meshHierarchy_(1).resY;
	resZ = meshHierarchy_(1).resZ;
	[eleX, eleY, eleZ] = Common_NodalizeDesignDomain([resX-1 resY-1 resZ-1], [1 1 1; resX resY resZ]);
	eleX = eleX(iEleListMapBack);
	eleY = eleY(iEleListMapBack);
	eleZ = eleZ(iEleListMapBack);
	tmpX = [eleX-1 eleX-1 eleX-1  eleX eleX eleX  eleX+1 eleX+1 eleX+1];
	tmpX = [tmpX tmpX tmpX]; tmpX = tmpX(:);
	tmpY = [eleY+1 eleY eleY-1  eleY+1 eleY eleY-1  eleY+1 eleY eleY-1];
	tmpY = [tmpY tmpY tmpY]; tmpY = tmpY(:);
	tmpZ = [eleZ eleZ eleZ eleZ eleZ eleZ eleZ eleZ eleZ];
	tmpZ = [tmpZ-1 tmpZ tmpZ+1]; tmpZ = tmpZ(:);
	xNegative = find(tmpX<1); xPositive = find(tmpX>resX);
	yNegative = find(tmpY<1); yPositive = find(tmpY>resY);
	zNegative = find(tmpZ<1); zPositive = find(tmpZ>resZ);
	allInvalidEles = unique([xNegative; xPositive; yNegative; yPositive; zNegative; zPositive]);
	tmpX(allInvalidEles) = []; tmpY(allInvalidEles) = []; tmpZ(allInvalidEles) = [];
	oEleListMapBack = resX*resY*(tmpZ-1) + resY*(tmpX-1) + resY-tmpY + 1;
	oEleList = meshHierarchy_(1).eleMapForward(oEleListMapBack);
	oEleList(oEleList<1) = []; oEleList = unique(oEleList);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [eleX, eleY, eleZ] = Common_NodalizeDesignDomain(numSeed, dd)
	% Returns integer x/y/z coordinates (1-based) for all elements in an
	% nx x ny x nz grid where nx=numSeed(1)+1, ny=numSeed(2)+1, nz=numSeed(3)+1
	nx = numSeed(1); ny = numSeed(2); nz = numSeed(3);
	xSeed = dd(1,1):(dd(2,1)-dd(1,1))/nx:dd(2,1);         % 1:resX
	ySeed = dd(2,2):(dd(1,2)-dd(2,2))/ny:dd(1,2);         % resY:-1:1
	zSeed = dd(1,3):(dd(2,3)-dd(1,3))/nz:dd(2,3);         % 1:resZ
	eleX = repmat(reshape(repmat(xSeed,ny+1,1), (nx+1)*(ny+1), 1), (nz+1), 1);
	eleY = repmat(repmat(ySeed,1,nx+1)', (nz+1), 1);
	eleZ = reshape(repmat(zSeed,(nx+1)*(ny+1),1), (nx+1)*(ny+1)*(nz+1), 1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ss = Solving_Operator4MultiGridRestrictionAndInterpolation(opt, spanWidth)
	switch spanWidth
		case 2
			ss = [  0		0		0		1		0		0		0		0
					0.5		0		0		0.5		0		0		0		0
					1		0		0		0		0		0		0		0
					0		0		0.5		0.5		0		0		0		0
					0.25	0.25	0.25	0.25	0		0		0		0
					0.5		0.5		0		0		0		0		0		0
					0		0		1		0		0		0		0		0
					0		0.5		0.5		0		0		0		0		0
					0		1		0		0		0		0		0		0
					0		0		0		0.5		0		0		0		0.5
					0.25	0		0		0.25	0.25	0		0		0.25
					0.5		0		0		0		0.5		0		0		0
					0		0		0.25	0.25	0		0		0.25	0.25
					0.125	0.125	0.125	0.125	0.125	0.125	0.125	0.125
					0.25	0.25	0		0		0.25	0.25	0		0
					0		0		0.5		0		0		0		0.5		0
					0		0.25	0.25	0		0		0.25	0.25	0
					0		0.5		0		0		0		0.5		0		0
					0		0		0		0		0		0		0		1
					0		0		0		0		0.5		0		0		0.5
					0		0		0		0		1		0		0		0
					0		0		0		0		0		0		0.5		0.5
					0		0		0		0		0.25	0.25	0.25	0.25
					0		0		0		0		0.5		0.5		0		0
					0		0		0		0		0		0		1		0
					0		0		0		0		0		0.5		0.5		0
					0		0		0		0		0		1		0		0];
		case 4
			ss = [  0			0			0			1			0			0			0			0
					0.25		0			0			0.75		0			0			0			0
					0.50		0			0			0.50		0			0			0			0
					0.75		0			0			0.25		0			0			0			0
					1			0			0			0			0			0			0			0
					0			0			0.2500		0.7500		0			0			0			0
					0.1875		0.0625		0.1875		0.5625		0			0			0			0
					0.3750		0.1250		0.1250		0.3750		0			0			0			0
					0.5625		0.1875		0.0625		0.1875		0			0			0			0
					0.7500		0.25		0			0			0			0			0			0
					0			0			0.50		0.5			0			0			0			0
					0.125		0.125		0.375		0.375		0			0			0			0
					0.250		0.250		0.250		0.250		0			0			0			0
					0.375		0.375		0.125		0.125		0			0			0			0
					0.500		0.500		0			0			0			0			0			0
					0			0			0.750		0.25		0			0			0			0
					0.0625		0.1875		0.5625		0.1875		0			0			0			0
					0.1250		0.3750		0.3750		0.1250		0			0			0			0
					0.1875		0.5625		0.1875		0.0625		0			0			0			0
					0.2500		0.7500		0			0			0			0			0			0
					0			0			1			0			0			0			0			0
					0			0.25		0.75		0			0			0			0			0
					0			0.50		0.50		0			0			0			0			0
					0			0.75		0.25		0			0			0			0			0
					0			1			0			0			0			0			0			0
					0			0			0			0.75		0			0			0			0.25
					0.1875		0			0			0.5625		0.0625		0			0			0.1875
					0.3750		0			0			0.3750		0.1250		0			0			0.1250
					0.5625		0			0			0.1875		0.1875		0			0			0.0625
					0.7500		0			0			0			0.25		0			0			0
					0			0			0.1875		0.5625		0			0			0.0625		0.1875
					0.140625	0.046875	0.140625	0.421875	0.046875	0.015625	0.046875	0.140625
					0.281250	0.093750	0.093750	0.281250	0.093750	0.031250	0.031250	0.093750
					0.421875	0.140625	0.046875	0.140625	0.140625	0.046875	0.015625	0.046875
					0.562500	0.187500	0			0			0.1875		0.0625		0			0
					0			0			0.375		0.375		0			0			0.125		0.125
					0.09375		0.09375		0.28125		0.28125		0.03125		0.03125		0.09375		0.09375
					0.18750		0.18750		0.18750		0.18750		0.06250		0.06250		0.06250		0.06250
					0.28125		0.28125		0.09375		0.09375		0.09375		0.09375		0.03125		0.03125
					0.37500		0.37500		0			0			0.125		0.125		0			0
					0			0			0.5625		0.1875		0			0			0.1875		0.0625
					0.046875	0.140625	0.421875	0.140625	0.015625	0.046875	0.140625	0.046875
					0.093750	0.281250	0.281250	0.093750	0.031250	0.093750	0.093750	0.031250
					0.140625	0.421875	0.140625	0.046875	0.046875	0.140625	0.046875	0.015625
					0.187500	0.562500	0			0			0.0625		0.1875		0			0
					0			0			0.75		0			0			0			0.25		0
					0			0.1875		0.5625		0			0			0.0625		0.1875		0
					0			0.3750		0.3750		0			0			0.1250		0.1250		0
					0			0.5625		0.1875		0			0			0.1875		0.0625		0
					0			0.75		0			0			0			0.25		0			0
					0			0			0			0.5	0		0			0			0.5
					0.125		0			0			0.375		0.125		0			0			0.375
					0.250		0			0			0.250		0.250		0			0			0.250
					0.375		0			0			0.125		0.375		0			0			0.125
					0.5			0			0			0			0.5			0			0			0
					0			0			0.125		0.375		0			0			0.125		0.375
					0.09375		0.03125		0.09375		0.28125		0.09375		0.03125		0.09375		0.28125
					0.18750		0.06250		0.06250		0.18750		0.18750		0.06250		0.06250		0.18750
					0.28125		0.09375		0.03125		0.09375		0.28125		0.09375		0.03125		0.09375
					0.37500		0.12500		0			0			0.375		0.125		0			0
					0			0			0.25		0.25		0			0			0.25		0.25
					0.0625		0.0625		0.1875		0.1875		0.0625		0.0625		0.1875		0.1875
					0.1250		0.1250		0.1250		0.1250		0.1250		0.1250		0.1250		0.1250
					0.1875		0.1875		0.0625		0.0625		0.1875		0.1875		0.0625		0.0625
					0.2500		0.2500		0			0			0.25		0.2500		0			0
					0			0			0.375		0.125		0			0			0.375		0.125
					0.03125		0.09375		0.28125		0.09375		0.03125		0.09375		0.28125		0.09375
					0.06250		0.18750		0.18750		0.06250		0.06250		0.18750		0.18750		0.06250
					0.09375		0.28125		0.09375		0.03125		0.09375		0.28125		0.09375		0.03125
					0.12500		0.37500		0			0			0.125		0.375		0			0
					0			0			0.5			0			0			0			0.5			0
					0			0.125		0.375		0			0			0.125		0.375		0
					0			0.250		0.250		0			0			0.250		0.250		0
					0			0.375		0.125		0			0			0.375		0.125		0
					0			0.500		0			0			0			0.5			0			0
					0			0			0			0.25		0			0			0			0.75
					0.0625		0			0			0.1875		0.1875		0			0			0.5625
					0.1250		0			0			0.1250		0.3750		0			0			0.3750
					0.1875		0			0			0.0625		0.5625		0			0			0.1875
					0.2500		0			0			0			0.75		0			0			0
					0			0			0.0625		0.1875		0			0			0.1875		0.5625
					0.046875	0.015625	0.046875	0.140625	0.140625	0.046875	0.140625	0.421875
					0.093750	0.031250	0.031250	0.093750	0.281250	0.093750	0.093750	0.281250
					0.140625	0.046875	0.015625	0.046875	0.421875	0.140625	0.046875	0.140625
					0.187500	0.062500	0			0			0.5625		0.1875		0			0
					0			0			0.125		0.125		0			0			0.375		0.375
					0.03125		0.03125		0.09375		0.09375		0.09375		0.09375		0.28125		0.28125
					0.06250		0.06250		0.06250		0.06250		0.18750		0.18750		0.18750		0.18750
					0.09375		0.09375		0.03125		0.03125		0.28125		0.28125		0.09375		0.09375
					0.12500		0.12500		0			0			0.37500		0.37500		0			0
					0			0			0.1875		0.0625		0			0			0.5625		0.1875
					0.015625	0.046875	0.140625	0.046875	0.046875	0.140625	0.421875	0.140625
					0.031250	0.093750	0.093750	0.031250	0.093750	0.281250	0.281250	0.093750
					0.046875	0.140625	0.046875	0.015625	0.140625	0.421875	0.140625	0.046875
					0.062500	0.187500	0			0			0.1875		0.5625		0			0
					0			0			0.25		0			0			0			0.7500		0
					0			0.0625		0.1875		0			0			0.1875		0.5625		0
					0			0.1250		0.1250		0			0			0.3750		0.3750		0
					0			0.1875		0.0625		0			0			0.5625		0.1875		0
					0			0.2500		0			0			0			0.75		0			0
					0			0			0			0			0			0			0			1
					0			0			0			0			0.25		0			0			0.75
					0			0			0			0			0.50		0			0			0.50
					0			0			0			0			0.75		0			0			0.25
					0			0			0			0			1			0			0			0
					0			0			0			0			0			0			0.25		0.75
					0			0			0			0			0.1875		0.0625		0.1875		0.5625
					0			0			0			0			0.3750		0.1250		0.1250		0.3750
					0			0			0			0			0.5625		0.1875		0.0625		0.1875
					0			0			0			0			0.7500		0.2500		0			0
					0			0			0			0			0			0			0.5			0.5
					0			0			0			0			0.125		0.125		0.375		0.375
					0			0			0			0			0.250		0.250		0.250		0.250
					0			0			0			0			0.375		0.375		0.125		0.125
					0			0			0			0			0.500		0.500		0			0
					0			0			0			0			0			0			0.75		0.25
					0			0			0			0			0.0625		0.1875		0.5625		0.1875
					0			0			0			0			0.1250		0.3750		0.3750		0.1250
					0			0			0			0			0.1875		0.5625		0.1875		0.0625
					0			0			0			0			0.2500		0.7500		0			0
					0			0			0			0			0			0			1			0
					0			0			0			0			0			0.25		0.75		0
					0			0			0			0			0			0.50		0.50		0
					0			0			0			0			0			0.75		0.25		0
					0			0			0			0			0			1			0			0];
		otherwise
			error('Wrong Span Width!');
	end
	if strcmp(opt, 'inDOF')
		ss = kron(ss, eye(3));
	end
	ss = sparse(ss);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = Solving_SubEleNodMat(spanWidth)
	switch spanWidth
		case 2
			val = [ 2 	5 	4 	1 	11 	14 	13 	10
					3 	6 	5 	2 	12 	15 	14 	11
					6 	9 	8 	5 	15 	18 	17 	14
					5 	8 	7 	4 	14 	17 	16 	13
					11 	14 	13 	10 	20 	23 	22 	19
					12 	15 	14 	11 	21 	24 	23 	20
					15 	18 	17 	14 	24 	27 	26 	23
					14 	17 	16 	13 	23 	26 	25 	22];
		case 4
			val = [ 2	7	6	1	27	32	31	26
					3	8	7	2	28	33	32	27
					4	9	8	3	29	34	33	28
					5	10	9	4	30	35	34	29
					7	12	11	6	32	37	36	31
					8	13	12	7	33	38	37	32
					9	14	13	8	34	39	38	33
					10	15	14	9	35	40	39	34
					12	17	16	11	37	42	41	36
					13	18	17	12	38	43	42	37
					14	19	18	13	39	44	43	38
					15	20	19	14	40	45	44	39
					17	22	21	16	42	47	46	41
					18	23	22	17	43	48	47	42
					19	24	23	18	44	49	48	43
					20	25	24	19	45	50	49	44
					27	32	31	26	52	57	56	51
					28	33	32	27	53	58	57	52
					29	34	33	28	54	59	58	53
					30	35	34	29	55	60	59	54
					32	37	36	31	57	62	61	56
					33	38	37	32	58	63	62	57
					34	39	38	33	59	64	63	58
					35	40	39	34	60	65	64	59
					37	42	41	36	62	67	66	61
					38	43	42	37	63	68	67	62
					39	44	43	38	64	69	68	63
					40	45	44	39	65	70	69	64
					42	47	46	41	67	72	71	66
					43	48	47	42	68	73	72	67
					44	49	48	43	69	74	73	68
					45	50	49	44	70	75	74	69
					52	57	56	51	77	82	81	76
					53	58	57	52	78	83	82	77
					54	59	58	53	79	84	83	78
					55	60	59	54	80	85	84	79
					57	62	61	56	82	87	86	81
					58	63	62	57	83	88	87	82
					59	64	63	58	84	89	88	83
					60	65	64	59	85	90	89	84
					62	67	66	61	87	92	91	86
					63	68	67	62	88	93	92	87
					64	69	68	63	89	94	93	88
					65	70	69	64	90	95	94	89
					67	72	71	66	92	97	96	91
					68	73	72	67	93	98	97	92
					69	74	73	68	94	99	98	93
					70	75	74	69	95	100	99	94
					77	82	81	76	102	107	106	101
					78	83	82	77	103	108	107	102
					79	84	83	78	104	109	108	103
					80	85	84	79	105	110	109	104
					82	87	86	81	107	112	111	106
					83	88	87	82	108	113	112	107
					84	89	88	83	109	114	113	108
					85	90	89	84	110	115	114	109
					87	92	91	86	112	117	116	111
					88	93	92	87	113	118	117	112
					89	94	93	88	114	119	118	113
					90	95	94	89	115	120	119	114
					92	97	96	91	117	122	121	116
					93	98	97	92	118	123	122	117
					94	99	98	93	119	124	123	118
					95	100	99	94	120	125	124	119];
		otherwise
			error('Wrong input of span width!')
	end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function tarBC = AdaptBCExternalMdl(srcBC, adjustedRes)
	global nelx_ nely_ nelz_;
	nullNodeVolume = zeros((nelx_+1)*(nely_+1)*(nelz_+1),1);
	adjustedNnlx = adjustedRes(1); adjustedNnly = adjustedRes(2); adjustedNnlz = adjustedRes(3);
	nnlx = nelx_+1; nnly = nely_+1; nnlz = nelz_+1;
	tmpBC = zeros(size(srcBC));
	[~, ia] = sort(srcBC(:,1)); srcBC = srcBC(ia,:);
	nodeVolume = nullNodeVolume;
	nodeVolume(srcBC(:,1)) = 1;
	nodeVolume = reshape(nodeVolume, nely_+1, nelx_+1, nelz_+1);
	nodeVolume(:,end+1:adjustedNnlx,:) = zeros(nnly,adjustedNnlx-nnlx,nnlz);
	nodeVolume(end+1:adjustedNnly,:,:) = zeros(adjustedNnly-nnly,adjustedNnlx,nnlz);
	nodeVolume(:,:,end+1:adjustedNnlz) = zeros(adjustedNnly,adjustedNnlx,adjustedNnlz-nnlz);
	newLoadedNodes = find(nodeVolume); tmpBC(:,1) = newLoadedNodes;
	for col = 2:4
		nodeVolume = nullNodeVolume;
		nodeVolume(srcBC(:,1)) = srcBC(:,col);
		nodeVolume = reshape(nodeVolume, nely_+1, nelx_+1, nelz_+1);
		nodeVolume(:,end+1:adjustedNnlx,:) = zeros(nnly,adjustedNnlx-nnlx,nnlz);
		nodeVolume(end+1:adjustedNnly,:,:) = zeros(adjustedNnly-nnly,adjustedNnlx,nnlz);
		nodeVolume(:,:,end+1:adjustedNnlz) = zeros(adjustedNnly,adjustedNnlx,adjustedNnlz-nnlz);
		nodeVolume = nodeVolume(:);
		tmpBC(:,col) = nodeVolume(newLoadedNodes);
	end
	tarBC = tmpBC;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function adjustedVoxelIndices = AdaptPassiveElementsExternalMdl(srcElesMapback, adjustedRes)
	global nelx_ nely_ nelz_;
	nullVoxelVolume = zeros(nelx_*nely_*nelz_,1);
	adjustedNelx = adjustedRes(1); adjustedNely = adjustedRes(2); adjustedNelz = adjustedRes(3);
	[~, ia] = sort(srcElesMapback(:,1)); srcElesMapback = srcElesMapback(ia,:);
	voxelVolume = nullVoxelVolume;
	voxelVolume(srcElesMapback(:,1)) = 1;
	voxelVolume = reshape(voxelVolume, nely_, nelx_, nelz_);
	voxelVolume(:,end+1:adjustedNelx,:) = zeros(nely_,adjustedNelx-nelx_,nelz_);
	voxelVolume(end+1:adjustedNely,:,:) = zeros(adjustedNely-nely_,adjustedNelx,nelz_);
	voxelVolume(:,:,end+1:adjustedNelz) = zeros(adjustedNely,adjustedNelx,adjustedNelz-nelz_);
	adjustedVoxelIndices = find(voxelVolume);
end
