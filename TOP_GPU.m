function TOP_GPU(inputModel, varargin)
% TOP_GPU  GPU-accelerated topology optimization (batch-friendly).
%
% Usage (rectangular domain):
%   TOP_GPU(true(nely,nelx,nelz), 'optCase',1, 'V0',0.12, ...)
%
% Usage (file-based model):
%   TOP_GPU('./data/femur.TopVoxel', 'V0',0.12, ...)
%
% Name-value optional parameters (with defaults):
%   'optCase'          1        Load/BC case (rectangular domain only)
%   'V0'              0.12     Target volume fraction
%   'rMin'            sqrt(3)  Filter radius (cells)
%   'nLoop'           50       Max optimization iterations
%   'ft'              1        Filter type: 1=sensitivity, 2=PDE density
%   'mixed_Precision' 0        0=double, 1=mixed precision
%   'super_element'   0        0=standard voxel, 1=super-element mode

p = inputParser;
addRequired(p, 'inputModel');
addParameter(p, 'optCase',          1);
addParameter(p, 'V0',               0.12);
addParameter(p, 'rMin',             sqrt(3));
addParameter(p, 'nLoop',            50);
addParameter(p, 'ft',               1);
addParameter(p, 'mixed_Precision',  0);
addParameter(p, 'super_element',    0);
parse(p, inputModel, varargin{:});

V0              = p.Results.V0;
rMin            = p.Results.rMin;
nLoop           = p.Results.nLoop;
ft              = p.Results.ft;
mixed_Precision = p.Results.mixed_Precision;
super_element   = p.Results.super_element;
optCase_in      = p.Results.optCase;

reset(gpuDevice);
global mixed_Precision_; mixed_Precision_ = mixed_Precision;
global Super_element_;   Super_element_   = super_element;
global optCase;          optCase          = optCase_in;
global coarsestResolutionControl_; coarsestResolutionControl_ = 20000; 
	global meshHierarchy_;
	global specifyPassiveRegions_;
	global passiveElements_;
	global modulus_;
	global modulusMin_; 
	global SIMPpenalty_;	
	global tol_;
	global maxIT_;	
	global densityLayout_; %%Result
	global F_;
	global d_F_;
	global cellSize_ maxIT_ typeVcycle_ nonDyadic_ poissonRatio_;
    global Num_;
	%% Build output folder name from parameters
	if islogical(inputModel)
		[nely_sz, nelx_sz, nelz_sz] = size(inputModel);
		folderName = sprintf('case%d_%dx%dx%d_V%.4g_r%.4g_ft%d_n%d_mp%d_se%d', ...
			optCase_in, nelx_sz, nely_sz, nelz_sz, V0, rMin, ft, nLoop, mixed_Precision, super_element);
		InitialSettings(nelx_sz);
        %InitialSettings(1);
	else
		[~, modelName, ~] = fileparts(inputModel);
		folderName = sprintf('%s_V%.4g_r%.4g_ft%d_n%d_mp%d_se%d', ...
			modelName, V0, rMin, ft, nLoop, mixed_Precision, super_element);
		InitialSettings(1);
	end
	outPath = ['./out/' folderName '/'];
	if exist(outPath, 'dir')
		tIdx = 1;
		while exist(['./out/' folderName '_t' num2str(tIdx) '/'], 'dir')
			tIdx = tIdx + 1;
		end
		folderName = [folderName '_t' num2str(tIdx)];
		outPath = ['./out/' folderName '/'];
	end
	mkdir(outPath);
	fid = fopen(strcat(outPath, 'RunLog.log'), 'w'); fclose(fid); diary(strcat(outPath, 'RunLog.log'));
	
	%%Displaying Inputs
	disp('==========================Displaying Inputs==========================');
	disp(['..............................................Volume Fraction: ', sprintf('%6.4f', V0)]);
	disp(['..........................................Filter Radius: ', sprintf('%6.4f', rMin), ' Cells']);
	disp(['................................................Cell Size: ', sprintf('%6.4e', cellSize_)]);
	disp(['...............................................#MGCG Iterations: ', sprintf('%4i', maxIT_)]);
	disp(strcat('.....................................................V-cycle: ', " ", typeVcycle_));
	disp(['...............................................Non-dyadic Strategy: ', sprintf('%1i', nonDyadic_)]);
	disp(['...........................................Youngs Modulus: ', sprintf('%6.4e', modulus_)]);
	disp(['....................................Youngs Modulus (Min.): ', sprintf('%6.4e', modulusMin_)]);
	disp(['...........................................Poissons Ratio: ', sprintf('%6.4e', poissonRatio_)]);	
	
	%%0. Modeling.
	tStart = tic;
	CreateVoxelFEAmodel(inputModel);
	disp(['Preparing Voxel-based FEA Model Costs ', sprintf('%10.1f',toc(tStart)), 's']);
	%% Get mesh dimensions (set by CreateVoxelFEAmodel for both input types)
	global nelx_ nely_ nelz_;
	nelx = nelx_; nely = nely_; nelz = nelz_;
	
	%%1. Pre. FEA
	FEA_ApplyBoundaryCondition();
	FEA_SetupVoxelBased();
	
	%%2. Setup PDE filter or Standard filter
	if Super_element_==0
	[PDEkernal4Filtering, diagPrecond4Filtering] = TopOpti_SetupPDEfilter_matrixFree(rMin,inputModel);
	d_PDEkernal4Filtering=gpuArray(PDEkernal4Filtering);
	d_diagPrecond4Filtering=gpuArray(diagPrecond4Filtering);
	clear PDEkernal4Filtering;
	clear diagPrecond4Filtering;
    else
		rMin=Num_*rMin;
		[map_old2new, map_new2old] = gen_index_mapping(nelx, nely, nelz, Num_);
        bcF=0;
        [dy,dx,dz]=meshgrid(-ceil(rMin)+1:ceil(rMin)-1,...
        -ceil(rMin)+1:ceil(rMin)-1,-ceil(rMin)+1:ceil(rMin)-1 );
        h = max( 0, rMin - sqrt( dx.^2 + dy.^2 + dz.^2 ) );
        Hs = gpuArray(imfilter( ones( Num_*nely, Num_*nelx, Num_*nelz ), h, bcF ));
        h = gpuArray(h);   % ensure imfilter executes on the GPU (prevents implicit gather to CPU)
	end
	%%3. prepare optimizer
	TopOpti_SetPassiveElements(specifyPassiveRegions_(1), specifyPassiveRegions_(2), specifyPassiveRegions_(3));
	numElements = meshHierarchy_(1).numElements;	
	passiveElements = passiveElements_;
	if Super_element_==0
	x = gpuArray(repmat(V0, numElements, 1));
	else
	x = gpuArray(repmat(V0, 4^3, numElements));
	end
    xPhys = x;
	loop = 0;
	change = 1.0;
	sharpness = 1.0;
	lssIts = [];
	cHist = [];
	volHist = [];
	sharpHist = [];
	consHist = [];
	tHist = [];
	%%4. Evaluate Compliance of Fully Solid Domain
	U = zeros(size(F_),'gpuArray');
    if Super_element_==0
	meshHierarchy_(1).eleModulus = repmat(modulus_, 1, numElements);
	meshHierarchy_(1).d_eleModulus = gpuArray(meshHierarchy_(1).eleModulus);
    else
    meshHierarchy_(1).eleModulus = repmat(modulus_, Num_^3, numElements);
	meshHierarchy_(1).d_eleModulus = gpuArray(meshHierarchy_(1).eleModulus);
    end
	tSolvingFEAssemblingClock = tic;
	Solving_AssembleFEAstencil();
	itSolvingFEAssembling = toc(tSolvingFEAssemblingClock);
	d_F_=gpuArray(F_);
    tSolvingFEAiterationClock = tic;
	for ii=1:size(F_,2), [U(:,ii), ~] = Solving_PCG(@Solving_KbyU_MatrixFree, @Solving_Vcycle, d_F_(:,ii), tol_, maxIT_, [0 1]);end	
	itSolvingFEAiteration = toc(tSolvingFEAiterationClock);
    %fprintf('%4i',itSolvingFEAiteration);
	if Super_element_==0
	ceList = TopOpti_ComputeUnitCompliance(U);
	cSolid = meshHierarchy_(1).eleModulus*ceList;
	else
    ceList = TopOpti_ComputeUnitCompliance_Super(U);
	cSolid=sum(ceList,1);
	end
	disp(['Compliance of Fully Solid Domain: ' sprintf('%16.6e',cSolid)]);
	disp([' It.: ' sprintf('%4i',0) ' Assembling Time: ', sprintf('%4i',itSolvingFEAssembling) 's;', ' Solver Time: ', sprintf('%4i',itSolvingFEAiteration) 's.']);

	if Super_element_==0
	SIMP = @(xPhys) modulusMin_+xPhys(:)'.^SIMPpenalty_ .* (modulus_-modulusMin_);
	DeSIMP = @(xPhys) SIMPpenalty_*(modulus_-modulusMin_)' .* xPhys.^(SIMPpenalty_-1);
	else
	SIMP = @(xPhys) modulusMin_+xPhys.^SIMPpenalty_ .* (modulus_-modulusMin_);
	DeSIMP = @(xPhys) SIMPpenalty_*(modulus_-modulusMin_) .* xPhys.^(SIMPpenalty_-1);
	end
	tStartTotal = tic;
	while loop < nLoop && change > 0.0001 && sharpness>0.01
		perIteCost = tic;
		loop = loop+1;
		%%5.1 & 5.2 FEA, objective and sensitivity analysis
		meshHierarchy_(1).d_eleModulus = SIMP(xPhys);
		meshHierarchy_(1).eleModulus = gather(meshHierarchy_(1).d_eleModulus);
		tSolvingFEAssemblingClock = tic;
	    Solving_AssembleFEAstencil();
		itSolvingFEAssembling = toc(tSolvingFEAssemblingClock);
		tSolvingFEAiterationClock = tic;
		for ii=1:size(F_,2), [U(:,ii), lssIts(end+1,1)] = Solving_PCG(@Solving_KbyU_MatrixFree, @Solving_Vcycle, d_F_(:,ii), tol_, maxIT_, [0 1], U(:,ii)); end	
		itSolvingFEAiteration = toc(tSolvingFEAiterationClock);
		tOptimizationClock = tic;
		if Super_element_==0
            %e1=tic;
		ceList = TopOpti_ComputeUnitCompliance(U);
        %e1=toc(e1);
        %fprintf("sensitivity computation time: %.3f\n",e1);
		ceNorm = ceList / cSolid;
		cObj = meshHierarchy_(1).eleModulus * ceNorm;
		cDesign = cObj*cSolid;
		V = double(sum(xPhys(:)) / numElements);
		dc = -DeSIMP(xPhys).*ceNorm;
		dv = gpuArray.ones(numElements,1);
		else
		ceList = TopOpti_ComputeUnitCompliance_Super(U);
		ceNorm = ceList / cSolid;
        cObj=sum(ceNorm,1);
		cDesign = cObj*cSolid;
		V = double(sum(xPhys(:)) / (numElements*Num_^3));
		dc=pre_compute_der(U)/cSolid;
		dc = -DeSIMP(xPhys).*dc;
		dv = gpuArray.ones(Num_^3,numElements);
		end
		itimeOptimization = toc(tOptimizationClock);
		%%5.3 filtering/modification of sensitivity
		tPDEfilteringClock = tic;
        if 1==ft
			dc(:) = TopOpti_ConductPDEFiltering_matrixFree(x(:).*dc(:), d_PDEkernal4Filtering, d_diagPrecond4Filtering)./max(1e-3,x(:));     
        elseif ft == 2
			if Super_element_==0
			dc(:) = TopOpti_ConductPDEFiltering_matrixFree(dc(:), d_PDEkernal4Filtering, d_diagPrecond4Filtering);
			dv(:) = TopOpti_ConductPDEFiltering_matrixFree(dv(:), d_PDEkernal4Filtering, d_diagPrecond4Filtering);
			else
			dc=dc(:);
			 dc = imfilter(reshape(dc(map_old2new), 4*nely, 4*nelx, 4*nelz ) ./ Hs, h, bcF );
			 dc=dc(:);
			 dc=reshape(dc(map_new2old),64,[]);
			dv=dv(:);
			 dv = imfilter( reshape(dv(map_old2new), 4*nely, 4*nelx, 4*nelz ) ./ Hs, h, bcF );
			 dv=dv(:);
			 dv=reshape(dv(map_new2old),64,[]);
			end
        end
		itimeDensityFiltering = toc(tPDEfilteringClock);
        tOptimizationClock = tic;
		%%5.4 solve the optimization probelm
		fval = mean(xPhys(:))-V0;
		l1 = 0; l2 = 1e9; move = 0.2;
		if Super_element_==0
		while (l2-l1)/(l1+l2) > 1e-6
			lmid = 0.5*(l2+l1);
			xnew = max(0,max(x-move,min(1,min(x+move,x.*sqrt(-dc./dv/lmid)))));
			xnew(passiveElements,1) = 1.0;
			gt=fval+mean(((xnew(:)-x(:))));
			if gt>0, l1 = lmid; else l2 = lmid; end				
		end
	else
			while (l2-l1)/(l1+l2) > 1e-6
			lmid = 0.5*(l2+l1);
			xnew = max(0,max(x-move,min(1,min(x+move,x.*sqrt(-dc./dv/lmid)))));
			xnew(Num_,passiveElements) = 1.0;
			gt=fval+mean(((xnew(:)-x(:))));
			if gt>0, l1 = lmid; else l2 = lmid; end				
    		end	
     end
		change = max(abs(xnew(:)-x(:)));
		x = xnew;	
		itimeOptimization = itimeOptimization + toc(tOptimizationClock);
        tPDEfilteringClock = tic;
		if ft == 1
			xPhys = xnew;
		elseif ft == 2
			if Super_element_==0
			xPhys(:) = TopOpti_ConductPDEFiltering_matrixFree(xnew(:), d_PDEkernal4Filtering, d_diagPrecond4Filtering);
			else
			xnew=xnew(:);
			xPhys=imfilter(reshape(xnew(map_old2new), 4*nely, 4*nelx, 4*nelz ), h, bcF ) ./ Hs;
			xPhys=xPhys(:);
			xPhys=reshape(xPhys(map_new2old),64,[]);
    	    end
        end
		itimeDensityFiltering = itimeDensityFiltering + toc(tPDEfilteringClock);
		if Super_element_==0		
		xPhys(passiveElements,1) = 1;
		sharpness = 4*sum(sum(xPhys.*(ones(numElements,1)-xPhys)))/numElements;
		else
		xPhys(Num_,passiveElements) = 1;
		sharpness = 4*sum(sum(xPhys(:).*(gpuArray.ones(numElements*Num_^3,1)-xPhys(:))))/(numElements*Num_^3);
        end
		itimeTotal = toc(perIteCost);

		%%5.5 write opti. history
		cHist(loop,1) = cDesign;
		volHist(loop,1) = V;
		consHist(loop,:) = fval;
		sharpHist(loop,1) = sharpness;
		iTimeStatistics = [itSolvingFEAssembling itSolvingFEAiteration itimeOptimization itimeDensityFiltering itimeTotal];
		tHist(loop,:) = iTimeStatistics;

		%%5.6 print results
		fprintf(' It.:%4i Obj.:%16.8e Vol.:%6.4e Sharp.:%6.4e Cons.:%4.2e Ch.:%4.2e\n',...
			loop,cDesign,V, sharpness, fval, change);			 
		disp([' It.: ' sprintf('%i',loop) ' (Time)... Total per-It.: ' sprintf('%8.2e',itimeTotal) 's;', ' Assemb.: ', ...
			sprintf('%8.2e',itSolvingFEAssembling), 's; CG: ', sprintf('%8.2e',itSolvingFEAiteration), ...
			's; Opti.: ', sprintf('%8.2e',itimeOptimization), 's; Filtering: ', sprintf('%8.2e',itimeDensityFiltering) 's.']);			
    end
	%%Output
	if Super_element_==0
	densityLayout_ = gather(xPhys(:));
	else
	densityLayout_ = gather(xPhys);
	% Export xPhys in small-element (spatial grid) order as N×1
	map_old2new_cpu = gather(map_old2new);
	xPhys_sorted = gather(xPhys(:));
	xPhys_sorted = xPhys_sorted(map_old2new_cpu);   % reorder to spatial grid order, N×1
	fid = fopen(strcat(outPath, 'xPhys_sorted.dat'), 'w');
	fprintf(fid, '%30.16e\n', xPhys_sorted);
	fclose(fid);
	end
	fileName = strcat(outPath, 'DesignVolume.nii');
	IO_ExportDesignInVolume_nii(fileName);
	disp(['..........Solving FEA Costs: ', sprintf('%10.4e', sum(sum(tHist(:,1:2)))), 's.']);
	disp(['..........Optimization (inc. sentivity analysis, update) Costs: ', sprintf('%10.4e', sum(tHist(:,3))), 's.']);
	disp(['..........Performing PDE Filtering Costs: ', sprintf('%10.4e', sum(tHist(:,4))), 's.']);
	disp(['..........Performing Topology Optimization Costs (in total): ', sprintf('%10.4e', toc(tStartTotal)), 's.']);
	fid = fopen(strcat(outPath, 'iters_Target.dat'), 'w');
	fprintf(fid, '%d\n', lssIts);
	fclose(fid);
	fid = fopen(strcat(outPath, 'c_Target.dat'), 'w');
	fprintf(fid, '%30.16e\n', cHist);
	fclose(fid);
	fid = fopen(strcat(outPath, 'sharp_Target.dat'), 'w');
	fprintf(fid, '%30.16e\n', sharpHist);
	fclose(fid);
	fid = fopen(strcat(outPath, 'timing_Target.dat'), 'w');
	fprintf(fid, '%16.6e\n', tHist(:,end));
	fclose(fid);
	fid=fopen(strcat(outPath, 'solvetiming_Target.dat'), 'w');
	fprintf(fid, '%16.6e\n', tHist(:,2));
	fclose(fid);

	%%Vis.
	if Super_element_==0
    allVoxels = zeros(size(meshHierarchy_(1).eleMapForward));
	allVoxels(meshHierarchy_(1).eleMapBack,1) = densityLayout_;
	isovals = reshape(allVoxels,meshHierarchy_(1).resY,meshHierarchy_(1).resX,meshHierarchy_(1).resZ);
    isovals = flip(isovals,1);
    isovals = smooth3(isovals,'box',1);    
	% figure;
	facesIsosurface = isosurface(isovals,0.5);
	facesIsocap = isocaps(isovals,0.5);
    patch(facesIsosurface,'FaceColor',[0 127 0]/255,'EdgeColor','none');
    patch(facesIsocap,'FaceColor',[0 127 0]/255,'EdgeColor','none');
    view(55,25); axis equal tight on; axis off; xlabel('X'); ylabel('Y'); zlabel('Z');
    lighting('gouraud');
    material('dull');
    camlight('headlight','infinite');
	fileName = strcat(outPath, 'DesignVolume.stl');
	IO_ExportDesignInTriSurface_stl(fileName, facesIsosurface, facesIsocap);
	else
	rx = meshHierarchy_(1).resX; 
    ry = meshHierarchy_(1).resY;   
    rz = meshHierarchy_(1).resZ;   
    up = Num_;                        

    allVoxelsHi = zeros(ry*up, rx*up, rz*up, 'single'); 

   definedBigLin = meshHierarchy_(1).eleMapBack(:);
N = numel(definedBigLin);

for t = 1:N
    [iy, ix, iz] = ind2sub([ry, rx, rz], definedBigLin(t));
    yRange = (iy-1)*up + (1:up);
    xRange = (ix-1)*up + (1:up);
    zRange = (iz-1)*up + (1:up);

    blk = reshape(densityLayout_(:, t), [up, up, up]);
    allVoxelsHi(yRange, xRange, zRange) = blk;
end
isovals = allVoxelsHi;
isovals = flip(isovals, 1);                 
isovals = smooth3(isovals, 'box', 3);   
beta=8;eta=0.5;
den = tanh(beta*eta) + tanh(beta*(1-eta));
isovals =(tanh(beta*eta) + tanh(beta*(isovals-eta))) / den;
% figure;
facesIsosurface = isosurface(isovals, 0.5);
facesIsocap     = isocaps(isovals, 0.5);
patch(facesIsosurface, 'FaceColor', [0 127 0]/255, 'EdgeColor', 'none');
patch(facesIsocap,     'FaceColor', [0 127 0]/255, 'EdgeColor', 'none');
view(55,25); axis equal tight on; axis off;
xlabel('X'); ylabel('Y'); zlabel('Z');
lighting gouraud; material dull; camlight('headlight','infinite');
fileName = strcat(outPath, 'DesignVolume.stl');
IO_ExportDesignInTriSurface_stl(fileName, facesIsosurface, facesIsocap);
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
function InitialSettings(nelx)
	%% Physical Property
	global modulus_; modulus_ = 1.0; %% Young's modulus	
	global poissonRatio_; poissonRatio_ = 0.3;	%% Poisson's ratio
	global modulusMin_; modulusMin_ = 1.0e-6 * modulus_;	
	global SIMPpenalty_; SIMPpenalty_ = 3;
	global cellSize_; cellSize_ = 1/nelx;
	global Num_;Num_=4;%size of super element
	%% Linear System Solver	
	global tol_; tol_ = 1.0e-3; %% convergence tolerance of iterative linear system solver
	global maxIT_; maxIT_ = 800; %% permitted maximum number of Conjugate Gradient iteartion
	global weightFactorJacobi_; weightFactorJacobi_ = 0.6; 
	global typeVcycle_; typeVcycle_ = 'Standard'; %% 'Standard', 'Adapted'	
	global nonDyadic_; nonDyadic_ = 1; %%True or False
	%% Optimization 
	global specifyPassiveRegions_; specifyPassiveRegions_ = [0 0 0]; %% [#LayersBoundary, #LayersLoads, #LayersFixations];
	end
function CreateVoxelFEAmodel(inputModel)
	global voxelizedVolume_;
	global nelx_; 
	global nely_; 
	global nelz_; 
	global fixingCond_; 
	global loadingCond_;
	global objWeightingList_;
	global meshHierarchy_;
	global passiveElements_;	
	global densityLayout_;
	global optCase;
	loadingCond_ = cell(1,1);
	if ischar(inputModel) || isstring(inputModel)
		fid = fopen(inputModel, 'r');
		fileHead = fscanf(fid, '%s %s %s %s', 4);
		versionHead = fscanf(fid, '%s', 1);
		versionID = fscanf(fid, '%f', 1);
		switch versionID
			case 1.0
				if ~strcmp(fscanf(fid, '%s', 1), 'Resolution:'), error('Incompatible Mesh Data Format!'); end
				resolutions = fscanf(fid, '%d %d %d', 3);
				nelx_ = resolutions(1); nely_ = resolutions(2); nelz_ = resolutions(3);
				densityValuesCheck = fscanf(fid, '%s %s', 2);
				checkDensityValuesIncluded = fscanf(fid, '%d', 1);
				startReadSolidVoxels = fscanf(fid, '%s %s', 2);
				numSolidVoxels = fscanf(fid, '%d', 1);
				if checkDensityValuesIncluded
					voxelsState = fscanf(fid, '%d %e', [2 numSolidVoxels])';
					solidVoxels = voxelsState(:,1);
					densityLayout_ = voxelsState(:,2);
				else
					solidVoxels = fscanf(fid, '%d', [1 numSolidVoxels])';
				end
				if ~strcmp(fscanf(fid, '%s', 1), 'Passive'), error('Incompatible Mesh Data Format!'); end
				if ~strcmp(fscanf(fid, '%s', 1), 'elements:'), error('Incompatible Mesh Data Format!'); end
				numPassiveElements = fscanf(fid, '%d', 1);
				passiveElements_ = fscanf(fid, '%d', [1 numPassiveElements])';
				if ~strcmp(fscanf(fid, '%s', 1), 'Fixations:'), error('Incompatible Mesh Data Format!'); end
				numFixedNodes = fscanf(fid, '%d', 1);
				if numFixedNodes>0		
					fixingCond_ = fscanf(fid, '%d %d %d %d', [4 numFixedNodes])';		
				end
				if ~strcmp(fscanf(fid, '%s', 1), 'Loads:'), error('Incompatible Mesh Data Format!'); end
				numLoadedNodes = fscanf(fid, '%d', 1);	
				if numLoadedNodes>0			
					loadingCond_{1} = fscanf(fid, '%d %e %e %e', [4 numLoadedNodes])';								
				end
				AdditionalLoads = fscanf(fid, '%s %s', 2);
				numAdditionalLoads = fscanf(fid, '%d', 1);
				if numAdditionalLoads>0
					for ii=1:numAdditionalLoads
						if ~strcmp(fscanf(fid, '%s', 1), 'Loads:'), error('Incompatible Mesh Data Format!'); end
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
		
		%%In case the resolution is slightly inconsistent	
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
	elseif islogical(inputModel) %%Built-in Cuboid Design Domain for Testing
		voxelizedVolume_ = inputModel;
		[nely_, nelx_, nelz_] = size(inputModel);
		if nelx_<3 || nely_<3 || nelz_<3 %% At least 3 layers solid elements are needed
			error('Inappropriate Input Model!');
		end
		FEA_VoxelBasedDiscretization();
		%%Apply Boundary Conditions
		nodeVolume4ApplyingBC = zeros(meshHierarchy_(1).resY+1, meshHierarchy_(1).resX+1, meshHierarchy_(1).resZ+1);
		switch optCase
			case 1 % cantilever beam
				nodeVolume4ApplyingBC(1:nely_+1,1,1:nelz_+1) = 1;
				fixingCond_ = find(1==nodeVolume4ApplyingBC);
		        fixingCond_ = double(meshHierarchy_(1).nodMapForward(fixingCond_));
		        fixingCond_ = [fixingCond_ ones(numel(fixingCond_),3)];
				nodeVolume4ApplyingBC = zeros(meshHierarchy_(1).resY+1, meshHierarchy_(1).resX+1, meshHierarchy_(1).resZ+1);
				nodeVolume4ApplyingBC(1:nely_+1,nelx_+1,1:round(nelz_/6+1)) = 1;
				iLoad = find(1==nodeVolume4ApplyingBC);
		        iLoad = double(meshHierarchy_(1).nodMapForward(iLoad));
		        iLoad = [iLoad repmat([0 0 -1]/numel(iLoad), numel(iLoad), 1)];
			case 2%MBB
		        nodeVolume4ApplyingBC(1,1,1) = 1;%MBB
                nodeVolume4ApplyingBC(1,nelx_+1,1) = 1;%MBB
                nodeVolume4ApplyingBC(nely_+1,1,1) = 1;%MBB
                nodeVolume4ApplyingBC(nely_+1,nelx_+1,1) = 1;%MBB
				fixingCond_ = find(1==nodeVolume4ApplyingBC);
		        fixingCond_ = double(meshHierarchy_(1).nodMapForward(fixingCond_));
		        fixingCond_ = [fixingCond_ ones(numel(fixingCond_),3)];
				nodeVolume4ApplyingBC = zeros(meshHierarchy_(1).resY+1, meshHierarchy_(1).resX+1, meshHierarchy_(1).resZ+1);
				nodeVolume4ApplyingBC(nely_/2+1,nelx_/2+1,nelz_+1) = 1;%MBB
				iLoad = find(1==nodeVolume4ApplyingBC);
		        iLoad = double(meshHierarchy_(1).nodMapForward(iLoad));
                iLoad = [iLoad [0,0,-1]];

		end
		loadingCond_{1} = iLoad;
		objWeightingList_ = 1;
		passiveElements_ = [];
	end
end

%%Key Features
function IO_ExportDesignInVolume_nii(fileName)
	global meshHierarchy_;
	global densityLayout_;
	global Super_element_;
	global Num_;
	if Super_element_==0
	V = zeros(numel(meshHierarchy_(1).eleMapForward),1);
	V(meshHierarchy_(1).eleMapBack,1) = densityLayout_;
	V = reshape(V, meshHierarchy_(1).resY, meshHierarchy_(1).resX, meshHierarchy_(1).resZ);
	else
	rx = meshHierarchy_(1).resX; 
    ry = meshHierarchy_(1).resY;   
    rz = meshHierarchy_(1).resZ;   
    up = Num_;                        

    allVoxelsHi = zeros(ry*up, rx*up, rz*up, 'single'); 

   definedBigLin = meshHierarchy_(1).eleMapBack(:);
N = numel(definedBigLin);

for t = 1:N
    [iy, ix, iz] = ind2sub([ry, rx, rz], definedBigLin(t));
    yRange = (iy-1)*up + (1:up);
    xRange = (ix-1)*up + (1:up);
    zRange = (iz-1)*up + (1:up);

    blk = reshape(densityLayout_(:, t), [up, up, up]);
    allVoxelsHi(yRange, xRange, zRange) = blk;
end
V=allVoxelsHi;
end
	niftiwrite(V,fileName);
end

function IO_ExportDesignInTriSurface_stl(fileName, facesIsosurface, facesIsocap)
	allFaces.vertices = [facesIsosurface.vertices; facesIsocap.vertices];
	allFaces.faces = [facesIsosurface.faces; size(facesIsosurface.vertices,1)+facesIsocap.faces];
    TR = triangulation(allFaces.faces,allFaces.vertices);
    stlwrite(TR, fileName);	  	
end

function ceList = TopOpti_ComputeUnitCompliance(U)
    global meshHierarchy_;
    global objWeightingList_;

    Ne        = meshHierarchy_(1).numElements;
    numNodes  = meshHierarchy_(1).numNodes;
    d_eNodMat = meshHierarchy_(1).d_eNodMat;       % [Ne×8], int32, GPU
    d_Ke      = gpuArray(meshHierarchy_(1).Ke);     % [24x24], GPU (576 doubles; transfer overhead negligible)
    nRHS      = size(U, 2);

    ceList_gpu = gpuArray.zeros(Ne, nRHS);
    for jj = 1:nRHS
        ithU = U(:, jj);
        if ~isa(ithU, 'gpuArray'), ithU = gpuArray(ithU); end
        uVec = reshape(ithU, 3, numNodes)';         % [numNodes x 3], GPU, zero-copy reshape
        iReshapedU = zeros(Ne, 24, 'gpuArray');     % [Ne×24], GPU
        Gathering_inplace(iReshapedU, uVec, d_eNodMat);
        ceList_gpu(:, jj) = sum((iReshapedU * d_Ke) .* iReshapedU, 2);
    end
    ceList = ceList_gpu* objWeightingList_(:);
end

function [y, varargout] = Solving_PCG(AtX, PtV, b, tol, maxIT, printP, varargin)
global mixed_Precision_;
	%%0. arguments introduction
	%%AtX --- function handle for the product of system matrix and vector
	%%b --- right hand section
	%%tol --- stopping condition: resnrm < discrepancy
	%%maxIT --- mAtXximum number of iterations
	normB = gather(norm(b));
    b=full(b);
	if 7==nargin, y = varargin{1}; else, y = zeros(size(b),'gpuArray'); end
	rVec1 =b - AtX(y);
	zVec = PtV(rVec1);
	pVec = zVec;
	x1Val = dot(zVec , rVec1);
	 enableLogResnorm = false;
     checkEvery = 5;   % check the true residual every 5 steps in mixed-precision mode
    if enableLogResnorm
        ndof = numel(b);
        resnormHist = zeros(maxIT, 1);
        logFolder = 'pcg_logs';   
        if ~exist(logFolder, 'dir')
            mkdir(logFolder);     
        end
        logFileName = sprintf('PCG_resnorm_ndof_%d.mat', ndof);
    end
	for its=1:maxIT
	
		vVec = AtX(pVec);
		lambda = x1Val / dot(pVec, vVec);
		y = y + double(lambda * pVec);
		rVec1 = rVec1 - lambda*vVec;
        %         % mixed precision: recompute the true residual every 5 steps
        % if mixed_Precision_ == 1 && mod(its, checkEvery) == 0
        %  rVec1 = b - AtX(y);
        % end
		resnorm = gather(norm(rVec1))/normB;
		if printP(1)
			disp([' It.: ' sprintf('%4i',its) ' Res.: ' sprintf('%16.6e',resnorm)]);
		end
		if enableLogResnorm
            resnormHist(its) = resnorm;
        end		
		if resnorm<tol
			if printP(2)
				disp(['CG solver converged at iteration' sprintf('%5i', its) ' to a solution with relative residual' sprintf('%16.6e',resnorm)]);
			end
			break;
		end
		zVec = PtV(rVec1);
		x2Val = dot(zVec, rVec1);
		pVec = zVec + x2Val / x1Val * pVec;
		x1Val = x2Val;
	end

	if its == maxIT
		warning('Exceed the maximum iterate numbers');
		disp(['The iterative process stops at residual = ' sprintf('%10.4f',resnorm)]);		
	end
	if 2==nargout, varargout{1} = its; end
	if enableLogResnorm
        
        resnormHist = resnormHist(1:its);
      
        save(fullfile(logFolder, logFileName), 'resnormHist', 'its', 'ndof');
  end
end

function Y = Solving_KbyU_MatrixFree(uVec, varargin)
	global meshHierarchy_;
	global uniqueKesFixed_;
	global d_uniqueKesFixed_;
	global uniqueKesFree_;	
	global d_uniqueKesFree_;	
	global mapUniqueKes_;
	global d_mapUniqueKes_;
	global d_eleWithFixedDOFs_;
	global mixed_Precision_;
	global Super_element_;
	global d_K_mat;
	if 1==nargin, iLevel = 1; end
	if 2==nargin, iLevel = varargin{1}; end
	if mixed_Precision_==1
	uVec =single(reshape(uVec,3,meshHierarchy_(iLevel).numNodes)');
	else
	uVec =reshape(uVec,3,meshHierarchy_(iLevel).numNodes)';
    end
	if 1==iLevel
	K_0 = meshHierarchy_(iLevel).d_Ks;
	numNodes = meshHierarchy_(1).numNodes;
	eNodMat = meshHierarchy_(1).d_eNodMat;
	E_e = meshHierarchy_(1).d_eleModulus;
	%%Step 1
	if mixed_Precision_==1
	Y=zeros(numNodes,3,'single','gpuArray');
	uMat = zeros(size(eNodMat,1),24,'single','gpuArray');
	else
	Y=zeros(numNodes,3,'gpuArray');
	uMat = zeros(size(eNodMat,1),24,'gpuArray');
	end
	Gathering_inplace(uMat, uVec, eNodMat);
N_b = numel(d_eleWithFixedDOFs_);%
if Super_element_==0
K_free  = reshape(d_uniqueKesFree_,  24, 24, N_b);
K_fixed = reshape(d_uniqueKesFixed_, 24, 24, N_b);
Ee_fixed   = reshape(E_e(d_eleWithFixedDOFs_), 1, 1, N_b);  % [1,1,N]
if mixed_Precision_==1
K_combined = single(K_free .* Ee_fixed + K_fixed);          % [24,24,N]
else
K_combined = K_free .* Ee_fixed + K_fixed;
end
U_fixed = uMat(d_eleWithFixedDOFs_, :);% [N, 24]
    % single RHS: reshape [N,24] -> [24,1,N]
U_fixed=permute(U_fixed, [2 3 1]);
Y_fixed = pagemtimes(K_combined, U_fixed);% [24,1,N]
Y_fixed = reshape(permute(Y_fixed, [3 1 2]), N_b, 24);
	%%Step 2 & 3
uMat=uMat*K_0.*E_e(:);
	%%Apply for Boundary Conditions
uMat(d_eleWithFixedDOFs_,:)=Y_fixed;
else
	% 1) [24,24,N]
K_free  = reshape(d_uniqueKesFree_,  24*24,64, N_b);%[24*24,64,N]
Ee_fixed   = reshape(E_e(:,d_eleWithFixedDOFs_), 1, 64, N_b); % [1,64,N]
K_combined = sum(K_free .* Ee_fixed, 2);%[24*24,1,N]
K_combined=squeeze(K_combined);
K_combined=reshape(K_combined,24,24,N_b);
K_fixed = reshape(d_uniqueKesFixed_, 24, 24, N_b);
% 2) K = Ee.*K_free + K_fixed 
if mixed_Precision_==1
K_combined = single(K_combined+ K_fixed);     % [24,24,N]
else
K_combined = K_combined+ K_fixed; 
end
%%Step 2 & 3
Ke3D = reshape(d_K_mat, 24, 24, []);     % [24×24×N]
Ke3D(:,:,d_eleWithFixedDOFs_)=K_combined;
U3D  = permute(uMat, [2 3 1]);     % [24×1×N]
res3D = pagemtimes(Ke3D, U3D);     % [24×1×N]
uMat = permute(res3D, [3 1 2]);     % [N×24]
end
%%Step 4
Scattering_inplace(Y, eNodMat, uMat);
Y=Y'; Y=double(Y(:));
wait(gpuDevice);
else
numNodes = meshHierarchy_(iLevel).numNodes;
eNodMat = meshHierarchy_(iLevel).d_eNodMat;
if mixed_Precision_==1
Y=zeros(numNodes,3,'single','gpuArray');
uMat = zeros(size(eNodMat,1),24,'single','gpuArray');
else
Y=zeros(numNodes,3,'gpuArray');
uMat = zeros(size(eNodMat,1),24,'gpuArray');
end
Gathering_inplace(uMat, uVec, eNodMat);
U3D = permute(uMat, [3 2 1]);        % 1 × 24 × numEle
% Ks: 24 × 24 × numEle
Uout = pagemtimes(U3D, meshHierarchy_(iLevel).d_Ks);
uMat = permute(Uout, [3 2 1]);     % numEle × 24
Scattering_inplace(Y, eNodMat, uMat);
Y = Y'; Y = double(Y(:));
wait(gpuDevice);	
end	
end
function x = Solving_Vcycle(r)
    global meshHierarchy_;
    global weightFactorJacobi_;
    global numLevels_;
    %global typeVcycle_;
	global cholFac_; global cholPermut_;
	global mixed_Precision_;
	global typeVcycle_; 
    % use cell datastruct
	if mixed_Precision_ == 1
        r = single(r);   % explicitly downcast to single precision inside the V-cycle
    end
    x_cell = cell(numLevels_, 1);
    r_cell = cell(numLevels_, 1);
    r_cell{1} = r;
    d_weightFactorJacobi=gpuArray(weightFactorJacobi_);
    %% 1. Restriction (fine -> coarse)
    for ii = 1:numLevels_-1
        x_cell{ii} = d_weightFactorJacobi * (r_cell{ii} ./ meshHierarchy_(ii).d_diagK);
            %r_cell{ii+1} = Solving_RestrictResidual(r_cell{ii}, ii+1);
			switch typeVcycle_
			case 'Adapted'
				r_cell{ii+1} = Solving_RestrictResidual(r_cell{ii}, ii+1);
			case 'Standard'
				d = r_cell{ii} - Solving_KbyU_MatrixFree(x_cell{ii}, ii);
				r_cell{ii+1} = Solving_RestrictResidual(d, ii+1);				
		end
    end
    
    %% 2. Coarsest level solve (Cholesky solver)
	c_r = gather(r_cell{numLevels_});
	if mixed_Precision_==1
	c_x = single(cholPermut_ * (cholFac_'\(cholFac_\(cholPermut_'*double(c_r)))));
	else
	c_x = cholPermut_ * (cholFac_'\(cholFac_\(cholPermut_'*c_r)));
	end
	x_cell{numLevels_} = gpuArray(c_x);
    %% 3. Interpolation (coarse -> fine)
    for ii = numLevels_-1:-1:1
        x_cell{ii} = x_cell{ii} + Solving_InterpolationDeviation(x_cell{ii+1}, ii+1);
         %x_cell{ii} = x_cell{ii} + d_weightFactorJacobi * r_cell{ii} ./ meshHierarchy_(ii).d_diagK;
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
function rCoaser = Solving_RestrictResidual(rFiner,ii)
	global meshHierarchy_;
	global mixed_Precision_;	
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
RI    = meshHierarchy_(ii).d_multiGridOperatorRI;   % [nInt x NPE x Ne]
Ne    = size(meshHierarchy_(ii).d_eNodMat, 1);
NPE   = size(meshHierarchy_(ii).d_eNodMat, 2);
nInt  = size(RI, 1);
keys  = meshHierarchy_(ii).d_eNodMat(:);     % 1-based
% 1) 
F_evals = rFiner1(meshHierarchy_(ii).d_transferMat, :);  % [(nInt*Ne) x 3]
% 2)  [NPE x nInt x Ne] * [nInt x 3 x Ne] -> [NPE x 3 x Ne]
F_pages = reshape(F_evals, [nInt, Ne, 3]);% [nInt x Ne x 3]
P3D     = pagemtimes(RI,'transpose', F_pages,'none');% [NPE x Ne x 3]
P_ne_npe_3 = permute(P3D, [2 1 3]);% [Ne x NPE x 3]
P2D        = reshape(P_ne_npe_3, [Ne*NPE, 3]);
%4)
scatter_accum3_inplace(keys, P2D, rCoaser);
rCoaser = reshape(rCoaser', 3*meshHierarchy_(ii).numNodes, 1);
end
function xFiner = Solving_InterpolationDeviation(xCoarser, ii)
	global meshHierarchy_;
	global mixed_Precision_;
	xCoarser = reshape(xCoarser,3,meshHierarchy_(ii).numNodes)';
	if mixed_Precision_==1
	xFiner = zeros(meshHierarchy_(ii).intermediateNumNodes,3,'single','gpuArray');
	else
	xFiner = zeros(meshHierarchy_(ii).intermediateNumNodes,3,'gpuArray');
	end
	transferMat = meshHierarchy_(ii).d_transferMat(:);
RI = meshHierarchy_(ii).d_multiGridOperatorRI;     % [nInt x NPE]
Ne = size(meshHierarchy_(ii).d_eNodMat,1);
NPE = size(meshHierarchy_(ii).d_eNodMat,2);
nInt = size(RI,1);

% 1)[(Ne*NPE) x 3] -> [NPE x Ne x 3]
Evals = xCoarser(meshHierarchy_(ii).d_eNodMat, :);     % [(Ne*NPE) x 3] 
Epages = reshape(Evals, [Ne, NPE, 3]);                 % [Ne x NPE x 3]
Epages = permute(Epages, [2 1 3]);                     % [NPE x Ne x 3]

% 2) 
Tmp = pagemtimes(RI, Epages);                         % [nInt x Ne x 3]
% 3) 
Tmp2D = reshape(Tmp, [nInt*Ne, 3]);                  % [(nInt*Ne) x 3]

%4) 
scatter_accum3_inplace(transferMat, Tmp2D, xFiner);
	xFiner = xFiner ./ meshHierarchy_(ii).d_transferMatCoeffi;
	xFiner = xFiner(meshHierarchy_(ii).d_solidNodeMapCoarser2Finer,:);	
	xFiner = reshape(xFiner', 3*meshHierarchy_(ii-1).numNodes, 1);
end

function [PDEkernal, diagPrecond] = TopOpti_SetupPDEfilter_matrixFree(filterRadius,model)
	global meshHierarchy_;
	global Super_element_;
	global Num_;
	global elemNew;
    global X;
    global Y;
    global Z;
	%% Gaussian Points
	s = [-1.0  1.0  1.0 -1.0 -1.0  1.0 1.0 -1.0]'/sqrt(3);
	t = [-1.0 -1.0  1.0  1.0 -1.0 -1.0 1.0  1.0]'/sqrt(3);
	p = [-1.0 -1.0 -1.0 -1.0  1.0  1.0 1.0  1.0]'/sqrt(3);
	w = [ 1.0  1.0  1.0  1.0  1.0  1.0 1.0  1.0]';
	
	%% Trilinear Shape Functions (N)
	N = zeros(size(s,1), 8);
	N(:,1) = 0.125*(1-s).*(1-t).*(1-p); N(:,2) = 0.125*(1+s).*(1-t).*(1-p); N(:,3) = 0.125*(1+s).*(1+t).*(1-p); N(:,4) = 0.125*(1-s).*(1+t).*(1-p);
	N(:,5) = 0.125*(1-s).*(1-t).*(1+p); N(:,6) = 0.125*(1+s).*(1-t).*(1+p); N(:,7) = 0.125*(1+s).*(1+t).*(1+p); N(:,8) = 0.125*(1-s).*(1+t).*(1+p);
	
	%% dN
	dN1ds = -0.125*(1-t).*(1-p); dN2ds = 0.125*(1-t).*(1-p); dN3ds = 0.125*(1+t).*(1-p);  dN4ds = -0.125*(1+t).*(1-p);
	dN5ds = -0.125*(1-t).*(1+p); dN6ds = 0.125*(1-t).*(1+p); dN7ds = 0.125*(1+t).*(1+p);  dN8ds = -0.125*(1+t).*(1+p);
	dN1dt = -0.125*(1-s).*(1-p); dN2dt = -0.125*(1+s).*(1-p); dN3dt = 0.125*(1+s).*(1-p);  dN4dt = 0.125*(1-s).*(1-p);
	dN5dt = -0.125*(1-s).*(1+p); dN6dt = -0.125*(1+s).*(1+p); dN7dt = 0.125*(1+s).*(1+p);  dN8dt = 0.125*(1-s).*(1+p);
	dN1dp = -0.125*(1-s).*(1-t); dN2dp = -0.125*(1+s).*(1-t); dN3dp = -0.125*(1+s).*(1+t); dN4dp = -0.125*(1-s).*(1+t);
	dN5dp = 0.125*(1-s).*(1-t);  dN6dp = 0.125*(1+s).*(1-t); dN7dp = 0.125*(1+s).*(1+t);  dN8dp = 0.125*(1-s).*(1+t);
	dShape = zeros(3*numel(s), 8);
	dShape(1:3:end,:) = [dN1ds dN2ds dN3ds dN4ds dN5ds dN6ds dN7ds dN8ds];
	dShape(2:3:end,:) = [dN1dt dN2dt dN3dt dN4dt dN5dt dN6dt dN7dt dN8dt];
	dShape(3:3:end,:) = [dN1dp dN2dp dN3dp dN4dp dN5dp dN6dp dN7dp dN8dp];
	
	%%Jacobian Matrix, corresponding to the commonly used 2x2x2 cubic element in natural coordinate system
	%%CellSize
	CellSize = 1; %%Always treated as a unit cell
	detJ = CellSize^3 /8 * ones(8,1); %%Sub-Volume
	wgt = w.*detJ;
	KEF0 = dShape'*dShape;
	KEF1 = N'*diag(wgt)*N;
    iRmin = (filterRadius * meshHierarchy_(1).eleSize(1))/2/sqrt(3);
    iKEF = iRmin^2*KEF0 + KEF1;
	PDEkernal = iKEF;
	
	%%Diagonal Preconditioner
	if Super_element_==0
	diagPrecond = zeros(meshHierarchy_(1).numNodes,1);
	numElements = meshHierarchy_(1).numElements;
	diagKe = diag(PDEkernal);
	blockIndex = Solving_MissionPartition(numElements, 1.0e7);
	for jj=1:size(blockIndex,1)
		rangeIndex = (blockIndex(jj,1):blockIndex(jj,2))';
		jElesNodMat = meshHierarchy_(1).eNodMat(rangeIndex,:)';
		diagKeBlock = diagKe(:) .* ones(1,numel(rangeIndex));
		jElesNodMat = jElesNodMat(:);
		diagKeBlockSingleDOF = diagKeBlock(:); 
		diagPrecond = diagPrecond + accumarray(jElesNodMat, diagKeBlockSingleDOF, [meshHierarchy_(1).numNodes, 1]);				
	end
else
	[Y,X,Z]=size(model);
	edomat=build_eNodMat_match(Num_*X,Num_*Y,Num_*Z);
	%elemNew = gpuArray(reorder_fine_elems(edomat, X, Y, Z, Num_));
    [map_old2new, map_new2old] = gen_index_mapping(X, Y, Z, Num_);
    elemNew=gpuArray(edomat(map_new2old,:));
	clear edomat;
	diagPrecond = zeros((Num_*X+1)*(Num_*Y+1)*(Num_*Z+1),1);
	numElements = (Num_*X)*(Num_*Y)*(Num_*Z);
	diagKe = diag(PDEkernal);
	blockIndex = Solving_MissionPartition(numElements, 1.0e7);
	for jj=1:size(blockIndex,1)
		rangeIndex = (blockIndex(jj,1):blockIndex(jj,2))';
		jElesNodMat = elemNew(rangeIndex,:)';
		diagKeBlock = diagKe(:) .* ones(1,numel(rangeIndex));
		jElesNodMat = jElesNodMat(:);
		diagKeBlockSingleDOF = diagKeBlock(:); 
		diagPrecond = diagPrecond + accumarray(jElesNodMat, diagKeBlockSingleDOF, [(Num_*X+1)*(Num_*Y+1)*(Num_*Z+1), 1]);				
	end
end	
	diagPrecond = diagPrecond.^(-1);
end

function tar = TopOpti_ConductPDEFiltering_matrixFree(src, PDEkernal, diagPrecond, varargin)
	global meshHierarchy_;
	global PDEkernal_;
    global maxIT_;
    global tol_;
	global Super_element_;
	global elemNew;
	global X;
    global Y;
    global Z;
    global Num_;
	PDEkernal_ = PDEkernal;
	%%Element to Node
	if 4==nargin
		passiveElements = varargin{1}; src(passiveElements) = 0.0; %%Exclude Volume Constributions from Passive Elements
	end	
	values = src(:)*(1/8);
	
if Super_element_==0
tmpVal = gpuArray.zeros(meshHierarchy_(1).numNodes,1);
iElesNodMat = meshHierarchy_(1).d_eNodMat;         % [Ne × 8] 
else
tmpVal = gpuArray.zeros((Num_*X+1)*(Num_*Y+1)*(Num_*Z+1),1);
iElesNodMat = elemNew;
end 
nPerEle     = size(iElesNodMat, 2);                % = 8
idx_all     = iElesNodMat(:, 1:nPerEle);           % [Ne × 8]
idx_all     = idx_all(:);                          % [Ne*8 × 1]

vals_all    = repmat(values(:), nPerEle, 1);       % [Ne*8 × 1]
if Super_element_==0
tmpVal = tmpVal + accumarray(idx_all, vals_all, ...
                             [meshHierarchy_(1).numNodes, 1]);
else
tmpVal = tmpVal + accumarray(idx_all, vals_all, ...
                             [(Num_*X+1)*(Num_*Y+1)*(Num_*Z+1), 1]); 
end
	src = tmpVal;
	
	%% Solving on Node
	PtV = @(x) diagPrecond .* x;
	%tar = pcg(@MatTimesVec_matrixFree_B, src, 1.0e-6, 200, PtV);
	tar = Solving_PCG(@MatTimesVec_matrixFree_B, PtV, src, tol_/10, maxIT_, [0 0]);

	%%Node to Element
    tmpVal = sum(tar(iElesNodMat), 2);
	tar = tmpVal*(1/8);
end

function Y1 = MatTimesVec_matrixFree_B(uVec)	
	global meshHierarchy_;
	global PDEkernal_;
	global Super_element_;
	global elemNew;
    global X;
    global Y;
    global Z;
    global Num_;

	PDEfilterkernal = PDEkernal_;
	if Super_element_==0
    iElesNodMat = meshHierarchy_(1).d_eNodMat;
	else
	iElesNodMat =elemNew;
	end
subDisVec   = uVec(iElesNodMat);           
subDisVec   = subDisVec * PDEfilterkernal;
if Super_element_==0
Y1 = accumarray(iElesNodMat(:), subDisVec(:), [meshHierarchy_(1).numNodes, 1]);  
else
Y1 = accumarray(iElesNodMat(:), subDisVec(:), [(Num_*X+1)*(Num_*Y+1)*(Num_*Z+1), 1]);
end
end

function Solving_AssembleFEAstencil()
	global meshHierarchy_;
	global numLevels_;
	global cholFac_; global cholPermut_;
	global isThisEle2ndLevelIncludingFixedDOFsOn1stLevel_;
	global uniqueKesFixed_;
	global uniqueKesFree_;
    global d_uniqueKesFixed_;
	global d_uniqueKesFree_;
	global sonElesWithFixedDOFs_;
	global mapUniqueKes_;
    global d_mapUniqueKes_;
	global mixed_Precision_;
	global Super_element_;
	global I_mat;
	global K_mat;
	global d_K_mat;
	%% Compute 'Ks' on Coarser Levels
	reOrdering = [1 9 17 2 10 18 3 11 19 4 12 20 5 13 21 6 14 22 7 15 23 8 16 24];
	t_asm  = zeros(1, numLevels_-1);       % Ks assembly time per level (index = ii-1)
	t_diag = zeros(1, numLevels_-1);       % diagK time per level (valid only for ii < numLevels_)
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
		%%Compute Element Stiffness Matrices on Coarser Levels
		t1 = tic;
		if 2==ii
			if Super_element_==0			
			iKe = meshHierarchy_(ii-1).Ke;
			iKs = reshape(iKe, 24*24, 1);
			eleModulus = meshHierarchy_(1).eleModulus;
			Ks = repmat(meshHierarchy_(ii).Ke, 1,1,numElements);
			isThisEle2ndLevelIncludingFixedDOFsOn1stLevel = isThisEle2ndLevelIncludingFixedDOFsOn1stLevel_;
			sonElesWithFixedDOFs = sonElesWithFixedDOFs_;
			mapAllElements2ElementsWithFixedDOFs = mapUniqueKes_;
			uniqueKeListFixedDOFs = uniqueKesFixed_;
			uniqueKeListFreeDOFs = uniqueKesFree_;	
           iKeCol = meshHierarchy_(1).Ke(:);
               d_Ks = assembleKs_level2_gpu( ...
                interpolatingKe, ...
               eDofMat4Finer2Coarser, ...
               meshHierarchy_(ii).elementUpwardMap, ...
                meshHierarchy_(1).eleModulus, ...
              iKeCol, ...
             d_mapUniqueKes_, ...
             d_uniqueKesFree_, ...
             d_uniqueKesFixed_);
            if ii==numLevels_, Ks = gather(d_Ks); end   % only the coarsest level needs a CPU Ks (for Cholesky)
            %wait(gpuDevice);
            %e1=toc(e1);
            %fprintf("L2 assembly time: %.3f\n",e1);
            else
            uniqueKeListFixedDOFs = uniqueKesFixed_;
			uniqueKeListFreeDOFs = uniqueKesFree_;	
            E_sub = meshHierarchy_(1).eleModulus;
			K_mat   = I_mat * E_sub;
            d_K_mat = gpuArray(K_mat);
            d_E_sub = gpuArray(E_sub);
            d_Ks = assembleKs_level2_superEle_gpu( ...
                interpolatingKe, ...
                eDofMat4Finer2Coarser, ...
                meshHierarchy_(ii).elementUpwardMap, ...
                d_K_mat, d_E_sub, ...
                d_mapUniqueKes_, ...
                d_uniqueKesFree_, ...
                d_uniqueKesFixed_);
            if ii==numLevels_, Ks = gather(d_Ks); end
	end
        else
            % e1=tic;
            % KsPrevious = Ks; clear Ks;
			% Ks = repmat(meshHierarchy_(ii).Ke, 1,1,numElements);	
			%   jj=1:numElements
			% 	iFinerEles = elementUpwardMap(jj,:);
			% 	solidEles = find(0~=iFinerEles);
			% 	iFinerEles = iFinerEles(solidEles);
			% 	sK = finerKes;
			% 	tarKes = KsPrevious(:,:,iFinerEles);
			% 	for kk=1:length(solidEles)
			% 		sK(:,solidEles(kk)) = reshape(tarKes(:,:,kk),24^2,1);
			% 	end
			% 	tmpK = sparse(iK, jK, sK, numProjectDOFs, numProjectDOFs);
			% 	tmpK = interpolatingKe' * tmpK * interpolatingKe;
			% 	Ks(:,:,jj) = full(tmpK);				
            % end		
    		% e1=toc(e1);
            % fprintf("L%d assembly time: %.3f\n",ii,e1);
            %e1 = tic;
                % ---- GPU path: use d_Ks from the previous level, already on the GPU ----
                % meshHierarchy_(ii-1).d_Ks was assigned at the end of the previous
                % iteration and already resides on the GPU; no extra transfer needed.
                d_Ks = assembleKs_higherLevel_gpu( ...
                    interpolatingKe, ...
                    eDofMat4Finer2Coarser, ...
                    elementUpwardMap, ...
                    meshHierarchy_(ii-1).d_Ks);
                if ii==numLevels_, Ks = gather(d_Ks); end   % only the coarsest level needs a CPU Ks (for Cholesky)
		end
		t_asm(ii-1) = toc(t1);
		meshHierarchy_(ii).d_Ks = d_Ks;
        clear d_Ks;
		%%Initialize Jacobian Smoother on Coarser Levels
		if ii<numLevels_
			t2 = tic;
			% d_Ks is [24x24xnumElements]; reshape to [576xnumElements], then extract diagonal with stride 25 -> [numElements x 24]
			d_Ks_flat     = reshape(meshHierarchy_(ii).d_Ks, 576, []);            % [576×Ne], GPU
			d_diagKeBlock = d_Ks_flat(1:25:end, :)';                              % [Ne×24],  GPU
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

	%%%Initialize Jacobian Smoother on Finest Level
	t_diagL1 = tic;
	diagK = zeros(meshHierarchy_(1).numNodes,3);
	numElements = meshHierarchy_(1).numElements;
	if Super_element_==0
	diagKe  = diag(meshHierarchy_(1).Ke);     % [24 x 1], CPU (diagonal of the element stiffness matrix)
	bEleIdx = find(mapUniqueKes_ > 0);        % [numBE x 1], CPU, boundary elements with fixed DOFs
	bKIdx   = mapUniqueKes_(bEleIdx);         % [numBE x 1], CPU, column indices into uniqueKes

	% Step 1: base diagonal block [Ne x 24] = d_eleModulus[Ne x 1] .* diagKe'[1 x 24] (GPU broadcast)
	d_diagKe      = gpuArray(diagKe(:)');                              % [1 x 24]
	d_diagKeBlock = meshHierarchy_(1).d_eleModulus(:) .* d_diagKe;    % [Ne x 24]

	% Step 2: overwrite diagonal entries for boundary elements (with fixed DOFs),
	% using the uniqueKes arrays already on the GPU.
	% d_uniqueKesFree_/Fixed_ are [576 x numBE]; extract diagonal with stride 25 -> [24 x numBE]
	if ~isempty(bEleIdx)
		d_diagFree_bd  = d_uniqueKesFree_(1:25:end, bKIdx);           % [24×numBE], GPU
		d_diagFixed_bd = d_uniqueKesFixed_(1:25:end, bKIdx);          % [24×numBE], GPU
		d_eleM_bd      = meshHierarchy_(1).d_eleModulus(bEleIdx);     % [numBE×1], GPU
		d_diagKeBlock(bEleIdx, :) = ...
			(d_diagFree_bd .* d_eleM_bd(:)' + d_diagFixed_bd)';       % [numBE×24]
	end

	% Step 3: scatter-add via Scattering_inplace MEX (same interface as Solving_KbyU_MatrixFree)
	% Y[eNodMat[e,n], comp] += d_diagKeBlock[e, 3*n+comp]
	d_diagK_out = zeros(meshHierarchy_(1).numNodes, 3, 'gpuArray');
	Scattering_inplace(d_diagK_out, meshHierarchy_(1).d_eNodMat, d_diagKeBlock);

	% Step 4: reshape into DOF vector format and store
	d_diagK_vec = reshape(d_diagK_out.', meshHierarchy_(1).numDOFs, 1);
	meshHierarchy_(1).diagK = gather(d_diagK_vec);
	if mixed_Precision_==1
		meshHierarchy_(1).d_diagK = single(d_diagK_vec);
	else
		meshHierarchy_(1).d_diagK = d_diagK_vec;
	end
else
	eleModulus = meshHierarchy_(1).eleModulus;
	blockIndex = Solving_MissionPartition(numElements, 1.0e7);
	for jj=1:size(blockIndex,1)
		rangeIndex = (blockIndex(jj,1):blockIndex(jj,2))';
		jElesNodMat = meshHierarchy_(1).eNodMat(rangeIndex,:)';
		K_mat_local=K_mat(:,rangeIndex);
		diagKeBlock =K_mat_local(int32(1:25:24*24),:);
		jTarEles = mapUniqueKes_(rangeIndex,:);
        eleWithFixedDOFs = find(jTarEles>0);
		eleWithFixedDOFsLocal = jTarEles(eleWithFixedDOFs);
		numTarEles = numel(eleWithFixedDOFs);
		for kk=1:numTarEles
			coeff = reshape(eleModulus(:,eleWithFixedDOFs(kk)), 1, 64, []);
			combined = sum(uniqueKeListFreeDOFs(:,:,eleWithFixedDOFsLocal(kk)) .* coeff, 2);%[24*24,1,1]
			kKeFreeDOFs=reshape(combined,24,24);
			kKeFixedDOFs = reshape(uniqueKeListFixedDOFs(:,eleWithFixedDOFsLocal(kk)), 24, 24);
			diagKeBlock(:,eleWithFixedDOFs(kk)) = diag(kKeFreeDOFs) + ...
				diag(kKeFixedDOFs);
		end
		jElesNodMat = jElesNodMat(:);
		diagKeBlockSingleDOF = diagKeBlock(1:3:end,:); diagKeBlockSingleDOF = diagKeBlockSingleDOF(:);
		diagK(:,1) = diagK(:,1) + accumarray(jElesNodMat, diagKeBlockSingleDOF, [meshHierarchy_(1).numNodes, 1]);
		diagKeBlockSingleDOF = diagKeBlock(2:3:end,:); diagKeBlockSingleDOF = diagKeBlockSingleDOF(:);
		diagK(:,2) = diagK(:,2) + accumarray(jElesNodMat, diagKeBlockSingleDOF, [meshHierarchy_(1).numNodes, 1]);
		diagKeBlockSingleDOF = diagKeBlock(3:3:end,:); diagKeBlockSingleDOF = diagKeBlockSingleDOF(:);
		diagK(:,3) = diagK(:,3) + accumarray(jElesNodMat, diagKeBlockSingleDOF, [meshHierarchy_(1).numNodes, 1]);
	end
	meshHierarchy_(1).diagK = reshape(diagK.',meshHierarchy_(1).numDOFs,1);
	if mixed_Precision_==1
		meshHierarchy_(1).d_diagK = single(gpuArray(meshHierarchy_(1).diagK));
	else
		meshHierarchy_(1).d_diagK = gpuArray(meshHierarchy_(1).diagK);
	end
end
	t_diagL1 = toc(t_diagL1);
	%% Assemble&Factorize Stiffness Matrix on Coarsest Level
	%t_coarsest = tic;
	[rowIndice, colIndice, ~] = find(ones(24));	
	t_coarsest = tic;
	sK = zeros(24^2, meshHierarchy_(end).numElements);
	for ii=1:meshHierarchy_(end).numElements
		iKe = Ks(:,:,ii);
		sK(:,ii) = iKe(:);
	end
	eNodMat = meshHierarchy_(end).eNodMat;
	eDofMat = [3*eNodMat-2 3*eNodMat-1 3*eNodMat];
	eDofMat = eDofMat(:,reOrdering);
	iK = eDofMat(:,rowIndice);
	jK = eDofMat(:,colIndice);
	KcoarsestLevel = sparse(iK, jK, sK');
	[cholFac_, ~, cholPermut_] = chol(KcoarsestLevel,'lower');
	t_coarsest = toc(t_coarsest);
	%% Print timing breakdown
	msg = '  [AssembleStencil]';
	for ii = 2:numLevels_
		msg = [msg sprintf(' L%d-asm:%.3fs', ii, t_asm(ii-1))];
	end
	for ii = 2:numLevels_-1
		msg = [msg sprintf(' L%d-diagK:%.3fs', ii, t_diag(ii-1))];
	end
	msg = [msg sprintf(' L1-diagK:%.3fs coarsest+Chol:%.3fs', t_diagL1, t_coarsest)];
	fprintf('%s\n', msg);
end

function blockIndex = Solving_MissionPartition(totalSize, blockSize)
	numBlocks = ceil(totalSize/blockSize);		
	blockIndex = ones(numBlocks,2);
	blockIndex(1:numBlocks-1,2) = (1:1:numBlocks-1)' * blockSize;
	blockIndex(2:numBlocks,1) = blockIndex(2:numBlocks,1) + blockIndex(1:numBlocks-1,2);
	blockIndex(numBlocks,2) = totalSize;	
end
function Solving_BuildingMeshHierarchy()
	global meshHierarchy_;
	global numLevels_;
	global eNodMatHalfTemp_;
	global nonDyadic_;
	
	if numel(meshHierarchy_)>1, return; end
	%%0. global ordering of nodes on each levels
	nodeVolume = 1:(int32(meshHierarchy_(1).resX)+1)*(int32(meshHierarchy_(1).resY)+1)*(int32(meshHierarchy_(1).resZ)+1);
	nodeVolume = reshape(nodeVolume(:), meshHierarchy_(1).resY+1, meshHierarchy_(1).resX+1, meshHierarchy_(1).resZ+1);

	if 1==nonDyadic_ && numLevels_>=4
		numLevels_ = numLevels_ - 1; 
	else
		nonDyadic_ = 0;
	end
	for ii=2:numLevels_
		%%1. adjust voxel resolution
		if ii==2 && 1==nonDyadic_
			spanWidth = 4;
		else
			spanWidth = 2;
		end
		nx = meshHierarchy_(ii-1).resX/spanWidth;
		ny = meshHierarchy_(ii-1).resY/spanWidth;
		nz = meshHierarchy_(ii-1).resZ/spanWidth;
		
		%%2. initialize mesh
		meshHierarchy_(ii) = Data_CartesianMeshStruct();
		meshHierarchy_(ii).resX = nx;
		meshHierarchy_(ii).resY = ny;
		meshHierarchy_(ii).resZ = nz;
		meshHierarchy_(ii).eleSize = meshHierarchy_(ii-1).eleSize*spanWidth;
		meshHierarchy_(ii).spanWidth = spanWidth;
		
		%%3. identify solid&void elements
		%%3.1 capture raw info.
		iEleVolume = reshape(meshHierarchy_(ii-1).eleMapForward, spanWidth*ny, spanWidth*nx, spanWidth*nz);
		iEleVolumeTemp = reshape((1:int32(spanWidth^3*nx*ny*nz))', spanWidth*ny, spanWidth*nx, spanWidth*nz);
		iFineNodVolumeTemp = reshape((1:int32((spanWidth*nx+1)*(spanWidth*ny+1)* (spanWidth*nz+1)))', spanWidth*ny+1, spanWidth*nx+1, spanWidth*nz+1); % temporary node index volume
	
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
		
		%%3.2 building the mapping relation for following tri-linear interpolation					
		%					 _______						 _______ _______
		%					|		|						|		|		|
		%			void	|solid	|						|void	|solid	|
		%					|		|						|		|		|
		%			 _______|_______|						|_______|_______|		
		%			|		|		|		<----->			|		|		|
		%			|solid	|solid	|						|solid	|solid	|	
		%			|		|		|						|		|		|
		%			|_______|_______|						|_______|_______|
		% elementsIncVoidLastLevelGlobalOrdering	elementsLastLevelGlobalOrdering
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
		meshHierarchy_(ii).d_transferMatCoeffi=gpuArray(meshHierarchy_(ii).transferMatCoeffi);
		elementsLastLevelGlobalOrdering = meshHierarchy_(ii-1).eleMapBack;
		nodesLastLevelGlobalOrdering = eNodMatHalfTemp_(elementsLastLevelGlobalOrdering,:);
		nodesLastLevelGlobalOrdering = Common_RecoverHalfeNodMat(nodesLastLevelGlobalOrdering);
		nodesLastLevelGlobalOrdering = unique(nodesLastLevelGlobalOrdering);
		[~,meshHierarchy_(ii).solidNodeMapCoarser2Finer] = intersect(nodesIncVoidLastLevelGlobalOrdering, nodesLastLevelGlobalOrdering);
		meshHierarchy_(ii).solidNodeMapCoarser2Finer = int32(meshHierarchy_(ii).solidNodeMapCoarser2Finer);
		meshHierarchy_(ii).d_solidNodeMapCoarser2Finer=gpuArray(meshHierarchy_(ii).solidNodeMapCoarser2Finer);
	
		%%3.3 initialize the solid elements 
		meshHierarchy_(ii).eleMapForward = zeros(nx*ny*nz,1,'int32');
		meshHierarchy_(ii).eleMapBack = int32(unemptyElements);
		meshHierarchy_(ii).numElements = length(unemptyElements);
		meshHierarchy_(ii).eleMapForward(unemptyElements) = (1:meshHierarchy_(ii).numElements)';
		elementUpwardMap = elementUpwardMap(unemptyElements,:);	
		meshHierarchy_(ii).elementUpwardMap = elementUpwardMap; clear elementUpwardMap
		
		%%4. discretize
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
		for jj=1:8
			eNodMat(:,jj) = meshHierarchy_(ii).nodMapForward(eNodMat(:,jj));
		end
		if 1==nonDyadic_, kk = ii; else, kk = ii-1; end
		tmp = nodeVolume(1:2^kk:meshHierarchy_(1).resY+1, 1:2^kk:meshHierarchy_(1).resX+1, 1:2^kk:meshHierarchy_(1).resZ+1);
		tmp = reshape(tmp,numel(tmp),1);
		meshHierarchy_(ii).nodMapBack = tmp(meshHierarchy_(ii).nodMapBack);
	
		%%5. initialize multi-grid Restriction&Interpolation operator
		meshHierarchy_(ii).multiGridOperatorRI = Solving_Operator4MultiGridRestrictionAndInterpolation('inNODE', spanWidth);
	    meshHierarchy_(ii).d_multiGridOperatorRI=gpuArray(full(meshHierarchy_(ii).multiGridOperatorRI));
		%%6. identify boundary info.
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
	
	%%Print Mesh Hierarchy
	disp('Mesh Hierarchy...');
	disp('             #Resolutions         #Elements   #DOFs');
	for ii=1:numel(meshHierarchy_)
		disp([sprintf('...Level %i', ii), sprintf(': %4i x %4i x %4i', [meshHierarchy_(ii).resX meshHierarchy_(ii).resY ...
			meshHierarchy_(ii).resZ]), sprintf(' %11i', meshHierarchy_(ii).numElements), sprintf(' %11i', meshHierarchy_(ii).numDOFs)]);
	end
end

function eNodMat = Common_RecoverHalfeNodMat(eNodMatHalf)
	if 4~=size(eNodMatHalf,2), eNodMat = []; return; end
	numEles = size(eNodMatHalf,1);
	eNodMat = zeros(numEles,8,'int32');
	eNodMat(:,[3 4 7 8]) = eNodMatHalf;
	eNodMat(:,[2 1 6 5]) = eNodMatHalf + 1;
end

function FEA_SetupVoxelBased()
	global meshHierarchy_;
	global mixed_Precision_;
	%%1. Initialize Element Stiffness Matrix
	meshHierarchy_(1).Ke = FEA_VoxelBasedElementStiffnessMatrix();	
	
	%%2. Initialize Solver
	%% Building Mesh Hierarchy for Geometric Multi-grid Solver
	Solving_BuildingMeshHierarchy();

	meshHierarchy_(1).Ks = meshHierarchy_(1).Ke;
	if mixed_Precision_==1
	meshHierarchy_(1).d_Ks =single(gpuArray(meshHierarchy_(1).Ks)) ;
	else
	meshHierarchy_(1).d_Ks =gpuArray(meshHierarchy_(1).Ks);
	end
	
	%%Preparation to apply for BCs directly on element stiffness matrix
	Solving_SetupKeWithFixedDOFs();
end

function KE = FEA_VoxelBasedElementStiffnessMatrix()
	global poissonRatio_;
	global cellSize_;
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

function FEA_ApplyBoundaryCondition()
	global meshHierarchy_;
	global F_;
	global loadingCond_;		
	global fixingCond_;
	%%Pre-Check
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
	
	%% Loading
	F_ = sparse(meshHierarchy_(1).numNodes*3, numel(loadingCond_));
	for ii=1:numel(loadingCond_)
		iLoad = loadingCond_{ii};
		iFarr = sparse(meshHierarchy_(1).numNodes, 3);
		iFarr(iLoad(:,1),:) = iLoad(:,2:end);
		F_(:,ii) = reshape(iFarr',meshHierarchy_(1).numDOFs,1);
	end
	
	%%Fixing
	meshHierarchy_(1).fixedDOFs = [];
	meshHierarchy_(1).freeDOFs = [];
	fixedDOFs = 3*fixingCond_(:,1);
	fixedDOFs = fixedDOFs - [2 1 0];
	fixedDOFs = reshape(fixedDOFs', numel(fixedDOFs), 1);
	fixingState = fixingCond_(:,2:end)';
	fixedDOFs = fixedDOFs(1==fixingState(:)); %% E.g., X-dir is fixed, Y-dir is not
	freeDOFs = true(meshHierarchy_(1).numDOFs,1);
	freeDOFs(fixedDOFs) = false;
	meshHierarchy_(1).freeDOFs = freeDOFs;
	meshHierarchy_(1).fixedDOFs = false(meshHierarchy_(1).numDOFs,1);
	meshHierarchy_(1).fixedDOFs(fixedDOFs) = true;
end

function Solving_SetupKeWithFixedDOFs()
	global meshHierarchy_;
	global isThisEle2ndLevelIncludingFixedDOFsOn1stLevel_;
	global uniqueKesFixed_;
	global d_uniqueKesFixed_;
	global uniqueKesFree_;
	global d_uniqueKesFree_;
	global sonElesWithFixedDOFs_;
	global mapUniqueKes_;
	global d_mapUniqueKes_;
	global fixingCond_;
	global modulus_;
	global d_eleWithFixedDOFs_;
	global Super_element_;
	global I_mat;
    global Num_;
	global mixed_Precision_;
	if Super_element_==0
	%%Identify Elements On 2nd Finest Level that Include Elements on Finest Level with Fixed DOFs
	%%Preparation
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
				if jiNode
					nodeBoundaryStruct(jiNode).arr(1,end+1) = ii;
				end
			end
		end
	end
	
	%fixedNodesStruct = nodeBoundaryStruct(fixingCond_(:,1),1);
	[~,fixedNodeIndices] = intersect(meshHierarchy_(1).nodesOnBoundary, fixingCond_(:,1));
	fixedNodesStruct = nodeBoundaryStruct(fixedNodeIndices,1);
	allElementsWithFixedDOFs = unique([fixedNodesStruct.arr]);
	numElementsWithFixedDOFs = numel(allElementsWithFixedDOFs);
	mapUniqueKes_ = zeros(meshHierarchy_(1).numElements,1,'int32');
	mapUniqueKes_(allElementsWithFixedDOFs,1) = (1:numElementsWithFixedDOFs)';
	KeCol = meshHierarchy_(1).Ke(:);
	% KeListFinestWithFixedDOFs_ = repmat(KeCol,1,numElementsWithFixedDOFs);
	uniqueKesFree_ = repmat(KeCol,1,numElementsWithFixedDOFs);
	uniqueKesFixed_ = zeros(size(uniqueKesFree_));
	for ii=1:numel(fixedNodesStruct)
        iFixation = fixingCond_(ii,:);
		% iNode = meshHierarchy_(1).nodesOnBoundary(iFixation(1));
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
	isThisEle2ndLevelIncludingFixedDOFsOn1stLevel_ = zeros(meshHierarchy_(2).numElements,1, 'int32');
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
	% if mixed_Precision_==1
	% d_uniqueKesFixed_=gpuArray(single(uniqueKesFixed_));
	% d_uniqueKesFree_=gpuArray(single(uniqueKesFree_));
	% d_mapUniqueKes_=gpuArray(int32(mapUniqueKes_));
	% d_eleWithFixedDOFs_=int32(find(d_mapUniqueKes_>0));
	% else
	d_uniqueKesFixed_=gpuArray(uniqueKesFixed_);
	d_uniqueKesFree_=gpuArray(uniqueKesFree_);
	d_mapUniqueKes_=gpuArray(int32(mapUniqueKes_));
	d_eleWithFixedDOFs_=int32(find(d_mapUniqueKes_>0));
	%end
    else
    I_mat=computeK_ij(Num_);
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
				if jiNode
					nodeBoundaryStruct(jiNode).arr(1,end+1) = ii;
				end
			end
		end
	end
	
	%fixedNodesStruct = nodeBoundaryStruct(fixingCond_(:,1),1);
	[~,fixedNodeIndices] = intersect(meshHierarchy_(1).nodesOnBoundary, fixingCond_(:,1));
	fixedNodesStruct = nodeBoundaryStruct(fixedNodeIndices,1);
	allElementsWithFixedDOFs = unique([fixedNodesStruct.arr]);
	numElementsWithFixedDOFs = numel(allElementsWithFixedDOFs);
	mapUniqueKes_ = zeros(meshHierarchy_(1).numElements,1,'int32');
	mapUniqueKes_(allElementsWithFixedDOFs,1) = (1:numElementsWithFixedDOFs)';
	uniqueKesFree_=repmat(I_mat,1,1,numElementsWithFixedDOFs);
	uniqueKesFixed_ = zeros(24*24,numElementsWithFixedDOFs);
	for ii=1:numel(fixedNodesStruct)
        iFixation = fixingCond_(ii,:);
		% iNode = meshHierarchy_(1).nodesOnBoundary(iFixation(1));
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
			[uniqueKesFree_(:,:,jEleLocally), uniqueKesFixed_(:,jEleLocally)] = ...
				Solving_ApplyBConEleStiffMat_B_Super(uniqueKesFree_(:,:,jEleLocally), uniqueKesFixed_(:,jEleLocally), fixedDOFsLocally, numElesOfThisNode);
		end
	end
	uniqueKesFixed_ = uniqueKesFixed_ * modulus_;
	isThisEle2ndLevelIncludingFixedDOFsOn1stLevel_ = zeros(meshHierarchy_(2).numElements,1, 'int32');
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
	% if mixed_Precision_==1
	% d_uniqueKesFixed_=gpuArray(single(uniqueKesFixed_));
	% d_uniqueKesFree_=gpuArray(single(uniqueKesFree_));
	% d_mapUniqueKes_=gpuArray(int32(mapUniqueKes_));
	% d_eleWithFixedDOFs_=int32(find(d_mapUniqueKes_>0));
	% else
	d_uniqueKesFixed_=gpuArray(uniqueKesFixed_);
	d_uniqueKesFree_=gpuArray(uniqueKesFree_);
	d_mapUniqueKes_=gpuArray(int32(mapUniqueKes_));
	d_eleWithFixedDOFs_=int32(find(d_mapUniqueKes_>0));
	%end
end
end

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

function tarBC = AdaptBCExternalMdl(srcBC, adjustedRes)
	global nelx_; global nely_; global nelz_;
	nullNodeVolume = zeros((nelx_+1)*(nely_+1)*(nelz_+1),1);
	
	adjustedNnlx = adjustedRes(1);
	adjustedNnly = adjustedRes(2); 
	adjustedNnlz = adjustedRes(3);
	nnlx = nelx_+1;
	nnly = nely_+1;
	nnlz = nelz_+1;
	
	tmpBC = zeros(size(srcBC));
	[~, ia] = sort(srcBC(:,1));
	srcBC = srcBC(ia,:);
	nodeVolume = nullNodeVolume; 
	nodeVolume(srcBC(:,1)) = 1; 
	nodeVolume = reshape(nodeVolume, nely_+1, nelx_+1, nelz_+1);
	nodeVolume(:,end+1:adjustedNnlx,:) = zeros(nnly,adjustedNnlx-nnlx,nnlz);
	nodeVolume(end+1:adjustedNnly,:,:) = zeros(adjustedNnly-nnly,adjustedNnlx,nnlz);
	nodeVolume(:,:,end+1:adjustedNnlz) = zeros(adjustedNnly,adjustedNnlx,adjustedNnlz-nnlz);
	newLoadedNodes = find(nodeVolume);
	tmpBC(:,1) = newLoadedNodes;
	
	nodeVolume = nullNodeVolume; 
	nodeVolume(srcBC(:,1)) = srcBC(:,2); 
	nodeVolume = reshape(nodeVolume, nely_+1, nelx_+1, nelz_+1);
	nodeVolume(:,end+1:adjustedNnlx,:) = zeros(nnly,adjustedNnlx-nnlx,nnlz);
	nodeVolume(end+1:adjustedNnly,:,:) = zeros(adjustedNnly-nnly,adjustedNnlx,nnlz);
	nodeVolume(:,:,end+1:adjustedNnlz) = zeros(adjustedNnly,adjustedNnlx,adjustedNnlz-nnlz);
	nodeVolume = nodeVolume(:);
	tmpBC(:,2) = nodeVolume(newLoadedNodes);
	
	nodeVolume = nullNodeVolume; 
	nodeVolume(srcBC(:,1)) = srcBC(:,3); 
	nodeVolume = reshape(nodeVolume, nely_+1, nelx_+1, nelz_+1);
	nodeVolume(:,end+1:adjustedNnlx,:) = zeros(nnly,adjustedNnlx-nnlx,nnlz);
	nodeVolume(end+1:adjustedNnly,:,:) = zeros(adjustedNnly-nnly,adjustedNnlx,nnlz);
	nodeVolume(:,:,end+1:adjustedNnlz) = zeros(adjustedNnly,adjustedNnlx,adjustedNnlz-nnlz);
	nodeVolume = nodeVolume(:);
	tmpBC(:,3) = nodeVolume(newLoadedNodes);		
	
	nodeVolume = nullNodeVolume; 
	nodeVolume(srcBC(:,1)) = srcBC(:,4); 
	nodeVolume = reshape(nodeVolume, nely_+1, nelx_+1, nelz_+1);
	nodeVolume(:,end+1:adjustedNnlx,:) = zeros(nnly,adjustedNnlx-nnlx,nnlz);
	nodeVolume(end+1:adjustedNnly,:,:) = zeros(adjustedNnly-nnly,adjustedNnlx,nnlz);
	nodeVolume(:,:,end+1:adjustedNnlz) = zeros(adjustedNnly,adjustedNnlx,adjustedNnlz-nnlz);
	nodeVolume = nodeVolume(:);
	tmpBC(:,4) = nodeVolume(newLoadedNodes);			
	tarBC = tmpBC;
end

function adjustedVoxelIndices = AdaptPassiveElementsExternalMdl(srcElesMapback, adjustedRes)
	global nelx_; global nely_; global nelz_;
	nullVoxelVolume = zeros(nelx_*nely_*nelz_,1);
	
	adjustedNelx = adjustedRes(1);
	adjustedNely = adjustedRes(2); 
	adjustedNelz = adjustedRes(3);

	tmpBC = zeros(size(srcElesMapback));
	[~, ia] = sort(srcElesMapback(:,1));
	srcElesMapback = srcElesMapback(ia,:);
	voxelVolume = nullVoxelVolume; 
	voxelVolume(srcElesMapback(:,1)) = 1; 
	voxelVolume = reshape(voxelVolume, nely_, nelx_, nelz_);
	voxelVolume(:,end+1:adjustedNelx,:) = zeros(nely_,adjustedNelx-nelx_,nelz_);
	voxelVolume(end+1:adjustedNely,:,:) = zeros(adjustedNely-nely_,adjustedNelx,nelz_);
	voxelVolume(:,:,end+1:adjustedNelz) = zeros(adjustedNely,adjustedNelx,adjustedNelz-nelz_);
	adjustedVoxelIndices = find(voxelVolume);
end

function [passiveElementsOnBoundary, passiveElementsNearLoads, passiveElementsNearFixation] = TopOpti_SetPassiveElements(numLayerboundary, numLayerLoads, numLayerFixation)
	global meshHierarchy_;
	global loadingCond_;
	global fixingCond_;
	global passiveElements_; 
	
	numLayerboundary = round(numLayerboundary);
	numLayerLoads = round(numLayerLoads);
	numLayerFixation = round(numLayerFixation);
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
		%%Relate Elements to Nodes
		allElements = zeros(meshHierarchy_(1).numElements,1);
		allElements(meshHierarchy_(1).elementsOnBoundary) = 1;
		nodeStruct_ = struct('arr', []);
		nodeStruct_ = repmat(nodeStruct_, meshHierarchy_(1).numNodes, 1);
		boundaryNodes_Temp = [];
		numNodsPerEle = 8;
		for ii=1:meshHierarchy_(1).numElements
			if 0 || allElements(ii) %%switch 0 to 1 for non-boundary fixation situations, efficiency loss
				iNodes = meshHierarchy_(1).eNodMat(ii,:);
				for jj=1:numNodsPerEle
					nodeStruct_(iNodes(jj)).arr(1,end+1) = ii;
				end
			end
		end
		
		%% Extract Elements near Loads
		passiveElementsNearLoads = [];
		if numLayerLoads>0
			passiveElementsNearLoads = [];
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

function oEleList = Common_IncludeAdjacentElements(iEleList)
	global meshHierarchy_;
	iEleListMapBack = meshHierarchy_(1).eleMapBack(iEleList);
	%%	1	4	7		10	 13	  16		19	 22	  25
	%%	2	5	8		11	 14*  17		20	 23	  26
	%%	3	6	9		12	 15   18		21	 24	  27
	%%	 bottom				middle				top
	resX = meshHierarchy_(1).resX;
	resY = meshHierarchy_(1).resY;
	resZ = meshHierarchy_(1).resZ;
	if 1
		[eleX, eleY, eleZ] = Common_NodalizeDesignDomain([resX-1 resY-1 resZ-1], [1 1 1; resX resY resZ]);
		eleX = eleX(iEleListMapBack);
		eleY = eleY(iEleListMapBack);
		eleZ = eleZ(iEleListMapBack);
	else
		numSeed = [resX-1 resY-1 resZ-1];
		nx = numSeed(1); ny = numSeed(2); nz = numSeed(3);
		dd = [1 1 1; resX resY resZ];
		xSeed = dd(1,1):(dd(2,1)-dd(1,1))/nx:dd(2,1);
		ySeed = dd(2,2):(dd(1,2)-dd(2,2))/ny:dd(1,2);
		zSeed = dd(1,3):(dd(2,3)-dd(1,3))/nz:dd(2,3);
		tmp = repmat(reshape(repmat(xSeed,ny+1,1), (nx+1)*(ny+1), 1), (nz+1), 1);
		eleX = tmp(iEleListMapBack);
		tmp = repmat(repmat(ySeed,1,nx+1)', (nz+1), 1);
		eleY = tmp(iEleListMapBack);
		tmp = reshape(repmat(zSeed,(nx+1)*(ny+1),1), (nx+1)*(ny+1)*(nz+1), 1);
		eleZ = tmp(iEleListMapBack);	
	end
	
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

function FEA_VoxelBasedDiscretization()	
	global nelx_; global nely_; global nelz_;
	global boundingBox_;
	global voxelizedVolume_; 
	global meshHierarchy_;
	global coarsestResolutionControl_;
	global eNodMatHalfTemp_;
	global numLevels_;
	%    z
	%    |__ x
	%   / 
	%  -y                            
	%            8--------------7      	
	%			/ |			   /|	
	%          5-------------6	|
	%          |  |          |  |
	%          |  |          |  |	
	%          |  |          |  |   
	%          |  4----------|--3  
	%     	   | /           | /
	%          1-------------2             
	%			Hexahedral element
	
	%%1. adjust voxel resolution for building Mesh Hierarchy
	[nely_, nelx_, nelz_] = size(voxelizedVolume_);
    numVoxels = nnz(voxelizedVolume_);%%numVoxels = numel(find(voxelizedVolume_));	
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
	if adjustedNelx>nelx_
		voxelizedVolume_(:,end+1:adjustedNelx,:) = false(nely_,adjustedNelx-nelx_,nelz_);
	end
	if adjustedNely>nely_
		voxelizedVolume_(end+1:adjustedNely,:,:) = false(adjustedNely-nely_,adjustedNelx,nelz_);
	end
	if adjustedNelz>nelz_
		voxelizedVolume_(:,:,end+1:adjustedNelz) = false(adjustedNely,adjustedNelx,adjustedNelz-nelz_);
	end
	%%2. initialize characteristic size
	boundingBox_ = [0 0 0; adjustedNelx adjustedNely adjustedNelz];
	
	%%3. initialize the finest mesh
	meshHierarchy_ = Data_CartesianMeshStruct();
	meshHierarchy_.resX = adjustedNelx; nx = meshHierarchy_.resX;
	meshHierarchy_.resY = adjustedNely; ny = meshHierarchy_.resY;
	meshHierarchy_.resZ = adjustedNelz; nz = meshHierarchy_.resZ;
	meshHierarchy_.eleSize = (boundingBox_(2,:) - boundingBox_(1,:)) ./ [nx ny nz];

	%%4. identify solid&void elements
	voxelizedVolume_ = voxelizedVolume_(:);
	meshHierarchy_.eleMapBack = int32(find(voxelizedVolume_));
	meshHierarchy_.numElements = numel(meshHierarchy_.eleMapBack);
	meshHierarchy_.eleMapForward = zeros(nx*ny*nz,1, 'int32');
	meshHierarchy_.eleMapForward(meshHierarchy_.eleMapBack) = (1:meshHierarchy_.numElements)';
		
	%%5. discretize
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
	meshHierarchy_.nodMapForward = zeros((nx+1)*(ny+1)*(nz+1),1, 'int32');
	meshHierarchy_.nodMapForward(meshHierarchy_.nodMapBack) = (1:meshHierarchy_.numNodes)';
	for ii=1:8
		eNodMat(:,ii) = meshHierarchy_.nodMapForward(eNodMat(:,ii));
	end	
	
	%%6. identify boundary info.	
	meshHierarchy_.numNod2ElesVec = zeros(meshHierarchy_.numNodes,1);
	for jj=1:8
		iNodes = eNodMat(:,jj);
		meshHierarchy_.numNod2ElesVec(iNodes) = meshHierarchy_.numNod2ElesVec(iNodes) + 1;
	end	

	meshHierarchy_.nodesOnBoundary = find(meshHierarchy_.numNod2ElesVec<8);
	meshHierarchy_.nodesOnBoundary = int32(meshHierarchy_.nodesOnBoundary);
	allNodes = zeros(meshHierarchy_.numNodes,1,'int32');
	allNodes(meshHierarchy_.nodesOnBoundary) = 1;	
	tmp = zeros(meshHierarchy_.numElements,1,'int32');
	for ii=1:8
		tmp = tmp + allNodes(eNodMat(:,ii));
	end
	meshHierarchy_.elementsOnBoundary = int32(find(tmp>0));
	meshHierarchy_.eNodMat = eNodMat;
	meshHierarchy_.d_eNodMat = gpuArray(eNodMat); % eNodMat contains node indices for solid elements only
end

function val = Data_CartesianMeshStruct()
	val = struct(...
		'resX',							0,	...
		'resY',							0,	...
		'resZ',							0,	...
		'spanWidth',					0,	...
		'eleSize',						[],	...
		'numElements',					0,	...		
		'numNodes',						0,	...
		'numDOFs',						0,	...
		'eNodMat',						[],	...
		'd_eNodMat',                    [], ...
		'numNod2ElesVec',				0,	...
		'freeDOFs',						0,	...
		'fixedDOFs',					0,	...
		'Ke',							[],	...
		'eleModulus',                   [],	...
		'd_eleModulus',                 [],	...
		'Ks',							[],	...
	    'd_Ks',							[],	...
		'storingState',					0,	...
		'diagK',						[],	...
		'd_diagK',						[],	...
		'eleMapBack',					[],	...
		'eleMapForward',				[],	...
		'nodMapBack',					[],	...
		'nodMapForward',				[],	...
		'solidNodeMapCoarser2Finer',	[], ...
		'd_solidNodeMapCoarser2Finer',	[], ...
		'intermediateNumNodes',			0,	...
		'nodesOnBoundary',				[],	...
		'boundaryNodeCoords',			[],	...
		'elementsOnBoundary',			[],	...
		'boundaryEleFaces',				[], ...
		'elementUpwardMap',				[],	...
		'multiGridOperatorRI',			[],	...
		'd_multiGridOperatorRI',			[],	...
		'transferMat',					[],	...
		'd_transferMat',				[],	...
		'transferMatCoeffi',			[],	...
		'd_transferMatCoeffi',			[]	...
	);
end

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

%% This MMA implementation is a translation of the c-version developed by Niles Aage at the DTU
function [xnew, xold1, xold2] = MMAseq(mm, nn, xvalTmp, xmin, xmax, xold1, xold2, dfdx, gx, dgdx)
	global n; n = nn;
	global m; m = mm;
	
	global asyminit; asyminit = 0.5;%%0.2;
	global asymdec; asymdec = 0.7;%%0.65;
	global asyminc; asyminc = 1.2;%%1.08;	
	
	global k; k = 0;
	
	global a; a = zeros(m,1);
	global c; c = 100.0 * ones(m,1);
	global d; d = zeros(m,1);
	
	global y; y = zeros(m,1);
	global z; z = 0;
	global lam; lam = zeros(m,1);
	
	global L; L = zeros(1,n);
	global U; U = zeros(1,n);
	
	global alpha; alpha = zeros(1,n);
	global beta; beta = zeros(1,n);
	
	global p0; p0 = zeros(1,n);
	global q0; q0 = zeros(1,n);
	global pij; pij = zeros(m,n);
	global qij; qij = zeros(m,n); 
	global b; b = zeros(m,1);
	
	global xo1; xo1 = xold1;
	global xo2; xo2 = xold2;
	
	global grad; grad = zeros(m,1);
	global mu; mu = zeros(m,1);
	global s; s = zeros(2*m,1);
	global Hess; Hess = zeros(m,m);
	
	global xVal; xVal = xvalTmp(:)';
	
	Update(dfdx(:)', gx(:), reshape(dgdx,n,m)', xmin(:)', xmax(:)');
	xnew = xVal(:); xold1 = xo1(:); xold2 = xo2(:);
	
	clear -global n m asyminit asymdec asyminc k a c d y z lam L U alpha beta p0 q0 pij qij b xo1 xo2 grad mu s Hess xVal
end

function [xnew, xold1, xold2] = Update(dfdx, gx, dgdx, xmin, xmax)
	global xo1 xo2 xVal;
	
	%% Generate the subproblem
	GenSub(dfdx,gx,dgdx,xmin,xmax);	%%Checked
	%% Update xolds
	xo2 = xo1;
	xo1 = xVal;
	%% Solve the dual with an interior point method
	SolveDIP();
end

function GenSub(dfdx,gx,dgdx,xmin,xmax)
	global k asyminit L U alpha beta p0 q0 pij qij b xVal;
	
	%% forward the iterator
	k = 1;
	%% Set asymptotes
	L = xVal - asyminit*(xmax - xmin);
	U = xVal + asyminit*(xmax - xmin);
	%% Set bounds and the coefficients for the approximation
	feps = 1.0e-6;
	alpha = 0.9*L+0.1*xVal;
	tmpBool = alpha-xmin<0; alpha(tmpBool) = xmin(tmpBool);
	beta = 0.9*U+0.1*xVal;
	tmpBool = beta-xmax>0; beta(tmpBool) = xmax(tmpBool);
	
	dfdxp = dfdx; dfdxp(dfdxp<0.0) = 0.0;
	dfdxm = -1.0*dfdx; dfdxm(dfdxm<0.0) = 0.0;
	p0 = (U-xVal).^2.0 .*(dfdxp + 0.001*abs(dfdx) + 0.5*feps./(U-L));
	q0 = (xVal-L).^2.0 .*(dfdxm + 0.001*abs(dfdx) + 0.5*feps./(U-L));
	
	dfdxp = dgdx; dfdxp(dfdxp<0.0) = 0.0;
	dfdxm = -1.0*dgdx; dfdxm(dfdxm<0.0) = 0.0;
	pij = (U-xVal).^2.0 .* dfdxp;
	qij = (xVal-L).^2.0 .* dfdxm;	
	%% The constant for the constraints
	b = -gx + sum(pij./(U-xVal) + qij./(xVal-L), 2);
end

function SolveDIP()
	global n m lam mu c grad s Hess;
	
	lam = c/2.0;
	mu = ones(size(mu));
	tol = 1.0e-9*sqrt(m+n);
	epsi = 1.0;
	err = 1.0;
	while epsi > tol
		loop = 0;
		while err>0.9*epsi && loop<100
			loop = loop + 1;	
			%% Set up newton system
			XYZofLAMBDA();		
			DualGrad();		
			for jj=1:m
				grad(jj) = -1.0 * grad(jj) - epsi/lam(jj);
			end
			DualHess();
			%% Solve Newton system
			s(1:m,1) = Hess\grad;
			%% Get the full search direction
			s(m+1:2*m,1) = -mu + epsi./lam(:) - s(1:m,1).*mu(:)./lam(:);
			%% Perform linesearch and update lam and mu
			DualLineSearch();
			XYZofLAMBDA();	
			%% Compute KKT res
			err = DualResidual(epsi);			
		end
		epsi=epsi*0.1;
	end
end

function XYZofLAMBDA()
	global lam a c y z p0 q0 U L alpha beta pij qij xVal;

	lam(lam<0.0) = 0;
	y = lam - c; y(y<0.0) = 0.0;
	lamai = lam(:)' * a;	
	z = max(0.0, 10.0*(lamai-1.0));	
	
	pjlam = p0 + sum(pij.*lam, 1);
	qjlam = q0 + sum(qij.*lam, 1);
	xVal = (sqrt(pjlam).*L + sqrt(qjlam).*U) ./ (sqrt(pjlam)+sqrt(qjlam));
	tmpBool = xVal-alpha<0; xVal(tmpBool) = alpha(tmpBool);
	tmpBool = xVal-beta>0; xVal(tmpBool) = beta(tmpBool);
	
	clear pjlam qjlam
end

function grad = DualGrad()
	global a y U L grad b z pij qij xVal;
	
	grad = -b - a*z - y;
	grad = grad + sum(pij./(U-xVal) + qij./(xVal-L),2);
end

function DualHess()
	global m p0 q0 pij qij U L lam alpha beta Hess a c mu xVal;
	
	pjlam = p0 + sum(pij.*lam, 1);
	qjlam = q0 + sum(qij.*lam, 1);
	PQ = pij./(U-xVal).^2.0 - qij./(xVal-L).^2.0;
	df2 = -1.0 ./(2.0*pjlam./(U-xVal).^3.0 + 2.0*qjlam./(xVal-L).^3.0);
	xp = (sqrt(pjlam).*L + sqrt(qjlam).*U) ./ (sqrt(pjlam)+sqrt(qjlam));
	df2(xp-alpha<0) = 0.0;
	df2(xp-beta>0) = 0.0;
	%% Create the matrix/matrix/matrix product: PQ^T * diag(df2) * PQ
	tmp = PQ .* df2;
	for ii=1:m
		for jj=1:m
			Hess(jj,ii) = 0;
			Hess(jj,ii) = Hess(jj,ii) + tmp(ii,:)*PQ(jj,:)';
		end
	end
	lamai=0.0;
	for jj=1:m
		if lam(jj)<0.0, lam(jj) = 0.0; end
		lamai = lamai + lam(jj)*a(jj);
		if lam(jj)>c(jj)
			Hess(jj,jj) = Hess(jj,jj) -1.0;
		end
		Hess(jj,jj) = Hess(jj,jj) - mu(jj)/lam(jj); 
	end
	if lamai>0.0
		for jj=1:m
			for kk=1:m
				Hess(kk,jj) = Hess(kk,jj) - 10.0*a(jj)*a(kk);
			end
		end
	end	
	%% pos def check
	HessTrace = trace(Hess);
	HessCorr = 1e-4*HessTrace/m;
	if -1.0*HessCorr < 1.0e-7, HessCorr = -1.0e-7; end
	Hess = Hess + diag(repmat(HessCorr,m,1));
	
	clear pjlam qjlam df2 PQ tmp
end

function DualLineSearch()
	global m s lam mu;
	theta=1.005;
	for jj=1:m
		if theta < -1.01*s(jj)/lam(jj), theta = -1.01*s(jj)/lam(jj); end
		if theta < -1.01*s(jj+m)/mu(jj), theta = -1.01*s(jj+m)/mu(jj); end
	end
	theta = 1.0/theta;
	lam = lam + theta*s(1:m,:);
	mu = mu + theta*s(m+1:2*m,1);
end

function nrI = DualResidual(epsi)
	global m b a z lam mu U L pij qij y xVal;
	res = zeros(2*m,1);
	res(1:m,1) = -b - a.*z -y + mu;
	res(m+1:2*m,1) = mu.*lam - epsi;
	res(1:m,1) = res(1:m,1) + sum(pij./(U-xVal),2) + sum(qij./(xVal-L),2);
	
	nrI=0.0;
	for jj=1:2*m
		if nrI<abs(res(jj))
			nrI = abs(res(jj));
		end
	end
end
function Kpt = hex8_BDB_at(xi, eta, zeta)
E = 1; nu = 0.30; a = 1; b = 1; c = 1;
sx = [-1  1  1 -1 -1  1  1 -1]';
sy = [-1 -1  1  1 -1 -1  1  1]';
sz = [-1 -1 -1 -1  1  1  1  1]';

fac_xi  = (1 + sy*eta).*(1 + sz*zeta);
fac_eta = (1 + sx*xi ).*(1 + sz*zeta);
fac_zet = (1 + sx*xi ).*(1 + sy*eta);

dN_dxi  = 0.125 * sx .* fac_xi;
dN_deta = 0.125 * sy .* fac_eta;
dN_dzet = 0.125 * sz .* fac_zet;

%J    = diag([a/2, b/2, c/2]);
detJ = (a*b*c)/8;
invJ = diag([2/a, 2/b, 2/c]);

dN_dx = invJ(1,1)*dN_dxi;
dN_dy = invJ(2,2)*dN_deta;
dN_dz = invJ(3,3)*dN_dzet;

B = zeros(6, 24);
for aN = 1:8
    iu = 3*(aN-1) + 1;  % u
    iv = iu + 1;        % v
    iw = iu + 2;        % w

    B(1,iu) = dN_dx(aN);                 % ex
    B(2,iv) = dN_dy(aN);                 % ey
    B(3,iw) = dN_dz(aN);                 % ez
    B(4,iv) = dN_dz(aN); B(4,iw) = dN_dy(aN); % gyz
    B(5,iu) = dN_dz(aN); B(5,iw) = dN_dx(aN); % gxz
    B(6,iu) = dN_dy(aN); B(6,iv) = dN_dx(aN); % gxy
end

coef = E / ((1+nu)*(1-2*nu));
D = coef * [ 1-nu,  nu,   nu,   0,             0,             0;
             nu,   1-nu,  nu,   0,             0,             0;
             nu,    nu,  1-nu,  0,             0,             0;
             0,     0,    0,  (1-2*nu)/2,      0,             0;
             0,     0,    0,    0,           (1-2*nu)/2,      0;
             0,     0,    0,    0,             0,           (1-2*nu)/2 ];

Kpt = (B.' * D * B) * detJ;
end
function Kij=computeK_ij(num)
global cellSize_;
	Kij=zeros(24*24,num^3);
	a = -1; b = 1; 
span= a + ((1:num) - 0.5)/num * (b - a);
eid=1;
	for z=1:num
		for x=1:num
			for y=1:num
				yy=span(num+1-y);
				xx=span(x);
				zz=span(z);
				Kij(:,eid)=reshape(hex8_BDB_at(xx,yy,zz),24*24,1);
				eid=eid+1;
			end
		end
	end
Kij=Kij/(num^3);
Kij=Kij*8*cellSize_; 
end
function [oKeFreeDOFs, oKeFixedDOFs] = Solving_ApplyBConEleStiffMat_B_Super(iKeFreeDOFs, iKeFixedDOFs, fixedDOFsLocally, nAdjEles)
	iKeFreeDOFs = reshape(iKeFreeDOFs,24,24,[]);
	iKeFixedDOFs = reshape(iKeFixedDOFs,24,24);
	oKeFreeDOFs = iKeFreeDOFs;
	oKeFixedDOFs = iKeFixedDOFs;
	oKeFreeDOFs(fixedDOFsLocally,:,:) = 0;
	oKeFreeDOFs(:,fixedDOFsLocally,:) = 0;
	for ii=1:numel(fixedDOFsLocally)
		oKeFixedDOFs(fixedDOFsLocally(ii), fixedDOFsLocally(ii)) = 1/nAdjEles;
	end
	oKeFreeDOFs = reshape(oKeFreeDOFs,24*24,[]);
	oKeFixedDOFs = oKeFixedDOFs(:);
end
function ceList = TopOpti_ComputeUnitCompliance_Super(U)
    global meshHierarchy_;	
	global objWeightingList_;
	global K_mat;
	blockIndex = Solving_MissionPartition(meshHierarchy_(1).numElements, 1.0e7);
	ceList = zeros(meshHierarchy_(1).numElements, size(U,2));
	for jj=1:size(U,2)
		ithU = U(:,jj);
		for ii=1:size(blockIndex,1)
			rangeIndex = (blockIndex(ii,1):blockIndex(ii,2))';
			iReshapedU = zeros(numel(rangeIndex),24);
			iElesNodMat = meshHierarchy_(1).eNodMat(rangeIndex,:);
			tmp = ithU(1:3:end,:); iReshapedU(:,1:3:24) = tmp(iElesNodMat);
			tmp = ithU(2:3:end,:); iReshapedU(:,2:3:24) = tmp(iElesNodMat);
			tmp = ithU(3:3:end,:); iReshapedU(:,3:3:24) = tmp(iElesNodMat);		
			Ke3D = reshape(K_mat(:,rangeIndex), 24, 24, []);   % [24 × 24 × nBlock]
            U3D  = permute(iReshapedU, [2 3 1]);            % [24 × 1  × nBlock]
            KU   = pagemtimes(Ke3D, U3D);                   % [24 × 1 × nBlock] -> K_i * u_i
            ceBlock3D = sum(KU .* U3D, 1);                  % row-wise dot product u_i^T (K_i u_i)
			ceList(rangeIndex,jj)=ceBlock3D(:);
		end		
	end
	ceList = ceList*objWeightingList_(:);
end
function dc = pre_compute_der(U)
    global meshHierarchy_;	
	global objWeightingList_;
	global I_mat;
	KeSet = reshape(I_mat, 24, 24, 64);
	dc=zeros(4^3,meshHierarchy_(1).numElements,size(U,2));
	blockIndex = Solving_MissionPartition(meshHierarchy_(1).numElements, 1.0e7);
	for jj=1:size(U,2)
		ithU = U(:,jj);
		for ii=1:size(blockIndex,1)
			rangeIndex = (blockIndex(ii,1):blockIndex(ii,2))';
			iReshapedU = zeros(numel(rangeIndex),24);
			iElesNodMat = meshHierarchy_(1).eNodMat(rangeIndex,:);
			tmp = ithU(1:3:end,:); iReshapedU(:,1:3:24) = tmp(iElesNodMat);
			tmp = ithU(2:3:end,:); iReshapedU(:,2:3:24) = tmp(iElesNodMat);
			tmp = ithU(3:3:end,:); iReshapedU(:,3:3:24) = tmp(iElesNodMat); 
            U3D  = permute(iReshapedU, [2 3 1]);            % [24 × 1  × nBlock]
			for s=1:64
            KU   = pagemtimes(KeSet(:,:,s), U3D);   % [24 × 1 × nBlock]
            e_s  = sum(KU .* U3D, 1);               % [1 × 1 × nBlock] = U^T K U
            dc(s, rangeIndex, jj) = e_s(:);
			end
		end		
	end
	dc = sum(dc .* reshape(objWeightingList_,1,1,[]), 3);
end

function [map_old2new, map_new2old] = gen_index_mapping(X, Y, Z, n)
    nX = n * X;
    nY = n * Y;
    nZ = n * Z;
    N  = nX * nY * nZ;
    i_old = (0:N-1)';
    iy = mod(i_old, nY);
    ix = mod(floor(i_old / nY), nX);
    iz = floor(i_old / (nY * nX));
    bigY   = floor(iy / n);
    localY = mod(iy, n);
    bigX   = floor(ix / n);
    localX = mod(ix, n);
    bigZ   = floor(iz / n);
    localZ = mod(iz, n);
    BigIdx = bigY + Y * (bigX + X * bigZ);  % 0-based
    LocalIdx = localY + n * (localX + n * localZ); % 0-based
    i_new = LocalIdx + (n^3) * BigIdx; % 0-based
    map_old2new = int32(i_new + 1);        
    map_old2new=gpuArray(map_old2new);
    map_new2old = zeros(N,1);
    map_new2old(map_old2new) = (1:N)';
    map_new2old=gpuArray(int32(map_new2old));

end

function eNodMat = build_eNodMat_match(nx, ny, nz)
% Build the element-to-node connectivity table for the finest level,
% matching the node/element ordering convention used throughout this code.
% Node linear numbering: x fastest, then y, then z (MATLAB column-major order).
% 8-node element vertex order:
%   bottom face: (1,0,0) (1,1,0) (0,1,0) (0,0,0)
%   top face:    (1,0,1) (1,1,1) (0,1,1) (0,0,1)

    nNodes  = (nx+1) * (ny+1) * (nz+1);
   % nEles   =  nx    *  ny     *  nz;

    % 3D node numbering table (x fastest)
    nodenrs = reshape(1:nNodes, nx+1, ny+1, nz+1);

    % “bottom-left-back” (0,0,0) base node of each element
    base = nodenrs(1:nx, 1:ny, 1:nz);   % nx x ny x nz
    eNodVec = base(:);                  % nEles x 1

    dx = 1;
    dy = (nx + 1);
    dz = (nx + 1) * (ny + 1);

    % 8-node vertex offsets in the element's local ordering
    off = [ ...
        dx, ...
        dx + dy, ...
        dy, ...
        0, ...
        dx + dz, ...
        dx + dy + dz, ...
        dy + dz, ...
        dz ...
    ];

    eNodMat = int32(repmat(eNodVec, 1, 8) + off);
end