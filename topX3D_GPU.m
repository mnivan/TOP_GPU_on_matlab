function topX3D_GPU_MF(nelx,nely,nelz,volfrac,penal,rmin,ft,varargin)
%TOPX3D_GPU_MF 3-D periodic material optimization using CUDA MEX kernels.
%   This is the 3-D counterpart of topX_GPU_MF. It uses trilinear H8
%   voxels, six unit macro-strain cases, periodic fluctuation fields, GPU
%   PCG, Gathering_inplace/Scattering_inplace, and no assembled global
%   stiffness matrix. It writes RunRecord.txt, Design.stl, and
%   DesignVolume.nii, then plots the final isosurface. It returns no density.

parser = inputParser;
addParameter(parser,'maxDesignIter',50,@(v)isnumeric(v) && isscalar(v) && v>=1);
addParameter(parser,'changeTol',0.01,@(v)isnumeric(v) && isscalar(v) && v>0);
addParameter(parser,'pcgTol',1e-12,@(v)isnumeric(v) && isscalar(v) && v>0);
addParameter(parser,'pcgMaxIter',800,@(v)isnumeric(v) && isscalar(v) && v>=1);
addParameter(parser,'verbose',true,@(v)islogical(v) && isscalar(v));
parse(parser,varargin{:});
opt = parser.Results;

validateattributes(nelx,{'numeric'},{'scalar','integer','>=',2});
validateattributes(nely,{'numeric'},{'scalar','integer','>=',2});
validateattributes(nelz,{'numeric'},{'scalar','integer','>=',2});
validateattributes(volfrac,{'numeric'},{'scalar','>',0,'<=',1});
validateattributes(penal,{'numeric'},{'scalar','>=',1});
validateattributes(rmin,{'numeric'},{'scalar','>',0});
validateattributes(ft,{'numeric'},{'scalar','integer','>=',1,'<=',2});

try
    gpu = gpuDevice;
catch exception
    error('topX3D_GPU_MF:NoGPU','A supported GPU is required: %s',exception.message);
end
toDevice = @(a) gpuArray(a);
deviceName = gpu.Name;
mexState = 'CUDA MEX (Gathering_inplace/Scattering_inplace)';
resultRoot = fullfile(fileparts(mfilename('fullpath')),'topX3D_results');
if ~exist(resultRoot,'dir'), mkdir(resultRoot); end
runStamp = char(datetime('now','Format','yyyyMMdd_HHmmss_SSS'));
caseName = sprintf('topX3D_%dx%dx%d_v%.4g_p%.4g_r%.4g_ft%d_%s', ...
    nelx,nely,nelz,volfrac,penal,rmin,ft,runStamp);
outputDirectory = fullfile(resultRoot,caseName);
mkdir(outputDirectory);
logPath = fullfile(outputDirectory,'RunRecord.txt');
stlPath = fullfile(outputDirectory,'Design.stl');
niiPath = fullfile(outputDirectory,'DesignVolume.nii');
logFid = fopen(logPath,'w','n','UTF-8');
if logFid<0
    error('topX3D_GPU_MF:LogOpen','Unable to create %s.',logPath);
end
fileCleanup = onCleanup(@() fclose(logFid));
totalClock = tic;
fprintf(logFid,'topX3D_GPU_MF Run Record\n');
fprintf(logFid,'Started: %s\n',char(datetime('now')));
fprintf(logFid,'Mesh: %d x %d x %d\n',nelx,nely,nelz);
fprintf(logFid,'volfrac=%.16g penal=%.16g rmin=%.16g ft=%d\n', ...
    volfrac,penal,rmin,ft);
fprintf(logFid,'Device: %s\nGather/scatter: %s\n',deviceName,mexState);
fprintf(logFid,'PCG tolerance: %.3e, maximum iterations: %d\n\n', ...
    opt.pcgTol,opt.pcgMaxIter);

E0 = 1;
Emin = 1e-9;
nu = 0.3;
KE = voxelElementStiffness(nu);

%% H8 connectivity in the same local-node convention as TOP_GPU.m
numElements = nelx*nely*nelz;
nodenrs = reshape(1:(nelx+1)*(nely+1)*(nelz+1),nely+1,nelx+1,nelz+1);
eNodVec = reshape(nodenrs(1:end-1,1:end-1,1:end-1)+1,numElements,1);
nodeOffsets = [0 nely+[1 0] -1 (nely+1)*(nelx+1)+[0 nely+[1 0] -1]];
eNodMat = repmat(eNodVec,1,8)+repmat(nodeOffsets,numElements,1);
edofMat = zeros(numElements,24);
edofMat(:,1:3:end) = 3*eNodMat-2;
edofMat(:,2:3:end) = 3*eNodMat-1;
edofMat(:,3:3:end) = 3*eNodMat;

% Map opposite faces, edges, and corners onto one periodic node lattice.
[fullY,fullX,fullZ] = ndgrid(0:nely,0:nelx,0:nelz);
fullNodeToPeriodic = mod(fullY(:),nely)+1 + ...
    nely*mod(fullX(:),nelx) + nely*nelx*mod(fullZ(:),nelz);
fullDofToPeriodic = reshape([3*fullNodeToPeriodic-2, ...
    3*fullNodeToPeriodic-1,3*fullNodeToPeriodic].',[],1);
eNodPeriodic = reshape(fullNodeToPeriodic(eNodMat),numElements,8);
edofPeriodic = fullDofToPeriodic(edofMat);
numPeriodicNodes = numElements;
numPeriodicDofs = 3*numElements;

flatDofs = edofPeriodic(:);
contributionCount = accumarray(flatDofs,1,[numPeriodicDofs 1]);
if any(contributionCount~=8)
    error('topX3D_GPU_MF:PeriodicMap','Unexpected periodic connectivity multiplicity.');
end
[~,scatterOrder] = sort(flatDofs);
scatterMap = reshape(scatterOrder,8,numPeriodicDofs).';

%% Six macro-strain cases: xx, yy, zz, xy, yz, xz
strain = zeros(3,3,6);
strain(:,:,1) = diag([1 0 0]);
strain(:,:,2) = diag([0 1 0]);
strain(:,:,3) = diag([0 0 1]);
strain(:,:,4) = [0 .5 0;.5 0 0;0 0 0];
strain(:,:,5) = [0 0 0;0 0 .5;0 .5 0];
strain(:,:,6) = [0 0 .5;0 0 0;.5 0 0];
% Match the H8 local-node convention inherited from TOP_GPU.m: x points
% right, y points from the last nodenrs row toward the first, and z points
% forward. Using fullY directly reverses the y macro-strain and changes the
% signs of Q12 and Q23.
coordinates = [fullX(:) nely-fullY(:) fullZ(:)];
macroUe = zeros(numElements,24,6);
for loadCase = 1:6
    macroNodeU = coordinates*strain(:,:,loadCase).';
    macroFullU = reshape(macroNodeU.',[],1);
    macroUe(:,:,loadCase) = macroFullU(edofMat);
end

%% Three-dimensional distance filter and initial spherical inclusion
filterSpan = -(ceil(rmin)-1):(ceil(rmin)-1);
[filterY,filterX,filterZ] = ndgrid(filterSpan,filterSpan,filterSpan);
filterKernel = max(0,rmin-sqrt(filterX.^2+filterY.^2+filterZ.^2));

KE = toDevice(KE);
eNodPeriodic = toDevice(int32(eNodPeriodic));
scatterMap = toDevice(int32(scatterMap));
macroUe = toDevice(macroUe);
filterKernel = toDevice(filterKernel);
Hs = convn(toDevice(ones(nely,nelx,nelz)),filterKernel,'same');

x = repmat(volfrac,nely,nelx,nelz);
[elementY,elementX,elementZ] = ndgrid(1:nely,1:nelx,1:nelz);
radius = sqrt((elementX-nelx/2-0.5).^2 + ...
    (elementY-nely/2-0.5).^2 + (elementZ-nelz/2-0.5).^2);
x(radius<min([nelx,nely,nelz])/3) = volfrac/2;
x = toDevice(x);
xPhys = x;
Ufluctuation = zeros(numPeriodicDofs,6,'like',x);
anchorDofs = int32([1 2 3]);

history.objective = nan(opt.maxDesignIter,1);
history.volume = nan(opt.maxDesignIter,1);
history.change = nan(opt.maxDesignIter,1);
history.pcgIterations = nan(opt.maxDesignIter,6);
history.pcgRelativeResidual = nan(opt.maxDesignIter,6);
if opt.verbose
    fprintf('topX3D GPU matrix-free on %s (%d x %d x %d elements, %s gather/scatter)\n', ...
        deviceName,nelx,nely,nelz,mexState);
    fprintf('Run record: %s\n',logPath);
end

change = 1;
loop = 0;
Q = zeros(6,6);
while change > opt.changeTol && loop < opt.maxDesignIter
    loop = loop+1;
    iterationClock = tic;
    elementModulus = Emin+xPhys(:).^penal*(E0-Emin);
    elementDiagonal = elementModulus.*diag(KE).';
    diagonalK = sum(elementDiagonal(scatterMap),2);
    diagonalK(anchorDofs) = 1;

    elementU = zeros(numElements,24,6,'like',x);
    pcgTimes = zeros(1,6);
    for loadCase = 1:6
        macroElementForce = (macroUe(:,:,loadCase)*KE).*elementModulus;
        rhs = -scatterElementForcesMex(macroElementForce,eNodPeriodic,numPeriodicNodes);
        rhs(anchorDofs) = 0;
        applyK = @(u) applyPeriodicK3DMex(u,elementModulus,KE, ...
            eNodPeriodic,numPeriodicNodes,anchorDofs);
        wait(gpu);
        pcgClock = tic;
        [Ufluctuation(:,loadCase),pcgIts,pcgRelRes] = matrixFreePCG3D( ...
            applyK,rhs,diagonalK,opt.pcgTol,opt.pcgMaxIter,Ufluctuation(:,loadCase));
        wait(gpu);
        pcgTimes(loadCase) = toc(pcgClock);
        fluctuationNodes = reshape(Ufluctuation(:,loadCase),3,numPeriodicNodes).';
        periodicElementU = zeros(numElements,24,'like',x);
        Gathering_inplace(periodicElementU,fluctuationNodes,eNodPeriodic);
        elementU(:,:,loadCase) = macroUe(:,:,loadCase)+periodicElementU;
        history.pcgIterations(loop,loadCase) = pcgIts;
        history.pcgRelativeResidual(loop,loadCase) = pcgRelRes;
    end

    dQ = cell(6,6);
    for i = 1:6
        for j = 1:6
            qe = sum((elementU(:,:,i)*KE).*elementU(:,:,j),2)/numElements;
            Q(i,j) = scalarGather3D(sum(elementModulus.*qe));
            dQ{i,j} = penal*(E0-Emin)*xPhys.^(penal-1).*reshape(qe,nely,nelx,nelz);
        end
    end

    objective = -sum(Q(1:3,1:3),'all');
    dc = zeros(size(xPhys),'like',xPhys);
    for i = 1:3
        for j = 1:3
            dc = dc-dQ{i,j};
        end
    end
    dv = ones(size(xPhys),'like',xPhys);
    if ft == 1
        dc = convn(x.*dc,filterKernel,'same')./Hs./max(1e-3,x);
    else
        dc = convn(dc./Hs,filterKernel,'same');
        dv = convn(dv./Hs,filterKernel,'same');
    end

    l1 = 0;
    l2 = 1e9;
    move = 0.2;
    while l2-l1 > 1e-9
        lmid = 0.5*(l2+l1);
        xnew = max(0,max(x-move,min(1,min(x+move,x.*sqrt(-dc./dv/lmid)))));
        if ft == 1
            trialXPhys = xnew;
        else
            trialXPhys = convn(xnew,filterKernel,'same')./Hs;
        end
        if scalarGather3D(mean(trialXPhys(:))) > volfrac
            l1 = lmid;
        else
            l2 = lmid;
        end
    end
    xPhys = trialXPhys;
    change = scalarGather3D(max(abs(xnew(:)-x(:))));
    x = xnew;

    history.objective(loop) = objective;
    history.volume(loop) = scalarGather3D(mean(xPhys(:)));
    history.change(loop) = change;
    iterationTime = toc(iterationClock);
    recordLine = sprintf(['It.:%4i Obj.:%11.4f Vol.:%7.3f ch.:%7.3f ' ...
        'PCGits:%s PCGtime:%s PCGtotal:%.3f s MaxRes:%.3e IterTime:%.3f s'], ...
        loop,objective,history.volume(loop),change, ...
        mat2str(history.pcgIterations(loop,:)),mat2str(pcgTimes,4), ...
        sum(pcgTimes),max(history.pcgRelativeResidual(loop,:)),iterationTime);
    fprintf(logFid,'%s\n',recordLine);
    if opt.verbose
        fprintf('%s\n',recordLine);
    end
end

xPhysFinal = gatherIfNeeded3D(xPhys);
% Match TOP_GPU's standard-voxel NIfTI export: preserve the final physical
% density in native [nely,nelx,nelz] order without display flip, smoothing,
% or thresholding.  The flip/smoothing below is visualization-only.
niftiwrite(xPhysFinal,niiPath);
isovals = flip(xPhysFinal,1);
isovals = smooth3(isovals,'box',1);
facesIsosurface = isosurface(isovals,0.5);
facesIsocap = isocaps(isovals,0.5);
[stlFaces,stlVertices] = combineTriangulatedSurfaces( ...
    facesIsosurface,facesIsocap);
if isempty(stlFaces)
    error('topX3D_GPU_MF:EmptySurface', ...
        'No 0.5 isosurface was found, so the STL could not be created.');
end
stlwrite(triangulation(stlFaces,stlVertices),stlPath);

figure('Name','topX3D GPU Matrix-Free Final Design','Color','w');
patch(facesIsosurface,'FaceColor',[0 127 0]/255,'EdgeColor','none');
if ~isempty(facesIsocap.faces)
    patch(facesIsocap,'FaceColor',[0 127 0]/255,'EdgeColor','none');
end
view(55,25); axis equal tight; axis off;
xlabel('X'); ylabel('Y'); zlabel('Z');
lighting gouraud; material dull; camlight('headlight','infinite'); drawnow;

totalTime = toc(totalClock);
fprintf(logFid,'\nCompleted: %s\n',char(datetime('now')));
fprintf(logFid,'Iterations: %d\nConverged: %d\nTotal time: %.6f s\n', ...
    loop,change<=opt.changeTol,totalTime);
fprintf(logFid,'Final effective Q (xx yy zz xy yz xz):\n');
fprintf(logFid,' %.16e %.16e %.16e %.16e %.16e %.16e\n',Q.');
fprintf(logFid,'STL: %s\nTriangles: %d\n',stlPath,size(stlFaces,1));
fprintf(logFid,'NIfTI: %s\nVoxel array: [%d %d %d]\n', ...
    niiPath,size(xPhysFinal,1),size(xPhysFinal,2),size(xPhysFinal,3));
clear fileCleanup;
fprintf('Completed in %.3f s\nRun record: %s\nSTL: %s\nNIfTI: %s\n', ...
    totalTime,logPath,stlPath,niiPath);
end

function [faces,vertices] = combineTriangulatedSurfaces(surface,caps)
surfaceFaces = triangulateFaceArray(surface.faces);
capFaces = triangulateFaceArray(caps.faces);
vertices = [surface.vertices;caps.vertices];
faces = [surfaceFaces;capFaces+size(surface.vertices,1)];
end

function triangles = triangulateFaceArray(faces)
if isempty(faces)
    triangles = zeros(0,3);
elseif size(faces,2)==3
    triangles = faces;
elseif size(faces,2)==4
    triangles = [faces(:,[1 2 3]);faces(:,[1 3 4])];
else
    error('topX3D_GPU_MF:UnsupportedFaces', ...
        'Surface faces must contain three or four vertices.');
end
end

function y = applyPeriodicK3DMex(u,elementModulus,KE,eNodPeriodic,numNodes,anchorDofs)
uNodes = reshape(u,3,numNodes).';
elementU = zeros(size(eNodPeriodic,1),24,'like',u);
Gathering_inplace(elementU,uNodes,eNodPeriodic);
elementForce = (elementU*KE).*elementModulus;
y = scatterElementForcesMex(elementForce,eNodPeriodic,numNodes);
y(anchorDofs) = u(anchorDofs);
end

function y = scatterElementForcesMex(elementForce,eNodPeriodic,numNodes)
nodalForce = zeros(numNodes,3,'like',elementForce);
Scattering_inplace(nodalForce,eNodPeriodic,elementForce);
y = reshape(nodalForce.',3*numNodes,1);
end

function [x,iteration,relativeResidual] = matrixFreePCG3D(applyA,b,diagonalA,tolerance,maxIterations,x)
normB = norm(b);
if scalarGather3D(normB) == 0
    x(:) = 0;
    iteration = 0;
    relativeResidual = 0;
    return;
end
r = b-applyA(x);
z = r./diagonalA;
p = z;
rz = dot(r,z);
relativeResidual = scalarGather3D(norm(r)/normB);
iteration = 0;
while iteration < maxIterations && relativeResidual > tolerance
    iteration = iteration+1;
    Ap = applyA(p);
    alpha = rz/dot(p,Ap);
    x = x+alpha*p;
    r = r-alpha*Ap;
    relativeResidual = scalarGather3D(norm(r)/normB);
    if relativeResidual <= tolerance
        break;
    end
    z = r./diagonalA;
    rzNew = dot(r,z);
    p = z+(rzNew/rz)*p;
    rz = rzNew;
end
end

function KE = voxelElementStiffness(nu)
% H8 stiffness matrix used by the inspected TOP_GPU implementation.
C = [2/9 1/18 1/24 1/36 1/48 5/72 1/3 1/6 1/12];
A11 = [-C(1) -C(3) -C(3) C(2) C(3) C(3); -C(3) -C(1) -C(3) -C(3) -C(4) -C(5); ...
    -C(3) -C(3) -C(1) -C(3) -C(5) -C(4); C(2) -C(3) -C(3) -C(1) C(3) C(3); ...
    C(3) -C(4) -C(5) C(3) -C(1) -C(3); C(3) -C(5) -C(4) C(3) -C(3) -C(1)];
B11 = [C(7) 0 0 0 -C(8) -C(8); 0 C(7) 0 C(8) 0 0; 0 0 C(7) C(8) 0 0; ...
    0 C(8) C(8) C(7) 0 0; -C(8) 0 0 0 C(7) 0; -C(8) 0 0 0 0 C(7)];
A22 = [-C(1) -C(3) C(3) C(2) C(3) -C(3); -C(3) -C(1) C(3) -C(3) -C(4) C(5); ...
    C(3) C(3) -C(1) C(3) C(5) -C(4); C(2) -C(3) C(3) -C(1) C(3) -C(3); ...
    C(3) -C(4) C(5) C(3) -C(1) C(3); -C(3) C(5) -C(4) -C(3) C(3) -C(1)];
B22 = [C(7) 0 0 0 -C(8) C(8); 0 C(7) 0 C(8) 0 0; 0 0 C(7) -C(8) 0 0; ...
    0 C(8) -C(8) C(7) 0 0; -C(8) 0 0 0 C(7) 0; C(8) 0 0 0 0 C(7)];
A12 = [C(6) C(3) C(5) -C(4) -C(3) -C(5); C(3) C(6) C(5) C(3) C(2) C(3); ...
    -C(5) -C(5) C(4) -C(5) -C(3) -C(4); -C(4) C(3) C(5) C(6) -C(3) -C(5); ...
    -C(3) C(2) C(3) -C(3) C(6) C(5); C(5) -C(3) -C(4) C(5) -C(5) C(4)];
B12 = [-C(9) 0 -C(9) 0 C(8) 0; 0 -C(9) -C(9) -C(8) 0 -C(8); C(9) C(9) -C(9) 0 C(8) 0; ...
    0 -C(8) 0 -C(9) 0 C(9); C(8) 0 -C(8) 0 -C(9) -C(9); 0 C(8) 0 -C(9) C(9) -C(9)];
A13 = [-C(4) -C(5) -C(3) C(6) C(5) C(3); -C(5) -C(4) -C(3) -C(5) C(4) -C(5); ...
    C(3) C(3) C(2) C(3) C(5) C(6); C(6) -C(5) -C(3) -C(4) C(5) C(3); ...
    C(5) C(4) -C(5) C(5) -C(4) -C(3); -C(3) C(5) C(6) -C(3) C(3) C(2)];
B13 = [0 0 C(8) -C(9) -C(9) 0; 0 0 C(8) C(9) -C(9) C(9); -C(8) -C(8) 0 0 -C(9) -C(9); ...
    -C(9) C(9) 0 0 0 -C(8); -C(9) -C(9) C(9) 0 0 C(8); 0 -C(9) -C(9) C(8) -C(8) 0];
A14 = [C(2) C(5) C(5) C(4) -C(5) -C(5); C(5) C(2) C(5) C(5) C(6) C(3); ...
    C(5) C(5) C(2) C(5) C(3) C(6); C(4) C(5) C(5) C(2) -C(5) -C(5); ...
    -C(5) C(6) C(3) -C(5) C(2) C(5); -C(5) C(3) C(6) -C(5) C(5) C(2)];
B14 = [-C(9) 0 0 -C(9) C(9) C(9); 0 -C(9) 0 -C(9) -C(9) 0; 0 0 -C(9) -C(9) 0 -C(9); ...
    -C(9) -C(9) -C(9) -C(9) 0 0; C(9) -C(9) 0 0 -C(9) 0; C(9) 0 -C(9) 0 0 -C(9)];
A23 = [C(2) C(5) -C(5) C(4) -C(5) C(5); C(5) C(2) -C(5) C(5) C(6) -C(3); ...
    -C(5) -C(5) C(2) -C(5) -C(3) C(6); C(4) C(5) -C(5) C(2) -C(5) C(5); ...
    -C(5) C(6) -C(3) -C(5) C(2) -C(5); C(5) -C(3) C(6) C(5) -C(5) C(2)];
B23 = [-C(9) 0 0 -C(9) C(9) -C(9); 0 -C(9) 0 -C(9) -C(9) 0; 0 0 -C(9) C(9) 0 -C(9); ...
    -C(9) -C(9) C(9) -C(9) 0 0; C(9) -C(9) 0 0 -C(9) 0; -C(9) 0 -C(9) 0 0 -C(9)];
KE = 1/(1+nu)/(2*nu-1)*([A11 A12 A13 A14; A12' A22 A23 A13'; ...
    A13' A23' A22 A12'; A14' A13 A12 A11] + ...
    nu*[B11 B12 B13 B14; B12' B22 B23 B13'; ...
    B13' B23' B22 B12'; B14' B13 B12 B11]);
end

function value = scalarGather3D(value)
if isa(value,'gpuArray')
    value = gather(value);
end
end

function value = gatherIfNeeded3D(value)
if isa(value,'gpuArray')
    value = gather(value);
end
end

