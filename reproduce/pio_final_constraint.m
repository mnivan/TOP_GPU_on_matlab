function result = pio_final_constraint(outputDir)
% Recompute PIO local-volume metrics from the exported final density, on CPU.
% Example (run later, when desired):
%   pio_final_constraint('./out/pio/case1_96x48x48_PIO_V0.5_r1.732_rh6_n200')
%
% Matches TOP_GPU.m's LOCAL constraint: nodal Q1 PDE filter, p=16,
% local limit V0=0.5, and f = mean((local/V0)^p)^(1/p)-1.
% Uses continuous exported density, exactly as the optimization constraint does.
% This case is a full 96x48x48 rectangular design domain.

if nargin < 1
    outputDir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'out', 'pio', ...
        'case1_96x48x48_PIO_V0.5_r1.732_rh6_n200');
end
densityFile = fullfile(outputDir, 'DesignVolume.nii');
assert(isfile(densityFile), 'Missing DesignVolume.nii: %s', densityFile);
rho = double(niftiread(densityFile));
assert(isequal(size(rho), [48 96 48]), 'Unexpected density dimensions.');
assert(all(isfinite(rho(:))) && all(rho(:)>=0) && all(rho(:)<=1), ...
    'Density must be finite and within [0,1].');

V0 = 0.5; rHat = 6; p = 16;
[ny,nx,nz] = size(rho);
ne = numel(rho);
nn = (ny+1)*(nx+1)*(nz+1);
nodeIds = reshape(1:nn, ny+1, nx+1, nz+1);
base = nodeIds(1:ny,1:nx,1:nz);
base = base(:);
eNod = base + [1, ny+2, ny+1, 0, ...
    (ny+1)*(nx+1)+1, (ny+1)*(nx+1)+ny+2, ...
    (ny+1)*(nx+1)+ny+1, (ny+1)*(nx+1)];

% TOP_GPU.m builds a [0,nx] x [0,ny] x [0,nz] mesh, so eleSize(1)=1.
g = [-1 1]/sqrt(3);
sgn = [-1 1 1 -1 -1 1 1 -1; ...
       -1 -1 1 1 -1 -1 1 1; ...
       -1 -1 -1 -1 1 1 1 1];
K0 = zeros(8); M = zeros(8);
for a=1:2
    for b=1:2
        for c=1:2
            q = [g(a);g(b);g(c)];
            N = prod(1 + sgn.*q, 1)/8;
            dN = zeros(3,8);
            for j=1:3
                other = setdiff(1:3,j);
                dN(j,:) = sgn(j,:).*prod(1+sgn(other,:).*q(other),1)/8;
            end
            K0 = K0 + dN'*dN;
            M = M + (N'*N)/8;
        end
    end
end
radiusScale = rHat/(2*sqrt(3));
Ke = radiusScale^2*K0 + M;

% Assemble the identical element matrix on the full nodal grid, in chunks.
A = sparse(nn,nn);
chunk = 10000;
for first=1:chunk:ne
    last = min(first+chunk-1,ne);
    ids = eNod(first:last,:);
    rows = repmat(ids,1,8);
    cols = repelem(ids,1,8);
    vals = repmat(reshape(Ke,1,64),last-first+1,1);
    A = A + sparse(rows(:),cols(:),vals(:),nn,nn);
end
b = accumarray(eNod(:),repmat(rho(:)/8,8,1),[nn 1]);
constantRhs = accumarray(eNod(:),ones(ne*8,1)/8,[nn 1]);
assert(norm(A*ones(nn,1)-constantRhs,inf)<1e-11, ...
    'PDE assembly failed the constant-field check.');
diagA = diag(A);
[nodal,flag,relres,iters] = pcg(A,b,1e-11,2000,spdiags(diagA,0,nn,nn));
assert(flag==0, 'CPU PDE solve failed: flag=%d, relres=%g',flag,relres);
local = mean(nodal(eNod),2);
aggregate = mean((local/V0).^p)^(1/p);
residual = aggregate-1;
maxLocal = max(local);
result = struct('aggregate',aggregate, 'constraintResidual',residual, ...
    'positiveViolation',max(0,residual), 'maxLocalFraction',maxLocal, ...
    'maxLocalExcess',maxLocal-V0, 'globalFraction',mean(rho(:)), ...
    'pcgRelativeResidual',relres, 'pcgIterations',iters);
fprintf('Final exported design: %s\n',densityFile);
fprintf('Density treatment: continuous exported density\n');
fprintf('Aggregate local-volume value : %.12g\n',aggregate);
fprintf('Constraint residual (g-1)    : %+.12g\n',residual);
fprintf('Positive violation           : %.12g\n',max(0,residual));
fprintf('Maximum local fraction       : %.12g\n',maxLocal);
fprintf('Maximum local excess over V0 : %+.12g\n',maxLocal-V0);
fprintf('Global volume fraction       : %.12g\n',mean(rho(:)));
fprintf('PDE PCG relative residual    : %.3e (%d iterations)\n',relres,iters);
end
