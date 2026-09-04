
% Paper experiment reproduction cases.

% Standard topology optimization on a cantilever beam (Paper, Section 5.1).
TOP_GPU(true(48,96,48),'optCase',1,'V0',0.12,'ft',2,'nLoop',200,'tol',1e-12);

% Topology optimization using the supplied voxel models (Paper, Section 5.1).
TOP_GPU('./data/Molar.TopVoxel','V0',0.4,'nLoop',200);
TOP_GPU('./data/femur.TopVoxel','V0',0.4,'nLoop',200);

% Topology optimization with 4-by-4-by-4 super-elements (Paper, Section 5.4).
TOP_GPU(true(48,96,48),'optCase',1,'V0',0.12,'ft',2,'nLoop',200, ...
	'super_element',1,'filter_method','distance');

% Porous infill optimization with a local volume constraint (Paper, Section 5.5).
TOP_GPU(true(48,96,48),'optCase',1,'V0',0.5,'nLoop',200,'consType','LOCAL');

% Periodic microstructure optimization (Paper, Section 5.5).
topX3D_GPU(50,50,50,0.5,5,2.5,1,'maxDesignIter',200);
