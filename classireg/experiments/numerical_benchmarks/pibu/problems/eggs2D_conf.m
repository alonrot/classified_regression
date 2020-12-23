noise_std = 0.01;
funNoise = @(x) fun(x) + noise_std*randn(size(x,1),1);

% start point
x0 = [0.1578, 0.558];

% ground truth solution
xOpt = [ 2./3 , 1./3 ];
yOpt = 98.;

% % search space grid
% m = 70;
% [t1, t2, t3] = ndgrid(linspace(-1,1,m),linspace(-1,1,m),linspace(-1,1,m));
% t = [t1(:) t2(:) t3(:)];


% % search space grid
% m = 3; % 3^10 = 59049
% grid1D = linspace(0,1,m);
% [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10] = ndgrid(grid1D);
% t = [t1(:) t2(:) t3(:) t4(:) t5(:) t6(:) t7(:) t8(:) t9(:) t10(:)];

dim = length(xOpt);

% grid that defines search space
m = 180;
grid1D = linspace(0.0,1.0,m);
[t1, t2] = ndgrid(grid1D);
t = [t1(:) t2(:)];

%% Add points right next to the initial point, otherwise...
Npoints = 7600;
XX = randn(Npoints,dim); % A normal dist is symmetrical
XX = XX./sqrt(sum(XX.^2,2)); % project to the surface of the Ndim-sphere
% That last line requires MATLAB R2016b or later. Earlier versions will use bsxfun or even repmat.
% radial scale factor
R = nthroot(rand(Npoints,1),dim);
% Combine
XX = XX.*R;

% Scale:
XX = XX * 0.15;

XX = XX + x0;

% Saturate:
XX(XX < 0.0) = 0.0;
XX(XX > 1.0) = 1.0;

t = [XX;t];

% keyboard;


% gp hyperparameter
% ellC = 1.5/(15 + 1.5);       % length scale 
ellC = 0.2;       % length scale 
% sf2C = sqrt(4.0);        % prior std of gpc
sf2C = 6.0;        % prior std of gpc -> amarco: it has to be 6.0, otherwise the algorithm fails

ellR = 1.5/(15 + 1.5);       % length scale
sf2R = sqrt(2.0);       % prior std of gpr
snR = 0.01;      % signal STD (corrected by alon; before it was wrong and it said 'variance') [smaller values lead to more exploitation]