noise_std = 0.01;
funNoise = @(x) fun(x) + noise_std*randn(size(x,1),1);

% start point
x0 = [0.7139, 0.6342, 0.2331, 0.8299, 0.7615, 0.8232, 0.9008, 0.1899, 0.6961, 0.3240];

% ground truth solution
xOpt = [2.202906, 1.570796, 1.284992, 1.923058, 1.720470, 1.570796, 1.454414, 1.756087, 1.655717, 1.570796] / pi;
yOpt = +9.6601517;

% % search space grid
% m = 70;
% [t1, t2, t3] = ndgrid(linspace(-1,1,m),linspace(-1,1,m),linspace(-1,1,m));
% t = [t1(:) t2(:) t3(:)];


% % search space grid
% m = 3; % 3^10 = 59049
% grid1D = linspace(0,1,m);
% [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10] = ndgrid(grid1D);
% t = [t1(:) t2(:) t3(:) t4(:) t5(:) t6(:) t7(:) t8(:) t9(:) t10(:)];

%% Points on a grid for D = 10 is a bad idea. We do random sampling:
sobseq = sobolset(10,'Skip',1e3,'Leap',1e2);
t = net(sobseq,3e5);

%% Add points right next to the initial point, otherwise...
Npoints = 1e5; dim = 10;
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