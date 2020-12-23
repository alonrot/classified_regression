% function to maximize
fun = @(x) exp(-(x(:,1).^2 + x(:,2).^2));
funNoise = @(x) fun(x) + sqrt(0.0001)*randn(size(x,1),1);

% safety function [1:sucess, -1:failure]
safety = @(x) (abs(x(:,1))<.8 & abs(x(:,2))<.6)*2-1;

% start point
x0 = [.5 .5]

% ground truth solution
xOpt = [0 0];

% grid that defines search space
m = 200;
[t1, t2] = ndgrid(linspace(-1,1,m),linspace(-1,1,m));
t = [t1(:) t2(:)];

% gp hyperparameter
ellC = .2;       % length scale
sf2C = 6;        % prior std of gpc

ellR = .2;       % length scale
sf2R = .5;       % prior std of gpr
snR = 0.05;      % signal variance [smaller values lead to more exploitation]
