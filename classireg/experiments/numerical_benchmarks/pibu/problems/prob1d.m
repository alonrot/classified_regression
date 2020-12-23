% function to maximize
fun = @(x) 1-exp(-x.^2);
funNoise = @(x) fun(x) + sqrt(0.0001)*randn(size(x,1),1);

% safety function [1:sucess, -1:failure]
safety = @(x) (x<0.8 & x>-.6)*2-1;

% start point
x0 = 0

% ground truth solution
xOpt = 0.8;

% search grid
t = ndgrid(-1:0.001:1);

% gp hyperparameter
ellC = 0.2;     % length scale 
sf2C = 6;       % prior std of gpc

ellR = .2;      % length scale
sf2R = .5  ;    % prior std of gpr
snR = 0.05;     % signal variance
