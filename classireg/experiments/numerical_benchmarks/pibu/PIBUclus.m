
function [out] = PIBUclus(my_seed)

	% clear;
	% myDefs;
	% addpath('gpml-matlab-v3.1-2010-09-27/');
	% addpath('gpml-matlab-v4.0-2016-10-19/');
	% start_GPML;

	addpath('gpml-matlab-master-2018-08-01/');
	startup


	addpath('problems/');

	%% options
	% problem = 'prob2d'; % 'prob1d', 'prob3d'
	% problem = 'prob3d';
	% problem = 'micha10D';

	randn('seed',my_seed)

	%% Algorithm parameters:
	e = 0; % Epislon for PI: c.PI(idxIR) = normcdf((c.yr(idxIR)-yMax-c.e)./sqrt(c.sr2(idxIR))); % prob. of improvement
	bOffset = 0.1;
	bThreshold = 0.55;
	verbose = 1;
	optM = 2;


	% fun = @micha10D;
	% eval('micha10D_conf');

	% fun = @hart6D;
	% eval('hart6D_conf');

	fun = @eggs2D;
	eval('eggs2D_conf');
	
	% Constraint function:
	safety = @cons_ball_regions;
	% eval('prob3d')

	%% Initialize class object
	n = length(x0);
	cbo = conBOpt(n,t,e,bOffset,bThreshold,verbose,optM,ellC,sf2C,ellR,sf2R,snR);

	% % Only needed for plotting:
	% if n <= 3
	% 	cbo.y_gt = fun(t);
	% end

	%% Add initial points:
	y = fun(x0);
	ys = safety(x0);
	cbo.addDataPoint(x0,y,ys);


	% s = warning('error', 'MATLAB:illConditionedMatrix');
	% warning('error', 'MATLAB:illConditionedMatrix');

	% keyboard;

	NBOiters = 100;

	% tic;
	fprintf('Starting loop\n')
	%% run algorithm
	for ii = 1:NBOiters

		fprintf('Computing next point ...\n')
		% profile on -history;
	    [x, isConv] = cbo.selectNextPoint();
	    % p = profile('info')
	    % keyboard;

	    y = funNoise(x);
	    ys = safety(x);

	    cbo.addDataPoint(x,y,ys);
	    
	    % Output more data:
	    [xBc,yBc,out] = cbo.stats();

	    fprintf( '  regret  : %f\n',yOpt - out.statBCy(end));
	    
	end

	% t_elapsed = toc;
	% disp(t_elapsed)

% warning(s);

end