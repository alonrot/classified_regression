% function to maximize
function [y] = eggs2D(xx)

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	% For function details and reference information, see:
	% http://www.sfu.ca/~ssurjano/
	%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	% INPUT:
	%
	% xx = [x1, x2]
	%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	assert( all(all(xx <= 1.0 , 1) & all(xx >= 0.0 , 1)) );
	assert(size(xx,1) == 1)

	% Scale domain:
	X = -5 + 7.5*xx(:, 1);
	Y = -2.5 + 7.5*xx(:, 2);

	scores = X.^2 + Y.^2 + (25 * (sin(X).^2 + sin(Y).^2));
	
	% Bring down:
	scores = scores - 98.0;

	% Flip sign:
	y = -scores;

end