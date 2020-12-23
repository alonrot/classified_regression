% function to maximize
function [y] = hart6D(xx)

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	% For function details and reference information, see:
	% http://www.sfu.ca/~ssurjano/
	%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	% INPUT:
	%
	% xx = [x1, x2, x3, x4, x5, x6]
	%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	assert( all(all(xx <= 1.0 , 1) & all(xx >= 0.0 , 1)) );
	assert(size(xx,1) == 1)

	alpha_ = [1.0, 1.2, 3.0, 3.2]';
	A = [10, 3, 17, 3.50, 1.7, 8;
			0.05, 10, 17, 0.1, 8, 14;
			3, 3.5, 1.7, 10, 17, 8;
			17, 8, 0.05, 10, 0.1, 14];
	P = 10^(-4) * [1312, 1696, 5569, 124, 8283, 5886;
					2329, 4135, 8307, 3736, 1004, 9991;
					2348, 1451, 3522, 2883, 3047, 6650;
					4047, 8828, 8732, 5743, 1091, 381];

	outer = 0;
	for ii = 1:4
		inner = 0;
		for jj = 1:6
			xj = xx(jj);
			Aij = A(ii, jj);
			Pij = P(ii, jj);
			inner = inner + Aij*(xj-Pij)^2;
		end


		new = alpha_(ii) * exp(-inner);
		outer = outer + new;
	end

	% Scale and return
	y = 10*outer;

end