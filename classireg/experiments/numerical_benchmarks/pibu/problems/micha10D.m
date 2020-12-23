% function to maximize
function [y] = micha10D(xx)

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	% For function details and reference information, see:
	% http://www.sfu.ca/~ssurjano/

	assert( all(all(xx <= 1.0 , 1) & all(xx >= 0.0 , 1)) );

    m = 10;

    % Domain [0,pi]^D
    xx = xx * pi;

	d = size(xx,2);
	my_sum = 0;

	% keyboard;

	for ii = 1:d
		xi = xx(:,ii);
		new = sin(xi) .* (sin(ii*(xi.^2)./pi)).^(2*m);
		my_sum  = my_sum + new;
	end

	y = my_sum;

	% y = -my_sum; % We want to maximize

end