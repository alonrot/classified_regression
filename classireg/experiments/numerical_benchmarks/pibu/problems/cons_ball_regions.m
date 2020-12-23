function [y] = cons_ball_regions(x)

	assert( all(all(x <= 1.0 , 1) & all(x >= 0.0 , 1)) );

	cons_val = prod(sin(x*2*pi),2) < 0.0;

	y = 2*cons_val - 1;

end