function [sig] = sigmoid(x)

	sig = 1./(1.0 + exp(-x));

end