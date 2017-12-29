function [J, grad] = costFunction(theta, X, y)

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

J = (-log(sigmoid(theta'*X'))*y-log(1-sigmoid(theta'*X'))*(1-y))/m;

grad = ((sigmoid(theta'*X') - y')*X)'/m;


end
