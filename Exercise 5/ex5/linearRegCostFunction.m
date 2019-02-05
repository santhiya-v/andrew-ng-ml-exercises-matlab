function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

htheta = X * theta;
reg = lambda / (2 * m) * sum(theta(2:size(theta)).^ 2);
J = (1 / (2 * m)) * sum(( htheta - y) .^ 2) + reg;
p = htheta - y;
grad_init = 1 / m * ( X' * p);
regGrad = lambda / m;

grad1 = grad_init(2:end) + (theta(2:end) * regGrad);
grad = [grad_init(1);grad1];









% =========================================================================

grad = grad(:);

end
