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

predictions = (X * theta) - y;
predictions_sq = predictions .^ 2;

  % Cost function and regularization
    penalty = sum( theta(2:end, :) .^ 2 ) * lambda;
    J = ( sum(predictions_sq) + penalty ) / (2 * m);

  % Gradient and regularization
    temp = sum( predictions .* X(:, 1) ) / m;
    grad(1) = temp;
    n = size(theta);
    for i = 2 : n
      penalty = (lambda * theta(i)) / m;
      temp = sum( predictions .* X(:, i) ) / m;
      grad(i) = temp + penalty;
    end


% =========================================================================

grad = grad(:);

end
