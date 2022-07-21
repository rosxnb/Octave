function J = computeCost(X, y, theta)

%   COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
% =========================================================================

% m represents the number of rows, i.e number of training set avialable
m = length(y);

% predicting all the values based on theta passed
predictions = X * theta;

% calculating squared error
errors = (predictions - y).^2;

% finally calulating J(theta) i.e. average error
J = 1 / (2 * m) * sum(errors);

end
