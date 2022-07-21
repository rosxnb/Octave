function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

%   GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    predictions = X * theta;
    errors1 = ( predictions - y ); 
    errors2 = ( predictions - y ) .* X(:, 2);
    
    delta1 = (1 / m) * sum(errors1);
    delta2 = (1 / m) * sum(errors2);
    
    val1 = theta(1) - alpha * delta1;
    val2 = theta(2) - alpha * delta2;
    theta(1) = val1;
    theta(2) = val2;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
