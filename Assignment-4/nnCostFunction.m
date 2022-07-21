function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

%   NNCOSTFUNCTION Implements the neural network cost function for a two layer
%   neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




  % Forward propagation to calculate the predicted outputs h_theta(X)^i
    a1s = [ones(m, 1) X];
    a2s = sigmoid(a1s * Theta1');
    
    a2s = [ones(m, 1) a2s];
  a3s = sigmoid(a2s * Theta2');

  % J without regularizaion
    % changing y (5000, 1) to (5000, 10) where each columns has value either 0
    % or 1 only repective to the label it contains
      Y = zeros(m, num_labels);
      for i = 1 : m
        Y(i, y(i)) = 1;
      end

    % summing all the columns elements (outputs) in each row (training set)
      vec = zeros(m, 1);
      for i = 1 : m
        % vec(i) = sum( -new_y(i, :) .* log(a3s(i, :)) - (1 - new_y(i, :)) .* log(1 - a3s(i, :)) );
        vec(i) = -Y(i, :) * log(a3s(i, :))' - (1 - Y(i, :)) * log(1 - a3s(i, :))';
      end

  J = sum(vec) / m;

  % J with regularization
    t1 = Theta1;
    t2 = Theta2;
    t1(:, 1) = 0;
    t2(:, 1) = 0;
    all_theta = [t1(:) ; t2(:)];
    penalty = lambda * sum( sum((all_theta.^ 2), 2) );
    penalty = penalty / (2 * m);

  J = J + penalty;

% -------------------------------------------------------------

  % backpropagation (errors/delta) calculation
  d3 = a3s - Y;

  d2 = d3 * Theta2;
  d2 = d2(:, [2:end]) .* sigmoidGradient(a1s * Theta1');

  Delta_1 = d2' * a1s;
  Delta_2 = d3' * a2s;
  
  Theta1_grad = Delta_1 ./ m;
  Theta2_grad = Delta_2 ./ m;

  % Regularization of gradient
  Theta1_penalty = lambda .* t1;
  Theta2_penalty = lambda .* t2;

  Theta1_grad = (Delta_1 + Theta1_penalty) ./ m;
  Theta2_grad = (Delta_2 + Theta2_penalty) ./ m;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
