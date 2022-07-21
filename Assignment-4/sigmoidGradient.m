function g = sigmoidGradient(z)

%   SIGMOIDGRADIENT returns the gradient of the sigmoid function
%   evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

% g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).


  all_values = [z(:)];
  count = size(all_values);

  for i = 1:count
    all_values(i) = sigmoid(all_values(i)) * ( 1 - sigmoid(all_values(i)));
  end

  g = reshape(all_values, size(z));










% =============================================================




end
