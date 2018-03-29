function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%Run the sigmoid function of each row in the matrix X. The hX matrix is m x
%1. hX is h(x)
hX = zeros(m, 1);
for i=1:m 
    hX(i,:) = sigmoid(sum(theta.*transpose(X(i,:))));
end

% The matrix y is m x 1
J = (1/m)*sum(transpose((-y.*log(hX) - (1-y).*log(1-hX))));

%hX - y
hXMinusY = hX - y;
for i=1:size(theta)
    grad(i,:) = (1/m)*(sum(transpose(hXMinusY.*X(:,i)))); 
end
% =============================================================

end
