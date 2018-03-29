function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
%Run the sigmoid function of each row in the matrix X. The hX matrix is m x
%1. hX is h(x)
hX = zeros(m, 1);
for i=1:m 
    hX(i,:) = sigmoid(sum(theta.*transpose(X(i,:))));
end

J = (1/m)*sum(transpose((-y.*log(hX) - (1-y).*log(1-hX)))) + (lambda/(2*m))*sum(theta(2:end,:).^2);

%hX - y
hXMinusY = hX - y;
for i=1:size(theta)
    thetaToUse = theta(i,:);
    if(i == 1)
        thetaToUse = 0;
    end
    grad(i,:) = (1/m)*(sum(transpose(hXMinusY.*X(:,i)))) + (lambda/m)*thetaToUse; 
end


% =============================================================

end
