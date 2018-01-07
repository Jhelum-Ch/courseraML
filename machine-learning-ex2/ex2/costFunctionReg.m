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


z = X*theta;

h = sigmoid(z); %hypothesis

r = log(ones(length(y),1) - h);

v = (ones(length(y),1) - y);

theta_trun = theta(2:end);

cost_reg = sum(theta_trun .^ 2);

J = (1/m)*(-y'*log(h) - v'*r) + (lambda/(2*m))*cost_reg;

err = h-y;

grad_reg = [0; theta(2:end)];

grad = (1/m)*X'*err + (lambda/m)*grad_reg;



% =============================================================

end
