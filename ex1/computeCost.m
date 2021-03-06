function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

##this is 1th version code
##S = 0;
##
##for i = 1:m
##    S += (theta(1) + theta(2)*X(i, 2) - y(i))^2;
##end
##
##J = S/(2*m);

J = sum((X*theta -y).^2)/(2*m);

% =========================================================================

end
