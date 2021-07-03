function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %
    % ============================================================

    theta = theta - alpha/m*X'*(X*theta - y);
    
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    
##    continue;
##    

## heta computed from gradient descent:
## 334302.063993
## 100087.116006
## 3673.548451
##
## Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):
## $289314.620338
 
##    S0 = 0;
##    S1 = 0;
##    S2 = 0;
##    
##    for i = 1:m
##      S0 +=(theta(1) + theta(2)*X(i, 2) + theta(3)*X(i, 3)- y(i)); 
##      S1 +=(theta(1) + theta(2)*X(i, 2) + theta(3)*X(i, 3) - y(i))* X(i,2); 
##      S2 +=(theta(1) + theta(2)*X(i, 2) + theta(3)*X(i, 3) - y(i))* X(i,3); 
##    endfor
##    
##    theta(1) -=  (alpha/m) * S0;
##    theta(2) -=  (alpha/m) * S1;
##    theta(3) -=  (alpha/m) * S2;
    
end


end
