function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_s = theta;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

##    1th version code
##    S0 = 0;
##    S1 = 0;
##    
##    for i = 1:m
##      S0 +=(theta(1) + theta(2)*X(i, 2) - y(i)); 
##      S1 +=(theta(1) + theta(2)*X(i, 2) - y(i))* X(i,2); 
##    endfor
##    
##    theta(1) -=  (alpha/m) * S0;
##    theta(2) -=  (alpha/m) * S1;


##    theta(1) = theta(1) - alpha*sum(X*theta_s - y)/m;
##    theta(2) = theta(2) - alpha*sum((X*theta_s - y).*X(:,2))/m;
##    
##    theta_s = theta;
##    
    theta = theta - alpha/m*X'*(X*theta - y);
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    

end
  
end


