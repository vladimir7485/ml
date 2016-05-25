function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

J0 = computeCost(X, y, theta);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    % th(1) = theta(1) - alpha * sum((theta'*X')' - y) / m;
    % th(2) = theta(2) - alpha * ((theta'*X')' - y)'*X(:,2) / m;
    th = theta' - alpha * ((theta'*X')' - y)'*X / m;
    
    if computeCost(X, y, th') > J0
        alpha = alpha / 10;
        continue;
    end
    
    theta = th';

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
    if J_history(iter) < alpha
        break;
    end

end

end
