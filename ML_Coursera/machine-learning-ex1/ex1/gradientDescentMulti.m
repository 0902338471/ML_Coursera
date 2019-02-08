function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
result=X*theta-y;
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    tmptheta=theta(1)-(alpha*sum(result.*X(:,1)))/m;
    tmptheta2=theta(2)-(alpha*sum(result.*X(:,2)))/m;
    theta3=theta(3)-(alpha*sum(result.*X(:,3)))/m;
    theta(1)=tmptheta;
    theta(2)=tmptheta2;
    result=X*theta-y;











    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
