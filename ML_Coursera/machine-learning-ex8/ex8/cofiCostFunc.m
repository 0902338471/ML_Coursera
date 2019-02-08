function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
tmpTheta=Theta.**2;
tmpX=X.**2;
J=(sum(sum(R.*((X*transpose(Theta)-Y).**2)))+lambda*sum(sum(tmpTheta))+lambda*sum(sum(tmpX)))/2;
for i=1:size(X,1)
 fix=X(i,:);
 tmpmovies=X(i,:);
 tmpdotproduct=tmpmovies*transpose(Theta);
 tmpR=R(i,:);
 tmpdotproduct-=Y(i,:);
 tmpdotproduct=tmpdotproduct.*tmpR;
 tmpfinal=tmpdotproduct*Theta;
 X_grad(i,:)=tmpfinal+lambda*fix;
for i=1:size(Theta_grad,1)
 fix=Theta(i,:);
 tmpusers=Theta(i,:);
 tmpdotproduct=tmpusers*transpose(X);
 tmpR=transpose(R)(i,:);
 tmpdotproduct=tmpdotproduct-transpose(Y)(i,:);
 tmpdotproduct=tmpdotproduct.*tmpR;
 tmpfinal=tmpdotproduct*X;
 Theta_grad(i,:)=tmpfinal+lambda*fix;
endfor














% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
