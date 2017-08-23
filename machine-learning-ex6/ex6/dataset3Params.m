function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_sigma_time = zeros(64, 3);
predictions = zeros(size(Xval,1), 64);
i = 0;
for C_i = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    for sigma_i = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
        i = i + 1;
        tm = tic;
        model_i = svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, sigma_i));
        elapsedTime = toc(tm);
        predictions(:,i) = svmPredict(model_i, Xval);
        C_sigma_time(i,:) = [C_i, sigma_i, elapsedTime];
    end
end

error = mean(double(predictions ~= repmat(yval, 1, 64)), 1);
[~, idx] = min(error, [], 2);
C = C_sigma_time(idx, 1);
sigma = C_sigma_time(idx, 2);

% =========================================================================

end
