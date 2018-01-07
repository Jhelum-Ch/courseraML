function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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

param = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

for i = 1:length(param) 
 c = param(i);
 for j = 1:length(param)
  sig = param(j);
  model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, sig));
  predictions = svmPredict(model, Xval);
  pred_error(i,j) = mean(double(predictions ~= yval));
 end
end

[B,I] = min(pred_error(:)); %find(pred_error==min(pred_error(:)));

[I_row, I_col] = ind2sub(size(pred_error),I);

C = param(I_row);
sigma = param(I_col);


%Q = I/(length(param));
%R = rem(I,(length(param)));

%if R == 0
 %idc = Q;
 %id_sig = length(param);
%else
 %idc = Q+1;
 %id_sig = R;
%end

%C = param(idc);
%sigma = param(id_sig);


% =========================================================================

end
