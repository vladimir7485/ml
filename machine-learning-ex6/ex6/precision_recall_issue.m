cc;

%% Train SVM model
load('spamTrain.mat');
% Prepare training set to contain 99% of non-spam and 1% of spam
spam = find(y==1);
nonspam = find(y==0);
Xtrain = [X(nonspam,:); X(spam(round(rand(27,1)*size(spam,1))),:)];
ytrain = [y(nonspam,:); y(spam(round(rand(27,1)*size(spam,1))),:)];
C = 0.1;
model = svmTrain(Xtrain, ytrain, C, @linearKernel);
save('model.mat', 'model');
% load('model.mat', 'model');

%% Calculate Precision and Recall for Train set
%% General case
load('spamTrain.mat');
Xtest = X;
ytest = y;
p = svmPredict(model, Xtest);
TruePos = sum((p+ytest)==2);
TrueNeg = sum((p+ytest)==0);
FalseNeg = sum((p-ytest)==-1);
FalsePos = sum((p-ytest)==1);
P = TruePos / (TruePos + FalsePos);
R = TruePos / (TruePos + FalseNeg);
A = (TruePos + TrueNeg) / size(Xtest, 1);
fprintf('\nGeneral case\nPrecision = %6.2f\nRecall = %6.2f\nAccuracy = %6.2f\n', P, R, A);

%% Apply only non-spam
load('spamTest.mat');
Xtest = X;
ytest = y;
nonspam = find(ytest==0);
Xtest = Xtest(nonspam, :);
ytest = ytest(nonspam, :);
p = svmPredict(model, Xtest);
TruePos = sum((p+ytest)==2);
TrueNeg = sum((p+ytest)==0);
FalseNeg = sum((p-ytest)==-1);
FalsePos = sum((p-ytest)==1);
P = TruePos / (TruePos + FalsePos);
R = TruePos / (TruePos + FalseNeg);
A = (TruePos + TrueNeg) / size(Xtest, 1);
fprintf('\nApply only for non-spam\nPrecision = %6.2f\nRecall = %6.2f\nAccuracy = %6.2f\n', P, R, A);

%% Apply only spam
load('spamTest.mat');
Xtest = X;
ytest = y;
spam = find(ytest==1);
Xtest = Xtest(spam, :);
ytest = ytest(spam, :);
p = svmPredict(model, Xtest);
TruePos = sum((p+ytest)==2);
TrueNeg = sum((p+ytest)==0);
FalseNeg = sum((p-ytest)==-1);
FalsePos = sum((p-ytest)==1);
P = TruePos / (TruePos + FalsePos);
R = TruePos / (TruePos + FalseNeg);
A = (TruePos + TrueNeg) / size(Xtest, 1);
fprintf('\nApply only for spam\nPrecision = %6.2f\nRecall = %6.2f\nAccuracy = %6.2f\n', P, R, A);

%% Calculate Precision and Recall for Test set
%% General case
load('spamTest.mat');
p = svmPredict(model, Xtest);
TruePos = sum((p+ytest)==2);
TrueNeg = sum((p+ytest)==0);
FalseNeg = sum((p-ytest)==-1);
FalsePos = sum((p-ytest)==1);
P = TruePos / (TruePos + FalsePos);
R = TruePos / (TruePos + FalseNeg);
A = (TruePos + TrueNeg) / size(Xtest, 1);
fprintf('\nGeneral case\nPrecision = %6.2f\nRecall = %6.2f\nAccuracy = %6.2f\n', P, R, A);

%% Apply only non-spam
load('spamTest.mat');
nonspam = find(ytest==0);
Xtest = Xtest(nonspam, :);
ytest = ytest(nonspam, :);
p = svmPredict(model, Xtest);
TruePos = sum((p+ytest)==2);
TrueNeg = sum((p+ytest)==0);
FalseNeg = sum((p-ytest)==-1);
FalsePos = sum((p-ytest)==1);
P = TruePos / (TruePos + FalsePos);
R = TruePos / (TruePos + FalseNeg);
A = (TruePos + TrueNeg) / size(Xtest, 1);
fprintf('\nApply only for non-spam\nPrecision = %6.2f\nRecall = %6.2f\nAccuracy = %6.2f\n', P, R, A);

%% Apply only spam
load('spamTest.mat');
spam = find(ytest==1);
Xtest = Xtest(spam, :);
ytest = ytest(spam, :);
p = svmPredict(model, Xtest);
TruePos = sum((p+ytest)==2);
TrueNeg = sum((p+ytest)==0);
FalseNeg = sum((p-ytest)==-1);
FalsePos = sum((p-ytest)==1);
P = TruePos / (TruePos + FalsePos);
R = TruePos / (TruePos + FalseNeg);
A = (TruePos + TrueNeg) / size(Xtest, 1);
fprintf('\nApply only for spam\nPrecision = %6.2f\nRecall = %6.2f\nAccuracy = %6.2f\n', P, R, A);
