% Linear Regression
% One Variable
% Gradient Descent

cc;

%% Generate training data
% x = 0:19;
% k = 0.7;
% b = 1.5;
% m = 20;
% 
% y = k.*x + b;
% 
% figure(); plot(x,y,'-r'); hold on;
% 
% noise = normrnd(4, 4, 1, 20) - 2;
% noisy_y = y + noise;
% plot(x,noisy_y,'*b');
% 
% % Training set
% y = noisy_y;

n = 2; % n = 4; % number of features
m = 2; % m = 4; % number of training examples

X = [[1, 2104, 5]; ... % X = [[1, 2104, 5, 1, 45]; ...
     [1, 1416, 3]; ... % [1, 1416, 3, 2, 40]; ...
     [1, 1534, 3]; ... % [1, 1534, 3, 2, 30]; ...
     [1, 852, 2]];     % [1, 852, 2, 1, 36]];
y = [460, 232, 315, 178]';    % y = [460, 232, 315, 178]';

%% Normal equation
Th = pinv(X'*X)*X'*y;

%% Hypothesis function
% th is a column vector
% X is a matrix (each column is one feature)
h = @(th, X)((th'*X')');

%% Cost function (MSE)
% h is a column vector
% y is a column vector
mse = @(h, y)(sum((h-y).^2) / (2*size(y,1)));

%% Feature scaling (mean normalization)
X = (X - repmat(mean(X,1), [size(X,1),1]))./repmat(max(X,[],1),[size(X,1),1]);
X(:,1) = X(:,1) + 1;

%% Initial parameters
Th_initial = [2, -2, 0.5]'; % Th_initial = [2, -2, 0.5, 5, -11]';
thresh = 0.01;
alpha = 0.001;

%% Gradient descent method
i = 0;
t = mse(h(Th_initial, X),y);
tic;
while t > thresh && i < 100000
    fprintf('i = %d, mse = %f\n', i, t);
    th0_ = Th_initial(1) - alpha .* sum(h(Th_initial, X) - y) ./ m; % dJ/dth0
    th1_ = Th_initial(2) - alpha .* sum((h(Th_initial, X) - y).*X(:,2)) ./ m; % dJ/dth1
    th2_ = Th_initial(3) - alpha .* sum((h(Th_initial, X) - y).*X(:,3)) ./ m; % dJ/dth2
    % th3_ = Th_initial(4) - alpha .* sum((h(Th_initial, X) - y).*X(:,4)) ./ m; % dJ/dth3
    % th4_ = Th_initial(5) - alpha .* sum((h(Th_initial, X) - y).*X(:,5)) ./ m; % dJ/dth4
    Th_initial(1) = th0_;
    Th_initial(2) = th1_;
    Th_initial(3) = th2_;
    % Th_initial(4) = th3_;
    % Th_initial(5) = th4_;
    i = i + 1;
    t = mse(h(Th_initial, X),y);
end
toc;
fprintf('N of iterations = %d', i);


