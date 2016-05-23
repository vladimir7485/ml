% Linear Regression
% One Variable
% Gradient Descent

cc;

%% Generate training data
x = 0:19;
k = 0.7;
b = 1.5;
m = 20;

y = k.*x + b;

figure(); plot(x,y,'-r'); hold on;

noise = normrnd(4, 4, 1, 20) - 2;
noisy_y = y + noise;
plot(x,noisy_y,'*b');

% Training set
y = noisy_y;

% x = [1, 2, 4, 0];
% y = [0.5, 1, 2, 0];
% m = 4;

figure(); plot(x,y,'*b'); hold on;

%% Hypothesis function
h = @(th0, th1, x)(th0 + th1.*x);

%% Cost function (MSE)
mse = @(h, y)(sum((h-y).^2) / (2*m));

%% Initial parameters
th0 = 2;
th1 = -2;
thresh = 0.01;
alpha = 0.002;

%% Gradient descent method
i = 0;
t = mse(h(th0, th1, x),y);
plot(x,h(th0, th1, x),'--g'); pause(0.01);
tic;
while t > thresh && i < 10000
    fprintf('i = %d, mse = %f, y = %f*x + %f\n', i, t, th1, th0);
    th0_ = th0 - alpha .* sum(h(th0, th1, x) - y) ./ m; % dJ/dth0
    th1_ = th1 - alpha .* sum((h(th0, th1, x) - y).*x) ./ m; % dJ/dth1
    th0 = th0_;
    th1 = th1_;
    plot(x,h(th0, th1, x),'--g'); pause(0.01);
    i = i + 1;
    t = mse(h(th0, th1, x),y);
end
toc;
fprintf('y = %f*x + %f\nN of iterations = %d', th1, th0, i);


