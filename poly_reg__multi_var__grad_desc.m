% Linear Regression
% One Variable
% Gradient Descent

cc;

%% Generate training data
m = 20;
x = 0:19;

% k = 0.7;
% b = 1.5;
% y = k.*x + b;

a1 = 0.1;
a2 = -2;
a3 = 1;
y = a1.*x.^2 + a2.*x + a3;

figure(); plot(x,y,'-r'); hold on;

noise = normrnd(1.5, 1, 1, 20) - 1.5;
noisy_y = y + noise;
plot(x,noisy_y,'*b');

% Training set
y = noisy_y;

% x = [1, 2, 4, 0];
% y = [0.5, 1, 2, 0];
% m = 4;

figure(); plot(x,y,'*b'); hold on;

%% Hypothesis function
h = @(th0, th1, th2, x)(th0 + th1.*x + th2.*x.^2);

%% Cost function (MSE)
mse = @(h, y)(sum((h-y).^2) / (2*m));

%% Initial parameters
th0 = 2;
th1 = -3;
th2 = 0.1;
thresh = 0.01;
alpha = 0.00001;

%% Gradient descent method
i = 0;
t0 = mse(h(th0, th1, th2, x),y);
t = t0;
plot(x,h(th0, th1, th2, x),'--g'); pause(0.01);
tic;
while t > thresh && i < 1000
    fprintf('i = %d, mse = %f, y = %f*x^2 + %f*x + %f\n', i, t, th2, th1, th0);
    th0_ = th0 - alpha .* sum(h(th0, th1, th2, x) - y) ./ m; % dJ/dth0
    th1_ = th1 - alpha .* sum((h(th0, th1, th2, x) - y).*x) ./ m; % dJ/dth1
    th2_ = th2 - alpha .* sum((h(th0, th1, th2, x) - y).*x.^2) ./ m; % dJ/dth1
    plot(x,h(th0_, th1_, th2_, x),'--g'); pause(0.01);
    t = mse(h(th0_, th1_, th2_, x),y);
    if t > t0
       alpha = alpha / 10;
       continue;
    end
    th0 = th0_;
    th1 = th1_;
    th2 = th2_;
    i = i + 1;
end
toc;
fprintf('y = %f*x^2 + %f*x + %f\nN of iterations = %d\n', th2, th1, th0, i);


