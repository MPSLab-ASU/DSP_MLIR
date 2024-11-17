% Constants
INPUT_LENGTH = 100000000;

% Main script
t = linspace(0, INPUT_LENGTH * 0.000125, INPUT_LENGTH);

f_sig = 500;
clean_sig = sin(2 * pi * f_sig * t);

f_noise = 3000;
noise = 0.5 * sin(2 * pi * f_noise * t);

noisy_sig = clean_sig + noise;

% LMS filter response
mu = 0.01;
filterSize = 32;

% Preallocate arrays
w = zeros(1, filterSize);
y = zeros(1, INPUT_LENGTH);

% Implement LMS filter
for n = filterSize:INPUT_LENGTH
    x = noisy_sig(n:-1:n-filterSize+1);
    y(n) = w * x';
    e = clean_sig(n) - y(n);
    w = w + mu * e * x;
end

sol = 10 * y;
fprintf('%f\n', sol);
