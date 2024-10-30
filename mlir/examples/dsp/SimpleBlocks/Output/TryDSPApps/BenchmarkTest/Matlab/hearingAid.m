% Define constants
INPUT_LENGTH = 100000000;
fs = 8000;
step = 1 / fs;

% Generate input range
input = (0:step:(INPUT_LENGTH-1)*step)';

% Generate clean signal
f_sig = 500;
clean_sig = sin(2 * pi * f_sig * input);

% Generate noise signal with frequency of 3000 Hz
f_noise = 3000;
noise = 0.5 * sin(2 * pi * f_noise * input);

% Create noisy signal by adding noise to clean signal
noisy_sig = clean_sig + noise;

% LMS filter response function
function y = lmsFilterResponse(noisy_sig, clean_sig, mu, filterSize)
    w = zeros(filterSize, 1);
    y = zeros(size(noisy_sig));
    
    for n = 1:length(noisy_sig)
        x = noisy_sig(max(1, n-filterSize+1):n);
        x = [zeros(filterSize - length(x), 1); x];
        y(n) = w' * x;
        e = clean_sig(n) - y(n);
        w = w + mu * e * x;
        y(n) = e;
    end
end

% Apply LMS filter
mu = 0.01;
filterSize = 32;
y = lmsFilterResponse(noisy_sig, clean_sig, mu, filterSize);

% Apply final gain factor G1 to the LMS filter output
G1 = 1002300;
sol = G1 * y;

% Display 
disp(sol);