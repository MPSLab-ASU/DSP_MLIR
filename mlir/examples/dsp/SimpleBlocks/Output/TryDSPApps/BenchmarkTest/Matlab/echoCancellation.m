INPUT_LENGTH = 1000;
fs = 8000;
step = 1 / fs;
t = (0:INPUT_LENGTH-1) * step;

f_sig = 500;
getSinDuration = gain(t, 2 * pi * f_sig);
clean_sig = sin(getSinDuration);

noise = delaySignal(clean_sig, 2);
noisy_sig = clean_sig + noise;

mu = 0.01;
filterSize = 32;
y = lmsFilterResponse(noisy_sig, clean_sig, mu, filterSize);

normalized_sol = normalizeSignal(y);

fprintf('%.6f\n', normalized_sol(6));

function output = gain(input, multiplier)
    output = input * multiplier;
end

function output = delaySignal(input, delaySamples)
    output = zeros(size(input));
    output((delaySamples+1):end) = input(1:end-delaySamples);
end

function y = lmsFilterResponse(noisy_sig, clean_sig, mu, filterSize)
    N = length(noisy_sig);
    y = zeros(1, N);
    w = zeros(1, filterSize);
    for n = 1:N
        y_n = 0;
        for i = 1:filterSize
            if n - i + 1 > 0
                y_n = y_n + w(i) * noisy_sig(n - i + 1);
            end
        end
        e = clean_sig(n) - y_n;
        for i = 1:filterSize
            if n - i + 1 > 0
                w(i) = w(i) + mu * e * noisy_sig(n - i + 1);
            end
        end
        y(n) = y_n;
    end
end

function output = normalizeSignal(input)
    min_val = min(input);
    max_val = max(input);
    output = (input - min_val) / (max_val - min_val);
end
