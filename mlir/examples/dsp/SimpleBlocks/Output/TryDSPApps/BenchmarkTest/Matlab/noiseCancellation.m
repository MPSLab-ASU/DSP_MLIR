INPUT_LENGTH = 1000;
fs = 8000;

t = getRangeOfVector(0, INPUT_LENGTH, 0.000125);

f_sig = 500;
getSinDuration = gain(t, 2 * pi * f_sig);
clean_sig = sin(getSinDuration);

f_noise = 3000;
getNoiseSinDuration = gain(t, 2 * pi * f_noise);
noise = sin(getNoiseSinDuration);
noise1 = gain(noise, 0.5);

noisy_sig = clean_sig + noise1;

y = lmsFilterResponse(noisy_sig, clean_sig, 0.01, 32);
sol = gain(y, 10);
normalized_sol = normalizeSignal(sol);

fprintf('%.6f\n', normalized_sol(6));

function vector = getRangeOfVector(start, length, increment)
    vector = start + (0:length-1) * increment;
end

function output = gain(input, multiplier)
    output = input * multiplier;
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
