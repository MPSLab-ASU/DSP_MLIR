INPUT_LENGTH = 40000;
fs = 8000;
dt = 1 / fs;

input = getRangeOfVector(0, INPUT_LENGTH, dt);
p = 2 * pi;
f_sig = 500;
getMultiplier = p * f_sig;
thresh = 0.4;

getSinDuration = gain(input, getMultiplier);
clean_sig = sin(getSinDuration);
binary_sig = thresholdUp(clean_sig, thresh, 0);
a = spaceModulate(binary_sig);
noisy_signal = addNoise(a);
b = spaceDemodulate(noisy_signal);
e = errorCorrection(b);

fprintf('%.6f\n', e(9));

function vec = getRangeOfVector(start, len, step)
    vec = start + (0:len-1) * step;
end

function out = gain(input, factor)
    out = input * factor;
end

function out = thresholdUp(input, threshold, returnOriginal)
    if returnOriginal == 0
        out = double(input >= threshold);
    else
        out = input;
        out(input < threshold) = 0;
    end
end

function out = spaceModulate(input)
    out = ones(1, length(input));
    out(input ~= 1) = -1;
end

function out = addNoise(input)
    out = input + sin(input);
end

function out = spaceDemodulate(input)
    out = double(input > 0);
end

function corrected = errorCorrection(data)
    corrected = zeros(1, length(data));
    idx = 1;
    for i = 1:8:length(data)
        chunk = data(i:min(i+7, end));
        if mod(sum(chunk), 2) == 0
            corrected(idx:idx+length(chunk)-1) = chunk;
        else
            chunk(1) = 0;
            corrected(idx:idx+length(chunk)-1) = chunk;
        end
        idx = idx + 8;
    end
end
