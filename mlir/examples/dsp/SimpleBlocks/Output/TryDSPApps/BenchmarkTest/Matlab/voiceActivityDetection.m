% Constants
INPUT_LENGTH = 100000000;

% Main script
fs = 1000;
input = getRangeOfVector(0, INPUT_LENGTH, 1);

getMultiplier = 2 * pi * 5;
getSinDuration = gain(input, getMultiplier);

signal = sine(getSinDuration);

noise = delay(signal, 5);

noisy_sig = add(signal, noise);

threshold_value = 0.8;
GetThresholdReal = threshold(noisy_sig, threshold_value);

zcr = zeroCrossCount(GetThresholdReal);

% Display results
disp(GetThresholdReal(4));

% Print zero-crossing count
fprintf('Zero-crossing count: %d\n', zcr);

% Function implementations
function vector = getRangeOfVector(start, length, increment)
    vector = (start : increment : start + (length-1)*increment)';
end

function output = gain(input, multiplier)
    output = input * multiplier;
end

function output = sine(input)
    output = sin(input);
end

function output = delay(input, delaySamples)
    output = [zeros(delaySamples, 1); input(1:end-delaySamples)];
end

function output = add(input1, input2)
    output = input1 + input2;
end

function output = threshold(input, thresholdValue)
    output = input;
    output(abs(input) < thresholdValue) = 0;
end

function count = zeroCrossCount(input)
    signs = sign(input);
    count = sum(abs(diff(signs)) == 2);
end