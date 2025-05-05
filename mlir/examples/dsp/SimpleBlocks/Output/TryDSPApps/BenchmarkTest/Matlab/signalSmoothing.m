INPUT_LENGTH = 10;
fs = 8000;
dt = 1 / fs;
WINDOW_SIZE = 3;

input = getRangeOfVector(0, INPUT_LENGTH, dt);
getMultiplier = 2 * pi * 500;
getSinDuration = gain(input, getMultiplier);
clean_sig = sin(getSinDuration);

getNoiseSinDuration = gain(input, 2 * pi * 3000);
noise = sin(getNoiseSinDuration);
noise1 = gain(noise, 0.5);

noisy_sig = clean_sig + noise1;
median = slidingMedianFilter(noisy_sig);
average = slidingAvgFilter(median);

fprintf('%.6f\n', average(4));

function vector = getRangeOfVector(start, len, increment)
    vector = start + (0:len-1) * increment;
end

function output = gain(input, multiplier)
    output = input * multiplier;
end

function avg_out = slidingAvgFilter(input)
    len = length(input) - 2;
    avg_out = zeros(1, len);
    for i = 1:len
        avg_out(i) = mean(input(i:i+2));
    end
end

function med_out = slidingMedianFilter(input)
    len = length(input) - 2;
    med_out = zeros(1, len);
    for i = 1:len
        window = input(i:i+2);
        med_out(i) = median(window);
    end
end
