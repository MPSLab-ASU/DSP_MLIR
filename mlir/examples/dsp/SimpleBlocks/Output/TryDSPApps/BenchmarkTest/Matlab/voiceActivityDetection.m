function main
    PI = 3.14159265359;
    SAMPLE_RATE = 1000;
    INPUT_LENGTH = 100;
    THRESHOLD = 0.01;

    input = (0:INPUT_LENGTH-1)' * 0.0125;

    getMultiplier = 2 * PI * 5;
    getSinDuration = input * getMultiplier;
    signal = sin(getSinDuration);

    delay_steps = 5;
    noise = [zeros(delay_steps,1); signal(1:end-delay_steps)];
    noisy_sig = signal + noise;

    GetThresholdReal = applyThreshold(noisy_sig, THRESHOLD);
    zcr = zeroCrossCount(GetThresholdReal);

    fprintf('%.6f\n', zcr);
end

function output = applyThreshold(input, threshold)
    output = input;
    output(abs(input) <= threshold) = 0;
end

function count = zeroCrossCount(signal)
    count = 0;
    for i = 2:length(signal)
        if (signal(i-1) > 0 && signal(i) < 0) || ...
           (signal(i-1) < 0 && signal(i) > 0)
            count = count + 1;
        end
    end
end
