function main
    PI = 3.14159265359;
    INPUT_LENGTH = 100;

    fs = 1000;
    input = getRangeOfVector(0, INPUT_LENGTH, 0.000125);

    getMultiplier = 2 * PI * 50;
    getSinDuration = input * getMultiplier;

    sig1 = sin(getSinDuration);

    getMultiplier2 = 2 * PI * 120;
    getSinDuration2 = input * getMultiplier2;

    sinsig2 = sin(getSinDuration2);
    sig2 = sinsig2 * 0.5;

    signal = sig1 + sig2;

    noise = delay(signal, 5);

    noisy_sig = signal + noise;

    threshold_value = 2;

    dft_output = fft(noisy_sig);

    fft_real = real(dft_output);
    fft_img = imag(dft_output);

    sq_abs = fft_real.^2 + fft_img.^2;
    magnitude = sqrt(sq_abs);
    GetThresholdReal = threshold(magnitude, threshold_value);

    disp(GetThresholdReal(1));
end

function vector = getRangeOfVector(start, len, increment)
    vector = start + (0:len-1) * increment;
end

function output = delay(input, delaySamples)
    output = zeros(size(input));
    output(delaySamples+1:end) = input(1:end-delaySamples);
end

function output = threshold(input, thresholdValue)
    output = input;
    output(input < thresholdValue) = 0;
end
