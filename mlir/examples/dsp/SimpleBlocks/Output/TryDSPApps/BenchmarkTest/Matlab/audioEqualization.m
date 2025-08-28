INPUT_LENGTH = 100;
FILTER_LENGTH = 101;
input = 0:INPUT_LENGTH-1;

pi = 3.14159265359;
Fs = 8000;
gainBass = 2;
gainMid = 0.8;
gainTreble = 0.8;

mid = (FILTER_LENGTH - 1) / 2;

hamming_window = hamming(FILTER_LENGTH);

wc1 = 2 * pi * 300 / Fs;
lpf = lowPassFIRFilter(wc1, FILTER_LENGTH);
lpf_w = elementWiseMultiply(lpf, hamming_window);

wc2 = 2 * pi * 1500 / Fs;
lpf2 = lowPassFIRFilter(wc2, FILTER_LENGTH);
hpf = -lpf2;
hpf(mid+1) = hpf(mid+1) + 1;
hpf_w = elementWiseMultiply(hpf, hamming_window);

bpf_w = elementWiseSubtract(elementWiseMultiply(lpf2, hamming_window), lpf_w);

resp_lpf = conv(input, lpf_w);
resp_hpf = conv(input, hpf_w);
resp_bpf = conv(input, bpf_w);

gain_lpf = applyGain(resp_lpf, gainBass);
gain_hpf = applyGain(resp_hpf, gainTreble);
gain_bpf = applyGain(resp_bpf, gainMid);

final_audio = gain_lpf + gain_hpf + gain_bpf;

fprintf('%.6f\n', final_audio(4));

function window = hamming(length)
    n = 0:length-1;
    window = 0.54 - 0.46 * cos(2 * pi * n / (length - 1));
end

function filter = lowPassFIRFilter(wc, length)
    mid = (length - 1) / 2;
    filter = zeros(1, length);
    for n = 0:length-1
        if n == mid
            filter(n+1) = wc / pi;
        else
            filter(n+1) = sin(wc * (n - mid)) / (pi * (n - mid));
        end
    end
end

function output = elementWiseMultiply(array1, array2)
    output = array1 .* array2;
end

function output = elementWiseSubtract(array1, array2)
    output = array1 - array2;
end

function output = applyGain(input, gainFactor)
    output = input * gainFactor;
end
