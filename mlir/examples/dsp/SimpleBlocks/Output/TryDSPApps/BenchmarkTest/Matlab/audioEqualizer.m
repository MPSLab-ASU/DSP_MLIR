% Constants
INPUT_LENGTH = 100000000;
pi = 3.14159265359;
fc = 300;
Fs = 8000;
gainForBass = 2;
gainForMid = 1.5;
gainForTreble = 0.8;
wc = 2 * pi * fc / Fs;
N = 101;

% Input signal
input = 0:(INPUT_LENGTH-1);

% Low-pass filter
lpf = lowPassFIRFilter(wc, N);
hamming_window = hamming(N)';
lpf_w = lpf .* hamming_window;
FIRfilterResponseForLpf = conv(input, lpf_w, 'same');
gainWithLpf = FIRfilterResponseForLpf * gainForBass;

% High-pass filter
fc2 = 1500;
wc2 = 2 * pi * fc2 / Fs;
hpf = highPassFIRFilter(wc2, N);
hpf_w = hpf .* hamming_window;
FIRfilterResponseForHpf = conv(input, hpf_w, 'same');
gainWithHpf = FIRfilterResponseForHpf * gainForTreble;

% Band-pass filter
lpf2 = lowPassFIRFilter(wc2, N);
lpf2_w = lpf2 .* hamming_window;
bpf_w = lpf2_w - lpf_w;
FIRfilterResponseForBpf = conv(input, bpf_w, 'same');
gainWithBpf = FIRfilterResponseForBpf * gainForMid;

% Final audio
final_audio = gainWithLpf + gainWithHpf + gainWithBpf;

% Print results
fprintf('Element at index 4: %f\n', final_audio(4));
disp(final_audio);

% Helper functions
function h = lowPassFIRFilter(wc, length)
    n = 0:(length-1);
    mid = (length - 1) / 2;
    h = zeros(1, length);
    h(n ~= mid) = sin(wc * (n(n ~= mid) - mid)) ./ (pi * (n(n ~= mid) - mid));
    h(mid+1) = wc / pi;
end

function h = highPassFIRFilter(wc, length)
    lpf = lowPassFIRFilter(wc, length);
    h = -lpf;
    h((length+1)/2) = h((length+1)/2) + 1;
end