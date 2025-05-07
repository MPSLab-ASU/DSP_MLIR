FS = 8000;
INPUT_LENGTH = 2000;
FILTER_SIZE = 101;
MAX_PEAKS = 950;

input = (0:INPUT_LENGTH-1) * 0.000125;

f_sig = 500;
clean_sig = sin(2 * pi * f_sig * input);

f_noise = 3000;
noise = 0.5 * sin(2 * pi * f_noise * input);

noisy_sig = clean_sig + noise;

fc1 = 1000;
fc2 = 7500;

wc1 = 2 * pi * fc1 / FS;
wc2 = 2 * pi * fc2 / FS;

hamming_window = hamming(FILTER_SIZE);

lpf1 = lowPassFIRFilter(wc1, FILTER_SIZE);
lpf2 = lowPassFIRFilter(wc2, FILTER_SIZE);

lpf1_w = elementWiseMultiply(lpf1, hamming_window);
lpf2_w = elementWiseMultiply(lpf2, hamming_window);

bpf_w = elementWiseSubtract(lpf2_w, lpf1_w);

FIRfilterResponseForBpf = conv(noisy_sig, bpf_w);

height = 0.3 * max(FIRfilterResponseForBpf);
r_peaks = findPeaks(FIRfilterResponseForBpf, height, 950, MAX_PEAKS);

len_r_peaks = r_peaks(end);
valid_peaks = r_peaks(1:len_r_peaks);

peak_intervals = diff(valid_peaks);
diff_mean = mean(peak_intervals);

avg_hr = (60 * FS) / diff_mean;

fprintf('%.6f\n', avg_hr);

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

function r_peaks = findPeaks(signal, threshold, min_distance, max_peaks)
    r_peaks = -ones(1, max_peaks);
    count = 0;
    for i = 2:length(signal)-1
        if signal(i) > signal(i-1) && signal(i) > signal(i+1) && signal(i) >= threshold
            if count == 0 || (i - r_peaks(count)) >= min_distance
                count = count + 1;
                r_peaks(count) = i;
                if count >= max_peaks - 1
                    break;
                end
            end
        end
    end
    r_peaks(max_peaks) = count;
end