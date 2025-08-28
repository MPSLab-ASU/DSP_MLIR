SAMPLE_RATE = 1000;
INPUT_LENGTH = 61;
DURATION = INPUT_LENGTH / SAMPLE_RATE;
CORRELATION_LENGTH = 2 * INPUT_LENGTH - 1;

person1 = generateVoiceSignature(100, 200, INPUT_LENGTH, SAMPLE_RATE);
person2 = generateVoiceSignature(150, 250, INPUT_LENGTH, SAMPLE_RATE);
person3 = generateVoiceSignature(120, 180, INPUT_LENGTH, SAMPLE_RATE);
unknown_signal = generateVoiceSignature(150, 250, INPUT_LENGTH, SAMPLE_RATE);

correlation1 = correlate(person1, unknown_signal);
correlation2 = correlate(person2, unknown_signal);
correlation3 = correlate(person3, unknown_signal);

total_maxes = [max(correlation1), max(correlation2), max(correlation3)];

[max_value, max_index] = max(total_maxes);

temp2 = total_maxes(1);
temp3 = total_maxes(2);
temp4 = total_maxes(3);

fprintf('%d\t', max_index - 1);
fprintf('%.6f\t', temp2);
fprintf('%.6f\t', max_value);
fprintf('%.6f\t', temp3);
fprintf('%.6f %.6f %.6f\t', total_maxes);
fprintf('%.6f\n', temp4);

function signal = generateVoiceSignature(freq1, freq2, len, fs)
    t = (0:len-1) / fs;
    signal = sin(2 * pi * freq1 * t) + sin(2 * pi * freq2 * t);
end

function result = correlate(signal1, signal2)
    len = length(signal1);
    corr_len = 2 * len - 1;
    result = zeros(1, corr_len);
    for lag = 0:corr_len-1
        for i = 1:len
            j = lag - len + i;
            if j >= 1 && j <= len
                result(lag+1) = result(lag+1) + signal1(i) * signal2(j);
            end
        end
    end
end
