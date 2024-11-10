def main() {
    # var voice_signature = generateVoiceSignature(150, 250, 5, 10);
    # print(voice_signature);
    var speaker_list = [
        [1,2,3,4,5,6,7,8,9,10],
        [11,12,13,14,15,16,17,18,19,20],
        [21,22,23,24,25,26,27,28,29,30]
    ];
    var speaker = identifySpeaker([1,2,3,4,5,6,7,8,9,10], speaker_list);
    print(speaker);
}
