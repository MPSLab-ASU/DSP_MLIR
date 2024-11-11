def main() {
  var sample_rate = 1000;
	var duration = 1220.703125;
  var person1 = generateVoiceSignature(100, 200, duration, sample_rate); #Alice
  var person2 = generateVoiceSignature(150, 250, duration, sample_rate); #Bob
  var person3 = generateVoiceSignature(120, 180, duration, sample_rate); #Charlie
  
  # var unknown_signal = generateVoiceSignature(100, 200, duration, sample_rate);
  var unknown_signal = generateVoiceSignature(150, 250, duration, sample_rate);
  # var unknown_signal = generateVoiceSignature(120, 180, duration, sample_rate);
  
  var max1 = max(correlate(person1, unknown_signal));
  var max2 = max(correlate(person2, unknown_signal));
  var max3 = max(correlate(person3, unknown_signal));
  
  var total_maxes = [0, 0, 0];

  var temp2 = setSingleElemAtIndx(total_maxes, 0, max1); #work
  var temp3 = setSingleElemAtIndx(total_maxes, 1, max2); #work
  var temp4 = setSingleElemAtIndx(total_maxes, 2, max3); #work
  
  var max_index = argmax(total_maxes,0);
  
  var max_value = getSingleElemAtIndx(total_maxes, max_index);
  
  print(max_index);
  print(temp2);
  print(max_value);
  print(temp3);
  print(total_maxes);
  print(temp4);
}
 
