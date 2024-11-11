def main() {
  #generate_voice_signature

  var person1 = [2.0, -1.0, 3.2, -2.4, 1.5, 1.0, 3.0];
  var person2 = [-2.1, -4.0, 1.0, -1.4, -2.5, 0.0, 0.1];
  var person3 = [-1.0, 1.0, 1.0, -1.0, 5.0, -1.0, -2.0];
  
  #var signature = [2.0, -1.0, 3.2,-2.4, 1.5, 1.0, 3.0]; # person 1
  #var signature = [-2.1, -4.0, 1.0, -1.4, -2.5, 0.0, 0.1]; # person 2
  var signature = [-1.0, 1.0, 1.0, -1.0, 5.0, -1.0, -2.0]; # person 3
  
  var max1 = max(correlate(person1, signature));
  var max2 = max(correlate(person2, signature));
  var max3 = max(correlate(person3, signature));
  
  var total_maxes = [0, 0, 0];

  #var temp2 = setElemAtIndx(total_maxes, 0, max1); #not work
  var temp2 = setSingleElemAtIndx(total_maxes, 0, max1); #work
  var temp3 = setSingleElemAtIndx(total_maxes, 1, max2); #work
  var temp4 = setSingleElemAtIndx(total_maxes, 2, max3); #work
  
  var max_index = argmax(total_maxes);
  
  var max_value = getSingleElemAtIndx(total_maxes, max_index);
  
  print(max_index);
  print(temp2);
  print(max_value);
  print(temp3);
  print(total_maxes);
  print(temp4);
}
 
