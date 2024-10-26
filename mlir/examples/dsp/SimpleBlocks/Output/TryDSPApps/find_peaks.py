def main() {

  var signal = [0.4, 0.3, 0.6, 1.8, 0.9, 0.5, 0.2, 0.7, 1.2, 0.8, 2.0, 1.9, 1.8, 1.7, 1.8, 1.7];
  var peaks = find_peaks(signal, 0.5, 1); 
  
  print(peaks);
  
}
