def main() {
  var x = [[1.0, 2.0], [4.0, 5.0]];
  var y = [[1.0, 2.0], [4.0, 5.0]];
  var z = matmul(x, y);
  print(z);
  
  
  var x2 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
  var y2 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];  
  var z2 = matmul(x2, y2);
  print(z2);  
}
