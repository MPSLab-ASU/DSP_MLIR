
def main() {
  # var a = [10,20,30];
  
  #size = 10
#   var input<2,5> = [10,20,30,40,50,60, 70, 80, 90 , 100];
#   var filter<2,5> = [10,20,30,40,50,60, 70, 80, 90 , 100];
  # var input = [10,20,30,40,50,60, 70, 80, 90 , 100];
  # var filter = [10,20,30,40,50,60, 70, 80, 90 , 100];

  var input =  [10,20,30]; # [1,2];
  var filter = [40,50,60]; # [1,3];

  # var filter =  [10,20,30]; # [1,2];
  # var input = [40,50,60]; # [1,3];

  var output = FIRFilter(input , filter);
  print(output); 
}