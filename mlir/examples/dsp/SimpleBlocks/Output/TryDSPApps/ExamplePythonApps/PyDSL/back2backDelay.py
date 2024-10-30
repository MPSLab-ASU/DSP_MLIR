
#fir filter design

# give some python code using similar api -- this is for audio compression and also 
def main() {
    
    # var a10 = [ 3.2, 1.5,  0.8, 2.9,  4.5,10 , 0,5,5.5, 1.1];
    # var a10 = getRangeOfVector(3.2, 10, 1);
    var input = getRangeOfVector(0, 1000000, 10);
    var b0 = 2;
    var b1 = 2;
    var b2 = 2;
    var b3 = 2;
    var b4 = 2;
    var b5 = 4;
    var b6 = 4;
    var b7 = 4;
    var b8 = 4;
    var b9 = 4;

    var out1 = delay(input, b0);
    var out2 = delay(out1, b1);
    var out3 = delay(out2, b2);
    var out4 = delay(out3, b3);
    var out5 = delay(out4, b4);
    var out6 = delay(out5, b5);
    var out7 = delay(out6, b6);
    var out8 = delay(out7, b7);
    var out9 = delay(out8, b8);
    var out10 = delay(out9, b9);
    var out11 = delay(out10, b0);
    var out12 = delay(out11, b1);
    var out13 = delay(out12, b2);
    var out14 = delay(out13, b3);
    var out15 = delay(out14, b4);
    var out16 = delay(out15, b5);
    var out17 = delay(out16, b6);
    var out18 = delay(out17, b7);
    var out19 = delay(out18, b8);
    var out20 = delay(out19, b9);
    print(out20);
}

