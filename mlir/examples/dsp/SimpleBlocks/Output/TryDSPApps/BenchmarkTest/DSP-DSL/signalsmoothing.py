def main() {
    var input = getRangeOfVector(0, 20, 1);
    var average = slidingWindowAvg(input);
    var median = slidingWindowAvg(average);
    print(median);
}