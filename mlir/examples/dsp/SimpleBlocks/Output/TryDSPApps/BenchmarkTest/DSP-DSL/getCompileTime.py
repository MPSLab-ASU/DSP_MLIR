import os
import subprocess
import time
import pandas as pd

# Path to the input file
input_files = [
    "spectralAnalysis",
    "audioCompression",
    "audioEqualization",
    "biomedicalSignalProcessing",
    "digitalModulation",
    "dtmfDetection",
    "echoCancellation",
    "FIRFilterDesign",
    "hearingAid",
    "lowPassFiltering",
    "noiseCancellation",
    "periodogram",
    "vibrationAnalysis",
    "radarSignalProcessing",
    "signalSmoothing",
    "spaceCommunication",
    "speakerIdentification",
    "targetDetection",
    "underWaterCommunication",
    "voiceActivityDetection"
]
data = []

for input_file_path in input_files:
    input_file_path +=".py"
    BasePathForLLVM = "/home/local/ASURITE/apkhedka/ForLLVM/"
    OutputScriptPath = (
        "mlir/examples/dsp/SimpleBlocks/Output/TryDSPApps/BenchmarkTest/DSP-DSL/"
        )

    try: 
        with open(input_file_path, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"File {input_file_path} not found")
        continue

    # Construct full output path
    OutputPath = os.path.join(BasePathForLLVM, OutputScriptPath, "Output")

    # Check if the Output folder exists, create it if it doesn't
    if not os.path.exists(OutputPath):
        os.makedirs(OutputPath)

    print(f"Running Application: {input_file_path}")
    print("InputPath: {}".format(BasePathForLLVM))
    print(f"OutputPath: {OutputPath}")

    # Input values dictionary
    inputValues = {
        "10": 10,
        "100": 100,
        "1K": 1000,
        "10K": 10000,
        "20K": 20000,
        "30K": 30000,
        "40K": 40000,
        "50K": 50000,
        "100K": 100000,
        "1M": 1000000,
        "10M": 10000000,
        "20M": 20000000,
        "30M": 30000000,
        "40M": 40000000,
        "50M": 50000000,
        "100M": 100000000,
    }

    NoOfIterations = 3

    # Base command
    commands_base = [
        f"{BasePathForLLVM}/build/bin/dsp1 {input_file_path} -emit=llvm",
    ]

    # Define the cases
    cases = [
        {
            "affineOpt": True,
            "canonOpt": False,
            "suffix": "fileAffineOpt.ll",
            "exe": "fileAffineOptExe",
        },
        {
            "affineOpt": True,
            "canonOpt": True,
            "suffix": "fileAffineCanonOpt.ll",
            "exe": "fileAffineCanonOptExe",
        },
    ]

    # Print header
    print("", end="\t")
    for case in cases:
        print(f"{case['suffix']}", end="\t")

    # Iterate over input sizes
    for key, value in inputValues.items():
        print(f"\n{key}", end="\t")

        # Update the input file with the current value
        with open(input_file_path, "r") as file:
            lines = file.readlines()

        with open(input_file_path, "w") as file:
            for line in lines:
                if line.strip().startswith("var input = getRangeOfVector("):
                    updated_line = (
                        f"\tvar input = getRangeOfVector(0, {value}, 0.000125);\n"
                    )
                    file.write(updated_line)
                else:
                    file.write(line)

        # Measure compiler time for each case
        for case in cases:
            command_llvm = commands_base[0]
            if case["affineOpt"]:
                command_llvm += " -affineOpt"
            if case["canonOpt"]:
                command_llvm += " -canonOpt"
            command_llvm += f" 2> {OutputPath}/{case['suffix']}"  # OutputPath

            sum_compile_time = 0
            for _ in range(NoOfIterations):
                # Record the start time
                start_time = time.time()

                # Execute the compilation command
                result = subprocess.run(command_llvm, shell=True, capture_output=True, text=True)

                # Record the end time
                end_time = time.time()

                # Calculate the elapsed time for compilation
                compile_time = end_time - start_time
                sum_compile_time += compile_time

                

            # Calculate average compile time
            avg_compile_time = sum_compile_time / NoOfIterations
            print(f"{avg_compile_time:.6f}", end="\t")
            data.append({
                    "filename": input_file_path,
                    "input size" : key,
                    # "opt": case['exe'],
                    f"{case['exe']}": avg_compile_time,
                })
        df = pd.DataFrame(data)
        
        df.to_csv("compile_time.csv", index=False)