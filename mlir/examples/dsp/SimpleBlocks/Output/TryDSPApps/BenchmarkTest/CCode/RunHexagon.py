import subprocess
import os

# Ensure the log directory exists
log_dir = "ServerExeLogs"
os.makedirs(log_dir, exist_ok=True)

# Corrected list of application names (without non-application entries)
app_names = [
    "speakerIdentification",
    "targetDetection",
    "underWaterCommunication",
    "voiceActivityDetection",
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
    "spaceCommunication"
]

# Loop through each application and execute the  script
for app_name in app_names:
    app_script = f"{app_name}.c"
    log_file = os.path.join(log_dir, f"{app_name}.log")
    with open(log_file, "w") as log:
        command = ["python", "HexagonClangResultScript.py", app_script, app_name]
        print("")
        print(f"Running command: {' '.join(command)}")
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        for line in process.stderr:
            print(line, end="")
            log.write(line)
        process.wait()

