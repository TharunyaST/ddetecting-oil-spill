import subprocess

# Step 1: Run model.py
try:
    print("Running model.py...")
    subprocess.run(["python", "intergration\model.py"], check=True)
    print("Completed model.py.\n")
except subprocess.CalledProcessError as e:
    print(f"Error while running model.py: {e}")
    exit(1)

# Step 2: Run path.py
try:
    print("Running path.py...")
    subprocess.run(["python", "intergration\path.py"], check=True)
    print("Completed path.py.\n")
except subprocess.CalledProcessError as e:
    print(f"Error while running path.py: {e}")
    exit(1)

# Step 3: Run Researchpaper.py
try:
    print("Running Researchpaper.py...")
    subprocess.run(["python", "intergration\Researchpaper.py"], check=True)
    print("Completed Researchpaper.py.\n")
except subprocess.CalledProcessError as e:
    print(f"Error while running Researchpaper.py: {e}")
    exit(1)

try:
    print("Running ais_collision.py...")
    subprocess.run(["python", r"G:\SIH_FINAL_AIS_SATE\intergration\ais_collision.py"], check=True)
    print("Completed ais_collision.py.\n")
except subprocess.CalledProcessError as e:
    print(f"Error while running ais_collision.py: {e}")
    exit(1)

try:
    print("Running merge.py...")
    subprocess.run(["python", "intergration\merge.py"], check=True)
    print("Completed merge.py.\n")
except subprocess.CalledProcessError as e:
    print(f"Error while running merge.py: {e}")
    exit(1)

# try:
#     print("Running test.py...")
#     subprocess.run(["python", r"G:\SIH_FINAL_AIS_SATE\intergration\test.py"], check=True)
#     print("Completed test.py.\n")
# except subprocess.CalledProcessError as e:
#     print(f"Error while running test.py: {e}")
#     exit(1)

try:
    print("Running satellite.py...")
    subprocess.run(["python", r"G:\SIH_FINAL_AIS_SATE\Final_satellite_intergration - Copy\ais_sat_intergration.py"], check=True)
    print("Completed satellite.py.\n")
except subprocess.CalledProcessError as e:
    print(f"Error while running satellite.py: {e}")
    exit(1)

print("Workflow completed successfully!")
