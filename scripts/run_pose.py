import os
import subprocess

# --- 1. DEFINE ALL YOUR PATHS (NOW RELATIVE) ---
openpose_root = r"../openpose"

openpose_exe = os.path.join(openpose_root, "bin/OpenPoseDemo.exe")

openpose_model_folder = os.path.join(openpose_root, "models/")

input_images = r"../our dataset/images-resized"

output_json = r"../our dataset/openpose-json/"
output_images = r"../our dataset/openpose-img/"
os.makedirs(output_json, exist_ok=True)
os.makedirs(output_images, exist_ok=True)

print(f"Starting OpenPose...")
print(f"Input folder: {input_images}")
print(f"JSON output: {output_json}")

# --- 3. BUILD THE COMMAND ---
# We've added the --model_folder argument
command = [
    openpose_exe,
    "--image_dir", input_images,
    "--write_json", output_json,
    "--write_images", output_images,
    "--model_folder", openpose_model_folder,  # <-- HERE IS THE FIX
    "--display", "0"  # Disables the pop-up GUI window
]

# --- 4. RUN THE COMMAND ---
try:
    print("\nRunning command:")
    print(" ".join(command))  # Print the command so you can see it

    subprocess.run(command, check=True)

    print("\n--- OpenPose processing finished successfully! ---")

except subprocess.CalledProcessError as e:
    print(f"\n--- ERROR: OpenPose failed to run. ---")
    print(f"Return code: {e.returncode}")
    print(f"Output: {e.stdout}")
    print(f"Error: {e.stderr}")
except FileNotFoundError:
    print(f"\n--- ERROR: Could not find 'OpenPoseDemo.exe' ---")
    print(f"Please check your 'openpose_exe' path: {openpose_exe}")
    print("Ensure the script is in your main 'vton-project' folder.")
