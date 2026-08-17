import json
import os

NOTEBOOKS = [
    "interactive_regression.ipynb",
    "road_following.ipynb",
    "road_following_live.ipynb"
]

ROBUST_CAMERA_CODE = [
    "import os\n",
    "import time\n",
    "# Restart the camera service to avoid freezing issues\n",
    "os.system('echo \"jetson\" | sudo -S systemctl restart nvargus-daemon')\n",
    "time.sleep(2)\n",
    "\n",
    "from jetcam.csi_camera import CSICamera\n",
    "# from jetcam.usb_camera import USBCamera\n",
    "\n",
    "# Stop any existing camera instance to prevent crashes when rerunning this cell\n",
    "try:\n",
    "    if 'camera' in globals():\n",
    "        camera.running = False\n",
    "        camera.unobserve_all()\n",
    "except:\n",
    "    pass\n",
    "\n",
    "camera = CSICamera(width=224, height=224, capture_fps=65)\n",
    "# camera = USBCamera(width=224, height=224, capture_width=640, capture_height=480, capture_device=0)\n"
]

for nb_file in NOTEBOOKS:
    filepath = os.path.join("/home/jetson/jetracer_official/notebooks", nb_file)
    if not os.path.exists(filepath):
        continue
        
    with open(filepath, "r") as f:
        nb = json.load(f)
        
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "from jetcam.csi_camera import CSICamera" in source:
                # Add camera.running = True if it was in the original cell
                new_source = list(ROBUST_CAMERA_CODE)
                if "camera.running = True" in source:
                    new_source.append("\ncamera.running = True")
                
                cell["source"] = new_source
                print(f"Updated {nb_file}")
                
    with open(filepath, "w") as f:
        json.dump(nb, f, indent=1)
        
print("Done fixing notebooks!")
