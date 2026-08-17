import json
import os

filepath = "/home/jetson/jetracer_official/notebooks/road_following_live.ipynb"
with open(filepath, "r") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "camera.close" in source:
            cell["source"] = ["# Tắt camera an toàn\n", "camera.running = False\n", "camera.unobserve_all()\n"]
            print("Fixed camera.close in road_following_live.ipynb")

with open(filepath, "w") as f:
    json.dump(nb, f, indent=1)
