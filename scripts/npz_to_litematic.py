"""Convert build.npz + build_palette.json to a Litematica schematic (.litematic).

    python npz_to_litematic.py --npz build.npz --palette build_palette.json --name "My Build" --out my_build.litematic
Note: schematics cannot store biomes; use export_world.py for the custom-biome datapack.
"""
import argparse
import json

import numpy as np

from litematic_writer import write_litematic

ap = argparse.ArgumentParser()
ap.add_argument("--npz", default="build.npz")
ap.add_argument("--palette", default="build_palette.json")
ap.add_argument("--name", default="Generated Build")
ap.add_argument("--out", default="build.litematic")
a = ap.parse_args()
arr = np.load(a.npz)["arr"]
pal = [(n, p) for n, p in json.load(open(a.palette))]
print(write_litematic(a.out, arr, pal, a.name, description="generated with minecraft-world-builder", data_version=4671))
