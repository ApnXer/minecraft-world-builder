"""Render review views of build.npz into renders/. Usage: python views.py [name ...]"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))

import json
import sys
import time

import numpy as np
from PIL import Image

from render import Renderer

C = 192
P, CP = 40, 72
VIEWS = {
    "overview": dict(kind="ortho", target=(C, 70, C), yaw=215, pitch=33, width=1600, height=1000, ppb=3.0),
    "castle": dict(kind="ortho", target=(C - 5, CP + 30, C), yaw=230, pitch=22, width=1500, height=1000, ppb=7.5),
    "castle_east": dict(kind="persp", eye=(C + 95, P + 22, C + 30), look=(C + 10, CP + 25, C - 5), fov=72),
    "market": dict(kind="persp", eye=(C + 40, P + 2.6, C + 99), look=(C + 66, P + 4, C + 78), fov=80),
    "street": dict(kind="persp", eye=(C - 2, P + 3, C + 100), look=(C, P + 14, C + 40), fov=80),
    "courtyard": dict(kind="persp", eye=(C + 28, CP + 4, C + 6), look=(C - 20, CP + 14, C - 4), fov=80),
    "throne": dict(kind="persp", eye=(C - 38, CP + 31, C - 15), look=(C - 38, CP + 28, C + 19), fov=85),
    "forest": dict(kind="persp", eye=(C + 140, 46, C + 40), look=(C + 120, 46, C - 60), fov=78),
    "mother": dict(kind="persp", eye=(C + 118, 38, C - 40), look=(C + 150, 52, C - 62), fov=80),
    "houses": dict(kind="ortho", target=(C - 60, P + 8, C + 75), yaw=200, pitch=30, width=1400, height=900, ppb=8),
    "north_street": dict(kind="persp", eye=(C + 1, P + 2.6, C - 100), look=(C + 1, P + 8, C - 60), fov=78),
    "east_street": dict(kind="persp", eye=(C + 100, P + 2.6, C + 1), look=(C + 60, P + 7, C + 1), fov=78),
    "town_close": dict(kind="ortho", target=(C + 40, P + 8, C - 75), yaw=200, pitch=32, width=1400, height=900, ppb=11),
    "town_close2": dict(kind="ortho", target=(C - 70, P + 8, C - 40), yaw=30, pitch=30, width=1400, height=900, ppb=11),
    "underside": dict(kind="persp", eye=(C + 230, 20, C + 60), look=(C, 20, C), fov=70),
}


def main(names):
    os.makedirs("renders", exist_ok=True)
    arr = np.load("build.npz")["arr"]
    pal = [(n, p) for n, p in json.load(open("build_palette.json"))]
    r = Renderer(pal)
    for name in names:
        v = dict(VIEWS[name])
        t = time.time()
        if v.pop("kind") == "ortho":
            img = r.ortho(arr, **v)
        else:
            img = r.persp(arr, width=1400, height=880, **v)
        Image.fromarray(img).save(f"renders/{name}.png")
        print(name, round(time.time() - t, 1), "s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(VIEWS))
