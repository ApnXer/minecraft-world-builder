"""Build the v3 world: terrain -> town -> castle -> biome decoration. Saves build.npz + build_palette.json."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))

import json
import time

import numpy as np

import mc

t0 = time.time()


def step(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


import terrain  # noqa: E402

terrain.write()
step("terrain")
import city  # noqa: E402

n_houses = city.build()
step(f"town ({n_houses} houses)")
import castle  # noqa: E402

castle.build()
step("castle")
import biome  # noqa: E402

n_trees = biome.build()
step(f"biome ({n_trees} trees)")

import post  # noqa: E402

print("weather", post.weather(terrain.gh, terrain.R <= terrain.PLATEAU), flush=True)
step("weathering")
print("stairs", post.stair_shapes(), "connections", post.connections(), flush=True)
step("shapes + connections")

np.savez_compressed("build.npz", arr=mc.arr)
with open("build_palette.json", "w") as fh:
    json.dump([[n, p] for n, p in zip(mc.names, mc.props)], fh)
step(f"saved: palette {len(mc.names)}, blocks {int(np.count_nonzero(mc.arr))}")
