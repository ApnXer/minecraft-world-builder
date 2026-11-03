"""Write blocks_<version>.txt (all block ids of the jar, one per line) next to this script."""
import os
import sys
import zipfile

from mcpaths import find_jar

jar = find_jar()
z = zipfile.ZipFile(jar)
ids = sorted(n.split("/")[-1][:-5] for n in z.namelist() if n.startswith("assets/minecraft/blockstates/"))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blocks_1_21_11.txt")
open(out, "w").write("\n".join(ids))
print(len(ids), "blocks ->", out, file=sys.stderr)
