"""Validate build_palette.json block names / property names / values against the 1.21.11 jar."""
import json, os, zipfile
from mcpaths import find_jar
z = zipfile.ZipFile(find_jar())
known = {}
for nm in z.namelist():
    if nm.startswith("assets/minecraft/blockstates/"):
        j = json.loads(z.read(nm)); b = "minecraft:" + nm.split("/")[-1][:-5]; ps = {}
        for v in j.get("variants", {}):
            for kv in filter(None, v.split(",")):
                a, c = kv.split("="); ps.setdefault(a, set()).add(c)
        def walk(w):
            if isinstance(w, dict):
                for a, c in w.items():
                    if a in ("OR", "AND"): [walk(x) for x in c]
                    else: ps.setdefault(a, set()).update(str(c).split("|"))
        for part in j.get("multipart", []): walk(part.get("when", {}))
        known[b] = ps
# properties that do not change the model (absent from blockstate json) but exist on the block
extra = {"waterlogged": {"true", "false"}, "distance": set("1234567"), "persistent": {"true", "false"},
         "powered": {"true", "false"}, "snowy": {"true", "false"}, "occupied": {"true", "false"},
         "signal_fire": {"true", "false"}, "type": {"single", "left", "right"}, "has_book": {"true", "false"},
         "age": {str(i) for i in range(26)}, "attachment": {"floor", "ceiling", "single_wall", "double_wall"},
         "honey_level": {str(i) for i in range(6)}, "cracked": {"true", "false"}, "bites": {str(i) for i in range(7)},
         "level": {str(i) for i in range(16)}, "tip": {"true", "false"}}
wall_none = {"none"}
# entity-rendered blocks: their states are not listed in the blockstate json
FACING4 = {"north", "south", "east", "west"}
for b in list(known):
    if b.endswith("_bed"):
        known[b].update({"facing": FACING4, "part": {"head", "foot"}})
for b in ("minecraft:chest", "minecraft:trapped_chest", "minecraft:decorated_pot"):
    known[b].setdefault("facing", set()).update(FACING4)
issues = []
for n, p in json.load(open("build_palette.json")):
    if n == "minecraft:air": continue
    if n not in known: issues.append(("unknown", n)); continue
    for a, v in p.items():
        vals = set(known[n].get(a, set())) | extra.get(a, set())
        if a in ("north", "south", "east", "west") and n.endswith("_wall"): vals |= wall_none
        if a in ("north", "south", "east", "west") and n.endswith("_fence"): vals |= {"true", "false"}
        if a == "up" and n.endswith("_wall"): vals |= {"true", "false"}
        if a in ("north", "south", "east", "west", "up", "down", "bottom"): vals |= {"true", "false", "none", "low", "tall"} & (vals or {"true", "false"})
        if not vals: issues.append((n, "prop?", a))
        elif v not in vals: issues.append((n, a, v, sorted(vals)[:8]))
print("palette:", len(json.load(open("build_palette.json"))), "issues:", len(issues))
for i in issues: print(" ", i)
