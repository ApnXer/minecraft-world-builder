"""Post-processing passes over the finished world:

1. weather(): re-texture building materials from physical cues instead of random noise —
   sun/shadow (raycast toward the sun), north-facing damp, height above ground (splash zone),
   shelter under eaves, rain streaks below windows/sills, exposed edges (wear/cracks),
   plus coherent large-scale noise for batches/patina. Keeps stair/slab/wall/log states.
2. stair_shapes(): inner/outer corner shapes exactly like Minecraft's StairBlock rules.
3. connections(): fence / wall / pane / bar connections from neighbours.
"""
import math
import os

import numpy as np
from numba import njit

import mc
from mc import B, fbm3, noise3

_BLOCKS = set(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "blocks_1_21_11.txt")).read().split())
DIRS = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}
CCW = {"north": "west", "west": "south", "south": "east", "east": "north"}


def short(i):
    return mc.names[i].replace("minecraft:", "")


def kind_of(n):
    if n.endswith("_stairs"):
        return "stairs"
    if n.endswith("_slab"):
        return "slab"
    if n.endswith("_wall") and "sign" not in n and "banner" not in n:
        return "wall"
    return "full"


def with_props(name, pr):
    return B(name + ("[" + ",".join(f"{k}={v}" for k, v in pr.items()) + "]" if pr else ""))


# ------------------------------------------------------------------ solidity table
NONSOLID_WORDS = ("_stairs", "_slab", "_wall", "_fence", "_pane", "_bars", "door", "carpet", "lantern", "torch", "rod",
                  "chain", "candle", "potted_", "flower_pot", "_bed", "vine", "lichen", "ladder", "leaves", "glass",
                  "grass", "fern", "flower", "petals", "litter", "bush", "sapling", "mushroom", "roots", "moss_carpet",
                  "hanging", "dripstone", "button", "pressure", "rail", "sign", "banner", "water", "lava", "campfire",
                  "chest", "bell", "cauldron", "anvil", "lectern", "grindstone", "trapdoor", "path", "snow", "cobweb",
                  "tulip", "daisy", "orchid", "allium", "poppy", "dandelion", "cornflower", "lily", "bluet", "blossom",
                  "eyeblossom", "peony", "lilac", "rose", "berry", "kelp", "seagrass", "cluster", "pot", "composter",
                  "cake", "scaffold", "farmland")


def solid_table():
    t = np.zeros(len(mc.names), np.uint8)
    for i in range(1, len(mc.names)):
        n = short(i)
        if not any(w in n for w in NONSOLID_WORDS):
            t[i] = 1
        if "leaves" in n:
            t[i] = 2          # leaves: cast shade but are not walls
    return t


# ------------------------------------------------------------------ features (numba)
@njit(cache=True)
def _features(arr, solid, ys, zs, xs, gh, sun):
    n = len(ys)
    Y, Z, X = arr.shape
    out = np.zeros((n, 9), np.float32)   # exp_up, exp_north, exp_h_count, shelter, shadow, streak, h, ao, exp_south_east
    for i in range(n):
        y, z, x = ys[i], zs[i], xs[i]

        def s(yy, zz, xx):
            if yy < 0 or zz < 0 or xx < 0 or yy >= Y or zz >= Z or xx >= X:
                return 0
            return solid[arr[yy, zz, xx]]

        up = 1 if s(y + 1, z, x) != 1 else 0
        nn = 1 if s(y, z - 1, x) != 1 else 0
        ss = 1 if s(y, z + 1, x) != 1 else 0
        ee = 1 if s(y, z, x + 1) != 1 else 0
        ww = 1 if s(y, z, x - 1) != 1 else 0
        hc = nn + ss + ee + ww
        ao = 0
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                for dx in range(-1, 2):
                    if s(y + dy, z + dz, x + dx) == 1:
                        ao += 1
        # shelter: solid overhead (within 7) above an exposed horizontal neighbour
        shelter = 0.0
        for (dz, dx, ex) in ((-1, 0, nn), (1, 0, ss), (0, 1, ee), (0, -1, ww)):
            if ex:
                for k in range(1, 8):
                    if s(y + k, z + dz, x + dx) >= 1:
                        shelter = max(shelter, 1.0 - (k - 1) / 8.0)
                        break
        # sun: march from the most exposed face toward the sun
        if up:
            ox, oy, oz = x + 0.5, y + 1.05, z + 0.5
        elif ss:
            ox, oy, oz = x + 0.5, y + 0.5, z + 1.05
        elif ee:
            ox, oy, oz = x + 1.05, y + 0.5, z + 0.5
        elif ww:
            ox, oy, oz = x - 0.05, y + 0.5, z + 0.5
        elif nn:
            ox, oy, oz = x + 0.5, y + 0.5, z - 0.05
        else:
            ox, oy, oz = -1.0, -1.0, -1.0
        shadow = 1.0
        if ox >= 0:
            shadow = 0.0
            if nn and not (up or ss or ee or ww):
                shadow = 0.7           # pure north faces never get direct sun
            for k in range(1, 70):
                px = ox + sun[0] * k * 0.7
                py = oy + sun[1] * k * 0.7
                pz = oz + sun[2] * k * 0.7
                if py >= Y:
                    break
                v = s(int(math.floor(py)), int(math.floor(pz)), int(math.floor(px)))
                if v == 1:
                    shadow = 1.0
                    break
                if v == 2:
                    shadow = max(shadow, 0.6)
        # rain streak: an opening (window) higher up in the same wall column
        streak = 0.0
        if hc > 0:
            for k in range(1, 8):
                if s(y + k, z, x) != 1:
                    if s(y + k + 1, z, x) != 1 or k > 1:
                        streak = 1.0 - (k - 1) / 7.0
                    break
        out[i, 0] = up
        out[i, 1] = nn
        out[i, 2] = hc
        out[i, 3] = shelter
        out[i, 4] = shadow
        out[i, 5] = streak
        out[i, 6] = y - gh[z, x]
        out[i, 7] = ao
        out[i, 8] = ss + ee
    return out


# ------------------------------------------------------------------ families
def fam(**cats):
    return cats


# per category: {kind: block name}; missing kinds fall back to the 'clean' entry
FAMILIES = {
    "stone": dict(
        members=["stone_bricks", "mossy_stone_bricks", "cracked_stone_bricks", "tuff_bricks", "polished_andesite",
                 "stone_brick_stairs", "mossy_stone_brick_stairs", "tuff_brick_stairs", "polished_andesite_stairs",
                 "stone_brick_slab", "mossy_stone_brick_slab", "tuff_brick_slab", "polished_andesite_slab",
                 "stone_brick_wall", "mossy_stone_brick_wall", "tuff_brick_wall"],
        clean=dict(full="stone_bricks", stairs="stone_brick_stairs", slab="stone_brick_slab", wall="stone_brick_wall"),
        aged=dict(full="polished_andesite", stairs="polished_andesite_stairs", slab="polished_andesite_slab", wall="andesite_wall"),
        dirty=dict(full="tuff_bricks", stairs="tuff_brick_stairs", slab="tuff_brick_slab", wall="tuff_brick_wall"),
        mossy=dict(full="mossy_stone_bricks", stairs="mossy_stone_brick_stairs", slab="mossy_stone_brick_slab", wall="mossy_stone_brick_wall"),
        deep=dict(full="mossy_cobblestone", stairs="mossy_cobblestone_stairs", slab="mossy_cobblestone_slab", wall="mossy_cobblestone_wall"),
        cracked=dict(full="cracked_stone_bricks"),
    ),
    "plaster": dict(
        members=["smooth_quartz", "calcite", "quartz_bricks", "polished_diorite", "diorite", "white_terracotta",
                 "pale_oak_planks", "mushroom_stem", "bone_block", "smooth_quartz_stairs", "quartz_stairs",
                 "smooth_quartz_slab", "quartz_slab", "polished_diorite_stairs", "diorite_stairs", "polished_diorite_slab"],
        clean=dict(full="smooth_quartz", stairs="smooth_quartz_stairs", slab="smooth_quartz_slab"),
        aged=dict(full="calcite", stairs="quartz_stairs", slab="quartz_slab"),
        dirty=dict(full="polished_diorite", stairs="polished_diorite_stairs", slab="polished_diorite_slab"),
        mossy=dict(full="diorite", stairs="diorite_stairs", slab="diorite_slab"),
        deep=dict(full="diorite", stairs="diorite_stairs", slab="diorite_slab"),
        cracked=dict(full="quartz_bricks"),
        exposed=dict(full="stone_bricks"),
    ),
    "slate": dict(
        members=["deepslate_tiles", "polished_deepslate", "deepslate_bricks", "cracked_deepslate_tiles", "cobbled_deepslate",
                 "deepslate_tile_stairs", "deepslate_brick_stairs", "polished_deepslate_stairs", "cobbled_deepslate_stairs",
                 "deepslate_tile_slab", "deepslate_brick_slab", "polished_deepslate_slab", "cobbled_deepslate_slab"],
        clean=dict(full="deepslate_tiles", stairs="deepslate_tile_stairs", slab="deepslate_tile_slab"),
        aged=dict(full="deepslate_bricks", stairs="deepslate_brick_stairs", slab="deepslate_brick_slab"),
        dirty=dict(full="polished_deepslate", stairs="polished_deepslate_stairs", slab="polished_deepslate_slab"),
        mossy=dict(full="cobbled_deepslate", stairs="cobbled_deepslate_stairs", slab="cobbled_deepslate_slab"),
        deep=dict(full="cobbled_deepslate", stairs="cobbled_deepslate_stairs", slab="cobbled_deepslate_slab"),
        cracked=dict(full="cracked_deepslate_tiles"),
    ),
    "brick": dict(
        members=["bricks", "resin_bricks", "terracotta", "polished_granite", "red_terracotta", "mud_bricks",
                 "brick_stairs", "resin_brick_stairs", "polished_granite_stairs", "mud_brick_stairs",
                 "brick_slab", "resin_brick_slab", "mud_brick_slab", "brick_wall"],
        clean=dict(full="bricks", stairs="brick_stairs", slab="brick_slab", wall="brick_wall"),
        aged=dict(full="bricks", stairs="brick_stairs", slab="brick_slab", wall="brick_wall"),
        dirty=dict(full="terracotta"),
        mossy=dict(full="mud_bricks", stairs="mud_brick_stairs", slab="mud_brick_slab", wall="mud_brick_wall"),
        deep=dict(full="mud_bricks", stairs="mud_brick_stairs", slab="mud_brick_slab", wall="mud_brick_wall"),
        cracked=dict(full="polished_granite", stairs="polished_granite_stairs", slab="polished_granite_slab"),
    ),
    "rubble": dict(
        members=["cobblestone", "mossy_cobblestone", "cobblestone_stairs", "mossy_cobblestone_stairs", "cobblestone_slab",
                 "mossy_cobblestone_slab", "cobblestone_wall", "mossy_cobblestone_wall"],
        clean=dict(full="cobblestone", stairs="cobblestone_stairs", slab="cobblestone_slab", wall="cobblestone_wall"),
        aged=dict(full="cobblestone", stairs="cobblestone_stairs", slab="cobblestone_slab", wall="cobblestone_wall"),
        dirty=dict(full="andesite", stairs="andesite_stairs", slab="andesite_slab", wall="andesite_wall"),
        mossy=dict(full="mossy_cobblestone", stairs="mossy_cobblestone_stairs", slab="mossy_cobblestone_slab", wall="mossy_cobblestone_wall"),
        deep=dict(full="mossy_cobblestone", stairs="mossy_cobblestone_stairs", slab="mossy_cobblestone_slab", wall="mossy_cobblestone_wall"),
        cracked=dict(full="stone"),
    ),
    "timber": dict(
        members=["dark_oak_log", "stripped_dark_oak_log"],
        clean=dict(full="dark_oak_log"),
        aged=dict(full="stripped_dark_oak_log"),
        dirty=dict(full="dark_oak_log"), mossy=dict(full="dark_oak_log"), deep=dict(full="dark_oak_log"),
        cracked=dict(full="stripped_dark_oak_log"),
    ),
}
CATS = ["clean", "aged", "dirty", "mossy", "deep", "cracked", "exposed"]


def _target(fam_def, cat, kind):
    c = fam_def.get(cat) or fam_def["clean"]
    name = c.get(kind) or fam_def["clean"].get(kind)
    if name is None or name not in _BLOCKS:
        name = fam_def["clean"].get(kind)
    return name


def build_maps():
    """fam_of[id] -> family index (-1 none); remap[cat][id] -> new id with identical state properties."""
    fam_names = list(FAMILIES)
    fam_of = np.full(len(mc.names), -1, np.int16)
    member = {}
    for fi, fn in enumerate(fam_names):
        for m in FAMILIES[fn]["members"]:
            member[m] = fi
    for i in range(1, len(mc.names)):
        fi = member.get(short(i))
        if fi is not None:
            fam_of[i] = fi
    n0 = len(mc.names)
    remap = np.zeros((len(CATS), n0), np.uint16)
    for c in range(len(CATS)):
        remap[c] = np.arange(n0)
    for i in range(1, n0):
        fi = fam_of[i]
        if fi < 0:
            continue
        fd = FAMILIES[fam_names[fi]]
        kind = kind_of(short(i))
        for c, cat in enumerate(CATS):
            tgt = _target(fd, cat, kind)
            if tgt is None:
                continue
            pr = dict(mc.props[i])
            if kind == "wall" and not tgt.endswith("_wall"):
                continue
            if "axis" in pr and not any(tgt.endswith(e) for e in ("_log", "_wood", "_pillar", "bone_block", "_stem", "basalt")):
                pr.pop("axis")        # e.g. bone_block -> smooth_quartz must not keep axis
            remap[c, i] = with_props("minecraft:" + tgt, pr)
    return fam_names, fam_of, remap


def weather(gh, exclude_ground_mask, sun=(0.35, 0.85, 0.40), seed=5):
    arr = mc.arr
    fam_names, fam_of, remap = build_maps()
    full = np.zeros(len(mc.names) + 4096, np.int16) - 1
    full[:len(fam_of)] = fam_of
    cand = full[arr] >= 0
    # keep street / courtyard surfaces (they get their own pattern pass)
    Yidx = np.arange(arr.shape[0])[:, None, None]
    cand &= ~((Yidx <= gh[None]) & exclude_ground_mask[None])
    ys, zs, xs = np.nonzero(cand)
    solid = solid_table()
    sv = np.array(sun, np.float32)
    sv /= np.linalg.norm(sv)
    F = _features(arr, solid, ys.astype(np.int64), zs.astype(np.int64), xs.astype(np.int64), gh.astype(np.int32), sv)
    up, north, hc, shelter, shadow, streak, h, ao, se = (F[:, k] for k in range(9))
    exposed = (up + hc) > 0
    nb = fbm3(xs, ys, zs, 6.0, seed, 3)                                  # medium patches
    nl = fbm3(xs, ys, zs, 22.0, seed + 1, 2)                             # large batches / patina
    nv = fbm3(xs, ys, zs, 3.0, seed + 2, 2, stretch=(1.0, 5.0, 1.0))    # vertical streak texture
    nc = noise3(xs, ys, zs, 2.2, seed + 3)                               # fine wear
    low = np.clip(1 - h / 5.0, 0, 1)
    moss = 0.95 * low + 0.35 * north + 0.35 * shadow + 0.15 * up + 0.55 * nb - 0.7 * shelter - 0.25 * se * (1 - shadow)
    dirt = 0.5 * shelter + 0.85 * streak * (0.6 + nv) + 0.25 * np.clip(1 - h / 10, 0, 1) + 0.35 * nv + 0.1 * (ao >= 20)
    crack = 0.42 * (hc >= 2) + 0.12 * up + 0.5 * nc + 0.18 * low + 0.25 * nb
    sunlit = (1 - shadow) * 0.6 + 0.25 * se + 0.3 * nl
    cat = np.where(nl > 0.18, 1, 0)                                      # clean vs aged batches
    fid = full[arr[ys, zs, xs]]
    is_timber = fid == fam_names.index("timber")
    cat = np.where(exposed & (dirt > 0.72), 2, cat)
    cat = np.where(exposed & (crack > 0.78), 5, cat)
    cat = np.where(exposed & (moss > 0.95), 3, cat)
    cat = np.where(exposed & (moss > 1.4), 4, cat)
    plaster = fid == fam_names.index("plaster")
    cat = np.where(plaster & exposed & (low > 0.4) & (crack > 0.85) & (nb > 0.15), 6, cat)   # plaster fallen off
    cat = np.where(is_timber, np.where(exposed & (sunlit > 0.55), 1, 0), cat)
    ids = arr[ys, zs, xs]
    arr[ys, zs, xs] = remap[cat, ids]
    counts = {CATS[c]: int((cat == c).sum()) for c in range(len(CATS))}
    return counts


# ------------------------------------------------------------------ stair shapes
def stair_shapes():
    arr = mc.arr
    info = {}
    for i in range(1, len(mc.names)):
        n = short(i)
        if n.endswith("_stairs"):
            p = mc.props[i]
            info[i] = (p.get("facing", "north"), p.get("half", "bottom"))
    is_st = np.zeros(len(mc.names), bool)
    is_st[list(info)] = True
    ys, zs, xs = np.nonzero(is_st[arr])
    Y, Z, X = arr.shape

    def st(y, z, x):
        if 0 <= z < Z and 0 <= x < X:
            b = int(arr[y, z, x])
            return info.get(b)
        return None

    changes = []
    for y, z, x in zip(ys, zs, xs):
        b = int(arr[y, z, x])
        f, h = info[b]
        dx, dz = DIRS[f]
        shape = "straight"
        b1 = st(y, z + dz, x + dx)
        if b1 and b1[1] == h and DIRS[b1[0]][0] * dx + DIRS[b1[0]][1] * dz == 0:
            d1 = b1[0]
            ox, oz = DIRS[{"north": "south", "south": "north", "east": "west", "west": "east"}[d1]]
            nb = st(y, z + oz, x + ox)
            if not (nb and nb[0] == f and nb[1] == h):
                shape = "outer_left" if d1 == CCW[f] else "outer_right"
        if shape == "straight":
            b2 = st(y, z - dz, x - dx)
            if b2 and b2[1] == h and DIRS[b2[0]][0] * dx + DIRS[b2[0]][1] * dz == 0:
                d2 = b2[0]
                ox, oz = DIRS[d2]
                nb = st(y, z + oz, x + ox)
                if not (nb and nb[0] == f and nb[1] == h):
                    shape = "inner_left" if d2 == CCW[f] else "inner_right"
        if shape != "straight":
            pr = dict(mc.props[b])
            pr["shape"] = shape
            changes.append((y, z, x, with_props(mc.names[b], pr)))
    for y, z, x, nid in changes:
        arr[y, z, x] = nid
    return len(changes)


# ------------------------------------------------------------------ connections
def connections():
    arr = mc.arr
    solid = solid_table()
    kind = np.zeros(len(mc.names), np.int8)   # 1 fence, 2 wall, 3 pane/bars
    for i in range(1, len(mc.names)):
        n = short(i)
        if n.endswith("_fence"):
            kind[i] = 1
        elif kind_of(n) == "wall":
            kind[i] = 2
        elif n.endswith("_pane") or n.endswith("_bars"):
            kind[i] = 3
    glassish = np.array([("glass" in short(i) and not short(i).endswith("pane")) for i in range(len(mc.names))])
    ys, zs, xs = np.nonzero(kind[arr] > 0)
    Y, Z, X = arr.shape
    changes = []
    for y, z, x in zip(ys, zs, xs):
        b = int(arr[y, z, x])
        k = kind[b]
        con = {}
        for side, (dx, dz) in DIRS.items():
            xx, zz = x + dx, z + dz
            c = False
            if 0 <= xx < X and 0 <= zz < Z:
                nbid = int(arr[y, zz, xx])
                nk = kind[nbid]
                if solid[nbid] == 1 or glassish[nbid]:
                    c = True
                elif k == 1 and nk == 1:
                    c = True
                elif k in (2, 3) and nk in (2, 3):
                    c = True
                elif "fence_gate" in short(nbid) and k in (1, 2):
                    c = True
            con[side] = c
        pr = dict(mc.props[b])
        if k == 2:
            above = int(arr[y + 1, z, x]) if y + 1 < Y else 0
            above_solid = solid[above] == 1
            for side in DIRS:
                pr[side] = ("tall" if above_solid else "low") if con[side] else "none"
            straight = (con["north"] and con["south"] and not con["east"] and not con["west"]) or \
                       (con["east"] and con["west"] and not con["north"] and not con["south"])
            pr["up"] = "false" if straight and kind[above] != 2 and above == 0 else "true"
        else:
            for side in DIRS:
                pr[side] = "true" if con[side] else "false"
        nid = with_props(mc.names[b], pr)
        if nid != b:
            changes.append((y, z, x, nid))
    for y, z, x, nid in changes:
        arr[y, z, x] = nid
    return len(changes)
