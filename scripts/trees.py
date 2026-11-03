"""Natural, varied trees: branching limbs with oriented logs, bark mixes, roots, lumpy leaf clusters,
hanging lianas (glow-berry cave vines, pale hanging moss), trunk vines, fallen logs, shrubs."""
import math

import numpy as np

import mc
from mc import B, Mix, arr, ball, get, hash3, put

AXIS = {0: "x", 1: "y", 2: "z"}


def L(name, axis):
    return B(f"{name}[axis={axis}]")


SPECIES = {
    "oak": dict(log="oak_log", wood=Mix([("oak_wood[axis=y]", 62), ("dark_oak_wood[axis=y]", 18), ("moss_block", 8),
                                         ("oak_log[axis=y]", 12)], seed=31, patch=2),
                leaves=Mix([("oak_leaves[distance=1,persistent=true,waterlogged=false]", 52),
                            ("dark_oak_leaves[distance=1,persistent=true,waterlogged=false]", 30),
                            ("azalea_leaves[distance=1,persistent=true,waterlogged=false]", 12),
                            ("flowering_azalea_leaves[distance=1,persistent=true,waterlogged=false]", 6)], seed=32, patch=2)),
    "darkoak": dict(log="dark_oak_log", wood=Mix([("dark_oak_wood[axis=y]", 70), ("dark_oak_log[axis=y]", 15),
                                                   ("moss_block", 10), ("mossy_cobblestone", 5)], seed=33, patch=2),
                    leaves=Mix([("dark_oak_leaves[distance=1,persistent=true,waterlogged=false]", 55),
                                ("azalea_leaves[distance=1,persistent=true,waterlogged=false]", 25),
                                ("flowering_azalea_leaves[distance=1,persistent=true,waterlogged=false]", 20)], seed=34, patch=2)),
    "cherry": dict(log="cherry_log", wood=Mix([("cherry_wood[axis=y]", 85), ("cherry_log[axis=y]", 15)], seed=35),
                   leaves=Mix([("cherry_leaves[distance=1,persistent=true,waterlogged=false]", 100)], seed=36)),
    "pale": dict(log="pale_oak_log", wood=Mix([("pale_oak_wood[axis=y]", 80), ("pale_oak_log[axis=y]", 12), ("pale_moss_block", 8)], seed=37, patch=2),
                 leaves=Mix([("pale_oak_leaves[distance=1,persistent=true,waterlogged=false]", 100)], seed=38)),
    "birch": dict(log="birch_log", wood=Mix([("birch_wood[axis=y]", 80), ("birch_log[axis=y]", 20)], seed=39),
                  leaves=Mix([("birch_leaves[distance=1,persistent=true,waterlogged=false]", 80),
                              ("oak_leaves[distance=1,persistent=true,waterlogged=false]", 20)], seed=40, patch=2)),
    "spruce": dict(log="spruce_log", wood=Mix([("spruce_wood[axis=y]", 85), ("spruce_log[axis=y]", 15)], seed=41),
                   leaves=Mix([("spruce_leaves[distance=1,persistent=true,waterlogged=false]", 100)], seed=42)),
    "willow": dict(log="mangrove_log", wood=Mix([("mangrove_wood[axis=y]", 55), ("dark_oak_wood[axis=y]", 30), ("moss_block", 15)], seed=43, patch=2),
                   leaves=Mix([("azalea_leaves[distance=1,persistent=true,waterlogged=false]", 50),
                               ("flowering_azalea_leaves[distance=1,persistent=true,waterlogged=false]", 25),
                               ("mangrove_leaves[distance=1,persistent=true,waterlogged=false]", 25)], seed=44, patch=2)),
}
CAVE_PLANT = {True: B("cave_vines_plant[berries=true]"), False: B("cave_vines_plant[berries=false]")}
CAVE_TIP = {True: B("cave_vines[age=25,berries=true]"), False: B("cave_vines[age=25,berries=false]")}
HMOSS, HMOSS_TIP = B("pale_hanging_moss[tip=false]"), B("pale_hanging_moss[tip=true]")
LITTER = [B(f"leaf_litter[facing={f},segment_amount={n}]") for f in ("north", "east") for n in (2, 3, 4)]
SOLID_GROUND = None


def branch(p0, p1, r0, r1, sp, seed):
    """Tapered limb from p0 to p1 (x, y, z floats). Thin parts use oriented logs."""
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    ln = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
    ax = int(np.argmax([abs(dx), abs(dy), abs(dz)]))
    logb = L(sp["log"], AXIS[ax])
    n = int(ln * 2) + 2
    for i in range(n + 1):
        t = i / n
        x, y, z = x0 + dx * t, y0 + dy * t, z0 + dz * t
        r = r0 + (r1 - r0) * t
        if r < 0.75:
            put(int(round(x)), int(round(y)), int(round(z)), logb)
        else:
            ball(x, y, z, r, r, r, sp["wood"], only_air=False)
            put(int(round(x)), int(round(y)), int(round(z)), logb)


def cluster(x, y, z, rx, ry, sp, seed):
    ball(x, y, z, rx, ry, rx * (0.85 + 0.3 * ((seed % 7) / 7)), sp["leaves"], only_air=True, rough=0.7, seed=seed, hollow_prob=0.06)


def hang_from(x0, x1, y0, y1, z0, z1, rng, prob=0.035, kinds=("cave", "moss"), maxlen=9):
    """Hang lianas from the undersides of leaves inside a box."""
    xa, xb = max(x0, 0), min(x1, mc.S - 1)
    za, zb = max(z0, 0), min(z1, mc.S - 1)
    ya, yb = max(y0, 1), min(y1, mc.Y - 1)
    if xa >= xb or za >= zb or ya >= yb:
        return
    sub = arr[ya:yb + 1, za:zb + 1, xa:xb + 1]
    isleaf = np.isin(sub, LEAF_IDS)
    below_air = np.zeros_like(isleaf)
    below_air[1:] = isleaf[1:] & (sub[:-1] == 0)
    ys, zs, xs = np.nonzero(below_air)
    for yy, zz, xx in zip(ys, zs, xs):
        if rng.random() > prob:
            continue
        x, y, z = xx + xa, yy + ya - 1, zz + za
        kind = kinds[rng.integers(len(kinds))]
        ln = int(rng.integers(2, maxlen + 1))
        for k in range(ln):
            if get(x, y - k, z) != 0:
                break
            last = k == ln - 1 or get(x, y - k - 1, z) != 0
            if kind == "cave":
                berry = rng.random() < 0.45
                put(x, y - k, z, CAVE_TIP[berry] if last else CAVE_PLANT[berry])
            elif kind == "leaf":
                put(x, y - k, z, LEAF_IDS[0])
            else:
                put(x, y - k, z, HMOSS_TIP if last else HMOSS)


def trunk_vines(x0, x1, y0, y1, z0, z1, rng, prob=0.18, lichen=0.05):
    """Vines and glow lichen on the sides of bark blocks."""
    for x in range(max(x0, 1), min(x1, mc.S - 2) + 1):
        for z in range(max(z0, 1), min(z1, mc.S - 2) + 1):
            for y in range(max(y0, 1), min(y1, mc.Y - 2) + 1):
                if arr[y, z, x] != 0:
                    continue
                for side, (ddx, ddz) in (("east", (1, 0)), ("west", (-1, 0)), ("south", (0, 1)), ("north", (0, -1))):
                    if arr[y, z + ddz, x + ddx] in BARK_IDS:
                        r = rng.random()
                        if r < prob:
                            put(x, y, z, B(f"vine[east={'true' if side == 'east' else 'false'},north={'true' if side == 'north' else 'false'},"
                                           f"south={'true' if side == 'south' else 'false'},up=false,west={'true' if side == 'west' else 'false'}]"))
                        elif r < prob + lichen:
                            put(x, y, z, B(f"glow_lichen[down=false,east={'true' if side == 'east' else 'false'},"
                                           f"north={'true' if side == 'north' else 'false'},south={'true' if side == 'south' else 'false'},"
                                           f"up=false,waterlogged=false,west={'true' if side == 'west' else 'false'}]"))
                        break


def roots(x, gy, z, r0, n, sp, rng, spread=7):
    for k in range(n):
        a = 2 * math.pi * k / n + rng.uniform(-0.3, 0.3)
        ln = rng.uniform(spread * 0.6, spread)
        branch((x, gy + 1.5, z), (x + ln * math.cos(a), gy - 1.2, z + ln * math.sin(a)), r0 * 0.75, 0.5, sp, k)


def litter(x, gy_fn, z, r, rng, n=30, kinds=None):
    for _ in range(n):
        a, d = rng.uniform(0, 2 * math.pi), rng.uniform(1, r)
        xx, zz = int(x + d * math.cos(a)), int(z + d * math.sin(a))
        if not (0 <= xx < mc.S and 0 <= zz < mc.S):
            continue
        y = gy_fn(xx, zz) + 1
        if get(xx, y, zz) == 0 and get(xx, y - 1, zz) in GROUND_IDS:
            put(xx, y, zz, (kinds or LITTER)[rng.integers(len(kinds or LITTER))])


def oak(x, gy, z, rng, size=1.0, sp_name="oak", hang=0.03):
    sp = SPECIES[sp_name]
    H = rng.uniform(9, 15) * size
    r0 = rng.uniform(1.4, 2.0) * size
    lean = rng.uniform(0, 2 * math.pi)
    lx, lz = math.cos(lean) * H * 0.18, math.sin(lean) * H * 0.18
    mid = (x + lx * 0.4 + rng.uniform(-1, 1), gy + H * 0.5, z + lz * 0.4 + rng.uniform(-1, 1))
    topp = (x + lx, gy + H, z + lz)
    branch((x, gy - 1, z), mid, r0, r0 * 0.8, sp, 1)
    branch(mid, topp, r0 * 0.8, r0 * 0.6, sp, 2)
    roots(x, gy, z, r0, int(rng.integers(5, 8)), sp, rng, spread=5 * size + 2)
    nl = int(rng.integers(3, 6))
    tips = []
    for k in range(nl):
        a = 2 * math.pi * k / nl + rng.uniform(-0.4, 0.4)
        el = math.radians(rng.uniform(22, 55))
        ln = rng.uniform(7, 12) * size
        start = topp if k % 2 == 0 else (mid[0] * 0.3 + topp[0] * 0.7, mid[1] * 0.3 + topp[1] * 0.7, mid[2] * 0.3 + topp[2] * 0.7)
        end = (start[0] + ln * math.cos(a) * math.cos(el), start[1] + ln * math.sin(el), start[2] + ln * math.sin(a) * math.cos(el))
        branch(start, end, r0 * 0.55, 0.45, sp, 10 + k)
        for j in range(int(rng.integers(2, 4))):
            a2 = a + rng.uniform(-0.9, 0.9)
            l2 = rng.uniform(3, 6) * size
            e2 = (end[0] + l2 * math.cos(a2), end[1] + rng.uniform(0.5, 3), end[2] + l2 * math.sin(a2))
            branch(end, e2, 0.5, 0.4, sp, 20 + j)
            tips.append(e2)
        tips.append(end)
    bx = [x]
    by = [gy]
    bz = [z]
    for i, (tx, ty, tz) in enumerate(tips):
        rr = rng.uniform(3.2, 5.0) * size ** 0.6
        cluster(tx, ty + 0.8, tz, rr, rr * 0.62, sp, int(rng.integers(1, 1 << 30)))
        bx.append(tx)
        by.append(ty)
        bz.append(tz)
    cluster(topp[0], topp[1] + 2.5, topp[2], 5.5 * size ** 0.6, 3.2 * size ** 0.6, sp, int(rng.integers(1, 1 << 30)))
    pad = 8
    box = (int(min(bx)) - pad, int(max(bx)) + pad, int(gy), int(max(by)) + pad, int(min(bz)) - pad, int(max(bz)) + pad)
    if hang:
        hang_from(box[0], box[1], box[2], box[3], box[4], box[5], rng, prob=hang)
    if size >= 1.1:
        trunk_vines(int(x - r0 - 2), int(x + r0 + 2), int(gy), int(gy + H * 0.7), int(z - r0 - 2), int(z + r0 + 2), rng)
    return box


def willow(x, gy, z, rng, size=1.0):
    sp = SPECIES["willow"]
    H = rng.uniform(6, 9) * size
    r0 = rng.uniform(1.2, 1.7) * size
    topp = (x + rng.uniform(-1.5, 1.5), gy + H, z + rng.uniform(-1.5, 1.5))
    branch((x, gy - 1, z), topp, r0, r0 * 0.7, sp, 1)
    roots(x, gy, z, r0, 6, sp, rng, spread=5)
    nl = int(rng.integers(5, 8))
    pts = []
    for k in range(nl):
        a = 2 * math.pi * k / nl + rng.uniform(-0.3, 0.3)
        ln = rng.uniform(6, 10) * size
        el = math.radians(rng.uniform(8, 28))
        end = (topp[0] + ln * math.cos(a) * math.cos(el), topp[1] + ln * math.sin(el), topp[2] + ln * math.sin(a) * math.cos(el))
        branch(topp, end, r0 * 0.5, 0.4, sp, k)
        cluster(end[0], end[1] + 1, end[2], rng.uniform(3.5, 4.8), 2.4, sp, int(rng.integers(1, 1 << 30)))
        pts.append(end)
    cluster(topp[0], topp[1] + 2.5, topp[2], 6.5 * size, 3.2, sp, int(rng.integers(1, 1 << 30)))
    # weeping curtain: leaf strands + glowing vines
    for _ in range(int(70 * size)):
        p = pts[rng.integers(len(pts))]
        a, d = rng.uniform(0, 2 * math.pi), rng.uniform(2.5, 5.5)
        sx, sz = int(round(p[0] + d * math.cos(a))), int(round(p[2] + d * math.sin(a)))
        sy = int(round(p[1] + 2))
        while sy > gy and get(sx, sy, sz) == 0:
            sy -= 1
        if get(sx, sy, sz) not in LEAF_IDS:
            continue
        ln = int(rng.integers(3, 9))
        for k in range(1, ln + 1):
            if get(sx, sy - k, sz) != 0 or sy - k <= gy + 2:
                break
            put(sx, sy - k, sz, sp["leaves"].ids[k % 3] if k < ln - 2 else (CAVE_TIP[True] if k == ln else CAVE_PLANT[True]))
    return (int(x - 16), int(x + 16), int(gy), int(gy + H + 8), int(z - 16), int(z + 16))


def cherry(x, gy, z, rng, size=1.0):
    sp = SPECIES["cherry"]
    H = rng.uniform(5, 8) * size
    lean = rng.uniform(0, 2 * math.pi)
    topp = (x + 3 * math.cos(lean), gy + H, z + 3 * math.sin(lean))
    branch((x, gy - 1, z), (x + 1.2 * math.cos(lean), gy + H * 0.5, z + 1.2 * math.sin(lean)), 1.1 * size, 0.9, sp, 1)
    branch((x + 1.2 * math.cos(lean), gy + H * 0.5, z + 1.2 * math.sin(lean)), topp, 0.9, 0.6, sp, 2)
    for k in range(int(rng.integers(2, 4))):
        a = lean + math.pi + k * 2.2 + rng.uniform(-0.4, 0.4)
        ln = rng.uniform(5, 8) * size
        end = (topp[0] + ln * math.cos(a), topp[1] + rng.uniform(1, 4), topp[2] + ln * math.sin(a))
        branch(topp, end, 0.6, 0.4, sp, 5 + k)
        ball(end[0], end[1] + 1, end[2], rng.uniform(4.5, 6.5), 2.2, rng.uniform(4.5, 6.5), sp["leaves"], rough=0.8,
             seed=int(rng.integers(1, 1 << 30)), hollow_prob=0.05)
        ball(end[0], end[1] - 0.5, end[2], 3.0, 1.2, 3.0, sp["leaves"], rough=0.8, seed=int(rng.integers(1, 1 << 30)))
    ball(topp[0], topp[1] + 1.5, topp[2], 5.5 * size, 2.4, 5.5 * size, sp["leaves"], rough=0.8, seed=int(rng.integers(1, 1 << 30)))
    return (int(x - 14), int(x + 14), int(gy), int(gy + H + 8), int(z - 14), int(z + 14))


def spruce(x, gy, z, rng, size=1.0):
    sp = SPECIES["spruce"]
    H = int(rng.uniform(16, 28) * size)
    Rm = rng.uniform(4.5, 6.5) * size
    logy = L("spruce_log", "y")
    thick = size > 1.15
    for h in range(-1, H):
        put(x, gy + h, z, logy)
        if thick and h < H * 0.35:
            for dx, dz in ((1, 0), (0, 1), (1, 1)):
                put(x + dx, gy + h, z + dz, sp["wood"].at(x + dx, gy + h, z + dz))
    h = int(H * 0.22)
    while h < H:
        f = (1 - h / H) ** 0.85
        rad = Rm * f * rng.uniform(0.8, 1.15)
        nb = int(rng.integers(5, 9))
        off = rng.uniform(0, 2 * math.pi)
        for k in range(nb):
            a = off + 2 * math.pi * k / nb
            ln = rad * rng.uniform(0.7, 1.0)
            ex, ez = x + ln * math.cos(a), z + ln * math.sin(a)
            droop = -ln * 0.25
            if ln > 2:
                branch((x, gy + h, z), (ex, gy + h + droop, ez), 0.45, 0.4, sp, k)
            ball((x + ex) / 2, gy + h + droop / 2 + 0.6, (z + ez) / 2, ln / 2 + 1.0, 1.3, ln / 2 + 1.0, sp["leaves"],
                 rough=0.6, seed=int(rng.integers(1, 1 << 30)))
        h += int(rng.integers(2, 4))
    for k in range(4):
        put(x, gy + H + k, z, sp["leaves"].ids[0])
    ball(x, gy + H - 1, z, 1.6, 2.0, 1.6, sp["leaves"], rough=0.3, seed=int(rng.integers(1, 1 << 30)))
    return (int(x - Rm - 2), int(x + Rm + 2), gy, gy + H + 4, int(z - Rm - 2), int(z + Rm + 2))


def birch(x, gy, z, rng, size=1.0):
    sp = SPECIES["birch"]
    H = rng.uniform(11, 17) * size
    lean = rng.uniform(0, 2 * math.pi)
    topp = (x + 1.5 * math.cos(lean), gy + H, z + 1.5 * math.sin(lean))
    branch((x, gy - 1, z), topp, 0.5, 0.45, sp, 1)
    for k in range(3):
        t = 0.55 + 0.15 * k
        p = (x + (topp[0] - x) * t, gy + H * t, z + (topp[2] - z) * t)
        a = rng.uniform(0, 2 * math.pi)
        e = (p[0] + 3.5 * math.cos(a), p[1] + 2.5, p[2] + 3.5 * math.sin(a))
        branch(p, e, 0.45, 0.4, sp, k)
        cluster(e[0], e[1] + 0.5, e[2], rng.uniform(2.4, 3.4), 2.4, sp, int(rng.integers(1, 1 << 30)))
    cluster(topp[0], topp[1] + 1, topp[2], 3.2, 3.6, sp, int(rng.integers(1, 1 << 30)))
    return (int(x - 8), int(x + 8), gy, int(gy + H + 6), int(z - 8), int(z + 8))


def pale(x, gy, z, rng, size=1.0):
    box = oak(x, gy, z, rng, size=size * 0.95, sp_name="pale", hang=0)
    hang_from(*box, rng, prob=0.09, kinds=("moss",), maxlen=7)
    return box


def mother(x, gy, z, rng):
    """One colossal ancient tree with buttress roots, huge limbs, glowing crown and many lianas."""
    sp = SPECIES["darkoak"]
    H = 30
    r0 = 5.5
    topp = (x + 2, gy + H, z - 1)
    branch((x, gy - 2, z), (x + 1, gy + H * 0.55, z), r0, r0 * 0.75, sp, 1)
    branch((x + 1, gy + H * 0.55, z), topp, r0 * 0.75, r0 * 0.55, sp, 2)
    for k in range(9):
        a = 2 * math.pi * k / 9 + rng.uniform(-0.2, 0.2)
        ln = rng.uniform(11, 16)
        branch((x, gy + 3, z), (x + ln * math.cos(a), gy - 2, z + ln * math.sin(a)), 2.4, 0.6,
               dict(sp, wood=Mix([("mangrove_roots[waterlogged=false]", 30), ("dark_oak_wood[axis=y]", 50), ("moss_block", 20)], seed=45)), k)
    tips = []
    for k in range(8):
        a = 2 * math.pi * k / 8 + rng.uniform(-0.3, 0.3)
        el = math.radians(rng.uniform(15, 45))
        ln = rng.uniform(16, 24)
        st = (topp[0], topp[1] - rng.uniform(0, 8), topp[2])
        end = (st[0] + ln * math.cos(a) * math.cos(el), st[1] + ln * math.sin(el), st[2] + ln * math.sin(a) * math.cos(el))
        branch(st, end, 2.6, 0.8, sp, 10 + k)
        for j in range(3):
            a2 = a + rng.uniform(-1.0, 1.0)
            l2 = rng.uniform(6, 10)
            e2 = (end[0] + l2 * math.cos(a2), end[1] + rng.uniform(0, 4), end[2] + l2 * math.sin(a2))
            branch(end, e2, 0.8, 0.45, sp, 30 + j)
            tips.append(e2)
        tips.append(end)
    for tx, ty, tz in tips:
        cluster(tx, ty + 1, tz, rng.uniform(5, 7.5), rng.uniform(3, 4.2), sp, int(rng.integers(1, 1 << 30)))
    cluster(topp[0], topp[1] + 4, topp[2], 11, 6, sp, int(rng.integers(1, 1 << 30)))
    # glowing shroomlights tucked in the crown
    for tx, ty, tz in tips[::2]:
        put(int(tx), int(ty) + 1, int(tz), B("shroomlight"))
    box = (int(x - 38), int(x + 38), int(gy), int(gy + H + 22), int(z - 38), int(z + 38))
    hang_from(*box, rng, prob=0.06, kinds=("cave", "cave", "moss"), maxlen=14)
    trunk_vines(int(x - 9), int(x + 9), int(gy), int(gy + H), int(z - 9), int(z + 9), rng, prob=0.3, lichen=0.1)
    return box


def fallen_log(x, gy, z, rng, name="oak"):
    sp = SPECIES[name]
    a = rng.uniform(0, math.pi)
    ln = rng.uniform(6, 12)
    ex, ez = x + ln * math.cos(a), z + ln * math.sin(a)
    axis = "x" if abs(math.cos(a)) > abs(math.sin(a)) else "z"
    logb = L(sp["log"], axis)
    n = int(ln) + 1
    for i in range(n):
        xx, zz = int(round(x + (ex - x) * i / n)), int(round(z + (ez - z) * i / n))
        put(xx, gy + 1, zz, logb)
        if rng.random() < 0.6:
            put(xx, gy + 2, zz, B("moss_carpet"), only_air=True)
        elif rng.random() < 0.2:
            put(xx, gy + 2, zz, B("brown_mushroom" if rng.random() < 0.5 else "red_mushroom"), only_air=True)


def shrub(x, gy, z, rng):
    mix = Mix([("azalea_leaves[distance=1,persistent=true,waterlogged=false]", 50),
               ("flowering_azalea_leaves[distance=1,persistent=true,waterlogged=false]", 25),
               ("oak_leaves[distance=1,persistent=true,waterlogged=false]", 25)], seed=int(rng.integers(1, 999)))
    r = rng.uniform(1.6, 2.8)
    ball(x, gy + r * 0.6, z, r, r * 0.75, r, mix, rough=0.6, seed=int(rng.integers(1, 1 << 30)))


LEAF_IDS = np.array(sorted({int(i) for s in SPECIES.values() for i in s["leaves"].ids}), np.uint16)
BARK_IDS = np.array(sorted({int(i) for s in SPECIES.values() for i in s["wood"].ids if "wood" in mc.names[int(i)]}), np.uint16)
GROUND_IDS = np.array([B("grass_block[snowy=false]"), B("moss_block"), B("podzol[snowy=false]"), B("coarse_dirt"),
                       B("rooted_dirt"), B("pale_moss_block"), B("dirt")], np.uint16)
