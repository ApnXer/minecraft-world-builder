"""Natural (non-symmetric) decoration of the island: forests by zone, groves, mother tree, ground cover,
boulders, cliff vines and lichen, hanging roots and glow vines under the island."""
import math

import numpy as np

import mc
import terrain as T
import trees
from mc import B, fbm, get, put

rng = np.random.default_rng(4242)
zone = fbm(17, ((70, 1.0), (30, 0.5)))     # forest type zones
dens = fbm(18, ((50, 1.0), (18, 0.5)))     # forest density


def ok_ground(x, z):
    if not (0 <= x < mc.S and 0 <= z < mc.S):
        return False
    return bool(T.ISL[z, x] and T.R[z, x] > T.PLATEAU + 3 and not T.STREAM[z, x] and not T.ponds[z, x]
                and T.ein[z, x] < 0.9 and T.slope[z, x] <= 2)


def forest():
    pts = []
    # mother tree in a clearing (north-east)
    mx, mz = mc.C + 150, mc.C - 62
    trees.mother(mx, int(T.gh[mz, mx]), mz, rng)
    pts.append((mx, mz, 40))
    for _ in range(26000):
        x, z = int(rng.integers(0, mc.S)), int(rng.integers(0, mc.S))
        if not ok_ground(x, z):
            continue
        d = dens[z, x]
        if rng.random() > 0.35 + d * 0.8:
            continue
        zn = zone[z, x]
        near_water = T.stream_d[z, x] < 9 or (T.gh[z, x] <= T.WL + 1)
        if near_water:
            kind, size, sp = "willow", rng.uniform(1.0, 1.3), 18
        elif zn > 0.35:
            kind, size, sp = "cherry", rng.uniform(1.1, 1.5), 15
        elif zn < -0.58:
            kind, size, sp = "pale", rng.uniform(0.9, 1.2), 15
        elif T.ein[z, x] > 0.8 or T.gh[z, x] > T.G + 10:
            kind, size, sp = "spruce", rng.uniform(1.0, 1.4), 9
        elif zn > 0.1:
            kind, size, sp = "birch", rng.uniform(1.0, 1.3), 8
        else:
            kind, size, sp = ("darkoak" if rng.random() < 0.4 else "oak"), rng.uniform(1.4, 2.1), 19
        if any((x - a) ** 2 + (z - b) ** 2 < max(sp, r) ** 2 for a, b, r in pts):
            continue
        gy = int(T.gh[z, x])
        if kind == "willow":
            trees.willow(x, gy, z, rng, size)
        elif kind == "cherry":
            trees.cherry(x, gy, z, rng, size)
        elif kind == "pale":
            trees.pale(x, gy, z, rng, size)
        elif kind == "spruce":
            trees.spruce(x, gy, z, rng, size)
        elif kind == "birch":
            trees.birch(x, gy, z, rng, size)
        else:
            trees.oak(x, gy, z, rng, size, sp_name=kind, hang=0.04)
        trees.litter(x, lambda xx, zz: int(T.gh[zz, xx]), z, 7 * size, rng, n=int(25 * size))
        pts.append((x, z, sp))
    # spruce on the castle crag slopes
    for _ in range(4000):
        x, z = int(rng.integers(mc.C - 90, mc.C + 90)), int(rng.integers(mc.C - 70, mc.C + 70))
        if not (T.cragmask[z, x] and not T.PLATFORM[z, x] and not T.RAMP[z, x] and T.gh[z, x] > T.P + 4):
            continue
        if T.sE[z, x] < 1.07 or any((x - a) ** 2 + (z - b) ** 2 < 49 for a, b, _ in pts):
            continue
        if abs(x - mc.C - 56) < 6 and z > mc.C:
            continue
        trees.spruce(x, int(T.gh[z, x]), z, rng, rng.uniform(0.6, 0.9))
        pts.append((x, z, 7))
    # fallen logs, shrubs
    for _ in range(140):
        x, z = int(rng.integers(0, mc.S)), int(rng.integers(0, mc.S))
        if ok_ground(x, z) and get(x, int(T.gh[z, x]) + 1, z) == 0:
            if rng.random() < 0.4:
                trees.fallen_log(x, int(T.gh[z, x]), z, rng, ["oak", "birch", "spruce", "darkoak"][int(rng.integers(4))])
            else:
                trees.shrub(x, int(T.gh[z, x]), z, rng)
    return len(pts)


def boulders():
    mix = mc.Mix([("mossy_cobblestone", 30), ("andesite", 20), ("stone", 20), ("cobblestone", 10), ("moss_block", 12),
                  ("tuff", 8)], seed=301, patch=2)
    for _ in range(90):
        x, z = int(rng.integers(0, mc.S)), int(rng.integers(0, mc.S))
        if not ok_ground(x, z):
            continue
        r = rng.uniform(1.6, 4.2)
        gy = int(T.gh[z, x])
        mc.ball(x, gy + r * 0.3, z, r * rng.uniform(0.9, 1.3), r * 0.8, r, mix, only_air=False, rough=0.5, seed=int(rng.integers(1, 1 << 30)))
        for _ in range(4):
            xx, zz = x + int(rng.integers(-3, 4)), z + int(rng.integers(-3, 4))
            yy = gy + 1
            while get(xx, yy, zz) != 0 and yy < gy + 8:
                yy += 1
            put(xx, yy, zz, B("moss_carpet"), only_air=True)


GROUND = {B("grass_block[snowy=false]"), B("moss_block"), B("podzol[snowy=false]"), B("rooted_dirt"), B("pale_moss_block"), B("coarse_dirt")}


def ground_cover():
    """Grass, ferns, flowers in drifts, mushrooms in shade; done per column after trees."""
    from scipy.ndimage import gaussian_filter
    flowers = fbm(19, ((26, 1.0), (9, 0.5)))
    ys = T.gh + 1
    # coherent clumps (ferns, grass, mushrooms grow in patches, not per-block random)
    h = (np.clip(mc.fbm3(T.X, 0, T.Z, 2.6, 91, 2) / 0.6, -1, 1) * 0.5 + 0.5) * 1000
    # canopy / roof shade map: anything solid 3..45 blocks above the ground, softened
    cover = np.zeros((mc.S, mc.S), bool)
    for y in range(int(T.gh.min()) + 3, mc.Y):
        cover |= (mc.arr[y] != 0) & (y > T.gh + 2) & (y < T.gh + 45)
    shade_map = gaussian_filter(cover.astype(np.float32), 2.2)
    # forest floor follows the shade: podzol / coarse dirt / moss under trees, grass in the sun
    soil = mc.fbm3(T.X, 0, T.Z, 4.0, 92, 2)
    deep = T.ISL & (T.R > T.PLATEAU) & (shade_map > 0.45)
    for z, x in zip(*np.nonzero(deep)):
        y = int(T.gh[z, x])
        if mc.arr[y, z, x] in GROUND and mc.arr[y + 1, z, x] == 0:
            v = soil[z, x]
            mc.arr[y, z, x] = B("podzol[snowy=false]") if v > 0.12 else (B("moss_block") if v > -0.18 else B("coarse_dirt") if v > -0.4 else B("rooted_dirt"))
    for z in range(mc.S):
        for x in range(mc.S):
            if not T.ISL[z, x] or T.R[z, x] <= T.PLATEAU or T.STREAM[z, x]:
                continue
            y = int(ys[z, x])
            if y >= mc.Y - 2 or mc.arr[y, z, x] != 0 or mc.arr[y - 1, z, x] not in GROUND:
                continue
            r = int(h[z, x])
            shade = shade_map[z, x] > 0.35
            fl = flowers[z, x]
            under = mc.arr[y - 1, z, x]
            if under == B("pale_moss_block"):
                if r < 600:
                    put(x, y, z, B("pale_moss_carpet[bottom=true,east=none,north=none,south=none,west=none]"))
                elif r < 640:
                    put(x, y, z, B("closed_eyeblossom" if r % 2 else "open_eyeblossom"))
                continue
            if shade:
                if r < 250:
                    put(x, y, z, B("fern"))
                elif r < 300:
                    put(x, y, z, B("large_fern[half=lower]"))
                    put(x, y + 1, z, B("large_fern[half=upper]"), only_air=True)
                elif r < 330:
                    put(x, y, z, B("brown_mushroom" if r % 2 else "red_mushroom"))
                elif r < 420:
                    put(x, y, z, B("moss_carpet"))
                elif r < 470:
                    put(x, y, z, B("short_grass"))
                continue
            if fl > 0.35 and r < 520:
                kinds = ["wildflowers[facing=north,flower_amount=4]", "cornflower", "oxeye_daisy", "azure_bluet",
                         "allium", "pink_petals[facing=east,flower_amount=3]", "lily_of_the_valley", "dandelion"]
                put(x, y, z, B(kinds[(r // 7 + int(fl * 10)) % len(kinds)]))
            elif fl < -0.4 and r < 300:
                put(x, y, z, B("tall_grass[half=lower]"))
                put(x, y + 1, z, B("tall_grass[half=upper]"), only_air=True)
            elif r < 420:
                put(x, y, z, B("short_grass"))
            elif r < 450:
                put(x, y, z, B("bush"))
            elif r < 462:
                put(x, y, z, B("firefly_bush"))
            elif r < 470:
                put(x, y, z, B("sweet_berry_bush[age=3]"))


def cliffs_and_underside():
    """Vines and glow lichen on steep rock; hanging roots, glow vines and dripstone below the island."""
    vine_sides = (("east", 1, 0), ("west", -1, 0), ("south", 0, 1), ("north", 0, -1))
    xs, zs = np.nonzero(((T.slope >= 3) & T.ISL).T)
    for x, z in zip(xs, zs):
        x, z = int(x), int(z)
        top = int(T.gh[z, x])
        for side, dx, dz in vine_sides:
            xx, zz = x - dx, z - dz
            if not (0 <= xx < mc.S and 0 <= zz < mc.S):
                continue
            for y in range(int(T.gh[zz, xx]) + 1, top + 1):
                if mc.arr[y, zz, xx] == 0 and mc.arr[y, z, x] != 0 and rng.random() < 0.22:
                    flags = {s: "false" for s, _, _ in vine_sides}
                    flags[side] = "true"
                    if rng.random() < 0.75:
                        put(xx, y, zz, B(f"vine[east={flags['east']},north={flags['north']},south={flags['south']},up=false,west={flags['west']}]"))
                    else:
                        put(xx, y, zz, B(f"glow_lichen[down=false,east={flags['east']},north={flags['north']},south={flags['south']},up=false,waterlogged=false,west={flags['west']}]"))
    # underside
    for _ in range(9000):
        x, z = int(rng.integers(0, mc.S)), int(rng.integers(0, mc.S))
        if not T.ISL[z, x]:
            continue
        yb = int(T.ybot[z, x]) - 1
        if yb < 3 or mc.arr[yb, z, x] != 0:
            continue
        r = rng.random()
        if r < 0.45:
            ln = int(rng.integers(1, 4))
            for k in range(ln):
                put(x, yb - k, z, B("hanging_roots[waterlogged=false]"), only_air=True)
        elif r < 0.8:
            ln = int(rng.integers(3, 16))
            for k in range(ln):
                if yb - k < 2:
                    break
                berry = rng.random() < 0.4
                put(x, yb - k, z, B(f"cave_vines[age=25,berries={'true' if berry else 'false'}]") if k == ln - 1
                    else B(f"cave_vines_plant[berries={'true' if berry else 'false'}]"), only_air=True)
        else:
            ln = int(rng.integers(4, 22))
            r0 = rng.uniform(1.2, 3.5)
            k = 0
            for k in range(ln):
                rr = r0 * (1 - k / ln)
                if rr < 0.6 or yb - k < 3:
                    break
                mc.disc(x, yb - k, z, rr, B("dripstone_block"), only_air=True)
            if get(x, yb - k + 1, z) == B("dripstone_block"):
                put(x, yb - k, z, B("pointed_dripstone[thickness=tip,vertical_direction=down,waterlogged=false]"), only_air=True)


def build():
    n = forest()
    boulders()
    ground_cover()
    cliffs_and_underside()
    return n
