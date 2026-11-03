"""Island terrain, city plateau, castle crag, moat, stream; writes ground blocks into mc.arr."""
import math

import numpy as np

import mc
from mc import B, C, Mix, S, Y, arr, fbm, hash3

P = 40            # city level
CP = P + 32       # castle platform level
G = P - 7         # mean biome ground
WL = P - 9        # biome pond / stream water level
ISL_R = 184

X, Z = mc.grid()
U, V = X - C, Z - C
R = np.maximum(np.abs(U), np.abs(V))
E = np.hypot(U, V)

# ------------------------------------------------------------------ city layout
WALL_IN, WALL_OUT = 104, 108          # Chebyshev radius of the outer wall
MOAT_IN, MOAT_OUT = 112, 121
PLATEAU = 124

# castle crag platform (rounded rectangle, slightly off-centre)
CU0 = -1
sE = ((np.abs(U - CU0) / 52.0) ** 4 + (np.abs(V) / 30.0) ** 4) ** 0.25
n_rock = fbm(11, ((24, 1.0), (9, 0.5), (4, 0.3)))
PLATFORM = sE <= 1.0
over = np.clip(sE - 1, 0, None)
crag = np.where(PLATFORM, CP, CP - over * 52 * (1.55 + 0.35 * n_rock) + n_rock * 3 * np.clip(over * 5, 0, 1))
# forecourt bastion east of the gatehouse
FORE = (U >= 50) & (U <= 62) & (np.abs(V) <= 9)
# ramp from the forecourt down to the market (south-east)
RAMP = (U >= 53) & (U <= 59) & (V >= 10) & (V <= 74)
RAMP_H = np.where(RAMP, CP - (V - 10) / 2.0, -999.0)

# streets (paved)
ring_s = ((np.abs(U - CU0) / 80.0) ** 4 + (np.abs(V) / 58.0) ** 4) ** 0.25
RING = (ring_s >= 0.955) & (ring_s <= 1.02)
STREET = RING.copy()
STREET |= (np.abs(V) <= 3) & (np.abs(U) >= 70) & (np.abs(U) <= WALL_OUT)       # east / west main streets
STREET |= (np.abs(U) <= 3) & (np.abs(V) >= 52) & (np.abs(V) <= WALL_OUT)       # north / south main streets
STREET |= (R >= 98) & (R <= 102)                                                # lane along the wall
STREET |= (np.abs(U - V) <= 2) & (U < -60) & (V < -45) & (R < 100)               # NW diagonal lane
STREET |= (np.abs(U + V) <= 2) & (U < -55) & (V > 45) & (R < 100)               # SW diagonal lane
STREET |= (np.abs(V + 80) <= 2) & (U > 10) & (U < 98)                          # NE cross lane
MARKET = (U >= 34) & (U <= 94) & (V >= 64) & (V <= 96)
GARDEN = (U <= -62) & (U >= -96) & (V >= -30) & (V <= 30)                       # palace garden (west)
CITY = (R <= WALL_IN - 1)

# ------------------------------------------------------------------ heights
n1 = fbm(1)
n2 = fbm(2, ((60, 1.0), (22, 0.5), (9, 0.2)))
edge_noise = fbm(3, ((48, 1.0), (16, 0.4)))
ISL = (E + 14 * edge_noise) <= ISL_R
ein = np.clip((E + 14 * edge_noise) / ISL_R, 0, 1)

biome_h = G + 10 * n1 + 6 * n2 - 9 * np.clip((ein - 0.8) / 0.2, 0, 1) ** 2
blend = np.clip((R - PLATEAU) / 14.0, 0, 1)
blend = blend * blend * (3 - 2 * blend)
gh_f = np.where(R <= PLATEAU, P, P * (1 - blend) + biome_h * blend)
cragmask = (crag > P) & (R <= WALL_IN - 6)
gh_f = np.where(cragmask, np.maximum(gh_f, crag), gh_f)
gh_f = np.where(FORE, CP, gh_f)
gh_f = np.where(RAMP, np.maximum(np.where(cragmask, gh_f, P), 0) * 0 + RAMP_H, gh_f)

# stream: meanders from a spring west of the moat to the island edge (waterfall)
rng = np.random.default_rng(7)
pts = [(-126.0, -52.0)]
ang = math.radians(200)
while True:
    x, z = pts[-1]
    if math.hypot(x, z) > ISL_R + 12:
        break
    ang += rng.uniform(-0.35, 0.35)
    ang = ang * 0.85 + math.atan2(z, x) * 0.15
    pts.append((x + 3 * math.cos(ang), z + 3 * math.sin(ang)))
STREAM_PTS = [(int(round(x + C)), int(round(z + C))) for x, z in pts]
stream_d = np.full((S, S), 99.0)
for x, z in STREAM_PTS:
    x0, x1, z0, z1 = max(x - 6, 0), min(x + 7, S), max(z - 6, 0), min(z + 7, S)
    d = np.hypot(X[z0:z1, x0:x1] - x, Z[z0:z1, x0:x1] - z)
    stream_d[z0:z1, x0:x1] = np.minimum(stream_d[z0:z1, x0:x1], d)
STREAM = (stream_d <= 2.2) & ISL & (R > PLATEAU)
BANK = (stream_d <= 4.5) & ISL & (R > PLATEAU)

gh = np.rint(gh_f).astype(np.int32)
gh[BANK] = np.minimum(gh[BANK], np.rint(gh_f[BANK] - (4.5 - stream_d[BANK]) * 0.6).astype(np.int32))
# stream water surface: monotone downstream so it reads as a flowing brook with small cascades
STREAM_W = np.full((S, S), -1, np.int32)
lvl = 999
for x, z in STREAM_PTS:
    if not (0 <= x < S and 0 <= z < S):
        continue
    lvl = min(lvl, int(gh[z, x]) - 1)
    x0, x1, z0, z1 = max(x - 3, 0), min(x + 4, S), max(z - 3, 0), min(z + 4, S)
    sub = STREAM[z0:z1, x0:x1]
    STREAM_W[z0:z1, x0:x1] = np.where(sub & (STREAM_W[z0:z1, x0:x1] < 0), lvl, STREAM_W[z0:z1, x0:x1])
gh[STREAM] = np.minimum(gh[STREAM], STREAM_W[STREAM] - 1)

ponds = ISL & (R > PLATEAU + 8) & (gh < WL) & (ein < 0.93)

depth_n = fbm(4, ((40, 1.0), (14, 0.6), (6, 0.3)))
ybot = np.rint(G - 4 - 34 * np.sqrt(np.clip(1 - ein ** 2, 0, 1)) * (0.75 + 0.35 * depth_n)).astype(np.int32)
ybot = np.clip(np.minimum(ybot, gh - 2), 2, None)

# slope for surface material choices
slope = np.zeros_like(gh)
for sh, a in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
    slope = np.maximum(slope, np.abs(gh - np.roll(gh, sh, axis=a)))

# ------------------------------------------------------------------ materials
class Strata:
    """Sedimentary layering: bands warped by 2D/3D noise, darker and denser with depth, thin bright calcite seams,
    weathered (cobbled/mossy) faces only near the surface."""
    SEQ = ["stone", "stone", "andesite", "tuff", "stone", "calcite", "andesite", "stone", "diorite", "tuff",
           "stone", "dripstone_block", "andesite", "stone", "tuff", "deepslate", "stone", "andesite"]

    def __init__(self, seed=21):
        self.ids = np.array([B(n) for n in self.SEQ], np.uint16)
        self.seed = seed

    def at(self, x, y, z):
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        warp = mc.fbm3(x, 0, z, 36.0, self.seed, 2) * 7 + mc.noise3(x, y, z, 7.0, self.seed + 1) * 1.6
        t = (y + warp) / 2.6
        k = np.floor(t).astype(np.int64)
        out = self.ids[k % len(self.ids)]
        deep = (y + warp * 0.5) < (G - 14)
        out = np.where(deep & (mc.noise3(x, y, z, 5.0, self.seed + 2) > 0.1), B("deepslate"), out)
        return out


ROCK = Strata()
UNDER = Mix([("stone", 35), ("tuff", 20), ("deepslate", 15), ("andesite", 10), ("dripstone_block", 12), ("calcite", 4),
             ("coal_ore", 2), ("copper_ore", 2)], seed=22, patch=4)
DIRT = Mix([("dirt", 60), ("coarse_dirt", 22), ("rooted_dirt", 18)], seed=23, patch=2)
COBBLE = Mix([("cobblestone", 22), ("stone", 14), ("andesite", 14), ("polished_andesite", 10), ("gravel", 6),
              ("tuff", 10), ("mossy_cobblestone", 10), ("stone_bricks", 8), ("packed_mud", 6)], seed=24, patch=2)
MARKET_P = Mix([("polished_andesite", 24), ("stone_bricks", 20), ("andesite", 12), ("polished_tuff", 14),
                ("tuff_bricks", 12), ("mossy_stone_bricks", 8), ("cracked_stone_bricks", 6), ("chiseled_stone_bricks", 4)], seed=25)
COURT = Mix([("polished_tuff", 26), ("tuff_bricks", 22), ("polished_andesite", 18), ("stone_bricks", 14),
             ("chiseled_tuff_bricks", 6), ("mossy_stone_bricks", 8), ("cobblestone", 6)], seed=26, patch=2)
WST = Mix([("stone_bricks", 32), ("mossy_stone_bricks", 14), ("cracked_stone_bricks", 10), ("tuff_bricks", 14),
           ("polished_andesite", 10), ("andesite", 8), ("mossy_cobblestone", 6), ("chiseled_stone_bricks", 2),
           ("cobblestone", 4)], seed=27, patch=3)

GRASS, MOSS, PMOSS, PODZOL, CDIRT, ROOTED = (B("grass_block[snowy=false]"), B("moss_block"), B("pale_moss_block"),
                                            B("podzol[snowy=false]"), B("coarse_dirt"), B("rooted_dirt"))
PATH, GRAVEL, SAND, CLAY, MUD = B("dirt_path"), B("gravel"), B("sand"), B("clay"), B("mud")
WATER = B("water[level=0]")

hs = (np.clip(mc.fbm3(X, 0, Z, 3.0, 55, 2) / 0.6, -1, 1) * 0.5 + 0.5) * 1000   # coherent, not per-block random
mm = fbm(6, ((30, 1.0), (11, 0.6), (5, 0.3)))
top = np.full((S, S), GRASS, np.uint16)
top[mm > 0.35] = MOSS
top[(mm > 0.55) & (n2 > 0.1)] = PMOSS
top[mm < -0.38] = PODZOL
top[(mm < -0.55)] = CDIRT
top[(hs < 40)] = ROOTED
top[(slope >= 3) & ~(R <= PLATEAU)] = ROCK.at(X, 0, Z)[(slope >= 3) & ~(R <= PLATEAU)]
top[BANK] = np.where(hs < 500, GRAVEL, np.where(hs < 800, MUD, CLAY))[BANK]
top[ponds] = np.where(hs < 600, SAND, CLAY)[ponds]
forest_path = ISL & (R > PLATEAU + 3) & ((np.abs(np.hypot(U, V) - 152) <= 1.3) | ((np.minimum(np.abs(U), np.abs(V)) <= 1) & (R > PLATEAU + 2)))
forest_path &= ~STREAM & (ein < 0.97)
top[forest_path] = np.where(hs < 750, PATH, CDIRT)[forest_path]

# city ground
city_ground = R <= PLATEAU
top[city_ground] = np.where(hs < 700, GRASS, np.where(hs < 850, PATH, CDIRT))[city_ground]
top[STREET & city_ground] = COBBLE.at(X, 1, Z)[STREET & city_ground]
top[MARKET] = MARKET_P.at(X, 2, Z)[MARKET]
mk_ring = MARKET & ((np.abs(np.hypot(U - 64, V - 80) - 9) < 0.8) | (np.abs(np.hypot(U - 64, V - 80) - 14) < 0.8))
top[mk_ring] = B("polished_tuff")
top[GARDEN] = np.where(hs < 820, GRASS, MOSS)[GARDEN]
garden_path = GARDEN & ((np.abs(V) <= 1) | (np.abs(U + 79) <= 1) | (np.abs(np.hypot(U + 79, V) - 9) <= 1))
top[garden_path] = np.where(hs < 600, GRAVEL, B("polished_andesite"))[garden_path]
EMB = (R > WALL_OUT) & (R < MOAT_IN)
top[EMB] = COBBLE.at(X, 3, Z)[EMB]
CRAGC = cragmask & ~PLATFORM & ~RAMP & ~FORE
gentle = CRAGC & (slope <= 1)
top[CRAGC] = ROCK.at(X, 4, Z)[CRAGC]
top[gentle] = np.where(hs < 600, GRASS, MOSS)[gentle]
top[PLATFORM | FORE] = COURT.at(X, 5, Z)[PLATFORM | FORE]


def write():
    """Write ground, water, moat; returns nothing (fills mc.arr)."""
    rampH_i = np.floor(RAMP_H).astype(np.int32)
    ramp_half = (RAMP_H - rampH_i) >= 0.5
    for y in range(Y):
        L = arr[y]
        solid = ISL & (y >= ybot) & (y <= gh)
        if not solid.any() and y > CP + 2:
            continue
        depth = gh - y
        ids = np.where(y < ybot + 6, UNDER.at(X, y, Z), np.where(depth > 4, ROCK.at(X, y, Z), DIRT.at(X, y, Z)))
        city_sub = (R <= PLATEAU) & (depth >= 1)
        ids = np.where(city_sub, np.where(depth <= 2, DIRT.at(X, y, Z), ROCK.at(X, y, Z)), ids)
        ids = np.where(cragmask & (depth >= 1), ROCK.at(X, y, Z), ids)
        ids = np.where(depth == 0, top, ids)
        L[solid] = ids[solid]
        # ramp: slabs on half steps, stone brick body
        rm = RAMP & (y <= rampH_i) & (y >= P - 2)
        L[rm] = WST.at(X, y, Z)[rm]
        rh = RAMP & ramp_half & (y == rampH_i + 1)
        L[rh] = B("stone_brick_slab[type=bottom,waterlogged=false]")
        rc = RAMP & (y > rampH_i + (ramp_half.astype(np.int32))) & (y <= rampH_i + 6)
        L[rc] = 0
        # forecourt bastion body
        fm = FORE & (y <= CP) & (y >= P - 3)
        L[fm] = WST.at(X, y, Z)[fm]
        L[FORE & (y == CP)] = COURT.at(X, y, Z)[FORE & (y == CP)]
        # water: moat, ponds, stream
        moat = (R >= MOAT_IN) & (R <= MOAT_OUT)
        if P - 8 <= y <= P:
            if y == P - 8:
                L[moat] = B("mud_bricks")
            elif y < P:
                L[moat] = WATER
            else:
                L[moat] = 0
            lin = ((R == MOAT_IN - 1) | (R == MOAT_OUT + 1)) & (y >= P - 8)
            L[lin] = WST.at(X, y, Z)[lin]
        pm = ponds & (y > gh) & (y <= WL)
        L[pm] = WATER
        sm = STREAM & (y > gh) & (y <= STREAM_W)
        L[sm] = WATER
        # plateau retaining wall (outside the moat)
        ret = (R >= PLATEAU - 1) & (R <= PLATEAU) & (y <= P) & (y >= np.minimum(gh, P) - 12) & ISL
        L[ret] = WST.at(X, y, Z)[ret]
