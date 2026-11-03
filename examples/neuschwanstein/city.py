"""The town inside the walls: wall + gatehouses + towers, bridges, market square with fountain and stalls,
half-timbered houses with interiors, chapel, smithy, tavern, stable, palace garden, street lamps, trees."""
import math

import numpy as np

import arch as A
import houses
import mc
import terrain as T
import trees
from mc import B, C, DV, OPP, Mix, StairMix, fill, get, put, slab, stairs
from terrain import P

rng = np.random.default_rng(1234)
X = lambda u: C + u          # noqa: E731
Zc = lambda v: C + v         # noqa: E731
LIT = A.LANT
OCC = np.zeros((mc.S, mc.S), bool)      # occupied footprint cells (houses etc.)

FRAME = Mix([("stripped_dark_oak_log[axis=y]", 60), ("dark_oak_log[axis=y]", 40)], seed=81)
INFILL = [Mix([("calcite", 40), ("mushroom_stem", 30), ("smooth_quartz", 15), ("white_terracotta", 15)], seed=82, patch=2),
          Mix([("white_terracotta", 40), ("calcite", 35), ("pale_oak_planks", 25)], seed=83, patch=2),
          Mix([("mushroom_stem", 45), ("diorite", 20), ("calcite", 35)], seed=84, patch=2),
          Mix([("smooth_sandstone", 50), ("cut_sandstone", 20), ("calcite", 30)], seed=85, patch=2)]
GROUND = Mix([("cobblestone", 30), ("stone_bricks", 20), ("mossy_cobblestone", 15), ("andesite", 15), ("tuff", 10),
              ("mossy_stone_bricks", 10)], seed=86, patch=2)
ROOFS = [(Mix([("dark_oak_planks", 1)]), StairMix(["dark_oak"]), "dark_oak"),
         (Mix([("spruce_planks", 1)]), StairMix(["spruce"]), "spruce"),
         (A.SLATE, A.SLATE_ST, "deepslate_tile"),
         (Mix([("bricks", 70), ("mud_bricks", 30)]), StairMix(["brick", "mud_brick"]), "brick"),
         (Mix([("mangrove_planks", 1)]), StairMix(["mangrove"]), "mangrove")]


# ------------------------------------------------------------------ walls, gates, bridges
def town_wall():
    y0, y1 = P + 1, P + 18
    wi, wo = T.WALL_IN, T.WALL_OUT
    for y in range(P - 3, y1 + 1):
        m = (T.R >= wi) & (T.R <= wo)
        mc.arr[y][m] = A.STONEWALL.at(T.X, y, T.Z)[m]
    # walkway, crenels with slab caps, corbel line
    ring_out = T.R == wo
    ring_in = T.R == wi
    cren = ring_out & (((T.U + T.V) // 2) % 2 == 0)
    mc.arr[y1 + 1][ring_out] = A.STONEWALL.at(T.X, y1 + 1, T.Z)[ring_out]
    mc.arr[y1 + 2][cren] = A.STONEWALL.at(T.X, y1 + 2, T.Z)[cren]
    mc.arr[y1 + 3][cren] = slab("stone_brick")
    mc.arr[y1 + 2][ring_out & ~cren] = slab("stone_brick")
    mc.arr[y1 + 1][ring_in] = A.POST_STONE
    corb = T.R == wo + 1
    for x, z in zip(*np.nonzero(corb.T)):
        put(int(x), y1 - 1, int(z), stairs("stone_brick", mc.vdir(C - x if abs(x - C) >= abs(z - C) else 0, C - z if abs(z - C) > abs(x - C) else 0), "top"))
    # arrow slits
    sl = (T.R == wo) & (np.minimum(np.abs(T.U), np.abs(T.V)) % 9 == 4)
    for y in range(y0 + 6, y0 + 9):
        mc.arr[y][sl] = 0
    # towers: corners + mid-sides (offset from the gates)
    for su in (-1, 1):
        for sv in (-1, 1):
            A.round_tower(X(su * 106), Zc(sv * 106), 8, P - 4, P + 30, wall=A.STONEWALL, crown=True, cone=16, floors=9)
    for (u, v) in ((-106, 52), (106, -48), (55, -106), (-50, 106), (-106, -60), (106, 58), (-58, -106), (48, 106)):
        A.round_tower(X(u), Zc(v), 5.5, P - 4, P + 24, wall=A.STONEWALL, crown=True, cone=12, floors=8)
    # four gatehouses with arched passage, red-brick upper storey, slate roof
    for side in ("north", "south", "east", "west"):
        dx, dz = DV[side]
        cx, cz = X(dx * 106), Zc(dz * 106)
        if dx:
            x0, x1, z0, z1 = cx - 5, cx + 5, cz - 8, cz + 8
        else:
            x0, x1, z0, z1 = cx - 8, cx + 8, cz - 5, cz + 5
        fill(x0, P - 3, z0, x1, P + 22, z1, A.STONEWALL)
        fill(x0, P + 14, z0, x1, P + 22, z1, A.REDB)
        for k in range(P + 14, P + 23, 4):
            fill(x0 - 1 if dx else x0, k, z0 - 1 if dz else z0, x1 + 1 if dx else x1, k, z1 + 1 if dz else z1, 0, only_air=False) if False else None
        # passage
        for y in range(P + 1, P + 10):
            hw = 3 if y < P + 8 else (2 if y < P + 9 else 1)
            if dx:
                fill(x0 - 1, y, cz - hw, x1 + 1, y, cz + hw, 0)
            else:
                fill(cx - hw, y, z0 - 1, cx + hw, y, z1 + 1, 0)
        A.gable(x0, x1, z0, z1, P + 23, axis=("z" if dx else "x"), rise=2, over=1, end_wall=A.REDB)
        for t in (-1, 1):
            if dx:
                A.round_tower(cx + 6 * dx, cz + t * 9, 3, P - 3, P + 26, wall=A.STONEWALL, crown=True, cone=9, floors=8)
            else:
                A.round_tower(cx + t * 9, cz + 6 * dz, 3, P - 3, P + 26, wall=A.STONEWALL, crown=True, cone=9, floors=8)
        # bridge across the moat
        for k in range(T.WALL_OUT + 1, T.MOAT_OUT + 3):
            for w in range(-3, 4):
                x, z = (cx + dx * (k - 106), cz + w) if dx else (cx + w, cz + dz * (k - 106))
                put(x, P, z, A.STONEWALL.at(x, P, z))
                if abs(w) == 3:
                    put(x, P + 1, z, A.POST_STONE if k % 3 else A.STONEWALL.at(x, P, z))
                    if k % 6 == 0:
                        put(x, P + 2, z, LIT)
            # arches under the bridge
            if k % 5 == 0:
                for w in range(-3, 4):
                    x, z = (cx + dx * (k - 106), cz + w) if dx else (cx + w, cz + dz * (k - 106))
                    fill(x, P - 8, z, x, P - 1, z, A.STONEWALL)


# ------------------------------------------------------------------ houses
def free(x0, x1, z0, z1, pad=1):
    xa, xb, za, zb = x0 - pad, x1 + pad, z0 - pad, z1 + pad
    if xa < 0 or za < 0 or xb >= mc.S or zb >= mc.S:
        return False
    sub = (slice(za, zb + 1), slice(xa, xb + 1))
    if OCC[sub].any() or T.STREET[sub].any() or T.MARKET[sub].any() or T.GARDEN[sub].any():
        return False
    if (T.gh[sub] != P).any() or (T.R[sub] >= T.WALL_IN - 7).any() or T.RAMP[sub].any():
        return False
    return True


def nearest_street_side(x0, x1, z0, z1):
    best, bd = "south", 99
    for side, (dx, dz) in DV.items():
        cx, cz = (x0 + x1) // 2, (z0 + z1) // 2
        for d in range(1, 14):
            x = (x1 + d) if dx > 0 else (x0 - d) if dx < 0 else cx
            z = (z1 + d) if dz > 0 else (z0 - d) if dz < 0 else cz
            if 0 <= x < mc.S and 0 <= z < mc.S and (T.STREET[z, x] or T.MARKET[z, x]):
                if d < bd:
                    best, bd = side, d
                break
    return best, bd


def house(x0, x1, z0, z1, front, storeys, kind="home"):
    OCC[z0 - 1:z1 + 2, x0 - 1:x1 + 2] = True
    infill = INFILL[int(rng.integers(len(INFILL)))]
    roof = ROOFS[int(rng.integers(len(ROOFS)))]
    y0 = P
    hs = 5
    ytop = y0 + storeys * hs
    # ground floor stone, upper floors half-timbered
    A.shell(x0, x1, z0, z1, y0 + 1, y0 + hs, GROUND, Mix([("spruce_planks", 1)]), thick=1)
    A.shell(x0, x1, z0, z1, y0 + hs + 1, ytop, infill, Mix([("spruce_planks", 1)]), thick=1)
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            edge = x in (x0, x1) or z in (z0, z1)
            if not edge:
                continue
            corner = x in (x0, x1) and z in (z0, z1)
            along = (x - x0) if z in (z0, z1) else (z - z0)
            if corner or along % 3 == 0:
                fill(x, y0 + hs + 1, z, x, ytop, z, FRAME)
            for k in range(1, storeys):
                axis = "x" if z in (z0, z1) else "z"
                put(x, y0 + k * hs, z, B(f"stripped_dark_oak_log[axis={axis}]") if not corner else B("dark_oak_log[axis=y]"))
            put(x, ytop, z, B(f"dark_oak_log[axis={'x' if z in (z0, z1) else 'z'}]"))
            # diagonal braces in some panels
            if not corner and along % 3 == 1 and storeys > 1 and int(mc.hash3(x0, z0, along)) % 3 == 0:
                for k in range(1, 3):
                    put(x, y0 + hs + k, z, B("dark_oak_planks"))
    # jetty corbels (upside-down stairs) under the timber storey on all sides
    for side in A.SIDES:
        line, (nx, nz) = A.side_line(x0, x1, z0, z1, side)
        for (x, z) in line[1:-1:2]:
            put(x + nx, y0 + hs, z + nz, stairs("dark_oak", OPP[side], "top"))
    # floors, windows, door
    for k in range(storeys):
        fill(x0 + 1, y0 + k * hs, z0 + 1, x1 - 1, y0 + k * hs, z1 - 1, A.FLOOR_MIX)
    for side in A.SIDES:
        line, (nx, nz) = A.side_line(x0, x1, z0, z1, side)
        n = len(line)
        axis = "x" if side in ("north", "south") else "z"
        for k in range(storeys):
            wy = y0 + k * hs + 2
            for c in range(2, n - 2, 4):
                if side == front and k == 0 and abs(c - n // 2) <= 2:
                    continue
                for cc in (c, c + 1):
                    if cc >= n - 1:
                        continue
                    x, z = line[cc]
                    put(x, wy, z, A.PANE[axis])
                    put(x, wy + 1, z, A.PANE[axis])
                    put(x + nx, wy - 1, z + nz, B(f"spruce_trapdoor[facing={side},half=top,open=false,powered=false,waterlogged=false]"))
                    pot = ["potted_red_tulip", "potted_azure_bluet", "potted_cornflower", "potted_poppy", "potted_allium", "potted_fern"]
                    put(x + nx, wy, z + nz, B(pot[int(rng.integers(len(pot)))]))
                for cc, sd in ((c - 1, side), (c + 2, side)):
                    if 0 < cc < n - 1:
                        x, z = line[cc]
                        for h in (0, 1):
                            put(x + nx, wy + h, z + nz, B(f"spruce_trapdoor[facing={side},half=bottom,open=true,powered=false,waterlogged=false]"))
    line, (nx, nz) = A.side_line(x0, x1, z0, z1, front)
    n = len(line)
    dx_, dz_ = line[n // 2]
    for h in range(1, 4):
        put(dx_, y0 + h, dz_, 0)
    put(dx_, y0 + 1, dz_, B(f"spruce_door[facing={OPP[front]},half=lower,hinge=left,open=false,powered=false]"))
    put(dx_, y0 + 2, dz_, B(f"spruce_door[facing={OPP[front]},half=upper,hinge=left,open=false,powered=false]"))
    put(dx_ + nx, y0, dz_ + nz, stairs("stone_brick", OPP[front]))
    put(dx_ + nx + DV[mc.CW[front]][0], y0 + 3, dz_ + nz + DV[mc.CW[front]][1], LIT)
    # roof
    axis = "x" if (x1 - x0) >= (z1 - z0) else "z"
    mat, st, cap = roof
    rtop = A.gable(x0, x1, z0, z1, ytop + 1, axis=axis, rise=1 if rng.random() < 0.55 else 2, over=1, mat=mat, st=st,
                   end_wall=infill, cap=cap if cap != "brick" else "brick")
    # chimney with smoke
    cxh = x1 - 2 if rng.random() < 0.5 else x0 + 2
    czh = z1 - 2 if rng.random() < 0.5 else z0 + 2
    fill(cxh, y0 + 1, czh, cxh, rtop + 2, czh, Mix([("bricks", 70), ("stone_bricks", 30)], seed=87))
    put(cxh, rtop + 3, czh, B("campfire[facing=north,lit=true,signal_fire=false,waterlogged=false]"))
    put(cxh, y0 + 1, czh, B("furnace[facing=north,lit=true]") if czh < (z0 + z1) / 2 else B("furnace[facing=south,lit=true]"))
    # interior
    ix0, ix1, iz0, iz1 = x0 + 1, x1 - 1, z0 + 1, z1 - 1
    put(ix0, y0 + 1, iz0, B("crafting_table"))
    put(ix0 + 1, y0 + 1, iz0, B("barrel[facing=up,open=false]"))
    put(ix1, y0 + 1, iz1, B("chest[facing=north,type=single,waterlogged=false]"))
    tx, tz = (ix0 + ix1) // 2, (iz0 + iz1) // 2
    put(tx, y0 + 1, tz, B("spruce_fence[east=false,north=false,south=false,west=false,waterlogged=false]"))
    put(tx, y0 + 2, tz, B("spruce_pressure_plate[powered=false]"))
    put(tx + 1, y0 + 1, tz, stairs("spruce", "east"))
    put(tx - 1, y0 + 1, tz, stairs("spruce", "west"))
    A.chandelier(tx, y0 + hs - 1, tz + 1, 1)
    if storeys > 1:
        put(ix1, y0 + 1, iz0, B("ladder[facing=west,waterlogged=false]") if False else 0)
        for h in range(1, hs + 1):
            put(ix1, y0 + h, iz0 + 1, B("ladder[facing=north,waterlogged=false]"))
        put(ix1, y0 + hs, iz0 + 1, B("ladder[facing=north,waterlogged=false]"))
        bedc = ["red", "blue", "yellow", "green", "white", "brown"][int(rng.integers(6))]
        A.bed(ix0, y0 + hs + 1, iz1 - 1, "north" if iz1 - 1 > iz0 + 1 else "south", bedc) if iz1 - iz0 >= 3 else None
        A.carpet_run(ix0 + 1, ix1 - 1, iz0 + 2, iz1 - 2, y0 + hs + 1, bedc)
        put(ix1 - 1, y0 + hs + 1, iz1, B("bookshelf"))
        A.chandelier(tx, ytop - 1, tz, 1)
    if kind == "tavern":
        for k in range(3):
            put(ix0, y0 + 1, iz0 + 2 + k, B("barrel[facing=east,open=false]"))
            put(ix0, y0 + 2, iz0 + 2 + k, B("barrel[facing=east,open=false]"))
    return rtop


def place_houses(n_target=80):
    """Lots along the streets; each House is randomised (style, size, roof, wing, details)."""
    placed = 0
    tries = 0
    while placed < n_target and tries < 60000:
        tries += 1
        w, d = int(rng.integers(6, 13)), int(rng.integers(7, 14))
        if rng.random() < 0.5:
            w, d = d, w
        x0, z0 = int(rng.integers(C - 100, C + 100)), int(rng.integers(C - 100, C + 100))
        x1, z1 = x0 + w - 1, z0 + d - 1
        if not free(x0, x1, z0, z1, pad=1):
            continue
        side, dist = nearest_street_side(x0, x1, z0, z1)
        if dist > (4 if placed < n_target * 0.6 else 6):
            continue
        h = houses.House(x0, z0, w, d, side, rng, P)
        if h.wing and not free(*h.wing, pad=1):
            h.wing = None
        OCC[z0 - 1:z1 + 2, x0 - 1:x1 + 2] = True
        if h.wing:
            wx0, wx1, wz0, wz1 = h.wing
            OCC[wz0 - 2:wz1 + 3, wx0 - 2:wx1 + 3] = True
        (x, z), (nx, nz) = h.build()
        # worn footpath from the door to the street
        for k in range(1, dist + 2):
            for w2 in (-1, 0, 1):
                xx, zz = (x + nx * k + (w2 if nz else 0), z + nz * k + (w2 if nx else 0))
                if 0 <= xx < mc.S and 0 <= zz < mc.S and not T.STREET[zz, xx] and get(xx, P + 1, zz) == 0:
                    v = mc.noise3(xx, 0, zz, 2.5, 77)
                    put(xx, P, zz, B("dirt_path") if (w2 == 0 or v > 0.25) else (B("coarse_dirt") if v > -0.2 else B("gravel")))
        placed += 1
    return placed


def streets():
    """Street surfaces from gradients, not noise: gutters (damp, mossy, drains, weeds) -> wheel ruts ->
    worn smooth crown; repair patches and puddles from low-frequency noise; curb line along the edge;
    market square laid in concentric courses around the fountain."""
    from scipy.ndimage import distance_transform_edt, maximum_filter
    st = T.STREET & (T.R <= T.WALL_IN - 1) & ~T.MARKET & (T.gh == P)
    d = distance_transform_edt(st)
    dloc = maximum_filter(d, size=9)
    ys, xs = np.nonzero(st)
    ID = {n: B(n) for n in ("cobblestone", "mossy_cobblestone", "stone", "andesite", "polished_andesite", "gravel",
                            "packed_mud", "coarse_dirt", "mud", "smooth_stone", "stone_bricks", "tuff", "cracked_stone_bricks")}
    for z, x in zip(ys, xs):
        dd, dl = d[z, x], max(dloc[z, x], 1.0)
        rel = dd / dl
        patch = mc.fbm3(x, 0, z, 9.0, 31, 2)
        fine = mc.noise3(x, 0, z, 2.0, 32)
        puddle = mc.fbm3(x, 0, z, 14.0, 33, 2)
        if dd <= 1.01:                                        # gutter
            b = ID["mossy_cobblestone"] if fine > 0.1 else (ID["gravel"] if fine < -0.45 else ID["cobblestone"])
            if (x * 7 + z * 13) % 31 == 0:
                b = B("iron_trapdoor[facing=north,half=top,open=false,powered=false,waterlogged=false]")
            elif fine > 0.55:
                put(x, P + 1, z, B("short_grass"), only_air=True)
        elif dl >= 3.5 and abs(dd - 2) < 0.5 and patch > -0.1 and fine > 0.0:   # faint wheel ruts
            b = ID["gravel"] if fine < 0.3 else ID["packed_mud"]
        elif rel > 0.8 and patch + fine * 0.6 > 0.05:         # worn crown, broken up by wear noise
            b = ID["polished_andesite"] if fine > 0.15 else ID["stone"]
        else:
            b = ID["cobblestone"] if patch < 0.15 else (ID["andesite"] if patch < 0.32 else ID["stone_bricks"])
            if patch < -0.35:
                b = ID["stone"]
        if dd >= 2 and puddle < -0.5:
            b = ID["mud"] if puddle > -0.62 else B("water[level=0]")
        mc.arr[P, z, x] = b
    # curb: one course of dressed stone where the street meets grass
    edge = (distance_transform_edt(~st) <= 1.0) & ~st & (T.R <= T.WALL_IN - 2) & (T.gh == P) & ~T.MARKET
    for z, x in zip(*np.nonzero(edge)):
        if mc.arr[P + 1, z, x] == 0:
            mc.arr[P, z, x] = B("stone_bricks") if mc.noise3(x, 0, z, 4, 34) > -0.3 else B("cracked_stone_bricks")
    # market: concentric courses around the fountain, wear lines between the stalls
    cx, cz = X(64), Zc(80)
    mk = T.MARKET
    for z, x in zip(*np.nonzero(mk)):
        r = math.hypot(x - cx, z - cz)
        band = int(r / 2.0) % 3
        b = (B("stone_bricks"), B("polished_andesite"), B("tuff_bricks"))[band]
        if int(r) % 6 == 5:
            b = B("chiseled_tuff_bricks") if (int(math.degrees(math.atan2(z - cz, x - cx))) // 15) % 2 else B("polished_tuff")
        wear = mc.fbm3(x, 0, z, 6, 35, 2)
        if wear > 0.38:
            b = B("smooth_stone") if wear < 0.5 else B("polished_andesite")
        elif wear < -0.45:
            b = B("mossy_stone_bricks")
        mc.arr[P, z, x] = b


# ------------------------------------------------------------------ market square
def fountain(cx, cz):
    for x, z, d in A.ring_cells(cx, cz, 0, 6.4):
        if d > 5.4:
            fill(x, P + 1, z, x, P + 1, z, A.STONEWALL)
            put(x, P + 2, z, slab("smooth_stone"))
        else:
            fill(x, P - 1, z, x, P, z, B("water[level=0]"))
            put(x, P - 2, z, B("prismarine_bricks"))
            put(x, P + 1, z, B("water[level=0]"))
    for x, z, d in A.ring_cells(cx, cz, 0, 1.6):
        fill(x, P - 1, z, x, P + 5, z, A.STONEWALL)
    for x, z, d in A.ring_cells(cx, cz, 1.6, 2.7):
        put(x, P + 4, z, slab("smooth_quartz", "top"))
        put(x, P + 5, z, B("water[level=0]"))
    fill(int(cx), P + 6, int(cz), int(cx), P + 7, int(cz), B("quartz_pillar[axis=y]"))
    put(int(cx), P + 8, int(cz), B("chiseled_quartz_block"))
    put(int(cx), P + 9, int(cz), B("water[level=0]"))
    for k in range(8):
        a = k * math.pi / 4
        x, z = int(round(cx + 8.5 * math.cos(a))), int(round(cz + 8.5 * math.sin(a)))
        put(x, P + 1, z, stairs("spruce", mc.vdir(-math.cos(a), -math.sin(a))))
        put(x + (1 if abs(math.sin(a)) > 0.5 else 0), P + 1, z + (1 if abs(math.cos(a)) > 0.5 else 0),
            stairs("spruce", mc.vdir(-math.cos(a), -math.sin(a))))


def stall(x, z, facing, colors, goods):
    """Market stall: fence posts, striped wool canopy, counter with goods, barrels behind."""
    dx, dz = DV[facing]
    rx, rz = (1, 0) if dz else (0, 1)
    OCC[min(z, z + 4 * dz) - 1:max(z, z + 4 * dz) + 2, min(x, x + 4 * rx) - 1:max(x, x + 4 * rx) + 2] = True
    for i in range(5):
        for j in range(4):
            xx, zz = x + i * rx - j * dx, z + i * rz - j * dz
            if i in (0, 4) and j in (0, 3):
                fill(xx, P + 1, zz, xx, P + 3, zz, B("spruce_fence[east=false,north=false,south=false,west=false,waterlogged=false]"))
            col = colors[i % 2]
            put(xx, P + 4, zz, B(f"{col}_wool") if j in (1, 2) else B(f"{col}_carpet"), only_air=False)
            if j == 0:
                put(xx, P + 4, zz, B(f"{col}_wool"))
                put(xx + dx, P + 4, zz + dz, B(f"{col}_carpet") if False else 0)
        xx, zz = x + i * rx, z + i * rz
        if 0 < i < 4:
            put(xx, P + 1, zz, B("barrel[facing=up,open=false]") if i == 2 else B("spruce_planks"))
            put(xx, P + 2, zz, B(goods[i % len(goods)]))
            put(xx - 3 * dx, P + 1, zz - 3 * dz, B("barrel[facing=up,open=false]"))
    put(x + 2 * rx - dx, P + 3, z + 2 * rz - dz, A.LANT_H)


def market():
    cx, cz = X(64), Zc(80)
    fountain(cx, cz)
    OCC[cz - 16:cz + 17, cx - 16:cx + 17] = True
    stalls = [(X(40), Zc(68), "south", ("red", "white"), ["melon", "pumpkin", "hay_block[axis=y]"]),
              (X(48), Zc(68), "south", ("yellow", "orange"), ["potted_red_tulip", "potted_cornflower", "potted_allium"]),
              (X(76), Zc(68), "south", ("blue", "white"), ["cake[bites=0]", "candle[candles=4,lit=false,waterlogged=false]", "honey_block"]),
              (X(84), Zc(68), "south", ("green", "lime"), ["melon", "carved_pumpkin[facing=south]", "beehive[facing=south,honey_level=0]"]),
              (X(40), Zc(93), "north", ("purple", "magenta"), ["amethyst_cluster[facing=up,waterlogged=false]", "decorated_pot[cracked=false,facing=north,waterlogged=false]", "lantern[hanging=false,waterlogged=false]"]),
              (X(48), Zc(93), "north", ("brown", "orange"), ["barrel[facing=up,open=false]", "hay_block[axis=y]", "dried_kelp_block"]),
              (X(84), Zc(93), "north", ("cyan", "light_blue"), ["sea_lantern", "prismarine", "dried_kelp_block"])]
    for x, z, f, cols, goods in stalls:
        stall(x, z, f, cols, goods)
    # trees in stone planters, benches, lamp posts
    for (u, v) in ((38, 80), (90, 80), (64, 99), (52, 75), (78, 87)):
        x, z = X(u), Zc(v)
        for xx, zz, d in A.ring_cells(x, z, 0, 2.6):
            put(xx, P, zz, B("podzol[snowy=false]") if d < 1.8 else A.STONEWALL.at(xx, P, zz))
            if d >= 1.8:
                put(xx, P + 1, zz, slab("stone_brick"))
        trees.oak(x, P, z, rng, size=0.7, hang=0.0)
        OCC[z - 3:z + 4, x - 3:x + 4] = True
    for (u, v) in ((44, 80), (84, 80), (64, 72), (64, 89)):
        for k in (-1, 0, 1):
            x, z = X(u) + (k if v in (72, 89) else 0), Zc(v) + (k if v not in (72, 89) else 0)
            put(x, P + 1, z, stairs("spruce", "north" if v == 89 else "south" if v == 72 else ("west" if u > 64 else "east")))
    for (u, v) in ((36, 66), (92, 66), (36, 95), (92, 95), (60, 66), (68, 95)):
        lamp(X(u), Zc(v))


def lamp(x, z, h=4):
    put(x, P + 1, z, B("polished_andesite"))
    fill(x, P + 2, z, x, P + h, z, A.POST_STONE)
    put(x, P + h + 1, z, LIT)


# ------------------------------------------------------------------ special buildings
def chapel(x0, z0):
    x1, z1 = x0 + 10, z0 + 18
    OCC[z0 - 2:z1 + 3, x0 - 2:x1 + 9] = True
    A.shell(x0, x1, z0, z1, P + 1, P + 12, A.PLASTER, A.PLASTER)
    for side in ("east", "west"):
        A.facade(x0, x1, z0, z1, side, P + 1, P + 12, [(P, 3, 5)], bay=4, style="arch2", pilasters=True)
    A.facade(x0, x1, z0, z1, "south", P + 1, P + 12, [(P, 3, 5)], bay=11, style="arch2", door_at=5, door_w=3)
    fill(x0 + 1, P, z0 + 1, x1 - 1, P, z1 - 1, Mix([("polished_andesite", 50), ("smooth_quartz", 50)], seed=88))
    for z in range(z0 + 6, z1 - 2, 2):
        fill(x0 + 2, P + 1, z, x0 + 4, P + 1, z, stairs("dark_oak", "south"))
        fill(x1 - 4, P + 1, z, x1 - 2, P + 1, z, stairs("dark_oak", "south"))
    A.carpet_run(x0 + 5, x1 - 5, z0 + 3, z1 - 2, P + 1, "red")
    fill(x0 + 4, P + 1, z0 + 2, x1 - 4, P + 2, z0 + 2, B("chiseled_quartz_block"))
    put(x0 + 5, P + 3, z0 + 2, B("gold_block"))
    for x in (x0 + 4, x1 - 4):
        put(x, P + 3, z0 + 2, B("candle[candles=4,lit=true,waterlogged=false]"))
    for z in (z0 + 6, z0 + 12):
        A.chandelier(x0 + 5, P + 11, z, 3)
    A.gable(x0, x1, z0, z1, P + 13, axis="z", rise=2, over=1)
    # bell tower at the entrance end
    tx0, tz0 = x1 + 1, z1 - 6
    A.shell(tx0, tx0 + 6, tz0, tz0 + 6, P + 1, P + 30, A.PLASTER, A.PLINTH)
    for side in A.SIDES:
        A.facade(tx0, tx0 + 6, tz0, tz0 + 6, side, P + 1, P + 30, [(P + 22, 2, 4)], bay=7, style="arch2", pilasters=False, plinth=3)
    put(tx0 + 3, P + 26, tz0 + 3, B("bell[attachment=ceiling,facing=north,powered=false]"))
    fill(tx0 + 3, P + 27, tz0 + 3, tx0 + 3, P + 29, tz0 + 3, A.CHAIN)
    for k in range(10):
        r = 3.9 * (1 - k / 10)
        A.cone_roof(tx0 + 3, tz0 + 3, 4.2, P + 31, 14) if k == 0 else None
    A.fin(tx0 + 3, P + 45, tz0 + 3)


def smithy(x0, z0):
    x1, z1 = x0 + 10, z0 + 8
    OCC[z0 - 2:z1 + 3, x0 - 2:x1 + 3] = True
    fill(x0, P, z0, x1, P, z1, Mix([("stone", 40), ("cobblestone", 40), ("gravel", 20)], seed=89))
    for x, z in ((x0, z0), (x1, z0), (x0, z1), (x1, z1), (x0 + 5, z0), (x0 + 5, z1)):
        fill(x, P + 1, z, x, P + 5, z, B("stripped_spruce_log[axis=y]"))
    fill(x0, P + 1, z0, x1, P + 5, z0, GROUND)
    fill(x0 + 1, P + 1, z0 + 1, x0 + 3, P + 1, z0 + 3, Mix([("bricks", 1)]))
    put(x0 + 2, P + 2, z0 + 2, B("lava_cauldron"))
    fill(x0 + 1, P + 2, z0 + 1, x0 + 1, P + 12, z0 + 1, Mix([("bricks", 70), ("stone_bricks", 30)], seed=90))
    put(x0 + 1, P + 13, z0 + 1, B("campfire[facing=north,lit=true,signal_fire=false,waterlogged=false]"))
    for i, b in enumerate(["anvil[facing=east]", "blast_furnace[facing=south,lit=true]", "grindstone[face=floor,facing=east]",
                           "smithing_table", "chipped_anvil[facing=north]", "barrel[facing=up,open=false]"]):
        put(x0 + 4 + i, P + 1, z0 + 2, B(b))
    put(x0 + 6, P + 1, z1 - 2, B("water_cauldron[level=3]"))
    A.gable(x0, x1, z0, z1, P + 6, axis="x", rise=1, over=1, mat=Mix([("spruce_planks", 1)]), st=StairMix(["spruce"]), end_wall=None)


def stable(x0, z0):
    x1, z1 = x0 + 18, z0 + 9
    OCC[z0 - 2:z1 + 10, x0 - 2:x1 + 3] = True
    for x in range(x0, x1 + 1, 3):
        fill(x, P + 1, z0, x, P + 4, z0, B("stripped_spruce_log[axis=y]"))
        fill(x, P + 1, z1, x, P + 4, z1, B("stripped_spruce_log[axis=y]"))
        fill(x, P + 1, z0 + 1, x, P + 2, z1 - 1, B("spruce_fence[east=false,north=true,south=true,west=false,waterlogged=false]"))
    fill(x0, P + 1, z0, x1, P + 3, z0, Mix([("spruce_planks", 1)]))
    fill(x0 + 1, P, z0 + 1, x1 - 1, P, z1 - 1, Mix([("hay_block[axis=y]", 20), ("coarse_dirt", 50), ("dirt_path", 30)], seed=91))
    for x in range(x0 + 1, x1, 3):
        put(x, P + 1, z0 + 1, B("hay_block[axis=y]"))
        put(x + 1, P + 1, z0 + 1, B("water_cauldron[level=3]"))
    A.gable(x0, x1, z0, z1, P + 5, axis="x", rise=1, over=1, mat=Mix([("spruce_planks", 1)]), st=StairMix(["spruce", "dark_oak"]), end_wall=None)
    # paddock
    for x in range(x0, x1 + 1):
        for z in (z1 + 9,):
            put(x, P + 1, z, B("spruce_fence[east=true,north=false,south=false,west=true,waterlogged=false]"))
    for z in range(z1 + 1, z1 + 10):
        put(x0, P + 1, z, B("spruce_fence[east=false,north=true,south=true,west=false,waterlogged=false]"))
        put(x1, P + 1, z, B("spruce_fence[east=false,north=true,south=true,west=false,waterlogged=false]"))
    for _ in range(14):
        put(int(rng.integers(x0 + 1, x1)), P + 1, int(rng.integers(z1 + 1, z1 + 9)), B("short_grass"), only_air=True)


def garden():
    """Palace garden west of the castle: hedges, flower beds, cherry/birch, pavilion, pond."""
    xs, zs = np.nonzero(T.GARDEN.T)
    for x, z in zip(xs, zs):
        x, z = int(x), int(z)
        u, v = x - C, z - C
        if mc.arr[P, z, x] in (B("gravel"), B("polished_andesite")):
            continue
        if abs(v) in (6, 7) or abs(u + 79) in (6, 7):
            put(x, P + 1, z, B("azalea_leaves[distance=1,persistent=true,waterlogged=false]") if (u + v) % 7 else B("flowering_azalea_leaves[distance=1,persistent=true,waterlogged=false]"))
        elif (abs(v) in (3, 4)) or abs(u + 79) in (3, 4):
            put(x, P + 1, z, B(["allium", "cornflower", "lily_of_the_valley", "red_tulip", "oxeye_daisy", "azure_bluet"][(u * 3 + v) % 6]))
    cx, cz = X(-79), Zc(0)
    for x, z, d in A.ring_cells(cx, cz, 0, 4.2):   # pavilion
        put(x, P + 1, z, B("smooth_quartz"))
    for k in range(8):
        a = k * math.pi / 4
        x, z = int(round(cx + 3.6 * math.cos(a))), int(round(cz + 3.6 * math.sin(a)))
        fill(x, P + 2, z, x, P + 6, z, B("quartz_pillar[axis=y]"))
    for x, z, d in A.ring_cells(cx, cz, 0, 4.4):
        put(x, P + 7, z, B("smooth_quartz"))
    A.cone_roof(cx, cz, 5.2, P + 8, 7)
    A.fin(cx, P + 15, cz)
    A.chandelier(cx, P + 6, cz, 1)
    for (u, v, kind) in ((-90, -18, "cherry"), (-68, -18, "cherry"), (-90, 18, "cherry"), (-68, 18, "cherry"),
                         (-79, -24, "birch"), (-79, 24, "birch"), (-94, 0, "birch"), (-64, 0, "birch")):
        x, z = X(u), Zc(v)
        (trees.cherry if kind == "cherry" else trees.birch)(x, P, z, rng, size=0.85)
    OCC[T.GARDEN] = True


def street_details():
    # lamps along the ring road and main streets, trees on grass verges
    ys, xs = np.nonzero(T.STREET & (T.R < T.WALL_IN - 6))
    pts = list(zip(xs, ys))
    rng.shuffle(pts)
    lamps = []
    for x, z in pts:
        x, z = int(x), int(z)
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            xx, zz = x + 2 * dx, z + 2 * dz
            if not (0 <= xx < mc.S and 0 <= zz < mc.S):
                continue
            if not T.STREET[zz, xx] and not OCC[zz, xx] and T.gh[zz, xx] == P and get(xx, P + 1, zz) == 0 \
                    and all(math.hypot(xx - a, zz - b) > 11 for a, b in lamps) and not T.MARKET[zz, xx]:
                lamp(xx, zz)
                lamps.append((xx, zz))
                break
    trees_placed = []
    for _ in range(3000):
        x, z = int(rng.integers(C - 100, C + 100)), int(rng.integers(C - 100, C + 100))
        if T.STREET[max(z - 5, 0):z + 6, max(x - 5, 0):x + 6].any() or OCC[max(z - 3, 0):z + 4, max(x - 3, 0):x + 4].any() or T.gh[z, x] != P or T.R[z, x] > 96:
            continue
        if T.MARKET[z, x] or any(math.hypot(x - a, z - b) < 10 for a, b in trees_placed):
            continue
        if T.cragmask[max(z - 4, 0):z + 5, max(x - 4, 0):x + 5].any():
            continue
        r = rng.random()
        if r < 0.55:
            trees.oak(x, P, z, rng, size=rng.uniform(0.5, 0.7), hang=0.01)
        elif r < 0.8:
            trees.birch(x, P, z, rng, size=0.8)
        else:
            trees.shrub(x, P, z, rng)
        OCC[z - 3:z + 4, x - 3:x + 4] = True
        trees_placed.append((x, z))
        if len(trees_placed) > 28:
            break


def build():
    houses.init_ground_ids()
    streets()
    town_wall()
    market()
    garden()
    chapel(X(18), Zc(-100) + 6)
    smithy(X(70), Zc(-72))
    stable(X(-92), Zc(56))
    # tavern next to the market
    tav = houses.House(X(70), Zc(50), 15, 11, "south", rng, P, trade="brewer")
    tav.storeys, tav.style, tav.wing = 3, "timber", None
    OCC[Zc(48):Zc(63), X(68):X(87)] = True
    tav.build()
    for (u, v) in ((72, 62), (76, 62), (80, 62)):
        A.table(X(u), X(u), Zc(v), Zc(v), P + 1, chairs=False)
        put(X(u) - 1, P + 1, Zc(v), stairs("spruce", "west"))
        put(X(u) + 1, P + 1, Zc(v), stairs("spruce", "east"))
    n = place_houses(95)
    street_details()
    return n
