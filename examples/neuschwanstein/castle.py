"""Neuschwanstein-style castle on the crag (asymmetric, after the real layout):
Palas (west) with towers, stair tower with spiral stairs, square tower, Ritterhaus, Kemenate,
upper/lower courtyard, red-brick gatehouse (east), forecourt; furnished interiors."""
import math

import numpy as np

import arch as A
import mc
import trees
from mc import B, C, DV, OPP, Mix, fill, get, put, slab, stairs
from terrain import CP, P

X = lambda u: C + u          # noqa: E731
Zc = lambda v: C + v         # noqa: E731
rng = np.random.default_rng(99)
LIT = B("lantern[hanging=false,waterlogged=false]")


def curtain(points, y0, h=8, thick=3):
    """Curtain wall with walkway and crenels along a polyline of (u, v)."""
    for (u0, v0), (u1, v1) in zip(points, points[1:]):
        n = int(max(abs(u1 - u0), abs(v1 - v0)))
        for i in range(n + 1):
            u = u0 + (u1 - u0) * i / max(n, 1)
            v = v0 + (v1 - v0) * i / max(n, 1)
            x, z = X(int(round(u))), Zc(int(round(v)))
            fill(x - thick // 2, y0 - 10, z - thick // 2, x + thick // 2, y0 + h, z + thick // 2, A.STONEWALL)
            if (i // 2) % 2 == 0:
                fill(x - thick // 2, y0 + h + 1, z - thick // 2, x + thick // 2, y0 + h + 1, z + thick // 2, A.STONEWALL)
                fill(x - thick // 2, y0 + h + 2, z - thick // 2, x + thick // 2, y0 + h + 2, z + thick // 2, slab("stone_brick"))
            else:
                fill(x - thick // 2, y0 + h + 1, z - thick // 2, x + thick // 2, y0 + h + 1, z + thick // 2, slab("stone_brick"))


def build_palas():
    x0, x1, z0, z1 = X(-50), X(-26), Zc(-19), Zc(19)
    y0, y1 = CP, CP + 40
    A.shell(x0, x1, z0, z1, y0 + 1, y1, A.PLASTER, Mix([("stripped_spruce_wood[axis=y]", 40), ("spruce_planks", 30), ("white_terracotta", 30)], seed=70))
    storeys = [(CP, 2, 4), (CP + 8, 2, 4), (CP + 16, 2, 4), (CP + 24, 3, 9)]
    for side in A.SIDES:
        A.facade(x0, x1, z0, z1, side, y0 + 1, y1, storeys, bay=7, style="bifora",
                 door_at=(23 if side == "east" else None), door_w=3)
    # floors with beams, partitions
    for fy in (CP, CP + 8, CP + 16, CP + 24):
        A.floor(x0 + 2, x1 - 2, z0 + 2, z1 - 2, fy, beams=fy > CP)
    A.floor(x0 + 2, x1 - 2, z0 + 2, z1 - 2, y1, beams=True)
    # --- ground floor: kitchen (south) + servants' hall (north)
    fill(x0 + 2, CP + 1, Zc(0), x1 - 2, CP + 7, Zc(0), Mix([("bricks", 60), ("stone_bricks", 40)], seed=71))
    fill(X(-40), CP + 1, Zc(0), X(-38), CP + 3, Zc(0), 0)
    for i, b in enumerate(["furnace[facing=east,lit=true]", "smoker[facing=east,lit=true]", "furnace[facing=east,lit=false]",
                           "smoker[facing=east,lit=false]", "blast_furnace[facing=east,lit=false]"]):
        put(x0 + 2, CP + 1, Zc(3 + 2 * i), B(b))
        put(x0 + 2, CP + 2, Zc(3 + 2 * i), B("stone_bricks"))
    fill(x0 + 2, CP + 3, Zc(2), x0 + 2, CP + 5, Zc(14), B("bricks"))
    for i in range(4):
        put(X(-36) + i * 2, CP + 1, z1 - 2, B("water_cauldron[level=3]") if i % 2 == 0 else B("barrel[facing=up,open=false]"))
    A.table(X(-44), X(-34), Zc(7), Zc(8), CP + 1, chairs=False)
    for x in range(X(-44), X(-33), 3):
        put(x, CP + 2, Zc(7), B("melon") if x % 2 else B("pumpkin"))
        put(x + 1, CP + 2, Zc(8), B("hay_block[axis=y]"))
    A.table(X(-46), X(-30), Zc(-10), Zc(-9), CP + 1)
    for x in range(X(-46), X(-29), 4):
        A.chandelier(x, CP + 7, Zc(-10), 2)
    A.chandelier(X(-38), CP + 7, Zc(8), 2)
    # --- first floor: guest rooms
    for k, zc in enumerate(range(Zc(-17), Zc(17), 9)):
        fill(x0 + 2, CP + 9, zc + 8, x1 - 2, CP + 15, zc + 8, Mix([("spruce_planks", 60), ("stripped_spruce_log[axis=y]", 40)], seed=72))
        fill(X(-36), CP + 9, zc + 8, X(-36), CP + 11, zc + 8, 0)
        A.bed(X(-46), CP + 9, zc + 3, "east", ["red", "blue", "green", "cyan"][k % 4])
        A.bed(X(-46), CP + 9, zc + 5, "east", ["red", "blue", "green", "cyan"][k % 4])
        put(X(-47), CP + 9, zc + 1, B("chest[facing=east,type=single,waterlogged=false]"))
        put(X(-30), CP + 9, zc + 2, B("barrel[facing=up,open=false]"))
        put(X(-30), CP + 10, zc + 2, B("potted_azalea_bush"))
        A.carpet_run(X(-44), X(-38), zc + 2, zc + 6, CP + 9, ["red", "blue", "green", "cyan"][k % 4])
        A.chandelier(X(-40), CP + 15, zc + 4, 2)
    # --- second floor: king's bedroom (north), library (centre), chapel (south)
    fill(x0 + 2, CP + 17, Zc(-6), x1 - 2, CP + 23, Zc(-6), B("dark_oak_planks"))
    fill(x0 + 2, CP + 17, Zc(7), x1 - 2, CP + 23, Zc(7), B("dark_oak_planks"))
    for zz in (Zc(-6), Zc(7)):
        fill(X(-39), CP + 17, zz, X(-37), CP + 20, zz, 0)
    A.carpet_run(X(-47), X(-29), Zc(-16), Zc(-8), CP + 17, "blue")
    A.bed(X(-46), CP + 17, Zc(-12), "east", "blue")
    A.bed(X(-46), CP + 17, Zc(-11), "east", "blue")
    for zz in (Zc(-14), Zc(-9)):
        fill(X(-47), CP + 17, zz, X(-47), CP + 20, zz, B("dark_oak_fence[east=false,north=false,south=false,west=false,waterlogged=false]"))
        put(X(-47), CP + 21, zz, B("dark_oak_planks"))
    fill(X(-47), CP + 21, Zc(-14), X(-47), CP + 21, Zc(-9), slab("dark_oak", "top"))
    put(X(-30), CP + 17, Zc(-15), B("chest[facing=west,type=single,waterlogged=false]"))
    put(X(-30), CP + 17, Zc(-10), B("lectern[facing=west,has_book=false,powered=false]"))
    A.chandelier(X(-38), CP + 23, Zc(-12), 2)
    A.shelf_wall(x0 + 2, x0 + 2, Zc(-4), Zc(5), CP + 17, CP + 22)
    A.shelf_wall(X(-30), X(-30), Zc(-4), Zc(5), CP + 17, CP + 22)
    put(X(-38), CP + 17, Zc(0), B("enchanting_table"))
    A.table(X(-44), X(-41), Zc(-1), Zc(2), CP + 17)
    put(X(-34), CP + 17, Zc(1), B("lectern[facing=east,has_book=false,powered=false]"))
    A.carpet_run(X(-45), X(-31), Zc(-4), Zc(5), CP + 17, "brown")
    A.chandelier(X(-38), CP + 23, Zc(0), 2)
    for z in range(Zc(9), Zc(16), 2):
        fill(X(-44), CP + 17, z, X(-34), CP + 17, z, stairs("dark_oak", "north"))
    fill(X(-41), CP + 17, Zc(9), X(-37), CP + 17, Zc(16), 0)
    A.carpet_run(X(-41), X(-37), Zc(9), Zc(16), CP + 17, "red")
    fill(X(-41), CP + 17, z1 - 2, X(-37), CP + 18, z1 - 2, B("chiseled_quartz_block"))
    for x in (X(-40), X(-38)):
        put(x, CP + 19, z1 - 2, B("candle[candles=3,lit=true,waterlogged=false]"))
    put(X(-39), CP + 19, z1 - 2, B("gold_block"))
    # --- throne hall (two storeys, CP+25 .. CP+39) with gallery and pillars
    fill(x0 + 2, CP + 24, z0 + 2, x1 - 2, CP + 24, z1 - 2, Mix([("polished_deepslate", 30), ("deepslate_tiles", 25), ("gold_block", 3),
                                                              ("lapis_block", 8), ("polished_diorite", 34)], seed=73))
    fill(X(-40), CP + 24, z0 + 3, X(-36), CP + 24, z1 - 3, Mix([("light_blue_glazed_terracotta[facing=north]", 50), ("lapis_block", 50)], seed=74))
    A.carpet_run(X(-39), X(-37), z0 + 3, z1 - 6, CP + 25, "blue")
    for x in (X(-45), X(-31)):
        for z in range(Zc(-15), Zc(16), 5):
            fill(x, CP + 25, z, x, CP + 38, z, B("quartz_pillar[axis=y]"))
            put(x, CP + 25, z, B("chiseled_quartz_block"))
            put(x, CP + 38, z, B("gold_block"))
    for x0g, x1g in ((x0 + 2, X(-45)), (X(-31), x1 - 2)):     # galleries
        fill(x0g, CP + 32, z0 + 2, x1g, CP + 32, z1 - 2, A.FLOOR_MIX)
    for x in (X(-44), X(-32)):
        fill(x, CP + 33, z0 + 2, x, CP + 33, z1 - 2, B("dark_oak_fence[east=false,north=true,south=true,west=false,waterlogged=false]"))
    # throne on a dais (south end)
    for k in range(3):
        fill(X(-42) + k, CP + 25 + k, z1 - 7 + k, X(-34) - k, CP + 25 + k, z1 - 2, B("smooth_quartz"))
        fill(X(-42) + k, CP + 25 + k, z1 - 7 + k, X(-34) - k, CP + 25 + k, z1 - 7 + k, stairs("smooth_quartz", "south"))
    put(X(-38), CP + 28, z1 - 3, stairs("quartz", "north"))
    put(X(-39), CP + 28, z1 - 3, stairs("quartz", "west"))
    put(X(-37), CP + 28, z1 - 3, stairs("quartz", "east"))
    fill(X(-39), CP + 29, z1 - 2, X(-37), CP + 32, z1 - 2, B("gold_block"))
    fill(X(-38), CP + 33, z1 - 2, X(-38), CP + 34, z1 - 2, B("gold_block"))
    for z in range(Zc(-14), Zc(13), 7):
        A.chandelier(X(-38), CP + 39, z, 4)
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            put(X(-38) + dx, CP + 35, z + dz, A.LANT_H)
            put(X(-38) + dx, CP + 36, z + dz, A.CHAIN)
    # --- roof with dormers, gable pinnacles
    top = A.gable(x0, x1, z0, z1, y1 + 1, axis="z", rise=2, over=1,
                  dormers=[(Zc(-12), "lo"), (Zc(0), "lo"), (Zc(12), "lo"), (Zc(-12), "hi"), (Zc(0), "hi"), (Zc(12), "hi")])
    for z in (z0, z1):
        A.fin(X(-38), top + 2, z)
        for x in (x0, x1):
            fill(x, y1 + 1, z, x, y1 + 6, z, B("quartz_pillar[axis=y]"))
            A.fin(x, y1 + 7, z)
    # west balcony on the 2nd floor, overlooking the gorge
    fill(x0 - 3, CP + 16, Zc(-8), x0 - 1, CP + 16, Zc(8), B("smooth_quartz"))
    for z in range(Zc(-8), Zc(9), 2):
        put(x0 - 3, CP + 15, z, stairs("smooth_quartz", "east", "top"))
    fill(x0 - 3, CP + 17, Zc(-8), x0 - 3, CP + 17, Zc(8), A.POST_WHITE)
    fill(x0 - 2, CP + 17, Zc(-8), x0 - 1, CP + 17, Zc(-8), A.POST_WHITE)
    fill(x0 - 2, CP + 17, Zc(8), x0 - 1, CP + 17, Zc(8), A.POST_WHITE)
    for z in (Zc(-2), Zc(0), Zc(2)):
        fill(x0, CP + 17, z, x0 + 1, CP + 20, z, 0)
    # base of the palas runs down into the rock (west cliff)
    for x in range(x0 - 1, x1 + 2):
        for z in range(z0 - 1, z1 + 2):
            if x in (x0 - 1, x1 + 1) or z in (z0 - 1, z1 + 1):
                y = CP - 2
                while y > P - 10 and get(x, y, z) == 0:
                    put(x, y, z, A.PLINTH.at(x, y, z))
                    y -= 1


def build_towers():
    # stair tower with spiral staircase reaching every palas floor
    tcx, tcz = X(-23), Zc(-11)
    A.round_tower(tcx, tcz, 5.5, CP + 1, CP + 66, crown=True, cone=16, floors=200, spiral=False)
    # step at the palas floor heights (CP + 8k) lands on the west side, facing the palas doors
    A.spiral_stairs(tcx, tcz, 3.5, CP + 1, CP + 64, per_turn=8, offset=math.pi - 2 * math.pi * 7 / 8)
    for fy in (CP, CP + 8, CP + 16, CP + 24, CP + 32):
        fill(X(-28), fy + 1, tcz - 1, X(-26), fy + 3, tcz + 1, 0)
    fill(X(-19), CP + 1, tcz - 1, X(-17), CP + 3, tcz + 1, 0)
    for k in range(0, 64, 4):  # slit windows climbing with the stairs
        a = (k / 8) * 2 * math.pi + math.pi
        x, z = int(round(tcx + 5 * math.cos(a))), int(round(tcz + 5 * math.sin(a)))
        if X(-25) < x:
            fill(x, CP + k + 2, z, x, CP + k + 3, z, B("gray_stained_glass"))
    # north-west tall tower and south-west tower on the palas corners
    A.round_tower(X(-51), Zc(-20), 5, CP - 6, CP + 74, crown=True, cone=20, floors=8)
    A.round_tower(X(-51), Zc(20), 4, CP - 6, CP + 52, crown=False, cone=15, floors=8)
    # corner turrets (Erker) on the courtyard side of the palas
    A.round_tower(X(-25), Zc(19), 2.5, CP + 30, CP + 46, wall=A.PLASTER, crown=False, cone=11, corbel=5, windows=False)
    # square tower with crown, corner turret
    sx0, sx1, sz0, sz1 = X(-10), X(0), Zc(-29), Zc(-19)
    A.shell(sx0, sx1, sz0, sz1, CP - 6, CP + 54, A.PLASTER, A.PLINTH)
    for side in A.SIDES:
        A.facade(sx0, sx1, sz0, sz1, side, CP + 1, CP + 54, [(CP + 8 * k, 2, 3) for k in range(1, 7)], bay=5, style="arch2",
                 plinth=3, pilasters=False, eave_cornice=False)
    for fy in range(CP, CP + 54, 8):
        A.floor(sx0 + 2, sx1 - 2, sz0 + 2, sz1 - 2, fy, beams=False)
        fill(sx0 + 3, fy, sz0 + 3, sx0 + 4, fy, sz0 + 4, 0)
        fill(sx0 + 3, fy + 1, sz0 + 2, sx0 + 3, fy + 8, sz0 + 2, B("ladder[facing=south,waterlogged=false]"))
    fill(sx0 + 2, CP + 1, sz1, sx0 + 4, CP + 4, sz1 - 1, 0)
    for x in range(sx0 - 1, sx1 + 2):
        for z in range(sz0 - 1, sz1 + 2):
            edge = x in (sx0 - 1, sx1 + 1) or z in (sz0 - 1, sz1 + 1)
            if edge:
                put(x, CP + 52, z, stairs("smooth_quartz", A.toward(X(-5), Zc(-24), x, z), "top"))
                fill(x, CP + 53, z, x, CP + 55, z, A.PLASTER)
                if (x + z) % 2 == 0:
                    put(x, CP + 56, z, A.PLASTER.at(x, 0, z))
                    put(x, CP + 57, z, slab("smooth_quartz"))
                else:
                    put(x, CP + 56, z, slab("smooth_quartz"))
    A.floor(sx0, sx1, sz0, sz1, CP + 55, beams=False)
    A.round_tower(sx1 + 1, sz0 - 1, 2.5, CP + 50, CP + 66, crown=False, cone=10, corbel=5, windows=False)
    A.gable(sx0 + 2, sx1 - 2, sz0 + 2, sz1 - 2, CP + 56, axis="x", rise=2, over=0, end_wall=A.PLASTER)


def build_wing(x0, x1, z0, z1, axis, storeys=3, door_side=None, kind="ritter"):
    y1 = CP + storeys * 8
    A.shell(x0, x1, z0, z1, CP + 1, y1, A.PLASTER, Mix([("spruce_planks", 50), ("stripped_spruce_wood[axis=y]", 50)], seed=75))
    for side in A.SIDES:
        n = (x1 - x0 + 1) if side in ("north", "south") else (z1 - z0 + 1)
        A.facade(x0, x1, z0, z1, side, CP + 1, y1, [(CP + 8 * k, 2, 4) for k in range(storeys)], bay=5, style="arch2",
                 door_at=(n // 2 if side == door_side else None))
    for k in range(storeys):
        A.floor(x0 + 2, x1 - 2, z0 + 2, z1 - 2, CP + 8 * k, beams=k > 0)
        fill(x0 + 3, CP + 8 * k, z0 + 3, x0 + 3, CP + 8 * k, z0 + 3, 0)
        fill(x0 + 3, CP + 8 * k + 1, z0 + 2, x0 + 3, CP + 8 * k + 8, z0 + 2, B("ladder[facing=south,waterlogged=false]"))
        cx, cz = (x0 + x1) // 2, (z0 + z1) // 2
        A.chandelier(cx, CP + 8 * k + 7, cz, 2)
        if kind == "ritter":
            if k == 0:
                for i, b in enumerate(["anvil[facing=north]", "grindstone[face=floor,facing=north]", "smithing_table",
                                       "barrel[facing=up,open=false]", "chest[facing=north,type=single,waterlogged=false]"]):
                    put(x0 + 4 + 2 * i, CP + 1, z1 - 2, B(b))
                A.table(cx - 3, cx + 3, cz, cz, CP + 1)
            else:
                for i in range(x0 + 4, x1 - 2, 3):
                    A.bed(i, CP + 8 * k + 1, z1 - 3, "south", "gray")
        else:
            if k == 0:
                for i, b in enumerate(["loom[facing=north]", "cartography_table", "loom[facing=north]", "barrel[facing=up,open=false]"]):
                    put(x0 + 4 + 3 * i, CP + 1, z0 + 2, B(b))
                A.table(cx - 2, cx + 2, cz, cz + 1, CP + 1, cloth="white_carpet")
            else:
                A.bed(cx - 4, CP + 8 * k + 1, z1 - 3, "south", "pink")
                A.bed(cx + 3, CP + 8 * k + 1, z1 - 3, "south", "magenta")
                A.carpet_run(cx - 5, cx + 5, cz - 2, cz + 2, CP + 8 * k + 1, "pink")
                put(cx, CP + 8 * k + 1, z0 + 2, B("potted_flowering_azalea_bush"))
    A.gable(x0, x1, z0, z1, y1 + 1, axis=axis, rise=2, over=1,
            dormers=[((x0 + x1) // 2, "lo"), ((x0 + x1) // 2, "hi")] if axis == "x" else None)


def build_gatehouse():
    x0, x1, z0, z1 = X(34), X(48), Zc(-21), Zc(21)
    y1 = CP + 24
    A.shell(x0, x1, z0, z1, CP + 1, y1, A.REDB, Mix([("stone_bricks", 60), ("spruce_planks", 40)], seed=76))
    storeys = [(CP, 2, 4), (CP + 8, 2, 4), (CP + 16, 2, 4)]
    for side in A.SIDES:
        A.facade(x0, x1, z0, z1, side, CP + 1, y1, storeys, bay=5, style="arch2", wall=A.REDB, pilasters=False,
                 skip=(15, 20) if side in ("east", "west") else ())
    # white quoins
    for x in (x0, x1):
        for z in (z0, z1):
            for y in range(CP + 1, y1 + 1):
                if (y // 2) % 2 == 0:
                    put(x, y, z, B("smooth_quartz"))
                    put(x + (1 if x == x0 else -1), y, z, B("smooth_quartz"))
                    put(x, y, z + (1 if z == z0 else -1), B("smooth_quartz"))
    for k in range(3):
        A.floor(x0 + 2, x1 - 2, z0 + 2, z1 - 2, CP + 8 * k, beams=k > 0)
    # gate passage (vaulted, 5 wide)
    for x in range(x0 - 1, x1 + 2):
        for z in range(Zc(-2), Zc(3)):
            fill(x, CP + 1, z, x, CP + 6, z, 0)
        for z, f in ((Zc(-2), "south"), (Zc(2), "north")):
            put(x, CP + 6, z, stairs("smooth_quartz", f, "top"))
        fill(x, CP, Zc(-2), x, CP, Zc(2), Mix([("polished_andesite", 50), ("stone_bricks", 50)], seed=77))
        fill(x, CP + 1, Zc(-3), x, CP + 6, Zc(-3), A.PLINTH)
        fill(x, CP + 1, Zc(3), x, CP + 6, Zc(3), A.PLINTH)
    fill(x1 + 1, CP + 7, Zc(-2), x1 + 1, CP + 7, Zc(2), B("iron_bars[east=true,west=true,north=false,south=false,waterlogged=false]"))
    put(x1 + 2, CP + 4, Zc(-4), LIT)
    put(x1 + 2, CP + 4, Zc(4), LIT)
    # guard rooms
    for z0g, z1g in ((z0 + 2, Zc(-4)), (Zc(4), z1 - 2)):
        for i, b in enumerate(["barrel[facing=up,open=false]", "chest[facing=west,type=single,waterlogged=false]", "grindstone[face=floor,facing=west]",
                               "smithing_table", "barrel[facing=up,open=false]"]):
            put(x1 - 2, CP + 1, z0g + 1 + 2 * i, B(b))
        A.table(x0 + 4, x0 + 6, (z0g + z1g) // 2, (z0g + z1g) // 2 + 1, CP + 1)
        A.chandelier((x0 + x1) // 2, CP + 7, (z0g + z1g) // 2, 2)
        for k in (1, 2):
            for i in range(z0g + 1, z1g, 3):
                A.bed(x1 - 3, CP + 8 * k + 1, i, "west", "red")
            A.chandelier((x0 + x1) // 2, CP + 8 * k + 7, (z0g + z1g) // 2, 2)
        fill(x0 + 2, CP + 1, z0g + 1, x0 + 2, CP + 23, z0g + 1, B("ladder[facing=east,waterlogged=false]"))
        for k in (1, 2):
            put(x0 + 2, CP + 8 * k, z0g + 1, 0)
            put(x0 + 2, CP + 8 * k, z0g + 1, B("ladder[facing=east,waterlogged=false]"))
    fill(X(35), CP + 1, Zc(-5), X(35), CP + 3, Zc(-4), 0)
    fill(X(35), CP + 1, Zc(4), X(35), CP + 3, Zc(5), 0)
    # roof (ridge N-S) and stepped gable over the gate (east)
    A.gable(x0, x1, z0, z1, y1 + 1, axis="z", rise=2, over=1, end_wall=A.REDB)
    for k in range(9):
        w = 8 - k
        fill(x1, y1 + 1 + 2 * k, Zc(-w), x1 + 1, y1 + 2 + 2 * k, Zc(w), A.REDB)
        fill(x1 + 1, y1 + 3 + 2 * k, Zc(-w), x1 + 1, y1 + 3 + 2 * k, Zc(-w), slab("smooth_quartz"))
        fill(x1 + 1, y1 + 3 + 2 * k, Zc(w), x1 + 1, y1 + 3 + 2 * k, Zc(w), slab("smooth_quartz"))
    for dz in range(-2, 3):
        for dy in range(-2, 3):
            if dz * dz + dy * dy <= 5:
                put(x1 + 1, y1 + 7 + dy, Zc(dz), B("gray_stained_glass"))
    A.fin(x1 + 1, y1 + 19, Zc(0))
    # flanking turrets
    for zz in (z0, z1):
        A.round_tower(x1 + 1, zz, 3, CP - 8, CP + 34, crown=True, cone=11, floors=8)


def courtyard():
    # upper courtyard: linden tree on a round bench, a well, flower beds, lanterns
    tx, tz = X(-4), Zc(2)
    for x, z, d in A.ring_cells(tx, tz, 3.5, 4.4):
        put(x, CP + 1, z, stairs("spruce", A.toward(tx, tz, x, z)))
    for x, z, d in A.ring_cells(tx, tz, 0, 3.4):
        put(x, CP, z, B("grass_block[snowy=false]"))
        if d > 1.6:
            put(x, CP + 1, z, B("moss_carpet"))
    trees.oak(tx, CP, tz, rng, size=0.85, hang=0.0)
    # well
    wx, wz = X(10), Zc(-6)
    fill(wx - 2, CP - 6, wz - 2, wx + 2, CP + 1, wz + 2, A.STONEWALL)
    fill(wx - 1, CP - 5, wz - 1, wx + 1, CP + 1, wz + 1, B("water[level=0]"))
    fill(wx - 1, CP + 1, wz - 1, wx + 1, CP + 1, wz + 1, 0)
    for dx, dz in ((-2, -2), (2, -2), (-2, 2), (2, 2)):
        fill(wx + dx, CP + 2, wz + dz, wx + dx, CP + 4, wz + dz, B("spruce_fence[east=false,north=false,south=false,west=false,waterlogged=false]"))
    A.gable(wx - 2, wx + 2, wz - 2, wz + 2, CP + 5, axis="x", rise=1, over=0, mat=Mix([("spruce_planks", 1)]),
            st=mc.StairMix(["spruce"]), end_wall=None, verge=False)
    fill(wx, CP + 4, wz - 1, wx, CP + 4, wz + 1, B("stripped_spruce_log[axis=z]"))
    put(wx, CP + 3, wz, A.CHAIN)
    # flower beds along the Kemenate
    for x in range(X(-14), X(8)):
        put(x, CP, Zc(14), B("grass_block[snowy=false]"))
        put(x, CP + 1, Zc(14), B(["rose_bush[half=lower]", "peony[half=lower]", "lilac[half=lower]"][x % 3]))
        put(x, CP + 2, Zc(14), B(["rose_bush[half=upper]", "peony[half=upper]", "lilac[half=upper]"][x % 3]))
    for x in range(X(-14), X(20), 6):
        for z in (Zc(-15), Zc(13)):
            fill(x, CP + 1, z, x, CP + 3, z, A.POST_STONE)
            put(x, CP + 4, z, LIT)
    # lower courtyard: carts of goods, hay, barrels
    for (u, v) in ((25, -12), (28, 11), (24, 8)):
        x, z = X(u), Zc(v)
        put(x, CP + 1, z, B("hay_block[axis=y]"))
        put(x + 1, CP + 1, z, B("barrel[facing=up,open=false]"))
        put(x, CP + 2, z, B("hay_block[axis=x]"))
        put(x + 1, CP + 1, z + 1, B("composter[level=0]"))
    # forecourt: parapet and lanterns
    for x in range(X(50), X(63)):
        for z in (Zc(-10), Zc(10)):
            if not (z == Zc(10) and X(53) <= x <= X(59)):
                fill(x, CP + 1, z, x, CP + 1, z, A.STONEWALL)
                put(x, CP + 2, z, slab("stone_brick") if x % 2 else LIT)
    for z in range(Zc(-10), Zc(11)):
        fill(X(63), CP - 2, z, X(63), CP + 1, z, A.STONEWALL)
        put(X(63), CP + 2, z, slab("stone_brick") if z % 3 else LIT)
    # ramp parapets with lanterns, buttressed retaining wall on the downhill side
    for v in range(10, 75):
        ytop = int(math.floor(CP - (v - 10) / 2.0))
        for u in (52, 60):
            x, z = X(u), Zc(v)
            fill(x, P - 4, z, x, ytop + 1, z, A.STONEWALL)
            put(x, ytop + 2, z, LIT if v % 8 == 0 else slab("stone_brick"))
        if v % 9 == 0:
            fill(X(61), P - 2, Zc(v), X(61), ytop - 1, Zc(v + 1), A.STONEWALL)
            put(X(61), ytop, Zc(v), stairs("stone_brick", "west"))


def curtains():
    y = CP
    curtain([(-27, -24), (-12, -24)], y, h=7)
    curtain([(1, -28), (2, -28)], y, h=7)
    curtain([(21, -26), (33, -24)], y, h=7)
    curtain([(9, 27), (22, 24), (33, 22)], y, h=7)
    curtain([(-27, 23), (-17, 26)], y, h=7)


def build():
    curtains()
    build_palas()
    build_towers()
    build_wing(X(3), X(20), Zc(-27), Zc(-17), "x", storeys=3, door_side="south", kind="ritter")
    build_wing(X(-16), X(8), Zc(16), Zc(27), "x", storeys=3, door_side="north", kind="kemenate")
    build_gatehouse()
    courtyard()
