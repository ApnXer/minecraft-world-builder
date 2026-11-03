"""Architecture kit: walls with relief (plinth, pilasters, cornices, arched windows, sills), gable roofs,
dormers, round towers with corbels and stair-block cones, spiral stairs, doors and furniture."""
import math

import numpy as np

import mc
from mc import CCW, CW, DV, OPP, B, Mix, StairMix, fill, get, hash3, put, slab, stairs, vdir

# ------------------------------------------------------------------ materials
PLASTER = Mix([("smooth_quartz", 30), ("calcite", 24), ("quartz_bricks", 18), ("polished_diorite", 9),
               ("white_terracotta", 5), ("pale_oak_planks", 5), ("mushroom_stem", 6), ("bone_block[axis=y]", 3)], seed=51, patch=2)
TRIM = Mix([("smooth_quartz", 60), ("quartz_bricks", 25), ("chiseled_quartz_block", 15)], seed=52)
PLINTH = Mix([("stone_bricks", 28), ("polished_andesite", 18), ("tuff_bricks", 20), ("mossy_stone_bricks", 16),
              ("cracked_stone_bricks", 10), ("chiseled_stone_bricks", 4), ("andesite", 4)], seed=53, patch=2)
PLINTH_ST = StairMix(["stone_brick", "tuff_brick", "mossy_stone_brick", "polished_andesite"], seed=54)
SLATE = Mix([("deepslate_tiles", 50), ("polished_deepslate", 18), ("deepslate_bricks", 16), ("cracked_deepslate_tiles", 10),
             ("cobbled_deepslate", 6)], seed=55, patch=2)
SLATE_ST = StairMix(["deepslate_tile", "deepslate_tile", "deepslate_brick", "polished_deepslate"], seed=56)
TRIM_ST = StairMix(["smooth_quartz", "quartz", "smooth_quartz"], seed=57)
REDB = Mix([("bricks", 42), ("resin_bricks", 22), ("terracotta", 12), ("polished_granite", 12), ("red_terracotta", 6),
            ("mud_bricks", 6)], seed=58, patch=2)
REDB_ST = StairMix(["brick", "resin_brick", "brick", "polished_granite"], seed=59)
STONEWALL = Mix([("stone_bricks", 32), ("mossy_stone_bricks", 14), ("cracked_stone_bricks", 10), ("tuff_bricks", 16),
                 ("polished_andesite", 10), ("andesite", 8), ("mossy_cobblestone", 6), ("cobblestone", 4)], seed=60, patch=3)
STONE_ST = StairMix(["stone_brick", "mossy_stone_brick", "tuff_brick", "cobblestone", "andesite"], seed=61)
FLOOR_MIX = Mix([("spruce_planks", 45), ("dark_oak_planks", 35), ("oak_planks", 20)], seed=62, patch=2)
POST_WHITE = mc.wallb("diorite")
POST_STONE = mc.wallb("stone_brick")
PANE = {"x": B("gray_stained_glass_pane[east=true,west=true,north=false,south=false,waterlogged=false]"),
        "z": B("gray_stained_glass_pane[east=false,west=false,north=true,south=true,waterlogged=false]")}
LANT_H = B("lantern[hanging=true,waterlogged=false]")
LANT = B("lantern[hanging=false,waterlogged=false]")
CHAIN = B("iron_chain[axis=y,waterlogged=false]")
SIDES = ("north", "south", "west", "east")


def _along(side):
    return "east" if side in ("north", "south") else "south"


def side_line(x0, x1, z0, z1, side):
    """Positions (x, z) on a side's wall line in along-order, and the outward normal (nx, nz)."""
    if side == "north":
        return [(x, z0) for x in range(x0, x1 + 1)], (0, -1)
    if side == "south":
        return [(x, z1) for x in range(x0, x1 + 1)], (0, 1)
    if side == "west":
        return [(x0, z) for z in range(z0, z1 + 1)], (-1, 0)
    return [(x1, z) for z in range(z0, z1 + 1)], (1, 0)


# ------------------------------------------------------------------ walls
def shell(x0, x1, z0, z1, y0, y1, outer=PLASTER, inner=None, thick=2):
    """Hollow rectangular walls from y0..y1; outer layer `outer`, inner layer `inner`."""
    inner = inner or outer
    for t in range(thick):
        mix = outer if t == 0 else inner
        for xa, xb, za, zb in ((x0 + t, x1 - t, z0 + t, z0 + t), (x0 + t, x1 - t, z1 - t, z1 - t),
                               (x0 + t, x0 + t, z0 + t, z1 - t), (x1 - t, x1 - t, z0 + t, z1 - t)):
            fill(xa, y0, za, xb, y1, zb, mix)
    fill(x0 + thick, y0, z0 + thick, x1 - thick, y1, z1 - thick, 0)


def facade(x0, x1, z0, z1, side, y0, y1, storeys, bay=7, style="bifora", plinth=3, pilasters=True,
           wall=PLASTER, sill_mat="smooth_quartz", skip=(), door_at=None, door_w=3, eave_cornice=True):
    """Relief on one side of a rectangular building.

    storeys: list of (floor_y, window_bottom_offset, window_height); windows are placed per bay.
    style: 'bifora' (2+post+2), 'arch2' (2 wide arched), 'slit', 'rect' (house window w/ shutters).
    """
    line, (nx, nz) = side_line(x0, x1, z0, z1, side)
    out, inw, alp = side, OPP[side], _along(side)
    alm = OPP[alp]
    n = len(line)
    axis = "x" if side in ("north", "south") else "z"

    def P(i, d):
        x, z = line[i]
        return x + nx * d, z + nz * d

    # plinth: one block proud, chamfered with stairs
    if plinth:
        for i in range(-1, n + 1):
            ii = min(max(i, 0), n - 1)
            x, z = P(ii, 1)
            if i < 0 or i >= n:
                x, z = x + (DV[alm][0] if i < 0 else DV[alp][0]), z + (DV[alm][1] if i < 0 else DV[alp][1])
            fill(x, y0 - 2, z, x, y0 + plinth - 1, z, PLINTH)
            put(x, y0 + plinth, z, PLINTH_ST.at(x, y0, z, inw))
    # pilasters at bay boundaries
    edges = [i for i in range(n) if i % bay == 0 or i == n - 1]
    if pilasters:
        for i in edges:
            x, z = P(i, 1)
            fill(x, y0 + plinth + 1, z, x, y1 - 1, z, B("quartz_pillar[axis=y]") if wall is PLASTER else mc.Mix([("stone_bricks", 1)]))
            put(x, y1, z, TRIM_ST.at(x, y1, z, inw, "top") if wall is PLASTER else STONE_ST.at(x, y1, z, inw, "top"))
    # storey cornices (upside-down stairs) and windows
    for fy, wb, wh in storeys:
        if fy > y0:
            for i in range(n):
                x, z = P(i, 1)
                put(x, fy, z, TRIM_ST.at(x, fy, z, inw, "top") if wall is PLASTER else STONE_ST.at(x, fy, z, inw, "top"))
        r0 = fy + wb
        for b0 in range(0, n - bay + 1, bay):
            if b0 in skip:
                continue
            if style == "bifora":
                cols = [b0 + 1, b0 + 2, b0 + 4, b0 + 5] if bay >= 7 else [b0 + 1, b0 + 2]
                post = [b0 + 3] if bay >= 7 else []
            elif style == "arch2":
                c = b0 + bay // 2 - 1
                cols, post = [c, c + 1], []
            elif style == "rect":
                c = b0 + bay // 2 - 1
                cols, post = [c, c + 1], []
            else:
                cols, post = [b0 + bay // 2], []
            cols = [c for c in cols if 0 < c < n - 1]
            if not cols:
                continue
            for c in cols + post:   # sill
                x, z = P(c, 1)
                put(x, r0 - 1, z, slab(sill_mat, "top") if style != "rect" else stairs("spruce", inw, "top"))
            for k in range(wh):
                for c in cols:
                    x, z = P(c, 0)
                    put(x, r0 + k, z, 0)
                    xi, zi = P(c, -1)
                    put(xi, r0 + k, zi, PANE[axis])
                for c in post:
                    x, z = P(c, 0)
                    put(x, r0 + k, z, POST_WHITE)
                    xi, zi = P(c, -1)
                    put(xi, r0 + k, zi, PANE[axis])
            top = r0 + wh
            if style in ("bifora", "arch2"):
                pairs = [cols[i:i + 2] for i in range(0, len(cols), 2)]
                for pr in pairs:
                    if len(pr) == 2:
                        x, z = P(pr[0], 0)
                        put(x, top, z, TRIM_ST.at(x, top, z, alm, "top"))
                        x, z = P(pr[1], 0)
                        put(x, top, z, TRIM_ST.at(x, top, z, alp, "top"))
                    else:
                        x, z = P(pr[0], 0)
                        put(x, top, z, TRIM_ST.at(x, top, z, out, "top"))
                for c in post:
                    x, z = P(c, 0)
                    put(x, top, z, B("chiseled_quartz_block"))
                for c in cols + post:
                    x, z = P(c, -1)
                    put(x, top, z, PANE[axis])
                # hood moulding above the arch
                for c in [cols[0] - 1] + cols + post + [cols[-1] + 1]:
                    if 0 <= c < n:
                        x, z = P(c, 1)
                        if get(x, top + 1, z) == 0:
                            put(x, top + 1, z, slab("smooth_quartz", "bottom"))
            elif style == "rect":
                for c in cols:
                    x, z = P(c, 0)
                    put(x, top, z, stairs("stripped_spruce" if False else "spruce", out, "top"))
                # shutters
                for c, sd in ((cols[0] - 1, alm), (cols[-1] + 1, alp)):
                    if 0 <= c < n:
                        x, z = P(c, 1)
                        for k in range(wh):
                            put(x, r0 + k, z, B(f"spruce_trapdoor[facing={out},half=bottom,open=true,powered=false,waterlogged=false]"))
                # flower box under window
                for c in cols:
                    x, z = P(c, 1)
                    put(x, r0 - 1, z, B(f"spruce_trapdoor[facing={inw},half=top,open=false,powered=false,waterlogged=false]"))
                    pot = ["potted_red_tulip", "potted_azure_bluet", "potted_cornflower", "potted_poppy", "potted_oxeye_daisy"]
                    put(x, r0, z, B(pot[int(hash3(x, r0, z) % len(pot))]))
    if door_at is not None:
        door(line, (nx, nz), side, door_at, y0 + 1, door_w)
    if eave_cornice:
        for i in range(-1, n + 1):
            ii = min(max(i, 0), n - 1)
            x, z = P(ii, 1)
            if i < 0 or i >= n:
                continue
            put(x, y1, z, TRIM_ST.at(x, y1, z, inw, "top") if wall is PLASTER else STONE_ST.at(x, y1, z, inw, "top"))


def door(line, normal, side, at, y, w=3, mat="spruce", h=4):
    nx, nz = normal
    alp, alm = _along(side), OPP[_along(side)]
    cols = list(range(at - w // 2, at - w // 2 + w))
    for c in cols:
        x, z = line[c]
        for k in range(h):
            put(x, y + k, z, 0)
            put(x - nx, y + k, z - nz, 0)
    # arch head
    x, z = line[cols[0]]
    put(x, y + h - 1, z, TRIM_ST.at(x, y, z, alm, "top"))
    x, z = line[cols[-1]]
    put(x, y + h - 1, z, TRIM_ST.at(x, y, z, alp, "top"))
    # double door (2 panels) in the outer layer
    inner = OPP[side]
    dcols = cols[1:-1] if w >= 4 else cols[:2] if w == 2 else [cols[0], cols[1]] if w == 3 else cols
    for j, c in enumerate(dcols[:2]):
        x, z = line[c]
        hinge = "left" if j == 0 else "right"
        put(x, y, z, B(f"{mat}_door[facing={inner},half=lower,hinge={hinge},open=false,powered=false]"))
        put(x, y + 1, z, B(f"{mat}_door[facing={inner},half=upper,hinge={hinge},open=false,powered=false]"))
    # lanterns on both sides
    for c in (cols[0] - 1, cols[-1] + 1):
        if 0 <= c < len(line):
            x, z = line[c]
            put(x + nx, y + 2, z + nz, LANT)


# ------------------------------------------------------------------ roofs
def roof(x0, x1, z0, z1, y, kind="gable", axis="x", rise=2, over=1, mat=SLATE, st=SLATE_ST, end_wall=PLASTER,
         cap="deepslate_tile", hip_cut=0.55, ridge_cap=True):
    """Height-field roof: 'gable' (ridge along axis), 'hip', 'halfhip' (Kruppelwalm). The surface is built from
    stairs facing up-slope with a solid body below, so eaves, verges and hips stay continuous; corner stair
    shapes are resolved later by post.stair_shapes(). Returns the ridge height."""
    xa, xb, za, zb = x0 - over, x1 + over, z0 - over, z1 + over
    xc = (x0 + x1) / 2.0 if x0 > 0 else 0.0
    zc = (z0 + z1) / 2.0 if z0 > 0 else 0.0
    hwx = (x1 - x0) / 2.0 + over
    hwz = (z1 - z0) / 2.0 + over

    def Hf(x, z):
        dx, dz = hwx - abs(x - xc), hwz - abs(z - zc)
        if kind == "hip":
            return y + rise * min(dx, dz)
        g = y + rise * (dz if axis == "x" else dx)
        if kind == "halfhip":
            o = dx if axis == "x" else dz
            w = hwz if axis == "x" else hwx
            return min(g, y + rise * (o + w * (1 - hip_cut)))
        return g

    H = {}
    for x in range(xa, xb + 1):
        for z in range(za, zb + 1):
            H[(x, z)] = Hf(x, z)
    top = y
    for (x, z), hv in H.items():
        h = int(math.floor(hv + 1e-6))
        best, bdir = 0.0, None
        for f, (ddx, ddz) in DV.items():
            nh = H.get((x + ddx, z + ddz))
            if nh is not None and nh - hv > best + 1e-6:
                best, bdir = nh - hv, f
        for yy in range(max(y, h - rise + 1), h):
            put(x, yy, z, mat.at(x, yy, z))
        if bdir is None:          # ridge / peak
            put(x, h, z, mat.at(x, h, z))
            if ridge_cap:
                put(x, h + 1, z, slab(cap, "bottom"))
            top = max(top, h + 1)
        else:
            put(x, h, z, st.at(x, h, z, bdir))
            top = max(top, h)
    # gable end walls fill the triangle under the roof (flush with the wall face)
    if end_wall is not None and kind in ("gable", "halfhip"):
        ends = ([(xe, z) for xe in ((x0, x1) if x0 > 0 else (x1,)) for z in range(z0, z1 + 1)] if axis == "x"
                else [(x, ze) for ze in ((z0, z1) if z0 > 0 else (z1,)) for x in range(x0, x1 + 1)])
        for (x, z) in ends:
            h = int(math.floor(H[(x, z)] + 1e-6))
            for yy in range(y, h):
                put(x, yy, z, end_wall.at(x, yy, z))
        # small upside-down stair trim under the verge overhang
        for (x, z) in ends:
            ox, oz = ((x - over if x == x0 else x + over), z) if axis == "x" else (x, (z - over if z == z0 else z + over))
            if (ox, oz) in H and over > 0:
                h = int(math.floor(H[(ox, oz)] + 1e-6))
                face = vdir(x - ox, z - oz)
                put(ox, h - 1, oz, st.at(ox, h, oz, face, "top"), only_air=True)
    return top


def gable(x0, x1, z0, z1, y, axis="x", rise=2, over=1, mat=SLATE, st=SLATE_ST, end_wall=PLASTER, cap="deepslate_tile",
          dormers=None, verge=True, kind="gable"):
    top = roof(x0, x1, z0, z1, y, kind=kind, axis=axis, rise=rise, over=over, mat=mat, st=st, end_wall=end_wall, cap=cap)
    for d_a, d_side in dormers or ():
        dormer(d_a, d_side, x0, x1, z0, z1, y, axis, rise, over, mat, st)
    return top


def dormer(a, side_b, x0, x1, z0, z1, y, axis, rise, over, mat, st, depth=4, h=4):
    """Gabled dormer on a roof slope (side_b 'lo' = the x/z-minimum side). Built as a tiny height-field gable
    whose ridge runs perpendicular to the main ridge, so it merges cleanly into the slope."""
    out_dir = ("north" if side_b == "lo" else "south") if axis == "x" else ("west" if side_b == "lo" else "east")
    odx, odz = DV[out_dir]
    wall_line = (z0 if side_b == "lo" else z1) if axis == "x" else (x0 if side_b == "lo" else x1)
    front = wall_line - (odz if axis == "x" else odx) * 1          # one block inside the wall line
    yb = y + 2 * rise
    for k in range(depth + 2):                                       # body: plaster cheeks + window
        b = front - (odz if axis == "x" else odx) * k
        for da in (-1, 0, 1):
            x, z = (a + da, b) if axis == "x" else (b, a + da)
            for yy in range(yb, yb + h):
                put(x, yy, z, PLASTER.at(x, yy, z) if (da != 0 or k > 0 or yy in (yb, yb + h - 1)) else 0)
        if k == 0:
            x, z = (a, b) if axis == "x" else (b, a)
            for yy in range(yb + 1, yb + h - 1):
                put(x, yy, z, PANE["x" if axis == "x" else "z"])
            ox, oz = x + odx, z + odz
            put(ox, yb, oz, slab("smooth_quartz", "top"))
    # little roof: ridge points outward
    if axis == "x":
        ra = (a - 2, a + 2, min(front, front - odz * (depth + 1)), max(front, front - odz * (depth + 1)))
        roof(ra[0], ra[1], ra[2] - (1 if odz < 0 else 0), ra[3] + (1 if odz > 0 else 0), yb + h, kind="gable",
             axis="z", rise=1, over=0, mat=mat, st=st, end_wall=PLASTER, ridge_cap=False)
    else:
        ra = (min(front, front - odx * (depth + 1)), max(front, front - odx * (depth + 1)), a - 2, a + 2)
        roof(ra[0] - (1 if odx < 0 else 0), ra[1] + (1 if odx > 0 else 0), ra[2], ra[3], yb + h, kind="gable",
             axis="x", rise=1, over=0, mat=mat, st=st, end_wall=PLASTER, ridge_cap=False)
    x, z = (a, front + odz) if axis == "x" else (front + odx, a)
    put(x, yb + h + 3, z, B("lightning_rod[facing=up,powered=false,waterlogged=false]"))


# ------------------------------------------------------------------ round towers
def ring_cells(cx, cz, r0, r1):
    rr = int(math.ceil(r1)) + 1
    out = []
    for x in range(int(cx) - rr, int(cx) + rr + 1):
        for z in range(int(cz) - rr, int(cz) + rr + 1):
            d = math.hypot(x - cx, z - cz)
            if r0 <= d <= r1:
                out.append((x, z, d))
    return out


def toward(cx, cz, x, z):
    return vdir(cx - x, cz - z)


def round_tower(cx, cz, r, y0, y1, wall=PLASTER, crown=True, cone=0, cone_r=None, floors=8, windows=True,
                spiral=False, corbel=0, finial=True, beam=False):
    """Round tower with plinth, string courses, arched windows, machicolated crown and stair-block cone."""
    # corbelled base (turret on a wall)
    if corbel:
        for k in range(corbel):
            rr = r * (k + 1) / (corbel + 1)
            for x, z, d in ring_cells(cx, cz, 0, rr):
                put(x, y0 - corbel + k, z, wall.at(x, y0, z) if d > rr - 1 else wall.at(x, y0, z))
            for x, z, d in ring_cells(cx, cz, rr - 0.5, rr + 0.6):
                put(x, y0 - corbel + k, z, TRIM_ST.at(x, y0, z, toward(cx, cz, x, z), "top"))
    for x, z, d in ring_cells(cx, cz, 0, r):
        if d > r - 2:
            fill(x, y0, z, x, y1, z, wall)
        else:
            fill(x, y0, z, x, y1, z, 0)
    if not corbel:
        for x, z, d in ring_cells(cx, cz, r - 0.2, r + 1.2):
            fill(x, y0 - 3, z, x, y0 + 1, z, PLINTH)
            put(x, y0 + 2, z, PLINTH_ST.at(x, y0, z, toward(cx, cz, x, z)))
    # floors, string courses, windows
    nw = max(4, int(round(r * 0.9)))
    for fy in range(y0 + floors, y1 - 2, floors):
        for x, z, d in ring_cells(cx, cz, 0, r - 1.9):
            put(x, fy, z, FLOOR_MIX.at(x, fy, z))
        for x, z, d in ring_cells(cx, cz, r + 0.2, r + 1.15):
            put(x, fy, z, TRIM_ST.at(x, fy, z, toward(cx, cz, x, z), "top"), only_air=True)
        if windows:
            wy = fy + 2
            for k in range(nw):
                a = 2 * math.pi * (k + 0.5 * ((fy // floors) % 2)) / nw
                wx, wz = cx + (r - 0.5) * math.cos(a), cz + (r - 0.5) * math.sin(a)
                ix, iz = int(round(wx)), int(round(wz))
                ox, oz = int(round(cx + (r + 0.5) * math.cos(a))), int(round(cz + (r + 0.5) * math.sin(a)))
                inx, inz = int(round(cx + (r - 1.5) * math.cos(a))), int(round(cz + (r - 1.5) * math.sin(a)))
                f_out = vdir(math.cos(a), math.sin(a))
                for h in range(3):
                    put(ix, wy + h, iz, 0)
                    put(inx, wy + h, inz, PANE["z" if f_out in ("east", "west") else "x"])
                put(ix, wy + 3, iz, TRIM_ST.at(ix, wy, iz, f_out, "top"))
                put(ox, wy - 1, oz, slab("smooth_quartz", "top"), only_air=True)
    if spiral:
        spiral_stairs(cx, cz, r - 2, y0 + 1, y1 - 1)
    top = y1
    if crown:
        # corbel table + projecting parapet + crenels
        for x, z, d in ring_cells(cx, cz, r - 0.2, r + 1.25):
            put(x, y1 - 3, z, TRIM_ST.at(x, y1, z, toward(cx, cz, x, z), "top"))
            fill(x, y1 - 2, z, x, y1, z, wall)
        for x, z, d in ring_cells(cx, cz, 0, r + 1.25):
            put(x, y1, z, FLOOR_MIX.at(x, y1, z) if d < r - 1 else TRIM.at(x, y1, z))
        cells = ring_cells(cx, cz, r + 0.3, r + 1.25)
        for x, z, d in cells:
            a = math.atan2(z - cz, x - cx)
            if math.sin(a * max(6, int(2 * r))) > 0:
                put(x, y1 + 1, z, wall.at(x, y1 + 1, z))
                put(x, y1 + 2, z, slab("smooth_quartz", "bottom"))
            else:
                put(x, y1 + 1, z, slab("smooth_quartz", "bottom"))
        top = y1 + 2
    if cone:
        cr = cone_r if cone_r else r + 1.3
        base = top + 1 if crown else y1 + 1
        if crown:  # drum inside the parapet up to the cone eave
            base = y1 + 3
            for x, z, d in ring_cells(cx, cz, 0, r - 0.6):
                fill(x, y1 + 1, z, x, y1 + 2, z, wall if d > r - 1.6 else 0)
        cone_roof(cx, cz, cr, base, cone)
        if finial:
            fin(cx, base + cone, cz)
        if beam:
            fill(int(cx), base + cone + 5, int(cz), int(cx), base + cone + 30, int(cz), B("light_blue_stained_glass"))
    return top


def cone_roof(cx, cz, r, y, h):
    """Cone of slate with stair blocks on its surface (stairs face the axis) and a flared eave."""
    for k in range(h):
        rad = r * (1 - k / h) ** 1.05
        nrad = r * (1 - (k + 1) / h) ** 1.05
        for x, z, d in ring_cells(cx, cz, 0, max(rad, 0.5)):
            if d > nrad - 0.1 and rad > 1.0:
                put(x, y + k, z, SLATE_ST.at(x, y + k, z, toward(cx, cz, x, z)))
            else:
                put(x, y + k, z, SLATE.at(x, y + k, z))
    for x, z, d in ring_cells(cx, cz, r + 0.2, r + 1.2):   # eave flare
        put(x, y - 1, z, SLATE_ST.at(x, y, z, toward(cx, cz, x, z), "top"), only_air=True)
        put(x, y, z, SLATE_ST.at(x, y, z, toward(cx, cz, x, z)), only_air=True)


def fin(x, y, z):
    x, z = int(round(x)), int(round(z))
    put(x, y, z, B("gold_block"))
    put(x, y + 1, z, B("lightning_rod[facing=up,powered=false,waterlogged=false]"))
    put(x, y + 2, z, B("lightning_rod[facing=up,powered=false,waterlogged=false]"))


def spiral_stairs(cx, cz, r, y0, y1, mat="stone_brick", per_turn=8, offset=0.0):
    """Walkable spiral staircase around a central column (`per_turn` steps per revolution).
    The step at height y sits at angle offset + (y - y0) / per_turn * 2pi."""
    icx, icz = int(round(cx)), int(round(cz))
    fill(icx, y0, icz, icx, y1, icz, B("quartz_pillar[axis=y]"))
    cells = ring_cells(cx, cz, 0.9, r + 0.3)
    for y in range(y0, y1 + 1):
        a0 = (offset + (y - y0) / per_turn * 2 * math.pi) % (2 * math.pi)
        for x, z, d in cells:
            a = math.atan2(z - cz, x - cx) % (2 * math.pi)
            da = (a - a0 + math.pi / per_turn) % (2 * math.pi)
            if da < 2 * math.pi / per_turn + 0.05:
                tang = vdir(-math.sin(a), math.cos(a))
                put(x, y, z, stairs(mat, tang))
                if get(x, y - 1, z) == 0:
                    put(x, y - 1, z, B("stone_bricks"))
                for hh in range(1, 4):
                    if get(x, y + hh, z) not in (0,) and y + hh <= y1:
                        put(x, y + hh, z, 0)


# ------------------------------------------------------------------ interiors
def floor(x0, x1, z0, z1, y, mix=FLOOR_MIX, beams=True):
    fill(x0, y, z0, x1, y, z1, mix)
    if beams:
        axis = "x" if (x1 - x0) < (z1 - z0) else "z"
        if axis == "x":
            for z in range(z0 + 2, z1, 4):
                fill(x0, y - 1, z, x1, y - 1, z, B("stripped_dark_oak_log[axis=x]"))
        else:
            for x in range(x0 + 2, x1, 4):
                fill(x, y - 1, z0, x, y - 1, z1, B("stripped_dark_oak_log[axis=z]"))


def chandelier(x, y, z, drop=3):
    for k in range(drop):
        put(x, y - k, z, CHAIN)
    put(x, y - drop, z, LANT_H)


def table(x0, x1, z0, z1, y, mat="dark_oak", chairs=True, cloth=None):
    """Long table of slabs on fence legs with stair chairs along the long sides."""
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            edge = x in (x0, x1) and z in (z0, z1)
            put(x, y, z, B(f"{mat}_fence[east=false,north=false,south=false,west=false,waterlogged=false]") if edge else 0)
            put(x, y + 1, z, slab(mat, "top") if cloth is None else B(cloth))
    if chairs:
        if (x1 - x0) >= (z1 - z0):
            for x in range(x0, x1 + 1, 2):
                put(x, y, z0 - 1, stairs("spruce", "north"))
                put(x, y, z1 + 1, stairs("spruce", "south"))
        else:
            for z in range(z0, z1 + 1, 2):
                put(x0 - 1, y, z, stairs("spruce", "west"))
                put(x1 + 1, y, z, stairs("spruce", "east"))


def bed(x, y, z, facing, color="red"):
    dx, dz = DV[facing]
    put(x, y, z, B(f"{color}_bed[facing={facing},occupied=false,part=foot]"))
    put(x + dx, y, z + dz, B(f"{color}_bed[facing={facing},occupied=false,part=head]"))


def carpet_run(x0, x1, z0, z1, y, color):
    fill(x0, y, z0, x1, y, z1, B(f"{color}_carpet"), only_air=True)


def shelf_wall(x0, x1, z0, z1, y0, y1):
    fill(x0, y0, z0, x1, y1, z1, Mix([("bookshelf", 70), ("chiseled_bookshelf[facing=north,slot_0_occupied=true,slot_1_occupied=true,slot_2_occupied=false,slot_3_occupied=true,slot_4_occupied=false,slot_5_occupied=true]", 15),
                                      ("spruce_planks", 15)], seed=x0 * 7 + z0))
