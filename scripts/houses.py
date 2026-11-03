"""Town house kit: every house is assembled from randomised parts so no two look alike.

Styles: half-timbered (frame patterns: plain / diagonal / X / K, jettied upper floors), rubble stone with ashlar
quoins, coloured plaster with stone trims, brick with stepped gable. Shapes: rectangle or L (wing) with a combined
height-field roof (valleys form where the parts meet). Roofs: gable (eaves- or gable-to-street), hip, half-hip;
slate, wood shingles, red tile, tan tile, green copper. Details: shutters & flower boxes in house colours,
door canopies, steps, oriel, balcony, chimneys with smoke, downspouts, drip-edge strip, firewood, barrels,
gardens; interiors by trade (baker, smith, weaver, scholar, brewer, cartographer, butcher, home)."""
import math

import numpy as np

import arch as A
import mc
from mc import B, DV, OPP, Mix, StairMix, fill, get, put, slab, stairs, vdir

PLASTERS = {
    "white": Mix([("calcite", 45), ("smooth_quartz", 30), ("white_terracotta", 10), ("mushroom_stem", 15)], seed=401, scale=5),
    "cream": Mix([("smooth_sandstone", 60), ("white_terracotta", 25), ("birch_planks", 15)], seed=402, scale=5),
    "ochre": Mix([("smooth_sandstone", 45), ("yellow_terracotta", 20), ("cut_sandstone", 20), ("white_terracotta", 15)], seed=403, scale=5),
    "rose": Mix([("white_terracotta", 70), ("calcite", 20), ("pink_terracotta", 10)], seed=404, scale=5),
    "grey": Mix([("polished_diorite", 40), ("calcite", 35), ("light_gray_terracotta", 25)], seed=405, scale=5),
    "sage": Mix([("pale_oak_planks", 40), ("calcite", 35), ("light_gray_terracotta", 25)], seed=406, scale=5),
}
RUBBLE = Mix([("cobblestone", 34), ("stone", 18), ("andesite", 18), ("mossy_cobblestone", 12), ("tuff", 10), ("diorite", 4),
              ("gravel", 4)], seed=407, scale=3)
BRICKW = Mix([("bricks", 70), ("resin_bricks", 12), ("terracotta", 10), ("polished_granite", 8)], seed=408, scale=6)
WOODS = {"dark_oak": ("dark_oak_log", "stripped_dark_oak_log", "dark_oak"),
         "spruce": ("spruce_log", "stripped_spruce_log", "spruce"),
         "oak": ("oak_log", "stripped_oak_log", "oak"),
         "mangrove": ("mangrove_log", "stripped_mangrove_log", "mangrove")}
ROOFMATS = {
    "slate": (A.SLATE, A.SLATE_ST, "deepslate_tile"),
    "spruce": (Mix([("spruce_planks", 1)]), StairMix(["spruce"]), "spruce"),
    "dark_oak": (Mix([("dark_oak_planks", 1)]), StairMix(["dark_oak"]), "dark_oak"),
    "redtile": (Mix([("bricks", 60), ("polished_granite", 40)], seed=409, scale=6), StairMix(["brick", "polished_granite", "brick"], seed=410, scale=6), "brick"),
    "tantile": (Mix([("mud_bricks", 70), ("packed_mud", 30)], seed=411, scale=6), StairMix(["mud_brick"]), "mud_brick"),
    "copper": (Mix([("waxed_weathered_cut_copper", 60), ("waxed_oxidized_cut_copper", 40)], seed=412, scale=7),
               StairMix(["waxed_weathered_cut_copper", "waxed_oxidized_cut_copper"], seed=413, scale=7), "waxed_weathered_cut_copper"),
    "mangrove": (Mix([("mangrove_planks", 1)]), StairMix(["mangrove"]), "mangrove"),
}
SHUTTERS = ["spruce", "dark_oak", "mangrove", "warped", "birch", "jungle", "crimson", "oak"]
DOORS = ["spruce", "dark_oak", "oak", "mangrove", "cherry", "birch", "jungle", "crimson"]
GLASS = ["glass_pane", "glass_pane", "brown_stained_glass_pane", "light_gray_stained_glass_pane", "gray_stained_glass_pane"]
POTS = ["potted_red_tulip", "potted_azure_bluet", "potted_cornflower", "potted_poppy", "potted_allium", "potted_fern",
        "potted_orange_tulip", "potted_lily_of_the_valley", "potted_oxeye_daisy", "potted_azalea_bush"]


# ------------------------------------------------------------------ combined roof
def roof_parts(parts, y, rise, over, mat, st, end_wall, cap):
    """parts: list of (x0, x1, z0, z1, kind, axis). Height field = max over parts -> clean valleys/hips."""
    H = {}
    fp = set()
    for (x0, x1, z0, z1, kind, axis) in parts:
        xc, zc = (x0 + x1) / 2.0, (z0 + z1) / 2.0
        hwx, hwz = (x1 - x0) / 2.0 + over, (z1 - z0) / 2.0 + over
        for x in range(x0 - over, x1 + over + 1):
            for z in range(z0 - over, z1 + over + 1):
                dx, dz = hwx - abs(x - xc), hwz - abs(z - zc)
                if kind == "hip":
                    h = y + rise * min(dx, dz)
                elif kind == "halfhip":
                    g = y + rise * (dz if axis == "x" else dx)
                    o, w = (dx, hwz) if axis == "x" else (dz, hwx)
                    h = min(g, y + rise * (o + w * 0.45))
                else:
                    h = y + rise * (dz if axis == "x" else dx)
                H[(x, z)] = max(H.get((x, z), -1e9), h)
        for x in range(x0, x1 + 1):
            for z in range(z0, z1 + 1):
                fp.add((x, z))
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
        if bdir is None:
            put(x, h, z, mat.at(x, h, z))
            put(x, h + 1, z, slab(cap, "bottom"))
            top = max(top, h + 1)
        else:
            put(x, h, z, st.at(x, h, z, bdir))
            top = max(top, h)
    # fill walls up to the roof surface (gable triangles), only on footprint edges
    for (x, z) in fp:
        edge = any((x + dx, z + dz) not in fp for dx, dz in DV.values())
        if edge:
            h = int(math.floor(H[(x, z)] + 1e-6))
            for yy in range(y, h):
                if get(x, yy, z) == 0:
                    put(x, yy, z, end_wall.at(x, yy, z))
    return top, H


# ------------------------------------------------------------------ the house
class House:
    def __init__(self, x0, z0, w, d, front, rng, P, trade=None):
        self.x0, self.z0, self.x1, self.z1 = x0, z0, x0 + w - 1, z0 + d - 1
        self.front, self.rng, self.P = front, rng, P
        r = rng.random
        self.style = rng.choice(["timber", "timber", "timber", "timber", "stone", "plaster", "plaster", "plaster", "brick"])
        self.storeys = int(rng.choice([1, 2, 2, 2, 3, 3, 4], p=[0.06, 0.3, 0.2, 0.14, 0.15, 0.1, 0.05]))
        self.sh = int(rng.choice([4, 5, 5]))
        self.plaster = PLASTERS[rng.choice(list(PLASTERS))]
        self.wood = WOODS[rng.choice(list(WOODS), p=[0.45, 0.3, 0.1, 0.15])]
        self.pattern = rng.choice(["plain", "diag", "x", "k", "diag"])
        self.post_sp = int(rng.choice([2, 3, 3, 4]))
        self.jetty = self.style == "timber" and r() < 0.7 and self.storeys >= 2
        roof_pool = ["slate", "spruce", "dark_oak", "redtile", "redtile", "tantile", "mangrove"] + (["copper"] if r() < 0.15 else [])
        self.roofmat = ROOFMATS[rng.choice(roof_pool)]
        self.roofkind = rng.choice(["gable", "gable", "gable", "halfhip", "hip"])
        self.rise = 2 if (self.roofkind == "gable" and r() < 0.45) else 1
        self.gable_to_street = r() < 0.5
        self.shutter = rng.choice(SHUTTERS)
        self.doorwood = rng.choice(DOORS)
        self.glass = rng.choice(GLASS)
        self.flowerboxes = r() < 0.55
        self.trade = trade or rng.choice(["home", "home", "home", "baker", "smith", "weaver", "scholar", "brewer",
                                          "cartographer", "butcher", "tailor"])
        self.win_w = int(rng.choice([1, 2, 2]))
        self.win_h = 2 if self.sh == 4 else int(rng.choice([2, 3]))
        self.wing = None
        if w >= 9 and d >= 9 and r() < 0.35:
            ww, wd = int(rng.integers(5, max(6, w // 2 + 1))), int(rng.integers(4, max(5, d // 2)))
            back = OPP[front]
            bx, bz = DV[back]
            if bx == 0:
                wx0 = self.x0 if r() < 0.5 else self.x1 - ww + 1
                wz0 = self.z1 + 1 if bz > 0 else self.z0 - wd
                self.wing = (wx0, wx0 + ww - 1, wz0, wz0 + wd - 1)
            else:
                wz0 = self.z0 if r() < 0.5 else self.z1 - ww + 1
                wx0 = self.x1 + 1 if bx > 0 else self.x0 - wd
                self.wing = (wx0, wx0 + wd - 1, wz0, wz0 + ww - 1)

    # ---------------------------------------------------------- walls
    def wall_mix(self, storey):
        if self.style == "stone":
            return RUBBLE
        if self.style == "brick":
            return BRICKW
        if self.style == "timber":
            return RUBBLE if storey == 0 else self.plaster
        return self.plaster

    def block(self, x0, x1, z0, z1, wing=False):
        P, sh, n = self.P, self.sh, self.storeys - (1 if wing and self.storeys > 1 else 0)
        top = P + n * sh
        for s in range(n):
            mix = self.wall_mix(s)
            j = 1 if (self.jetty and s >= 1) else 0
            xa, xb, za, zb = x0 - j, x1 + j, z0 - j, z1 + j
            y0, y1 = P + s * sh + 1, P + (s + 1) * sh
            A.shell(xa, xb, za, zb, y0, y1, mix, Mix([(self.wood[2] + "_planks", 1)]), thick=1)
            fill(xa + 1, P + s * sh, za + 1, xb - 1, P + s * sh, zb - 1, Mix([(self.wood[2] + "_planks", 70), ("spruce_planks", 30)], seed=414, scale=4))
            if j and s == 1:
                for side in A.SIDES:
                    line, (nx, nz) = A.side_line(xa, xb, za, zb, side)
                    for i, (x, z) in enumerate(line[1:-1]):
                        if i % 2 == 0:
                            put(x, y0 - 1, z, stairs(self.wood[2], OPP[side], "top"))
            if self.style == "timber" and s >= 1:
                self.frame(xa, xb, za, zb, y0, y1)
            if self.style in ("stone", "plaster", "brick") or (self.style == "timber" and s == 0):
                self.quoins(xa, xb, za, zb, y0, y1)
            if self.style == "plaster" and s == 0:
                for side in A.SIDES:         # stone plinth band
                    line, _ = A.side_line(xa, xb, za, zb, side)
                    for (x, z) in line:
                        put(x, y0, z, RUBBLE.at(x, y0, z))
            self.windows(xa, xb, za, zb, s, y0)
        # cornice under the eaves for masonry styles
        if self.style in ("plaster", "brick", "stone"):
            for side in A.SIDES:
                line, (nx, nz) = A.side_line(x0, x1, z0, z1, side)
                for (x, z) in line:
                    put(x + nx, top, z + nz, stairs("smooth_quartz" if self.style == "plaster" else "stone_brick", OPP[side], "top"), only_air=True)
        return top

    def frame(self, x0, x1, z0, z1, y0, y1):
        log_v, strip, plank = self.wood
        for side in A.SIDES:
            line, (nx, nz) = A.side_line(x0, x1, z0, z1, side)
            n = len(line)
            axis = "x" if side in ("north", "south") else "z"
            posts = [i for i in range(n) if i % self.post_sp == 0 or i == n - 1]
            for i, (x, z) in enumerate(line):
                put(x, y0, z, B(f"{strip}[axis={axis}]"))        # sill beam
                put(x, y1, z, B(f"{log_v}[axis={axis}]"))        # head beam
                if i in posts:
                    fill(x, y0 + 1, z, x, y1 - 1, z, B(f"{log_v}[axis=y]"))
            ph = y1 - y0 - 1
            for a, b in zip(posts, posts[1:]):
                pw = b - a - 1
                if pw < 2 or ph < 2:
                    continue
                pat = self.pattern
                if pat == "plain":
                    continue
                seed = (a * 7 + y0 * 3) % 5
                if pat == "diag" or (pat == "k" and seed % 2):
                    cells = [(a + 1 + i, y0 + 1 + int(round(i * (ph - 1) / max(pw - 1, 1)))) for i in range(pw)]
                    if (a // self.post_sp) % 2:
                        cells = [(a + 1 + i, y0 + ph - int(round(i * (ph - 1) / max(pw - 1, 1)))) for i in range(pw)]
                elif pat == "x":
                    cells = [(a + 1 + i, y0 + 1 + int(round(i * (ph - 1) / max(pw - 1, 1)))) for i in range(pw)]
                    cells += [(a + 1 + i, y0 + ph - int(round(i * (ph - 1) / max(pw - 1, 1)))) for i in range(pw)]
                else:
                    continue
                for i, yy in cells:
                    x, z = line[i]
                    put(x, yy, z, B(f"{strip}[axis=y]"))

    def quoins(self, x0, x1, z0, z1, y0, y1):
        trim = {"stone": "stone_bricks", "brick": "smooth_sandstone", "plaster": "smooth_quartz", "timber": "stone_bricks"}[self.style]
        for x in (x0, x1):
            for z in (z0, z1):
                for yy in range(y0, y1 + 1):
                    put(x, yy, z, B(trim))
                    if (yy - y0) % 2 == 0:       # alternating long / short stones
                        dx = 1 if x == x0 else -1
                        put(x + dx, yy, z, B(trim))
                    else:
                        dz = 1 if z == z0 else -1
                        put(x, yy, z + dz, B(trim))

    def windows(self, x0, x1, z0, z1, s, y0):
        wy = y0 + (1 if self.sh == 4 else 1)
        for side in A.SIDES:
            line, (nx, nz) = A.side_line(x0, x1, z0, z1, side)
            n = len(line)
            if n < 5:
                continue
            if self.style == "timber" and s >= 1:
                # windows sit in the frame panels between posts (never on a post)
                posts = [i for i in range(n) if i % self.post_sp == 0 or i == n - 1]
                starts = []
                for a, b in zip(posts, posts[1:]):
                    pw = b - a - 1
                    if pw >= 1 and (a // self.post_sp) % 2 == (s % 2):
                        starts.append(a + 1 + max(0, (pw - self.win_w) // 2))
            else:
                step = int(self.rng.choice([3, 4]))
                starts = list(range(2, n - 2 - self.win_w + 1, max(step, self.win_w + 1)))
            for c in starts:
                if side == self.front and s == 0 and abs(c - n // 2) <= 2:
                    continue
                if self.rng.random() < 0.15:
                    continue                                  # a blank bay now and then
                cols = [c + k for k in range(self.win_w) if c + k < n - 1]
                for cc in cols:
                    x, z = line[cc]
                    for k in range(self.win_h):
                        put(x, wy + k, z, B(self.glass))
                    if self.style != "timber" or s == 0:     # lintel / sill in stone or quartz
                        put(x, wy + self.win_h, z, B("stripped_" + self.wood[2] + "_log[axis=" + ("x" if side in ("north", "south") else "z") + "]")
                            if self.style == "timber" else B("smooth_quartz" if self.style == "plaster" else "stone_bricks"))
                    put(x + nx, wy - 1, z + nz, slab("smooth_stone" if self.style != "timber" else self.wood[2], "top"))
                # shutters
                if self.style in ("timber", "plaster") or self.rng.random() < 0.4:
                    for cc in (cols[0] - 1, cols[-1] + 1):
                        if 0 < cc < n - 1:
                            x, z = line[cc]
                            for k in range(self.win_h):
                                put(x + nx, wy + k, z + nz, B(f"{self.shutter}_trapdoor[facing={side},half=bottom,open=true,powered=false,waterlogged=false]"), only_air=True)
                if self.flowerboxes and s >= 1:
                    for cc in cols:
                        x, z = line[cc]
                        put(x + nx, wy, z + nz, B(self.rng.choice(POTS)), only_air=True)

    # ---------------------------------------------------------- features
    def door(self):
        line, (nx, nz) = A.side_line(self.x0, self.x1, self.z0, self.z1, self.front)
        n = len(line)
        c = n // 2 + int(self.rng.integers(-1, 2))
        c = min(max(c, 2), n - 3)
        double = self.trade != "home" and self.rng.random() < 0.6
        cols = [c, c + 1] if double else [c]
        P = self.P
        for j, cc in enumerate(cols):
            x, z = line[cc]
            for k in range(1, 4):
                put(x, P + k, z, 0)
            hinge = "left" if j == 0 else "right"
            put(x, P + 1, z, B(f"{self.doorwood}_door[facing={OPP[self.front]},half=lower,hinge={hinge},open=false,powered=false]"))
            put(x, P + 2, z, B(f"{self.doorwood}_door[facing={OPP[self.front]},half=upper,hinge={hinge},open=false,powered=false]"))
            put(x, P + 3, z, B(f"stripped_{self.wood[2]}_log[axis={'x' if self.front in ('north', 'south') else 'z'}]"))
            put(x + nx, P, z + nz, stairs("stone_brick" if self.rng.random() < 0.6 else "cobblestone", OPP[self.front]))
        # canopy over the door
        if self.rng.random() < 0.6:
            for cc in range(cols[0] - 1, cols[-1] + 2):
                x, z = line[cc]
                put(x + nx, P + 4, z + nz, stairs(self.wood[2], OPP[self.front]))
            x, z = line[cols[0]]
            put(x + nx, P + 3, z + nz, B("lantern[hanging=true,waterlogged=false]"))
        else:
            x, z = line[cols[-1] + 1]
            put(x + nx, P + 3, z + nz, B("lantern[hanging=true,waterlogged=false]"))
            put(x + nx, P + 4, z + nz, slab(self.wood[2], "bottom"))
        return line[c], (nx, nz)

    def oriel(self, top):
        """Corbelled corner bay (Erker) on the front from the 2nd storey with its own little hip roof."""
        if self.storeys < 2 or self.rng.random() > 0.3:
            return
        line, (nx, nz) = A.side_line(self.x0, self.x1, self.z0, self.z1, self.front)
        n = len(line)
        c0 = 1 if self.rng.random() < 0.5 else n - 4
        cells = [line[c0 + i] for i in range(3)]
        y0, y1 = self.P + self.sh + 1, top - 1
        for (x, z) in cells:
            ox, oz = x + nx, z + nz
            put(ox, y0 - 1, oz, stairs("smooth_quartz" if self.style == "plaster" else self.wood[2], OPP[self.front], "top"))
            for yy in range(y0, y1 + 1):
                put(ox, yy, oz, self.wall_mix(1).at(ox, yy, oz))
            for yy in range(y0 + 1, y0 + 1 + self.win_h):
                put(ox, yy, oz, B(self.glass))
        xs = [x + nx for x, _ in cells]
        zs = [z + nz for _, z in cells]
        roof_parts([(min(xs), max(xs), min(zs), max(zs), "hip", "x")], y1 + 1, 1, 0, self.roofmat[0], self.roofmat[1],
                   self.plaster, self.roofmat[2])

    def balcony(self):
        if self.storeys < 2 or self.rng.random() > 0.3:
            return
        line, (nx, nz) = A.side_line(self.x0, self.x1, self.z0, self.z1, self.front)
        n = len(line)
        c = n // 2
        y = self.P + self.sh + (1 if self.jetty else 0)
        j = 1 if self.jetty else 0
        for cc in range(c - 2, c + 3):
            x, z = line[cc]
            ox, oz = x + nx * (1 + j), z + nz * (1 + j)
            put(ox, y, oz, slab(self.wood[2], "top"))
            put(ox + nx, y, oz + nz, slab(self.wood[2], "top"))
            put(ox + nx, y + 1, oz + nz, B(f"{self.wood[2]}_fence[east=false,north=false,south=false,west=false,waterlogged=false]"))
            put(ox, y - 1, oz, stairs(self.wood[2], OPP[self.front], "top"), only_air=True)
        for cc in (c - 2, c + 2):
            x, z = line[cc]
            put(x + nx * (1 + j), y + 1, z + nz * (1 + j), B(f"{self.wood[2]}_fence[east=false,north=false,south=false,west=false,waterlogged=false]"))
        x, z = line[c]
        x, z = x + nx * j, z + nz * j
        put(x, y + 1, z, B(f"{self.doorwood}_door[facing={OPP[self.front]},half=lower,hinge=left,open=false,powered=false]"))
        put(x, y + 2, z, B(f"{self.doorwood}_door[facing={OPP[self.front]},half=upper,hinge=left,open=false,powered=false]"))

    def chimneys(self, Hmap):
        for k in range(1 if self.rng.random() < 0.7 else 2):
            x = int(self.rng.integers(self.x0 + 1, self.x1))
            z = int(self.rng.integers(self.z0 + 1, self.z1))
            mix = BRICKW if self.rng.random() < 0.6 else RUBBLE
            ytop = int(math.floor(Hmap.get((x, z), self.P + self.storeys * self.sh))) + int(self.rng.integers(2, 4))
            fill(x, self.P + 1, z, x, ytop, z, mix)
            put(x, ytop + 1, z, slab("stone_brick" if mix is RUBBLE else "brick", "bottom") if self.rng.random() < 0.4 else
                B("campfire[facing=north,lit=true,signal_fire=false,waterlogged=false]"))
            put(x, self.P + 1, z, B("furnace[facing=north,lit=true]") if k == 0 else B("smoker[facing=south,lit=true]"))

    def downspouts(self, eave):
        for x in (self.x0 - (1 if self.jetty else 0), self.x1 + (1 if self.jetty else 0)):
            for z in (self.z0 - 1,):
                if self.rng.random() < 0.5:
                    fill(x, self.P + 1, z, x, eave - 1, z, B("copper_chain[axis=y,waterlogged=false]"))
                    put(x, self.P, z, B("cauldron") if get(x, self.P, z) else 0) if False else None

    def surroundings(self):
        """Drip-edge strip, firewood, barrels, a small garden behind."""
        P, rng = self.P, self.rng
        for side in A.SIDES:
            line, (nx, nz) = A.side_line(self.x0, self.x1, self.z0, self.z1, side)
            for (x, z) in line:
                gx, gz = x + nx, z + nz
                if get(gx, P + 1, gz) == 0 and get(gx, P, gz) in GROUNDY:
                    put(gx, P, gz, B("gravel") if mc.noise3(gx, 0, gz, 3, 9) > -0.2 else B("coarse_dirt"))
        side = rng.choice([s for s in A.SIDES if s != self.front])
        line, (nx, nz) = A.side_line(self.x0, self.x1, self.z0, self.z1, side)
        if rng.random() < 0.5 and len(line) > 5:     # firewood stack under the eave
            axis = "x" if side in ("north", "south") else "z"
            for i in range(1, min(5, len(line) - 1)):
                x, z = line[i]
                for k in range(1, 3):
                    put(x + nx, P + k, z + nz, B(f"stripped_spruce_log[axis={axis}]") if (i + k) % 3 else B(f"oak_log[axis={axis}]"), only_air=True)
        if rng.random() < 0.6:
            x, z = line[-2]
            put(x + nx, P + 1, z + nz, B("barrel[facing=up,open=false]"), only_air=True)
            if rng.random() < 0.5:
                put(x + nx, P + 2, z + nz, B("barrel[facing=up,open=false]"), only_air=True)
        if rng.random() < 0.3:
            x, z = line[1]
            put(x + nx, P + 1, z + nz, B("composter[level=3]"), only_air=True)

    def interior(self):
        P, sh, rng = self.P, self.sh, self.rng
        ix0, ix1, iz0, iz1 = self.x0 + 1, self.x1 - 1, self.z0 + 1, self.z1 - 1
        if ix1 - ix0 < 2 or iz1 - iz0 < 2:
            return
        jobs = {"baker": ["smoker[facing=north,lit=true]", "furnace[facing=north,lit=false]", "hay_block[axis=y]", "barrel[facing=up,open=false]"],
                "smith": ["blast_furnace[facing=north,lit=true]", "anvil[facing=east]", "grindstone[face=floor,facing=north]", "smithing_table"],
                "weaver": ["loom[facing=north]", "white_wool", "red_wool", "barrel[facing=up,open=false]"],
                "scholar": ["lectern[facing=north,has_book=false,powered=false]", "bookshelf", "bookshelf", "enchanting_table"],
                "brewer": ["brewing_stand[has_bottle_0=false,has_bottle_1=false,has_bottle_2=false]", "water_cauldron[level=3]", "barrel[facing=up,open=false]", "barrel[facing=up,open=false]"],
                "cartographer": ["cartography_table", "lectern[facing=north,has_book=false,powered=false]", "barrel[facing=up,open=false]", "bookshelf"],
                "butcher": ["smoker[facing=north,lit=true]", "barrel[facing=up,open=false]", "chest[facing=north,type=single,waterlogged=false]", "hay_block[axis=y]"],
                "tailor": ["loom[facing=north]", "cyan_wool", "yellow_wool", "chest[facing=north,type=single,waterlogged=false]"],
                "home": ["crafting_table", "barrel[facing=up,open=false]", "chest[facing=north,type=single,waterlogged=false]", "furnace[facing=north,lit=false]"]}
        items = jobs[self.trade]
        for i, it in enumerate(items):
            x = ix0 + i
            if x <= ix1:
                put(x, P + 1, iz0, B(it), only_air=True)
        tx, tz = (ix0 + ix1) // 2, (iz0 + iz1) // 2 + 1
        put(tx, P + 1, tz, B(f"{self.wood[2]}_fence[east=false,north=false,south=false,west=false,waterlogged=false]"), only_air=True)
        put(tx, P + 2, tz, B(f"{self.wood[2]}_pressure_plate[powered=false]"), only_air=True)
        put(tx - 1, P + 1, tz, stairs(self.wood[2], "west"), only_air=True)
        put(tx + 1, P + 1, tz, stairs(self.wood[2], "east"), only_air=True)
        put(tx, P + sh - 1, tz, B("lantern[hanging=true,waterlogged=false]"), only_air=True)
        if self.storeys > 1:
            for h in range(1, sh + 1):
                put(ix1, P + h, iz1, B("ladder[facing=north,waterlogged=false]"))
            for s in range(1, self.storeys):
                y = P + s * sh
                put(ix1, y, iz1, B("ladder[facing=north,waterlogged=false]"))
                for h in range(1, sh + 1):
                    if s < self.storeys - 1:
                        put(ix1, y + h, iz1, B("ladder[facing=north,waterlogged=false]"))
                col = rng.choice(["red", "blue", "yellow", "green", "white", "brown", "cyan", "lime", "purple", "orange"])
                if iz1 - iz0 >= 3:
                    A.bed(ix0, y + 1, iz1 - 1, "north", col)
                put(ix0 + 1, y + 1, iz0, B("chest[facing=south,type=single,waterlogged=false]"), only_air=True)
                fill(ix0 + 1, y + 1, iz0 + 1, ix1 - 1, y + 1, iz1 - 2, B(f"{col}_carpet"), only_air=True)
                put(ix0 + 2, y + 1, iz0, B(rng.choice(["bookshelf", "barrel[facing=up,open=false]", "flower_pot", "potted_fern"])), only_air=True)
                put((ix0 + ix1) // 2, y + sh - 1, (iz0 + iz1) // 2, B("lantern[hanging=true,waterlogged=false]"), only_air=True)

    def build(self):
        top = self.block(self.x0, self.x1, self.z0, self.z1)
        parts_axis = ("x" if self.front in ("north", "south") else "z")
        if self.gable_to_street:
            parts_axis = "z" if parts_axis == "x" else "x"
        j = 1 if self.jetty else 0
        parts = [(self.x0 - j, self.x1 + j, self.z0 - j, self.z1 + j, self.roofkind, parts_axis)]
        if self.wing:
            wx0, wx1, wz0, wz1 = self.wing
            wtop = self.block(wx0, wx1, wz0, wz1, wing=True)
            waxis = "z" if parts_axis == "x" else "x"
            # wing roof starts at its own eave but the combined height field keeps the valleys clean
            roof_parts([(wx0, wx1, wz0, wz1, "gable", waxis)], wtop + 1, self.rise if self.rise == 1 else 1, 1,
                       self.roofmat[0], self.roofmat[1], self.wall_mix(1), self.roofmat[2])
        door_cell, normal = self.door()
        rtop, Hmap = roof_parts(parts, top + 1, self.rise, 1, self.roofmat[0], self.roofmat[1], self.wall_mix(1), self.roofmat[2])
        self.oriel(top)
        self.balcony()
        self.chimneys(Hmap)
        self.surroundings()
        self.interior()
        return door_cell, normal


GROUNDY = set()


def init_ground_ids():
    for n in ("grass_block[snowy=false]", "dirt_path", "coarse_dirt", "dirt", "rooted_dirt", "moss_block", "podzol[snowy=false]"):
        GROUNDY.add(B(n))
