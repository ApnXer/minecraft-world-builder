"""Textured voxel raytracer for checking builds visually.

Uses the real block textures and models from the local Minecraft 1.21.11 jar.
Shapes: full cubes, stairs, slabs, walls, fences, panes, carpets, plants (crossed planes),
small boxes (lanterns, rods, ...). Alpha-tested leaves/plants, blended glass and water,
sun shadows, emissive light blocks.

    from render import Renderer
    r = Renderer(palette)            # palette: list of (name, props)
    img = r.ortho(arr, target=(x, y, z), yaw=35, pitch=30, width=1400, height=900, ppb=6)
    img = r.persp(arr, eye=(x, y, z), look=(x, y, z), fov=70, width=1400, height=900)
    Image.fromarray(img).save("view.png")
"""
import io
import json
import math
import os
import zipfile

import numpy as np
from numba import njit, prange
from PIL import Image

from mcpaths import find_jar  # noqa: E402

JAR = None  # resolved lazily by Assets()
GRASS_TINT = (0x5f, 0xd1, 0x8a)
FOLIAGE_TINT = (0x4c, 0xc2, 0x7a)
WATER_TINT = (0x3f, 0xd0, 0xc9)

GRASS_TINTED = {"grass_block", "short_grass", "tall_grass", "fern", "large_fern", "sugar_cane", "potted_fern"}
FOLIAGE_TINTED = {"oak_leaves", "dark_oak_leaves", "jungle_leaves", "acacia_leaves", "mangrove_leaves", "vine"}
FIXED_TINT = {"spruce_leaves": (0x61, 0x99, 0x61), "birch_leaves": (0x80, 0xa7, 0x55), "lily_pad": (0x20, 0x80, 0x30)}
EMISSIVE = ("lantern", "sea_lantern", "glowstone", "shroomlight", "froglight", "torch", "campfire", "candle",
            "end_rod", "jack_o_lantern", "redstone_lamp", "beacon", "magma", "lava", "glow_berries", "fire", "copper_bulb",
            "firefly_bush")
CROSS = ("short_grass", "fern", "tall_grass", "large_fern", "dandelion", "poppy", "cornflower", "lily_of_the_valley",
         "azure_bluet", "oxeye_daisy", "allium", "tulip", "eyeblossom", "blue_orchid", "bush", "sweet_berry_bush",
         "sapling", "cave_vines", "pale_hanging_moss", "hanging_roots", "dead_bush", "dry_grass", "rose_bush", "lilac",
         "peony", "sunflower", "torchflower", "pitcher", "spore_blossom", "weeping_vines", "twisting_vines", "cobweb",
         "sugar_cane", "wheat", "carrots", "potatoes", "beetroots", "brown_mushroom", "red_mushroom", "seagrass",
         "kelp", "big_dripleaf_stem", "small_dripleaf", "glow_lichen_x", "closed_eyeblossom", "open_eyeblossom",
         "pointed_dripstone", "flowering_azalea_x")
FLAT = ("leaf_litter", "pink_petals", "wildflowers", "lily_pad")


# ------------------------------------------------------------------ jar model / texture lookup
class Assets:
    def __init__(self, jar=None):
        self.z = zipfile.ZipFile(jar or find_jar())
        self.names = set(self.z.namelist())
        self._models = {}
        self._tex = {}

    def json(self, path):
        return json.loads(self.z.read(path)) if path in self.names else None

    def model_textures(self, ref):
        ref = ref.replace("minecraft:", "")
        if ref in self._models:
            return self._models[ref]
        m = self.json(f"assets/minecraft/models/{ref}.json") or {}
        tex = {}
        if "parent" in m:
            tex.update(self.model_textures(m["parent"]))
        tex.update(m.get("textures", {}))
        for _ in range(4):  # resolve #refs
            for k, v in list(tex.items()):
                if isinstance(v, str) and v.startswith("#") and v[1:] in tex:
                    tex[k] = tex[v[1:]]
        self._models[ref] = tex
        return tex

    def block_model(self, name, props):
        bs = self.json(f"assets/minecraft/blockstates/{name}.json")
        if not bs:
            return None
        if "variants" in bs:
            best, score = None, -1
            for key, val in bs["variants"].items():
                conds = dict(kv.split("=") for kv in key.split(",") if kv)
                if all(props.get(k, v) == v for k, v in conds.items()):
                    sc = sum(props.get(k) == v for k, v in conds.items())
                    if sc > score:
                        best, score = val, sc
            if best is None:
                best = next(iter(bs["variants"].values()))
            best = best[0] if isinstance(best, list) else best
            return best["model"]
        for part in bs.get("multipart", []):
            ap = part["apply"]
            ap = ap[0] if isinstance(ap, list) else ap
            return ap["model"]
        return None

    def texture(self, ref):
        ref = ref.replace("minecraft:", "")
        if ref in self._tex:
            return self._tex[ref]
        p = f"assets/minecraft/textures/{ref}.png"
        if p not in self.names:
            self._tex[ref] = None
            return None
        im = Image.open(io.BytesIO(self.z.read(p))).convert("RGBA")
        w, h = im.size
        if h > w:
            im = im.crop((0, 0, w, w))
        im = im.resize((16, 16), Image.NEAREST)
        self._tex[ref] = np.asarray(im, np.float32) / 255.0
        return self._tex[ref]


def _pick(tex, keys):
    for k in keys:
        if k in tex and isinstance(tex[k], str) and not tex[k].startswith("#"):
            return tex[k]
    return None


# ------------------------------------------------------------------ shapes
def _stairs(pr):
    top = pr.get("half") == "top"
    a = (0, .5, 0, 1, 1, 1) if top else (0, 0, 0, 1, .5, 1)
    y0, y1 = (0, .5) if top else (.5, 1)
    f = pr.get("facing", "north")
    b = {"north": (0, y0, 0, 1, y1, .5), "south": (0, y0, .5, 1, y1, 1),
         "east": (.5, y0, 0, 1, y1, 1), "west": (0, y0, 0, .5, y1, 1)}[f]
    return [a, b]


def _arms(pr, w, h, post):
    out = [post]
    lo, hi = .5 - w, .5 + w
    for side, box in (("north", (lo, 0, 0, hi, h, .5)), ("south", (lo, 0, .5, hi, h, 1)),
                      ("west", (0, 0, lo, .5, h, hi)), ("east", (.5, 0, lo, 1, h, hi))):
        v = pr.get(side, "none")
        if v in ("true", "low", "tall"):
            hh = 1.0 if v == "tall" else h
            out.append((box[0], 0, box[2], box[3], hh, box[5]))
    return out


def shape_for(name, pr):
    """Return (kind, boxes). kind: 0 boxes opaque, 1 boxes alpha-test, 2 cross, 3 blend cube, 4 flat alpha."""
    n = name.replace("minecraft:", "")
    if n.endswith("_stairs"):
        return 0, _stairs(pr)
    if n.endswith("_slab"):
        t = pr.get("type", "bottom")
        return 0, [(0, 0, 0, 1, 1, 1)] if t == "double" else [(0, .5, 0, 1, 1, 1)] if t == "top" else [(0, 0, 0, 1, .5, 1)]
    if n.endswith("_wall") and not n.endswith("wall_torch") and "sign" not in n and "banner" not in n:
        return 0, _arms(pr, .19, .875, (.25, 0, .25, .75, 1, .75))
    if n.endswith("_fence"):
        return 0, _arms(pr, .07, .94, (.375, 0, .375, .625, 1, .625))
    if n.endswith("_pane") or n.endswith("_bars"):
        return 1, _arms(pr, .07, 1, (.4375, 0, .4375, .5625, 1, .5625))
    if n.endswith("carpet"):
        return 0, [(0, 0, 0, 1, 1 / 16, 1)]
    if any(n.endswith(f) for f in FLAT):
        return 4, [(0, 0, 0, 1, 1 / 16, 1)]
    if n.endswith("door") and "trap" not in n:
        f = pr.get("facing", "north")
        return 0, [{"north": (0, 0, .8125, 1, 1, 1), "south": (0, 0, 0, 1, 1, .1875),
                    "east": (0, 0, 0, .1875, 1, 1), "west": (.8125, 0, 0, 1, 1, 1)}[f]]
    if n.endswith("trapdoor"):
        if pr.get("open") == "true":
            return 0, [{"north": (0, 0, .8125, 1, 1, 1), "south": (0, 0, 0, 1, 1, .1875),
                        "east": (0, 0, 0, .1875, 1, 1), "west": (.8125, 0, 0, 1, 1, 1)}[pr.get("facing", "north")]]
        return 0, [(0, .8125, 0, 1, 1, 1)] if pr.get("half") == "top" else [(0, 0, 0, 1, .1875, 1)]
    if "lantern" in n and "sea" not in n and "jack" not in n:
        return 0, [(.3125, 0, .3125, .6875, .5625, .6875)] if pr.get("hanging") != "true" else [(.3125, .0625, .3125, .6875, .625, .6875)]
    if n in ("lightning_rod", "end_rod", "iron_chain", "copper_chain") or n.endswith("_chain") or n == "torch" or n.endswith("_torch"):
        return 0, [(.4375, 0, .4375, .5625, 1 if "torch" not in n else .625, .5625)]
    if "candle" in n and "cake" not in n:
        return 0, [(.375, 0, .375, .625, .4, .625)]
    if n.startswith("potted_") or n == "flower_pot":
        return 0, [(.3125, 0, .3125, .6875, .375, .6875)]
    if n.endswith("_bed"):
        return 0, [(0, 0, 0, 1, .5625, 1)]
    if n in ("chest", "trapped_chest", "ender_chest"):
        return 0, [(.0625, 0, .0625, .9375, .875, .9375)]
    if n in ("campfire", "soul_campfire"):
        return 0, [(0, 0, 0, 1, .4375, 1)]
    if n in ("dirt_path", "farmland"):
        return 0, [(0, 0, 0, 1, .9375, 1)]
    if n in ("vine", "glow_lichen", "ladder"):
        for side, box in (("north", (0, 0, 0, 1, 1, .06)), ("south", (0, 0, .94, 1, 1, 1)),
                          ("west", (0, 0, 0, .06, 1, 1)), ("east", (.94, 0, 0, 1, 1, 1))):
            if pr.get(side) == "true" or pr.get("facing") == {"north": "south", "south": "north", "west": "east", "east": "west"}[side]:
                return 4 if n != "ladder" else 1, [box]
        return 4, [(0, .94, 0, 1, 1, 1)]
    if n in ("water", "ice") or "stained_glass" in n and not n.endswith("pane"):
        return 3, [(0, 0, 0, 1, 1, 1)]
    if any(c in n for c in CROSS):
        return 2, []
    if n.endswith("leaves") or n in ("glass", "azalea", "flowering_azalea", "mangrove_roots", "spawner", "tinted_glass_x"):
        return 1, [(0, 0, 0, 1, 1, 1)]
    if n in ("bell", "brewing_stand", "grindstone"):
        return 0, [(.25, 0, .25, .75, .75, .75)]
    return 0, [(0, 0, 0, 1, 1, 1)]


# ------------------------------------------------------------------ renderer
class Renderer:
    def __init__(self, palette):
        self.assets = A = Assets()
        atlas = [np.zeros((16, 16, 4), np.float32)]
        atlas[0][..., :3] = (1, 0, 1)
        atlas[0][..., 3] = 1
        tex_index = {}

        def tid(ref):
            if ref is None:
                return 0
            if ref not in tex_index:
                t = A.texture(ref)
                if t is None:
                    return 0
                tex_index[ref] = len(atlas)
                atlas.append(t)
            return tex_index[ref]

        N = len(palette)
        self.kind = np.zeros(N, np.int32)
        self.boxes = np.zeros((N, 6, 6), np.float32)
        self.nbox = np.zeros(N, np.int32)
        self.tex = np.zeros((N, 3), np.int32)
        self.tint = np.ones((N, 3), np.float32)
        self.emit = np.zeros(N, np.float32)
        self.solid = np.zeros(N, np.bool_)
        for i, (name, pr) in enumerate(palette):
            n = name.replace("minecraft:", "")
            if n in ("air", "cave_air", "void_air", "barrier", "light"):
                continue
            self.solid[i] = True
            kind, boxes = shape_for(name, pr)
            self.kind[i] = kind
            for k, b in enumerate(boxes[:6]):
                self.boxes[i, k] = b
            self.nbox[i] = len(boxes[:6])
            model = A.block_model(n, pr)
            t = A.model_textures(model) if model else {}
            top = _pick(t, ["top", "end", "up", "all", "texture", "cross", "plant", "flowerbed", "pane", "wall", "lantern", "particle"])
            side = _pick(t, ["side", "all", "texture", "cross", "plant", "flowerbed", "pane", "wall", "front", "north", "lantern", "particle"])
            bot = _pick(t, ["bottom", "end", "down", "all", "texture", "cross", "plant", "particle"])
            if n == "water":
                top = side = bot = "block/water_still"
            if "lantern" in n and "sea" not in n and "jack" not in n:
                top = side = bot = "block/glowstone"
            if n == "pointed_dripstone":
                top = side = bot = "block/pointed_dripstone_down_tip"
            if n == "grass_block":
                side = "block/grass_block_side"
            if pr.get("axis") in ("x", "z") and "end" in t:
                top, side = t.get("side"), t.get("end")
            self.tex[i] = (tid(top), tid(side), tid(bot))
            if n in GRASS_TINTED:
                self.tint[i] = np.array(GRASS_TINT) / 255
            elif n in FOLIAGE_TINTED:
                self.tint[i] = np.array(FOLIAGE_TINT) / 255
            elif n in FIXED_TINT:
                self.tint[i] = np.array(FIXED_TINT[n]) / 255
            elif n == "water":
                self.tint[i] = np.array(WATER_TINT) / 255
            if any(e in n for e in EMISSIVE) or (n.startswith("cave_vines") and pr.get("berries") == "true"):
                self.emit[i] = 1.0
        self.grass_top_only = tex_index.get("block/grass_block_side", -1)
        self.atlas = np.stack(atlas)

    def _run(self, arr, org, dirs, W, H, sun, fog):
        img = _render(arr, org, dirs, self.kind, self.boxes, self.nbox, self.tex, self.tint, self.emit, self.solid,
                      self.atlas, np.array(sun, np.float32), np.float32(fog), self.grass_top_only)
        return (np.clip(img.reshape(H, W, 3), 0, 1) * 255).astype(np.uint8)

    def ortho(self, arr, target, yaw=35, pitch=30, width=1200, height=800, ppb=6, sun=(-0.45, 0.8, -0.35), aa=2):
        W, H = width * aa, height * aa
        yr, pr = math.radians(yaw), math.radians(pitch)
        d = np.array([math.sin(yr) * math.cos(pr), -math.sin(pr), math.cos(yr) * math.cos(pr)], np.float32)
        right = np.cross(d, [0, 1, 0]).astype(np.float32)
        right /= np.linalg.norm(right)
        up = np.cross(right, d)
        u = (np.arange(W) - W / 2) / (ppb * aa)
        v = (H / 2 - np.arange(H)) / (ppb * aa)
        uu, vv = np.meshgrid(u, v)
        org = (np.array(target, np.float32)[None] - d[None] * 2000 + uu.reshape(-1, 1) * right[None]
               - vv.reshape(-1, 1) * 0 + vv.reshape(-1, 1) * up[None]).astype(np.float32)
        dirs = np.broadcast_to(d, org.shape).astype(np.float32).copy()
        img = self._run(arr, org, dirs, W, H, _norm(sun), 0.0)
        return _down(img, aa)

    def persp(self, arr, eye, look, fov=70, width=1200, height=800, sun=(-0.45, 0.8, -0.35), aa=2, fog=0.004):
        W, H = width * aa, height * aa
        f = np.array(look, np.float32) - np.array(eye, np.float32)
        f /= np.linalg.norm(f)
        right = np.cross(f, [0, 1, 0]).astype(np.float32)
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        tanf = math.tan(math.radians(fov) / 2)
        u = (np.arange(W) - W / 2) / (W / 2) * tanf
        v = (H / 2 - np.arange(H)) / (W / 2) * tanf
        uu, vv = np.meshgrid(u, v)
        dirs = f[None] + uu.reshape(-1, 1) * right[None] + vv.reshape(-1, 1) * up[None]
        dirs = (dirs / np.linalg.norm(dirs, axis=1, keepdims=True)).astype(np.float32)
        org = np.broadcast_to(np.array(eye, np.float32), dirs.shape).astype(np.float32).copy()
        img = self._run(arr, org, dirs, W, H, _norm(sun), fog)
        return _down(img, aa)


def _norm(v):
    v = np.array(v, np.float32)
    return v / np.linalg.norm(v)


def _down(img, aa):
    if aa == 1:
        return img
    H, W, _ = img.shape
    return img.reshape(H // aa, aa, W // aa, aa, 3).mean(axis=(1, 3)).astype(np.uint8)


@njit(cache=True, fastmath=True)
def _sample(atlas, t, u, v):
    iu = min(15, max(0, int(u * 16)))
    iv = min(15, max(0, int(v * 16)))
    return atlas[t, iv, iu]


@njit(cache=True, fastmath=True)
def _box_hit(ox, oy, oz, dx, dy, dz, b0, b1, b2, b3, b4, b5, tmin, tmax):
    """Ray vs box in cell-local coordinates; returns (t, axis, sign) or t<0."""
    t0, t1 = tmin, tmax
    ax = -1
    sg = 0.0
    lo = (b0, b1, b2)
    hi = (b3, b4, b5)
    o = (ox, oy, oz)
    d = (dx, dy, dz)
    for a in range(3):
        if abs(d[a]) < 1e-9:
            if o[a] < lo[a] or o[a] > hi[a]:
                return -1.0, -1, 0.0
        else:
            ta = (lo[a] - o[a]) / d[a]
            tb = (hi[a] - o[a]) / d[a]
            s = -1.0
            if ta > tb:
                ta, tb = tb, ta
                s = 1.0
            if ta > t0:
                t0 = ta
                ax = a
                sg = s
            if tb < t1:
                t1 = tb
            if t0 > t1:
                return -1.0, -1, 0.0
    if ax == -1:
        ax = 1
        sg = 1.0
    return t0, ax, sg


@njit(cache=True, fastmath=True)
def _trace(arr, px, py, pz, dx, dy, dz, kind, boxes, nbox, tex, tint, emit, solid, atlas, shadow, grass_side):
    """Returns (hit, r, g, b, t, nx, ny, nz, emissive, transmit)."""
    Y, Z, X = arr.shape
    # clip to grid
    tmin, tmax = 0.0, 1e9
    o = (px, py, pz)
    d = (dx, dy, dz)
    hi = (X, Y, Z)
    for a in range(3):
        if abs(d[a]) < 1e-9:
            if o[a] < 0 or o[a] >= hi[a]:
                return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
        else:
            ta = (0 - o[a]) / d[a]
            tb = (hi[a] - o[a]) / d[a]
            if ta > tb:
                ta, tb = tb, ta
            tmin = max(tmin, ta)
            tmax = min(tmax, tb)
    if tmin > tmax:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
    t = tmin + 1e-4
    ix = min(X - 1, max(0, int(math.floor(px + dx * t))))
    iy = min(Y - 1, max(0, int(math.floor(py + dy * t))))
    iz = min(Z - 1, max(0, int(math.floor(pz + dz * t))))
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    sz = 1 if dz > 0 else -1
    tdx = abs(1.0 / dx) if abs(dx) > 1e-9 else 1e30
    tdy = abs(1.0 / dy) if abs(dy) > 1e-9 else 1e30
    tdz = abs(1.0 / dz) if abs(dz) > 1e-9 else 1e30
    tmx = ((ix + (1 if sx > 0 else 0)) - px) / dx if abs(dx) > 1e-9 else 1e30
    tmy = ((iy + (1 if sy > 0 else 0)) - py) / dy if abs(dy) > 1e-9 else 1e30
    tmz = ((iz + (1 if sz > 0 else 0)) - pz) / dz if abs(dz) > 1e-9 else 1e30
    acc_r, acc_g, acc_b, trans = 0.0, 0.0, 0.0, 1.0
    tcell = t
    for _ in range(4000):
        b = arr[iy, iz, ix]
        tnext = min(tmx, tmy, tmz)
        if b != 0 and solid[b]:
            k = kind[b]
            lx, ly, lz = px - ix, py - iy, pz - iz
            if k == 2:  # crossed planes
                best = -1.0
                bu = 0.0
                bv = 0.0
                den1 = dx - dz
                if abs(den1) > 1e-9:
                    tt = (lz - lx) / den1
                    if tcell - 1e-6 <= tt <= tnext + 1e-6:
                        hx, hy = lx + dx * tt, ly + dy * tt
                        if 0.15 <= hx <= 0.85 and 0 <= hy <= 1:
                            c = _sample(atlas, tex[b, 1], hx, 1 - hy)
                            if c[3] > 0.5:
                                best, bu, bv = tt, hx, hy
                den2 = dx + dz
                if abs(den2) > 1e-9:
                    tt = (1 - lx - lz) / den2
                    if tcell - 1e-6 <= tt <= tnext + 1e-6 and (best < 0 or tt < best):
                        hx, hy = lx + dx * tt, ly + dy * tt
                        if 0.15 <= hx <= 0.85 and 0 <= hy <= 1:
                            c = _sample(atlas, tex[b, 1], hx, 1 - hy)
                            if c[3] > 0.5:
                                best, bu, bv = tt, hx, hy
                if best >= 0:
                    if shadow:
                        return True, 0.0, 0.0, 0.0, best, 0.0, 0.0, 0.0, 0.0, trans * 0.5
                    c = _sample(atlas, tex[b, 1], bu, 1 - bv)
                    return (True, acc_r + trans * c[0] * tint[b, 0], acc_g + trans * c[1] * tint[b, 1],
                            acc_b + trans * c[2] * tint[b, 2], best, 0.0, 1.0, 0.0, emit[b], trans)
            else:
                nb = nbox[b] if nbox[b] > 0 else 1
                best = -1.0
                bax = 0
                bsg = 0.0
                for q in range(nb):
                    if nbox[b] > 0:
                        bb = boxes[b, q]
                        b0, b1, b2, b3, b4, b5 = bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]
                    else:
                        b0, b1, b2, b3, b4, b5 = 0.0, 0.0, 0.0, 1.0, 1.0, 1.0
                    th, a_, s_ = _box_hit(lx, ly, lz, dx, dy, dz, b0, b1, b2, b3, b4, b5, tcell - 1e-5, tnext + 1e-5)
                    if th >= 0 and (best < 0 or th < best):
                        hx, hy, hz = lx + dx * th, ly + dy * th, lz + dz * th
                        if a_ == 1:
                            ti, u, v = (tex[b, 0] if s_ > 0 else tex[b, 2]), hx, hz
                        elif a_ == 0:
                            ti, u, v = tex[b, 1], hz, 1 - hy
                        else:
                            ti, u, v = tex[b, 1], hx, 1 - hy
                        u = u - math.floor(u)
                        v = v - math.floor(v) if v < 1 else 0.999
                        c = _sample(atlas, ti, u, v)
                        if k == 1 or k == 4:
                            if c[3] < 0.5:
                                continue
                        best, bax, bsg = th, a_, s_
                        cr, cg, cb = c[0], c[1], c[2]
                        if not (ti == grass_side and a_ != 1):
                            cr, cg, cb = cr * tint[b, 0], cg * tint[b, 1], cb * tint[b, 2]
                if best >= 0:
                    nx = sg_axis(bax, 0, bsg)
                    ny = sg_axis(bax, 1, bsg)
                    nz = sg_axis(bax, 2, bsg)
                    if k == 3:  # blended glass / water
                        a = 0.6 if tint[b, 2] < 0.9 or tint[b, 0] < 0.9 else 0.45
                        if shadow:
                            trans *= 0.7
                        else:
                            acc_r += trans * a * cr
                            acc_g += trans * a * cg
                            acc_b += trans * a * cb
                            trans *= (1 - a)
                            if trans < 0.05:
                                return True, acc_r, acc_g, acc_b, best, nx, ny, nz, 0.0, trans
                        # skip through same-type neighbours quickly
                    else:
                        if shadow:
                            return True, 0.0, 0.0, 0.0, best, nx, ny, nz, 0.0, 0.0
                        return (True, acc_r + trans * cr, acc_g + trans * cg, acc_b + trans * cb, best,
                                nx, ny, nz, emit[b], trans)
        # advance
        if tmx < tmy and tmx < tmz:
            ix += sx
            tcell = tmx
            tmx += tdx
        elif tmy < tmz:
            iy += sy
            tcell = tmy
            tmy += tdy
        else:
            iz += sz
            tcell = tmz
            tmz += tdz
        if ix < 0 or iy < 0 or iz < 0 or ix >= X or iy >= Y or iz >= Z or tcell > tmax:
            break
    return False, acc_r, acc_g, acc_b, 0.0, 0.0, 0.0, 0.0, 0.0, trans


@njit(cache=True)
def sg_axis(ax, a, s):
    return s if ax == a else 0.0


@njit(parallel=True, cache=True, fastmath=True)
def _render(arr, org, dirs, kind, boxes, nbox, tex, tint, emit, solid, atlas, sun, fog, grass_side):
    n = org.shape[0]
    out = np.zeros((n, 3), np.float32)
    for i in prange(n):
        px, py, pz = org[i, 0], org[i, 1], org[i, 2]
        dx, dy, dz = dirs[i, 0], dirs[i, 1], dirs[i, 2]
        sky_t = max(0.0, min(1.0, 0.5 + dy * 0.8))
        sr, sg_, sb = 0.62 + 0.2 * sky_t, 0.78 + 0.12 * sky_t, 0.98
        hit, r, g, b, t, nx, ny, nz, em, trans = _trace(arr, px, py, pz, dx, dy, dz, kind, boxes, nbox, tex, tint,
                                                        emit, solid, atlas, False, grass_side)
        if not hit:
            out[i, 0] = r + trans * sr
            out[i, 1] = g + trans * sg_
            out[i, 2] = b + trans * sb
            continue
        hx, hy, hz = px + dx * t, py + dy * t, pz + dz * t
        ndl = nx * sun[0] + ny * sun[1] + nz * sun[2]
        light = 1.0
        if ndl > 0:
            sh = _trace(arr, hx + nx * 2e-3 + sun[0] * 1e-3, hy + ny * 2e-3 + sun[1] * 1e-3, hz + nz * 2e-3 + sun[2] * 1e-3,
                        sun[0], sun[1], sun[2], kind, boxes, nbox, tex, tint, emit, solid, atlas, True, grass_side)
            if sh[0]:
                light = sh[9]
        else:
            light = 0.0
        face = 0.55 + 0.25 * ny + 0.08 * abs(nx)
        shade = 0.42 * face + 0.75 * max(ndl, 0.0) * light
        shade = max(shade, em)
        r, g, b = r * shade, g * shade, b * shade
        if fog > 0:
            f = 1 - math.exp(-fog * t)
            r, g, b = r * (1 - f) + sr * f, g * (1 - f) + sg_ * f, b * (1 - f) + sb * f
        out[i, 0], out[i, 1], out[i, 2] = r, g, b
    return out
