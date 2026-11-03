"""Core voxel library: palette, world array, block-state helpers, texture mixes, noise.

Coordinates: arr[y, z, x]; x, z in 0..S-1 with the island centre at (C, C).
"""
import math
import re

import numpy as np
from scipy.ndimage import zoom

S, C, Y = 385, 192, 256
arr = np.zeros((Y, S, S), np.uint16)

# ------------------------------------------------------------------ palette
names, props, _idx = ["minecraft:air"], [{}], {"minecraft:air": 0}


def B(s):
    """Block id for 'name[k=v,...]' (namespace optional). Cached."""
    if ":" not in s.split("[")[0]:
        s = "minecraft:" + s
    if s in _idx:
        return _idx[s]
    m = re.match(r"([^\[]+)(?:\[(.*)\])?$", s)
    pr = {}
    if m.group(2):
        for kv in m.group(2).split(","):
            k, v = kv.split("=")
            pr[k] = v
    _idx[s] = len(names)
    names.append(m.group(1))
    props.append(pr)
    return _idx[s]


AIR = 0
DV = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
CW = {"north": "east", "east": "south", "south": "west", "west": "north"}
CCW = {v: k for k, v in CW.items()}


def vdir(dx, dz):
    """Facing name of the dominant horizontal component of (dx, dz)."""
    if abs(dx) >= abs(dz):
        return "east" if dx > 0 else "west"
    return "south" if dz > 0 else "north"


def stairs(mat, facing, half="bottom"):
    return B(f"{mat}_stairs[facing={facing},half={half},shape=straight,waterlogged=false]")


def slab(mat, typ="bottom"):
    return B(f"{mat}_slab[type={typ},waterlogged=false]")


def wallb(mat):
    return B(f"{mat}_wall[up=true,east=none,west=none,north=none,south=none,waterlogged=false]")


def leaves(n):
    return B(f"{n}_leaves[distance=1,persistent=true,waterlogged=false]")


def log(n, axis="y"):
    return B(f"{n}[axis={axis}]")


# ------------------------------------------------------------------ hashing / mixes
_U = np.uint64


def hash3(x, y, z, seed=0):
    x, y, z = (np.asarray(v).astype(np.int64).astype(_U) for v in (x, y, z))
    h = x * _U(374761393) + y * _U(668265263) + z * _U(2147483647) + _U(seed * 2654435761 % (1 << 32))
    h = (h ^ (h >> _U(13))) * _U(1274126177)
    h = h ^ (h >> _U(16))
    return (h & _U(0x7FFFFFFF)).astype(np.int64)


# ---- coherent value noise (periodic lattice table, trilinear, smoothstep)
_TAB = np.random.default_rng(20261102).uniform(-1, 1, (64, 64, 64)).astype(np.float32)


def noise3(x, y, z, scale=8.0, seed=0, stretch=(1.0, 1.0, 1.0)):
    """Smooth 3D value noise in [-1, 1]; stretch > 1 elongates features along that axis."""
    x = np.asarray(x, np.float32) / (scale * stretch[0]) + seed * 17.13
    y = np.asarray(y, np.float32) / (scale * stretch[1]) + seed * 5.71
    z = np.asarray(z, np.float32) / (scale * stretch[2]) + seed * 11.37
    x0, y0, z0 = np.floor(x), np.floor(y), np.floor(z)
    tx, ty, tz = x - x0, y - y0, z - z0
    tx, ty, tz = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty), tz * tz * (3 - 2 * tz)
    xi, yi, zi = x0.astype(np.int64) & 63, y0.astype(np.int64) & 63, z0.astype(np.int64) & 63
    xj, yj, zj = (xi + 1) & 63, (yi + 1) & 63, (zi + 1) & 63
    c000, c100 = _TAB[xi, yi, zi], _TAB[xj, yi, zi]
    c010, c110 = _TAB[xi, yj, zi], _TAB[xj, yj, zi]
    c001, c101 = _TAB[xi, yi, zj], _TAB[xj, yi, zj]
    c011, c111 = _TAB[xi, yj, zj], _TAB[xj, yj, zj]
    a = c000 + (c100 - c000) * tx
    b = c010 + (c110 - c010) * tx
    c = c001 + (c101 - c001) * tx
    d = c011 + (c111 - c011) * tx
    e = a + (b - a) * ty
    f = c + (d - c) * ty
    return e + (f - e) * tz


def fbm3(x, y, z, scale=8.0, seed=0, octaves=3, stretch=(1.0, 1.0, 1.0)):
    out, amp, tot, sc = 0.0, 1.0, 0.0, scale
    for o in range(octaves):
        out = out + noise3(x, y, z, sc, seed + o * 31, stretch) * amp
        tot += amp
        amp *= 0.5
        sc /= 2.0
    return out / tot


def _ncdf(v, sigma):
    """Map roughly gaussian noise to a ~uniform [0, 1] value (keeps mix proportions)."""
    from scipy.special import erf
    return 0.5 * (1 + erf(np.asarray(v) / (sigma * 1.41421356)))


class Mix:
    """Weighted texture mix chosen by coherent noise: contiguous patches (scale = patch size in blocks)
    plus a faint high-frequency term so borders are ragged rather than blobby."""

    def __init__(self, pairs, seed=0, patch=0, scale=None, stretch=(1.0, 1.0, 1.0)):
        self.ids = np.array([B(n) if isinstance(n, str) else n for n, _ in pairs], np.uint16)
        w = np.cumsum([w for _, w in pairs]).astype(float)
        self.cum = w / w[-1]
        self.seed = seed
        self.scale = float(scale if scale is not None else (patch * 2.5 if patch else 4.0))
        self.stretch = stretch

    def value(self, x, y, z):
        n = fbm3(x, y, z, self.scale, self.seed, octaves=2, stretch=self.stretch)
        n = n + noise3(x, y, z, 1.6, self.seed + 99) * 0.12
        return _ncdf(n, 0.30)

    def at(self, x, y, z):
        k = np.searchsorted(self.cum, self.value(x, y, z), side="right")
        return self.ids[np.minimum(k, len(self.ids) - 1)]


class StairMix:
    """Stairs whose material varies in coherent patches (same shape)."""

    def __init__(self, mats, seed=0, scale=5.0):
        self.mats, self.seed, self.scale = mats, seed, scale

    def at(self, x, y, z, facing, half="bottom"):
        v = float(_ncdf(fbm3(x, y, z, self.scale, self.seed, 2), 0.30))
        k = min(int(v * len(self.mats)), len(self.mats) - 1)
        return stairs(self.mats[k], facing, half)


# ------------------------------------------------------------------ writing
def inb(x, y, z):
    return 0 <= x < S and 0 <= y < Y and 0 <= z < S


def put(x, y, z, b, only_air=False, replace=None):
    if not inb(x, y, z):
        return
    if only_air and arr[y, z, x] != 0:
        return
    if replace is not None and arr[y, z, x] not in replace:
        return
    arr[y, z, x] = b if not isinstance(b, Mix) else b.at(x, y, z)


def get(x, y, z):
    return int(arr[y, z, x]) if inb(x, y, z) else 0


def fill(x0, y0, z0, x1, y1, z1, b, only_air=False, mask=None):
    """Inclusive box fill; b is an id, a Mix, or a callable(xs, ys, zs) -> ids."""
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    za, zb = sorted((z0, z1))
    xa, ya, za = max(xa, 0), max(ya, 0), max(za, 0)
    xb, yb, zb = min(xb, S - 1), min(yb, Y - 1), min(zb, S - 1)
    if xa > xb or ya > yb or za > zb:
        return
    sub = arr[ya:yb + 1, za:zb + 1, xa:xb + 1]
    m = np.ones(sub.shape, bool) if mask is None else mask
    if only_air:
        m = m & (sub == 0)
    if isinstance(b, (int, np.integer)):
        sub[m] = b
        return
    ys, zs, xs = np.mgrid[ya:yb + 1, za:zb + 1, xa:xb + 1]
    ids = b.at(xs, ys, zs) if isinstance(b, Mix) else b(xs, ys, zs)
    sub[m] = ids[m]


def ball(cx, cy, cz, rx, ry, rz, b, only_air=True, rough=0.0, seed=0, hollow_prob=0.0):
    x0, x1 = int(math.floor(cx - rx)), int(math.ceil(cx + rx))
    y0, y1 = int(math.floor(cy - ry)), int(math.ceil(cy + ry))
    z0, z1 = int(math.floor(cz - rz)), int(math.ceil(cz + rz))
    xa, ya, za = max(x0, 0), max(y0, 0), max(z0, 0)
    xb, yb, zb = min(x1, S - 1), min(y1, Y - 1), min(z1, S - 1)
    if xa > xb or ya > yb or za > zb:
        return
    ys, zs, xs = np.mgrid[ya:yb + 1, za:zb + 1, xa:xb + 1]
    d = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 + ((zs - cz) / rz) ** 2
    if rough:
        d = d + (hash3(xs, ys, zs, seed) % 1000 / 1000.0 - 0.5) * rough
    m = d <= 1
    if hollow_prob:
        m &= (hash3(xs, ys, zs, seed + 3) % 1000) >= hollow_prob * 1000
    sub = arr[ya:yb + 1, za:zb + 1, xa:xb + 1]
    if only_air:
        m &= sub == 0
    ids = b.at(xs, ys, zs) if isinstance(b, Mix) else np.full(xs.shape, b, np.uint16)
    sub[m] = ids[m]


def disc(cx, y, cz, r, b, only_air=False):
    if not 0 <= y < Y:
        return
    rr = int(math.ceil(r))
    xa, xb = max(int(cx) - rr, 0), min(int(cx) + rr, S - 1)
    za, zb = max(int(cz) - rr, 0), min(int(cz) + rr, S - 1)
    zs, xs = np.mgrid[za:zb + 1, xa:xb + 1]
    m = (xs - cx) ** 2 + (zs - cz) ** 2 <= r * r + 0.25
    sub = arr[y, za:zb + 1, xa:xb + 1]
    if only_air:
        m &= sub == 0
    ids = b.at(xs, np.full(xs.shape, y), zs) if isinstance(b, Mix) else np.full(xs.shape, b, np.uint16)
    sub[m] = ids[m]


# ------------------------------------------------------------------ noise
def fbm(seed, scales=((96, 1.0), (40, 0.5), (16, 0.25), (7, 0.12)), size=S):
    """Smooth 2D noise in roughly [-1, 1] (non-symmetric, seeded)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((size, size))
    tot = 0
    for sc, w in scales:
        n = size // sc + 3
        g = rng.uniform(-1, 1, (n, n))
        up = zoom(g, sc, order=3)[:size, :size]
        out += up * w
        tot += w
    return out / tot


def grid():
    ax = np.arange(S)
    X, Z = np.meshgrid(ax, ax)
    return X, Z
