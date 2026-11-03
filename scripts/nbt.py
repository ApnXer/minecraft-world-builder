"""Minimal typed NBT reader/writer. Every value is a Tag(type_id, value).

Compound value: dict name -> Tag. List value: (element_type, [Tag, ...]).
Arrays (7/11/12) are numpy arrays (big-endian dtypes).
"""
import gzip
import struct

import numpy as np

END, BYTE, SHORT, INT, LONG, FLOAT, DOUBLE, BARR, STR, LIST, COMP, IARR, LARR = range(13)
_FMT = {BYTE: ">b", SHORT: ">h", INT: ">i", LONG: ">q", FLOAT: ">f", DOUBLE: ">d"}
_ARR = {BARR: ">i1", IARR: ">i4", LARR: ">i8"}


class Tag:
    __slots__ = ("t", "v")

    def __init__(self, t, v):
        self.t, self.v = t, v

    def __getitem__(self, k):
        return self.v[k]

    def __setitem__(self, k, val):
        self.v[k] = val

    def __contains__(self, k):
        return k in self.v

    def __repr__(self):
        return f"Tag({self.t}, {self.v!r})"


# ---- constructors
def b(v): return Tag(BYTE, int(v))
def s(v): return Tag(SHORT, int(v))
def i(v): return Tag(INT, int(v))
def l(v): return Tag(LONG, int(v))
def f(v): return Tag(FLOAT, float(v))
def d(v): return Tag(DOUBLE, float(v))
def st(v): return Tag(STR, str(v))
def c(**kw): return Tag(COMP, dict(kw))
def cd(dct): return Tag(COMP, dict(dct))
def lst(t, items): return Tag(LIST, (t if items else END, list(items)))
def la(arr): return Tag(LARR, np.asarray(arr, dtype=">i8"))
def ia(arr): return Tag(IARR, np.asarray(arr, dtype=">i4"))


# ---- reading
class _R:
    def __init__(self, data):
        self.d, self.p = data, 0

    def take(self, n):
        out = self.d[self.p:self.p + n]
        self.p += n
        return out

    def unpack(self, fmt):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.take(size))[0]

    def string(self):
        n = self.unpack(">H")
        return self.take(n).decode("utf-8", errors="surrogatepass")

    def payload(self, t):
        if t in _FMT:
            return Tag(t, self.unpack(_FMT[t]))
        if t in _ARR:
            n = self.unpack(">i")
            dt = np.dtype(_ARR[t])
            return Tag(t, np.frombuffer(self.take(n * dt.itemsize), dtype=dt).copy())
        if t == STR:
            return Tag(STR, self.string())
        if t == LIST:
            et = self.unpack(">b")
            n = self.unpack(">i")
            return Tag(LIST, (et, [self.payload(et) for _ in range(n)]))
        if t == COMP:
            out = {}
            while True:
                tt = self.unpack(">b")
                if tt == END:
                    return Tag(COMP, out)
                name = self.string()
                out[name] = self.payload(tt)
        raise ValueError(f"bad tag type {t}")


def loads(data):
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    r = _R(data)
    t = r.unpack(">b")
    r.string()
    return r.payload(t)


def load(path):
    with open(path, "rb") as fh:
        return loads(fh.read())


# ---- writing
def _ns(name):
    raw = name.encode("utf-8")
    return struct.pack(">H", len(raw)) + raw


def _payload(tag, out):
    t, v = tag.t, tag.v
    if t in _FMT:
        out.append(struct.pack(_FMT[t], v))
    elif t in _ARR:
        arr = np.asarray(v, dtype=_ARR[t])
        out.append(struct.pack(">i", len(arr)))
        out.append(arr.tobytes())
    elif t == STR:
        out.append(_ns(v))
    elif t == LIST:
        et, items = v
        out.append(struct.pack(">bi", et if items else END, len(items)))
        for it in items:
            _payload(it, out)
    elif t == COMP:
        for name, child in v.items():
            out.append(struct.pack(">b", child.t))
            out.append(_ns(name))
            _payload(child, out)
        out.append(b"\x00")
    else:
        raise ValueError(f"bad tag type {t}")


def dumps(tag, name=""):
    out = [struct.pack(">b", tag.t), _ns(name)]
    _payload(tag, out)
    return b"".join(out)


def save(path, tag, gz=True):
    raw = dumps(tag)
    with open(path, "wb") as fh:
        fh.write(gzip.compress(raw) if gz else raw)


def to_py(tag):
    """Plain-python view (for printing / inspection)."""
    t, v = tag.t, tag.v
    if t == COMP:
        return {k: to_py(x) for k, x in v.items()}
    if t == LIST:
        return [to_py(x) for x in v[1]]
    if t in _ARR:
        return f"<array {len(v)}>"
    return v
