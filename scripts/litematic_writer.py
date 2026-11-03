"""Fast .litematic writer for huge schematics (numpy only, no dependencies).

arr: numpy array shaped (Y, Z, X) of palette indices (index 0 must be air).
palette: list of (block_name, {property: value}) tuples.
"""
import gzip
import struct
import time

import numpy as np


def _n(s):
    b = s.encode("utf-8")
    return struct.pack(">H", len(b)) + b


def _int(name, v):
    return b"\x03" + _n(name) + struct.pack(">i", v)


def _long(name, v):
    return b"\x04" + _n(name) + struct.pack(">q", v)


def _str(name, v):
    return b"\x08" + _n(name) + _n(v)


def _comp(name, body):
    return b"\x0a" + _n(name) + body + b"\x00"


def _list(name, tid, items):
    return b"\x09" + _n(name) + struct.pack(">bi", tid, len(items)) + b"".join(items)


def _larr(name, data, count):
    return b"\x0c" + _n(name) + struct.pack(">i", count) + data


def _xyz(name, x, y, z):
    return _comp(name, _int("x", x) + _int("y", y) + _int("z", z))


def pack_states(vals, bits):
    """Pack palette indices tightly (Litematica style, values may span longs)."""
    n = len(vals)
    chunk = 64 * 65536
    shifts = np.arange(bits, dtype=np.uint8)
    out = []
    for s in range(0, n, chunk):
        part = vals[s:s + chunk].astype(np.uint16)
        flat = ((part[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
        pad = (-len(flat)) % 64
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, np.uint8)])
        out.append(np.packbits(flat, bitorder="little").view("<u8"))
    longs = np.concatenate(out)
    return longs.view(np.int64).astype(">i8").tobytes(), len(longs)


def write_litematic(path, arr, palette, name, author="Claude", description="",
                    data_version=3465, region="Main"):
    Y, Z, X = arr.shape
    bits = max(2, (len(palette) - 1).bit_length())
    t0 = time.time()
    data, count = pack_states(arr.reshape(-1), bits)
    total = int(np.count_nonzero(arr))
    now = int(time.time() * 1000)

    pal_items = []
    for bname, props in palette:
        body = _str("Name", bname)
        if props:
            body += _comp("Properties", b"".join(_str(k, v) for k, v in props.items()))
        pal_items.append(body + b"\x00")

    reg = (
        _xyz("Position", 0, 0, 0)
        + _xyz("Size", X, Y, Z)
        + _list("BlockStatePalette", 10, pal_items)
        + _larr("BlockStates", data, count)
        + _list("TileEntities", 10, [])
        + _list("Entities", 10, [])
        + _list("PendingBlockTicks", 10, [])
        + _list("PendingFluidTicks", 10, [])
    )
    meta = (
        _xyz("EnclosingSize", X, Y, Z)
        + _str("Author", author)
        + _str("Description", description)
        + _str("Name", name)
        + _int("RegionCount", 1)
        + _long("TimeCreated", now)
        + _long("TimeModified", now)
        + _int("TotalBlocks", total)
        + _int("TotalVolume", X * Y * Z)
    )
    root = (
        _int("MinecraftDataVersion", data_version)
        + _int("Version", 6)
        + _comp("Metadata", meta)
        + _comp("Regions", _comp(region, reg))
    )
    raw = b"\x0a\x00\x00" + root + b"\x00"
    with gzip.open(path, "wb", compresslevel=6) as f:
        f.write(raw)
    return {"bits": bits, "blocks": total, "volume": X * Y * Z, "palette": len(palette),
            "seconds": round(time.time() - t0, 1)}
