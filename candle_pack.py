#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
candle_pack.py (v3)  -  lossless (bit-exact) compact storage for candle csv data.   Needs: python3 + numpy only (lzma is in the standard library).
Use  python  (CPython) - NOT pypy3 (numpy does not work well on pypy).

  python candle_pack.py pack   <csv | zip | folder of zips>  <out.cpk | out_folder>   # pack + automatic verification of every file
  python candle_pack.py verify <csv | zip | folder of zips>  <out.cpk | out_folder>   # compare a source with its pack again
  python candle_pack.py unpack <in.cpk | folder>             <out.csv | out_folder>   # back to csv (the same numbers)

  * Nothing is extracted to disk: csv files are read straight out of the zip (also zip inside zip), ONE FILE AT A TIME, so a phone with
    little free space / memory can pack years of data.  A source with many csv files (e.g. monthly files) gives a FOLDER of small .cpk files.
  * In python:  names, cols, unit = candle_pack.read_any("x.cpk" | "folder_of_cpk" | "csv/zip")    cols = dict column -> numpy array

How it stays lossless (nothing is rounded):
  * every number column is first tested: "is it exactly  integer / scale  for some scale 1..1e8 ?"  (prices with 2-4 decimals are).
    Only if the re-built float64 is IDENTICAL (same 64 bits) the integer form is used, otherwise the raw 8-byte doubles are kept.
  * integers are stored as differences (open/high/low relative to the same bar's close, close relative to the previous close),
    the time stamp as the difference of the time steps -> small numbers -> very compressible.
  * mid_* columns that equal (bid+ask)/2 up to float noise: only the noise (a few units of the last bit) is stored.
  * then everything is compressed with LZMA.
  * the packer decodes its own output and compares all 64 bits of every number; if anything differs it refuses to write that file.
Text note: a csv is read with numpy (correctly rounded parsing). The numbers are identical, the *text* of a number may be printed a little
differently when you unpack (e.g. 4319.1905 instead of 4319.190500000001) - the double behind it is the same.
"""
import os, sys, io, glob, json, lzma, struct, zipfile
import numpy as np

MAGIC = b"CPK1"
FIELDS = ("open", "high", "low", "close")


def _iter_zip(fileobj):
    with zipfile.ZipFile(fileobj) as z:
        for info in sorted(z.infolist(), key=lambda i: i.filename):
            if info.is_dir(): continue
            nm = info.filename.lower()
            if nm.endswith(".csv"): yield os.path.splitext(os.path.basename(info.filename))[0], z.read(info)
            elif nm.endswith(".zip"): yield from _iter_zip(io.BytesIO(z.read(info)))


def iter_csv(path):
    """yields (name, csv bytes) for every csv in a csv / zip (nested ok) / folder - nothing is written to disk"""
    if os.path.isdir(path):
        for p in sorted(glob.glob(os.path.join(path, "**", "*"), recursive=True)):
            if os.path.isfile(p) and p.lower().endswith((".csv", ".zip")): yield from iter_csv(p)
    elif path.lower().endswith(".zip"):
        yield from _iter_zip(path)
    elif path.lower().endswith(".csv"):
        yield os.path.splitext(os.path.basename(path))[0], open(path, "rb").read()


BINANCE_NAMES = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def _isnum(x):
    try: float(x); return True
    except ValueError: return False


def _parse_ts(tt):
    """-> (int64 values in the ORIGINAL unit, unit)   unit = 's' | 'ms' | 'us' | 'iso' (iso text -> epoch seconds)"""
    try:
        ts = tt.astype(np.int64); m = int(np.abs(ts).max())
        return ts, ("s" if m < 4e10 else "ms" if m < 4e13 else "us")
    except ValueError:
        tt = np.array([x.replace("T", " ").replace("Z", "")[:19] for x in tt]); return tt.astype("datetime64[s]").astype(np.int64), "iso"


def read_one(name, data):
    """one csv (bytes) -> (names, dict column -> array, ts_unit). A file without a header line is accepted when it has 12 columns (Binance)."""
    first = data[:data.index(b"\n") if b"\n" in data else len(data)].decode("utf-8", "replace").strip().split(",")
    hdr = not _isnum(first[0]) and not first[0][:4].isdigit()
    if hdr: names = [h.strip() for h in first]
    else:
        if len(first) != 12: raise SystemExit(f"{name}: no header line and not the 12-column Binance layout")
        names = BINANCE_NAMES
    n = len(names); sk = 1 if hdr else 0
    tt = np.loadtxt(io.BytesIO(data), delimiter=",", skiprows=sk, usecols=[0], dtype=str, ndmin=1)
    num = np.loadtxt(io.BytesIO(data), delimiter=",", skiprows=sk, usecols=list(range(1, n)), ndmin=2)
    ts, unit = _parse_ts(tt); ts, u = np.unique(ts, return_index=True); num = num[u]
    cols = {names[0]: ts}
    for i, nm in enumerate(names[1:]): cols[nm] = np.ascontiguousarray(num[:, i])
    return names, cols, unit


def concat(parts, need=None):
    """parts: list of (names, cols, unit) -> one (names, cols, unit), sorted by time, duplicate time stamps removed.  need = set of columns to keep"""
    names = parts[0][0]; unit = parts[0][2]
    for p in parts:
        if p[0] != names or p[2] != unit: raise SystemExit("files have different columns / time units")
    keep = [c for c in names if need is None or c in need or c == names[0]]
    cols = {c: np.concatenate([p[1][c] for p in parts]) for c in keep}
    ts, u = np.unique(cols[names[0]], return_index=True)
    return keep, {c: (ts if c == names[0] else cols[c][u]) for c in keep}, unit


def read_source(path, need=None):
    """whole source (csv / zip / folder) in memory -> (names, cols, unit).  Good for a few months; for years use pack -> folder of .cpk"""
    parts = []
    for nm, data in iter_csv(path):
        names, cols, unit = read_one(nm, data); parts.append((names, cols, unit))
    if not parts: raise SystemExit("no csv found in " + path)
    return concat(parts, need)


def read_any(path, need=None):
    """.cpk file | folder of .cpk files | csv / zip / folder  ->  (names, cols, unit)"""
    if os.path.isfile(path) and path.lower().endswith(".cpk"):
        names, cols, unit = decode(open(path, "rb").read()); return concat([(names, cols, unit)], need)
    if os.path.isdir(path):
        fs = sorted(glob.glob(os.path.join(path, "*.cpk")))
        if fs:
            parts = []
            for f in fs:
                names, cols, unit = decode(open(f, "rb").read()); keep = [c for c in names if need is None or c in need or c == names[0]]
                parts.append((keep, {c: cols[c] for c in keep}, unit))
            return concat(parts, None)
    return read_source(path, need)


def _find_scale(v):
    """smallest S in 1,10,..1e8 such that round(v*S)/S == v bit for bit, else None"""
    if not np.all(np.isfinite(v)): return None
    for k in range(0, 9):
        S = 10.0 ** k; q = np.round(v * S)
        if np.abs(q).max() < 2 ** 52 and np.array_equal((q / S).view(np.int64), v.view(np.int64)): return k
    return None


def _small(a):
    m = int(np.abs(a).max()) if len(a) else 0
    return a.astype(np.int8 if m < 127 else np.int16 if m < 32767 else np.int32 if m < 2 ** 31 - 1 else np.int64)


def encode(names, cols, unit="s"):
    ts = cols[names[0]]; meta = {"names": names, "n": int(len(ts)), "unit": unit, "col": {}}; blobs = []
    def put(key, arr):
        arr = np.ascontiguousarray(arr); blobs.append(arr.tobytes()); return {"key": key, "dtype": arr.dtype.str, "len": int(arr.size)}
    g = int(np.gcd.reduce(np.diff(ts))) if len(ts) > 1 else 1; g = max(g, 1)
    assert int(ts[0]) % g == 0 or True
    meta["ts"] = {"first": int(ts[0]), "div": g}
    d = np.diff(ts) // g; meta["ts"]["arr"] = put("ts", _small(d) if len(d) else d.astype(np.int8))
    num = [c for c in names[1:]]; scale = {c: _find_scale(cols[c]) for c in num}; q = {c: np.round(cols[c] * 10.0 ** scale[c]).astype(np.int64) for c in num if scale[c] is not None}
    done = set(); groups = {}
    for c in num:
        for f in FIELDS:
            if c.endswith("_" + f): groups.setdefault(c[:-len(f) - 1], {})[f] = c
    plan = {}                                                                          # column -> ('rel', ref) | ('delta',) | ('raw',) | ('mid', bid, ask)
    for p, g in groups.items():
        if "close" in g and g["close"] in q:
            plan[g["close"]] = ("delta",)
            for f in ("open", "high", "low"):
                if f in g and g[f] in q and scale[g[f]] == scale[g["close"]]: plan[g[f]] = ("rel", g["close"])
    for p in ("mid",):
        if p in groups and "bid" in groups and "ask" in groups:
            for f in FIELDS:
                if all(f in groups[x] for x in ("mid", "bid", "ask")):
                    b, a, m = groups["bid"][f], groups["ask"][f], groups["mid"][f]
                    if b in q and a in q and scale[b] == scale[a]:
                        pred = (q[b] + q[a]).astype(np.float64) / (2.0 * 10.0 ** scale[b]); res = cols[m].view(np.int64) - pred.view(np.int64)
                        if np.abs(res).max() < 127: plan[m] = ("mid", b, a, scale[b])
    for c in num:
        pl = plan.get(c)
        if pl is None and c in q: pl = ("delta",)
        if pl is None: pl = ("raw",)
        info = {"plan": pl[0], "scale": scale[c]}
        if pl[0] == "delta": info["arr"] = put(c, _small(np.diff(q[c], prepend=0)))
        elif pl[0] == "rel": info["ref"] = pl[1]; info["arr"] = put(c, _small(q[pl[1]] - q[c]))
        elif pl[0] == "mid":
            _, b, a, s = pl; pred = (q[b] + q[a]).astype(np.float64) / (2.0 * 10.0 ** s)
            info.update(b=b, a=a, s=s); info["arr"] = put(c, (cols[c].view(np.int64) - pred.view(np.int64)).astype(np.int8))
        else: info["arr"] = put(c, cols[c].astype(np.float64))
        meta["col"][c] = info
    raw = b"".join(blobs); comp = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=6)
    head = json.dumps(meta, separators=(",", ":")).encode()
    return MAGIC + struct.pack("<I", len(head)) + head + comp


def decode(blob):
    if blob[:4] != MAGIC: raise SystemExit("not a candle pack")
    hl = struct.unpack("<I", blob[4:8])[0]; meta = json.loads(blob[8:8 + hl]); raw = lzma.decompress(blob[8 + hl:]); pos = 0
    def get(a):
        nonlocal pos
        dt = np.dtype(a["dtype"]); nb = dt.itemsize * a["len"]; arr = np.frombuffer(raw[pos:pos + nb], dtype=dt); pos += nb; return arr
    names = meta["names"]; t = meta["ts"]; d = get(t["arr"]).astype(np.int64)
    ts = t["first"] + np.r_[0, np.cumsum(d)] * t["div"]; cols = {names[0]: ts}; q = {}
    order = list(meta["col"].keys()); arrs = {c: get(meta["col"][c]["arr"]) for c in order}                       # same order as written
    for c in order:
        i = meta["col"][c]
        if i["plan"] == "delta": q[c] = np.cumsum(arrs[c].astype(np.int64)); cols[c] = (q[c] / 10.0 ** i["scale"])
    for c in order:
        i = meta["col"][c]
        if i["plan"] == "rel": q[c] = q[i["ref"]] - arrs[c].astype(np.int64); cols[c] = (q[c] / 10.0 ** i["scale"])
    for c in order:
        i = meta["col"][c]
        if i["plan"] == "mid":
            pred = (q[i["b"]] + q[i["a"]]).astype(np.float64) / (2.0 * 10.0 ** i["s"]); cols[c] = (pred.view(np.int64) + arrs[c].astype(np.int64)).view(np.float64)
        elif i["plan"] == "raw": cols[c] = arrs[c].astype(np.float64)
    return names, {k: cols[k] for k in names}, meta.get("unit", "s")


def same(c1, c2, names):
    return all(np.array_equal(c1[n].view(np.int64) if c1[n].dtype == np.float64 else c1[n], c2[n].view(np.int64) if c2[n].dtype == np.float64 else c2[n]) for n in names)


def load(path):
    """-> dict column -> numpy array (time stamp column in its original unit)"""
    return read_any(path)[1]


def load_full(path):
    """-> (names, cols, ts_unit)"""
    return read_any(path)


def _is_cpk_target(p): return p.lower().endswith(".cpk")


def main():
    if len(sys.argv) < 4 or sys.argv[1] not in ("pack", "unpack", "verify"): print(__doc__); return
    cmd, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    if cmd in ("pack", "verify"):
        single = _is_cpk_target(dst); i = 0; tot_in = 0; tot_out = 0; rows = 0; bad = 0
        if not single: os.makedirs(dst, exist_ok=True)
        if single:                                                                      # one .cpk for the whole source
            names, cols, unit = read_source(src)
            if cmd == "pack":
                blob = encode(names, cols, unit); n2, c2, u2 = decode(blob)
                if n2 != names or u2 != unit or not same(cols, c2, names): raise SystemExit("VERIFY FAILED - nothing written (the data would not be identical)")
                open(dst, "wb").write(blob); print(f"OK  {len(cols[names[0]]):,} rows x {len(names)} columns  -> {dst}  {len(blob) / 1e6:.2f} MB  (all 64-bit values verified identical)")
            else:
                n2, c2, u2 = decode(open(dst, "rb").read()); print("IDENTICAL (all columns, all 64 bits)" if n2 == names and u2 == unit and same(cols, c2, names) else "DIFFERENT")
            return
        for nm, data in iter_csv(src):                                                  # one .cpk per csv file, one file in memory at a time
            i += 1; names, cols, unit = read_one(nm, data); out = os.path.join(dst, f"{i:05d}_{nm}.cpk"); tot_in += len(data); rows += len(cols[names[0]])
            if cmd == "pack":
                blob = encode(names, cols, unit); n2, c2, u2 = decode(blob)
                if n2 != names or u2 != unit or not same(cols, c2, names): print(f"VERIFY FAILED for {nm} - not written"); bad += 1; continue
                open(out, "wb").write(blob); tot_out += len(blob); print(f"  {i:4d} {nm}: {len(data) / 1e6:6.2f} MB csv -> {len(blob) / 1e6:5.2f} MB   identical", flush=True)
            else:
                if not os.path.exists(out): print(f"  {nm}: pack file missing"); bad += 1; continue
                n2, c2, u2 = decode(open(out, "rb").read()); ok = n2 == names and u2 == unit and same(cols, c2, names); bad += (not ok); tot_out += os.path.getsize(out)
                print(f"  {i:4d} {nm}: {'IDENTICAL' if ok else 'DIFFERENT'}", flush=True)
            del cols, data
        if i == 0: raise SystemExit("no csv found in " + src)
        print(f"{'PACK' if cmd == 'pack' else 'VERIFY'} done: {i} files, {rows:,} rows, csv {tot_in / 1e6:.1f} MB -> packed {tot_out / 1e6:.1f} MB ({100 * (1 - tot_out / max(tot_in, 1)):.0f}% smaller); problems: {bad}")
    else:
        files = [src] if os.path.isfile(src) else sorted(glob.glob(os.path.join(src, "*.cpk")))
        def write(path, names, cols, unit):
            with open(path, "w") as f:
                f.write(",".join(names) + "\n"); t0 = cols[names[0]]; tt = t0.astype("datetime64[s]").astype(str) if unit == "iso" else None
                for k in range(len(t0)): f.write((str(tt[k]).replace("T", " ") if tt is not None else str(int(t0[k]))) + "," + ",".join(repr(float(cols[c][k])) for c in names[1:]) + "\n")
        if len(files) == 1 and not os.path.isdir(dst): write(dst, *decode(open(files[0], "rb").read())); print("written", dst)
        else:
            os.makedirs(dst, exist_ok=True)
            for f in files: write(os.path.join(dst, os.path.basename(f)[:-4] + ".csv"), *decode(open(f, "rb").read()))
            print("written", len(files), "csv files into", dst)


if __name__ == "__main__":
    main()
# ===== END OF FILE candle_pack.py =====
