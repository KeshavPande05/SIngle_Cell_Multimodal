#!/usr/bin/env python3
"""
Open Problems - Multimodal Single-Cell Integration  --  DATASET INSPECTOR
=========================================================================
Checks EVERYTHING in the competition data folder:

  1. File inventory (names + sizes)
  2. HDF5 layout of the 6 matrix files (groups, axes, dtypes, shapes)
     + value statistics (min/max/mean/sparsity/negatives) via row-sampling
  3. metadata.csv  -> donor x day x technology x cell_type breakdown
     + split verification against the official split description
  4. Organizer patch file test_cite_inputs_day_2_donor_27678.h5
     (cell overlap + feature-set differences vs the main files)
  5. Consistency checks (cell-id overlap, target counts, feature alignment)
  6. evaluation_ids.csv (chunked scan) - row_id range, unique cells/genes,
     per-technology split, 15%-column sampling check
  7. sample_submission.csv format + alignment with evaluation_ids

IMPORTANT -- HDF5 READER NOTES:
  The competition files are pandas HDFStore files compressed with the
  blosc filter. Plain h5py CANNOT decompress them unless plugin filters
  are registered -> 'pip install hdf5plugin'. This script tries in order:
    (a) h5py + hdf5plugin           (if the hdf5plugin package is installed)
    (b) h5py without plugins        (works for uncompressed / mock files)
    (c) PyTables (pip install tables, has blosc built in - pandas' read_hdf
        uses it, so if you can run pd.read_hdf you already have it)
  The active reader is shown per file ("read via: h5py | pytables").

CONFIRMED REAL-DATA NUMBERS (verified against the 2022-11 dataset):
  train_cite_inputs   70,988 x 22,050   RNA lib-size norm + log1p
  train_cite_targets  70,988 x    140   ADT, dsb-normalised (range ~ -22 .. +45)
  test_cite_inputs    48,663 x 22,050  (public: donor 27678 d2/3/4 + private: all d7)
  train_multi_inputs 105,942 x 228,942 ATAC TF-IDF (97.7% zeros)
  train_multi_targets105,942 x  23,418 RNA (82% zeros)
  test_multi_inputs   55,935 x 228,942 (public: donor 27678 d2/3/7 + private: all d10)
  evaluation_ids.csv  65,744,180 rows  (cite: all 140 targets x 48,663 cells;
                                        multi: 30% of cells x ~15% of 23,418 genes)
  metadata.csv        281,528 cells;  CITE test cell_type VISIBLE, multiome test hidden
  patch file          7,016 x 22,085  (35 extra genes vs main files -> align by gene id!)

Usage:
  python check_data.py            # auto-locates data (env OPM_DATA_DIR, /kaggle/input,
                                  #  ~/.cache/kagglehub cache, else kagglehub download)
  python check_data.py --data_dir /path/to/competition/files
  python check_data.py --full-scan       # stream stats over ALL rows (slow)
  python check_data.py --make-mock ./mock_data   # tiny synthetic copy (for testing)
  python check_data.py --data_dir ./mock_data

kagglehub note: competition files sit DIRECTLY in
  ~/.cache/kagglehub/competitions/open-problems-multimodal/
(no extra nesting). If nothing is found locally and the kagglehub package is
installed, kagglehub.competition_download('open-problems-multimodal') is used
as automatic fallback (needs kaggle.json credentials + accepted comp rules).

Dependencies: numpy, pandas, h5py   (optional: tables, hdf5plugin, kagglehub)
"""

import argparse
import os
import re
import sys
import numpy as np
import pandas as pd

try:
    import h5py
except ImportError:
    sys.exit("h5py is required:  pip install h5py")

try:
    import hdf5plugin  # noqa: F401 - registers blosc/lz4/zstd/... filters for h5py
    HAVE_HDF5PLUGIN = True
except ImportError:
    HAVE_HDF5PLUGIN = False

try:
    import importlib.util
    HAVE_TABLES = importlib.util.find_spec("tables") is not None
except Exception:
    HAVE_TABLES = False

# --------------------------------------------------------------------------- #
# Expected structure (from the official Dataset Description + verified run)
# --------------------------------------------------------------------------- #
TRAIN_DONORS = {13176, 31800, 32606}
PUBLIC_DONOR = 27678
ALL_DONORS = TRAIN_DONORS | {PUBLIC_DONOR}

EXPECT = {
    "cite": {
        "inputs":  "train_cite_inputs.h5",   # RNA, 22,050 genes
        "targets": "train_cite_targets.h5",  # ADT, 140 proteins (dsb)
        "train_days": {2, 3, 4},
        "public_days": {2, 3, 4},
        "private_day": 7,
        "n_targets": 140,
        "test_celltype_hidden": False,       # CITE test cell types are VISIBLE
    },
    "multi": {
        "inputs":  "train_multi_inputs.h5",  # ATAC, 228,942 peaks
        "targets": "train_multi_targets.h5", # RNA, 23,418 genes
        "train_days": {2, 3, 4, 7},
        "public_days": {2, 3, 7},
        "private_day": 10,
        "test_celltype_hidden": True,        # multiome test cell types are hidden
    },
}

H5_FILES = {
    "cite":   {"train_inputs": "train_cite_inputs.h5",
               "train_targets": "train_cite_targets.h5",
               "test_inputs": "test_cite_inputs.h5"},
    "multi":  {"train_inputs": "train_multi_inputs.h5",
               "train_targets": "train_multi_targets.h5",
               "test_inputs": "test_multi_inputs.h5"},
}

PATCH_H5 = "test_cite_inputs_day_2_donor_27678.h5"
PATCH_META = "metadata_cite_day_2_donor_27678.csv"

BAR = "=" * 78


def hdr(title):
    print(f"\n{BAR}\n{title}\n{BAR}")


def human(nbytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def dec(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def dec_list(arr):
    return [dec(x) for x in arr]


# --------------------------------------------------------------------------- #
# Data-folder resolution
# --------------------------------------------------------------------------- #
def find_data_root(p, depth=0):
    """Accept p (or a single unambiguous subfolder of it) if metadata.csv is
    there. Handles both flat and nested download layouts."""
    if depth > 3 or not os.path.isdir(p):
        return None
    if os.path.exists(os.path.join(p, "metadata.csv")):
        return p
    subs = [s for s in sorted(os.listdir(p)) if os.path.isdir(os.path.join(p, s))]
    found = [f for f in (find_data_root(os.path.join(p, s), depth + 1) for s in subs) if f]
    return found[0] if len(found) == 1 else None


def resolve_data_dir():
    """Try the usual locations; fall back to kagglehub (locates cache or downloads)."""
    home = os.path.expanduser("~")
    candidates = [
        os.environ.get("OPM_DATA_DIR", ""),
        "/kaggle/input/open-problems-multimodal",
        os.path.join(home, ".cache", "kagglehub", "competitions", "open-problems-multimodal"),
        ".",
    ]
    for c in candidates:
        if not c:
            continue
        root = find_data_root(c)
        if root:
            return root
    try:
        import kagglehub
        print("kagglehub: locating / downloading 'open-problems-multimodal' ...")
        path = kagglehub.competition_download("open-problems-multimodal")
        root = find_data_root(path)
        if root:
            return root
        print(f"kagglehub returned a path without metadata.csv: {path}")
    except ImportError:
        print("kagglehub not installed (pip install kagglehub) -- skipping auto-download.")
    except Exception as e:
        print(f"kagglehub failed: {type(e).__name__}: {e}")
        print("(needs kaggle.json credentials and accepted rules at")
        print(" https://www.kaggle.com/competitions/open-problems-multimodal/rules)")
    sys.exit("Pass the folder explicitly:  python check_data.py --data_dir /path/to/files")


# --------------------------------------------------------------------------- #
# Unified HDF5 reader:  h5py -> PyTables fallback (blosc-compressed files)
# --------------------------------------------------------------------------- #
class H5Backend:
    """Reader for pandas HDFStore files.

    The competition files are blosc-compressed. h5py can only read them if
    filter plugins are registered (hdf5plugin package); PyTables has blosc
    built in, so we probe h5py first and fall back to PyTables on the
    'missing plugin' OSError.
    """

    def __init__(self, path):
        self.path = path
        self.backend = None
        self._f = None
        try:
            self._f = h5py.File(path, "r")
            self._probe_h5py()
            self.backend = "h5py"
            return
        except OSError as e:
            if self._f is not None:
                try:
                    self._f.close()
                except Exception:
                    pass
                self._f = None
            if "plugin" not in str(e).lower():
                raise                      # real corruption -> propagate
        # ---- PyTables fallback (blosc built in) ----
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import tables
            self._tables = tables
            self._f = tables.open_file(path, "r")
        self._probe_pytables()
        self.backend = "pytables"

    # ---- probes: force the filter error NOW, not mid-report ----
    def _probe_h5py(self):
        key = list(self._f.keys())[0]
        g = self._f[key]
        _ = g["axis0"][:1]
        blocks = sorted([k for k in g.keys() if re.fullmatch(r"block\d+_values", k)])
        if blocks:
            _ = g[blocks[0]][0:1, 0:1]

    def _probe_pytables(self):
        key = self.keys()[0]
        _ = self.get_array(key, "axis0")[:1]
        blocks = sorted([k for k in self.subkeys(key) if re.fullmatch(r"block\d+_values", k)])
        if blocks:
            _ = self.get_array(key, blocks[0])[0:1]

    # ---- common API ----
    def keys(self):
        if self.backend == "h5py":
            return list(self._f.keys())
        return [n for n, _ in self._f.root._v_children.items()
                if isinstance(self._f.root._v_children[n], self._tables.Group)]

    def subkeys(self, key):
        if self.backend == "h5py":
            return list(self._f[key].keys())
        return list(self._f.root._v_children[key]._v_children)

    def get_array(self, key, name):
        """Return a node supporting [:] / row-slicing."""
        if self.backend == "h5py":
            return self._f[key][name]
        return self._f.get_node("/" + key + "/" + name)

    def read_full(self, key, name):
        return self.get_array(key, name)[:]

    def shape_dtype(self, key, name):
        node = self.get_array(key, name)
        return tuple(node.shape), str(node.dtype)

    def slice_rows(self, key, name, start, stop):
        return np.asarray(self.get_array(key, name)[start:stop])

    def storage(self, key, name):
        """Storage details (chunks + compression filter) for a dataset.
        Reading metadata does NOT require the filter plugin -- only data reads do."""
        out = {"chunks": None, "filter": "none (uncompressed)",
               "level": None, "shuffle": None}
        try:
            if self.backend == "h5py":
                d = self._f[key][name]
                out["chunks"] = d.chunks
                pl = d.id.get_create_plist()
                for i in range(pl.get_nfilters()):
                    code, flags, cdvals, fname = pl.get_filter(i)
                    nm = fname.decode() if isinstance(fname, bytes) else str(fname)
                    out["filter"] = f"{nm} (id {code})"
                    if code == 32001 and len(cdvals) >= 6:   # blosc cd_values
                        out["level"] = cdvals[4]
                        out["shuffle"] = {0: False, 1: True, 2: "bitshuffle"}.get(cdvals[5])
            else:
                node = self.get_array(key, name)
                cs = getattr(node, "chunkshape", None)
                out["chunks"] = tuple(int(c) for c in cs) if cs else None
                fl = getattr(node, "filters", None)
                if fl is not None and getattr(fl, "complib", None):
                    out["filter"] = str(fl.complib)
                    out["level"] = fl.complevel
                    out["shuffle"] = bool(fl.shuffle)
        except Exception as e:
            out["filter"] = f"unknown ({e})"
        return out

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def read_h5_axes(path):
    """Return layout info without loading the big matrix."""
    out = {"path": path, "ok": False}
    try:
        b = H5Backend(path)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        if "plugin" in out["error"].lower():
            out["error"] += ("  (blosc-compressed file: pip install hdf5plugin "
                             "OR pip install tables)")
        return out
    with b:
        out["backend"] = b.backend
        out["top_keys"] = b.keys()
        key = out["top_keys"][0]
        sub = b.subkeys(key)
        out["group"], out["subkeys"] = key, sub

        blocks = sorted([k for k in sub if re.fullmatch(r"block\d+_values", k)])
        items = sorted([k for k in sub if re.fullmatch(r"block\d+_items", k)])
        if not blocks:
            out["error"] = "no block*__values datasets found"
            return out

        # axis0 = columns (features), axis1 = rows (cells)  [pandas HDFStore layout]
        feat_ax = b.read_full(key, "axis0")
        cell_ax = b.read_full(key, "axis1")
        shapes = [tuple(int(x) for x in b.shape_dtype(key, blk)[0]) for blk in blocks]
        dtypes = [b.shape_dtype(key, blk)[1] for blk in blocks]

        n_rows = shapes[0][0]
        n_cols = sum(s[1] for s in shapes)      # multi-block files: total width
        if len(feat_ax) == n_cols and len(cell_ax) == n_rows:
            out["features"], out["cells"] = dec_list(feat_ax), dec_list(cell_ax)
            out["orientation"] = "rows=cells, cols=features"
        elif len(feat_ax) == n_rows and len(cell_ax) == n_cols:
            out["features"], out["cells"] = dec_list(cell_ax), dec_list(feat_ax)
            out["orientation"] = "TRANSPOSED (rows=features)"
        else:
            out["features"], out["cells"] = dec_list(feat_ax), dec_list(cell_ax)
            out["orientation"] = (f"UNMATCHED axis0={len(feat_ax)} "
                                  f"axis1={len(cell_ax)} block={shapes[0]}")

        out["ok"] = True
        out["total_shape"] = (int(n_rows), int(n_cols))
        out["block_names"], out["block_shapes"], out["block_dtypes"] = blocks, shapes, dtypes
        out["n_items"] = [len(b.read_full(key, i)) for i in items] if items else []
        try:
            out["storage"] = b.storage(key, blocks[0])
        except Exception:
            pass
    return out


def h5_value_stats(path, n_sample_rows=4096, full=False, peek=5):
    """Stream value statistics. By default samples contiguous row-slabs
    (fast); --full-scan reads every row."""
    with H5Backend(path) as b:
        key = b.keys()[0]
        blocks = sorted([k for k in b.subkeys(key) if re.fullmatch(r"block\d+_values", k)])
        shape, dtype = b.shape_dtype(key, blocks[0])
        n_rows = shape[0]

        def read_slab(s, e):
            """Row-slab across ALL blocks (multi-block files split columns by dtype)."""
            parts = [np.asarray(b.slice_rows(key, blk, s, e)) for blk in blocks]
            return parts[0] if len(parts) == 1 else np.hstack(parts)

        def slabs():
            if full or n_rows <= n_sample_rows:
                for s in range(0, n_rows, 4096):
                    yield s, read_slab(s, s + 4096)
            else:
                n_slabs, width = 8, max(256, n_sample_rows // 8)
                starts = np.linspace(0, n_rows - width, n_slabs).astype(int)
                for s in starts:
                    yield s, read_slab(s, s + width)

        cnt = ssum = nnz = neg = nan = 0
        vmin, vmax = np.inf, -np.inf
        for _, arr in slabs():
            a = np.asarray(arr)
            cnt += a.size
            ssum += float(a.sum(dtype=np.float64))
            nnz += int(np.count_nonzero(a))
            neg += int((a < 0).sum())
            nan += int(np.isnan(a).sum())
            if a.size:
                vmin = min(vmin, float(np.nanmin(a)))
                vmax = max(vmax, float(np.nanmax(a)))

        sampled = ("ALL rows" if (full or n_rows <= n_sample_rows)
                   else f"{n_sample_rows} sampled rows")
        blabel = "+".join(n.replace("_values", "") for n in blocks)
        print(f"     values[{blabel}] via {b.backend}  dtype={dtype}  scanned={sampled}")
        print(f"     min={vmin:.4g}  max={vmax:.4g}  mean={ssum/max(cnt,1):.4g}  "
              f"sparsity={1 - nnz/max(cnt,1):.2%}  negatives={neg}  NaN={nan}")
        if peek:
            a = read_slab(0, peek)[:, :peek]
            print(f"     top-left {a.shape[0]}x{a.shape[1]} block:\n"
                  + "".join("       " + "  ".join(f"{v:8.3f}" for v in row) + "\n"
                             for row in a))


def inspect_h5_file(label, path, n_sample, full, do_stats=True):
    hdr(f"HDF5  {label}:  {os.path.basename(path)}")
    if not os.path.exists(path):
        print("  !! file not found -- skipped")
        return None
    print(f"  size: {human(os.path.getsize(path))}")
    info = read_h5_axes(path)
    if not info["ok"]:
        print("  !! could not parse:", info.get("error"))
        return info
    print(f"  read via       : {info['backend']}"
          + ("  (hdf5plugin active)" if info["backend"] == "h5py" and HAVE_HDF5PLUGIN
             else "  (blosc fallback)" if info["backend"] == "pytables" else ""))
    print(f"  top-level keys : {info['top_keys']}")
    print(f"  group '{info['group']}' contains: {info['subkeys']}")
    print(f"  orientation    : {info['orientation']}")
    st = info.get("storage", {})
    if st:
        extra = ""
        if st.get("level") is not None:
            extra = f"  level={st['level']}  shuffle={st['shuffle']}"
        print(f"  storage        : chunks={st.get('chunks')}  filter={st.get('filter')}{extra}")
    shp = info.get("total_shape", info["block_shapes"][0])
    print(f"  MATRIX SHAPE   : {shp}  "
          f"({shp[0]:,} cells x {shp[1]:,} features)  "
          f"dtype={info['block_dtypes'][0]}")
    if len(info["block_names"]) > 1:
        det = " | ".join(f"{n.replace('_values', '')}{s} {t}" for n, s, t in
                         zip(info["block_names"], info["block_shapes"], info["block_dtypes"]))
        print(f"  blocks         : {len(info['block_names'])} column blocks: {det}")
    nf, nc = len(info["features"]), len(info["cells"])
    print(f"  axes           : axis0={nf:,} features, axis1={nc:,} cells")
    print(f"  first features : {info['features'][:3]}")
    print(f"  last  features : {info['features'][-3:]}")
    print(f"  first cells    : {info['cells'][:3]}")
    uniq_f = len(set(info["features"])) == nf
    uniq_c = len(set(info["cells"])) == nc
    print(f"  unique ids     : features={'OK' if uniq_f else 'DUPLICATES!'}  "
          f"cells={'OK' if uniq_c else 'DUPLICATES!'}")
    if do_stats:
        h5_value_stats(path, n_sample_rows=n_sample, full=full)
    return info


# --------------------------------------------------------------------------- #
# Split verification
# --------------------------------------------------------------------------- #
def crosstab(df, rows, cols, title):
    print(f"\n  {title}")
    ct = pd.crosstab(df[rows], df[cols], margins=True)
    print(ct.to_string().replace("\n", "\n  "))


def verify_split(meta, h5info, tech):
    hdr(f"SPLIT VERIFICATION  --  {tech.upper()}  (vs official Dataset Description)")
    exp = EXPECT[tech]
    ok = lambda b: "OK" if b else "MISMATCH!!"

    tr_ids = set(h5info[tech]["train_inputs"]["cells"])
    te_ids = set(h5info[tech]["test_inputs"]["cells"])
    tech_name = {"cite": "citeseq", "multi": "multiome"}[tech]
    m = meta[meta.technology == tech_name].copy()
    m["split"] = np.where(m.cell_id.isin(tr_ids), "train",
                  np.where(m.cell_id.isin(te_ids), "test", "not-in-h5!"))

    for s in ["train", "test"]:
        d = m[m.split == s]
        crosstab(d, "donor", "day", f"{s} cells:  {len(d):,} cells, donor x day")

    tr, te = m[m.split == "train"], m[m.split == "test"]
    print(f"\n  checks:")
    print(f"   - train donors == {{13176,31800,32606}}                : "
          f"{ok(set(tr.donor) == TRAIN_DONORS)}  (found {sorted(set(tr.donor))})")
    print(f"   - train days == expected {sorted(exp['train_days'])} : "
          f"{ok(set(tr.day) <= exp['train_days'])}  (found {sorted(set(tr.day))})")
    pub = te[(te.donor == PUBLIC_DONOR) & (te.day != exp["private_day"])]
    print(f"   - public part  (donor {PUBLIC_DONOR}) days == {sorted(exp['public_days'])} : "
          f"{ok(set(pub.day) == exp['public_days'])}  (found {sorted(set(pub.day))})")
    priv = te[te.day == exp["private_day"]]
    print(f"   - private part (day {exp['private_day']} only), donors == all four : "
          f"{ok(set(priv.donor) == ALL_DONORS)}  (found {sorted(set(priv.donor))})")
    print(f"   - no cell in both train & test                          : "
          f"{ok(len(tr_ids & te_ids) == 0)}")
    print(f"   - every {tech} cell in metadata appears in an h5        : "
          f"{ok((m.split != 'not-in-h5!').all())}")
    if (m.split == "not-in-h5!").any():
        print(f"     !! {sum(m.split=='not-in-h5!'):,} metadata cells missing from h5 files")

    # cell_type visibility: multiome test hidden; CITE test visible (verified on real data)
    if exp["test_celltype_hidden"]:
        hidden_ok = set(te.cell_type) == {"hidden"}
        print(f"   - test cell_type == hidden (as expected for multiome)   : "
              f"{ok(hidden_ok)}  (found {sorted(set(te.cell_type))[:5]})")
    else:
        n_vis = int((te.cell_type != "hidden").sum())
        top = te.cell_type.value_counts().head(3).to_dict()
        print(f"   - test cell_type VISIBLE (as expected for cite)         : "
              f"{'OK' if n_vis == len(te) else 'PARTIAL'}  ({n_vis:,}/{len(te):,} labelled; top: {top})")
        print(f"     -> cell_type is a usable FEATURE for CITE inference")

    ts = h5info[tech]["train_targets"]
    n_tg = ts.get("total_shape", ts.get("block_shapes", [[None, None]])[0])[1]
    if "n_targets" in exp and n_tg is not None:
        print(f"   - n target columns == {exp['n_targets']}                        : "
              f"{ok(n_tg == exp['n_targets'])}  (found {n_tg:,})")
    return m


# --------------------------------------------------------------------------- #
# Organizer patch file (re-processed CITE day-2 donor-27678 batch)
# --------------------------------------------------------------------------- #
def inspect_patch(d, h5info, meta):
    hdr(f"PATCH FILE  --  {PATCH_H5}  (organizer re-processed batch)")
    path = os.path.join(d, PATCH_H5)
    if not os.path.exists(path):
        print("  not present in this folder -- skipped")
        return None
    print(f"  size: {human(os.path.getsize(path))}")
    info = read_h5_axes(path)
    if not info["ok"]:
        print("  !! could not parse:", info.get("error"))
        return None
    print(f"  read via       : {info['backend']}")
    shp = info.get("total_shape", info["block_shapes"][0])
    print(f"  MATRIX SHAPE   : {shp}  "
          f"({shp[0]:,} cells x {shp[1]:,} features)")
    if len(info.get("block_names", [])) > 1:
        det = " | ".join(f"{n.replace('_values', '')}{s} {t}" for n, s, t in
                         zip(info["block_names"], info["block_shapes"], info["block_dtypes"]))
        print(f"  blocks         : {len(info['block_names'])} column blocks: {det}")
    st = info.get("storage", {})
    if st:
        extra = f"  level={st['level']}  shuffle={st['shuffle']}" if st.get("level") is not None else ""
        print(f"  storage        : chunks={st.get('chunks')}  filter={st.get('filter')}{extra}")

    main_cells = set(h5info["cite"]["test_inputs"].get("cells", []))
    patch_cells = set(info["cells"])
    inter = patch_cells & main_cells
    print(f"  cells          : {len(patch_cells):,} total; "
          f"{len(inter):,} already in test_cite_inputs; "
          f"{len(patch_cells - main_cells):,} NEW (not in main test file)")

    main_feats = set(h5info["cite"]["train_inputs"].get("features", []))
    pfeats = set(info["features"])
    extra, missing = pfeats - main_feats, main_feats - pfeats
    print(f"  features       : {len(pfeats):,} vs {len(main_feats):,} in main files  "
          f"(+{len(extra)} extra, -{len(missing)} missing)")
    if extra:
        print(f"     extra examples : {sorted(extra)[:4]}")
    if missing:
        print(f"     missing examples: {sorted(missing)[:4]}")

    mp = os.path.join(d, PATCH_META)
    if os.path.exists(mp):
        pm = pd.read_csv(mp, dtype={"cell_id": str})
        donors = sorted(int(x) for x in pm.donor.unique()) if "donor" in pm else "?"
        days = sorted(int(x) for x in pm.day.unique()) if "day" in pm else "?"
        print(f"  {PATCH_META}: {len(pm):,} rows; donors={donors}; days={days}")
        mm = meta.set_index("cell_id")
        got = pm.cell_id.isin(mm.index).sum()
        print(f"     patch cell_ids found in main metadata: {got:,}/{len(pm):,}")

    print("  GUIDANCE: on the real dataset these are brand-NEW cells (not scored in")
    print("            evaluation_ids) -- optional extra INPUT data. Align any use by")
    print("            cell_id AND gene_id, never by column position (feature sets differ).")
    return set(info["cells"])


# --------------------------------------------------------------------------- #
# evaluation_ids.csv / sample_submission.csv  (chunked, memory-safe)
# --------------------------------------------------------------------------- #
def inspect_evaluation_ids(path, cell_sets, gene_sets, patch_cells=None):
    hdr(f"evaluation_ids.csv  ({human(os.path.getsize(path))})")
    n, rmin, rmax = 0, None, None
    cells, genes = set(), set()
    cite_cells, multi_cells = set(), set()
    cite_rows = multi_rows = 0
    for ch in pd.read_csv(path, usecols=["row_id", "cell_id", "gene_id"],
                          dtype={"cell_id": str, "gene_id": str}, chunksize=5_000_000):
        n += len(ch)
        rmin = ch.row_id.min() if rmin is None else min(rmin, ch.row_id.min())
        rmax = ch.row_id.max() if rmax is None else max(rmax, ch.row_id.max())
        in_cite = ch.cell_id.isin(cell_sets["cite_test"])
        in_multi = ch.cell_id.isin(cell_sets["multi_test"])
        cite_rows += int(in_cite.sum()); multi_rows += int(in_multi.sum())
        cite_cells.update(ch.cell_id[in_cite])   # sets: chunk boundaries cannot double-count
        multi_cells.update(ch.cell_id[in_multi])
        cells.update(ch.cell_id.unique()); genes.update(ch.gene_id.unique())
    print(f"  rows            : {n:,}   (row_id {rmin} .. {rmax}, "
          f"{'contiguous OK' if n == rmax - rmin + 1 else 'NON-CONTIGUOUS'})")
    print(f"  unique cells    : {len(cells):,}   unique genes/targets: {len(genes):,}")
    print(f"  cite  eval rows : {cite_rows:,}  ({len(cite_cells):,} cells, "
          f"{cite_rows/max(len(cite_cells),1):.1f} targets/cell -- expect ALL 140)")
    print(f"  multi eval rows : {multi_rows:,}  ({len(multi_cells):,} cells, "
          f"{multi_rows/max(len(multi_cells),1):.1f} targets/cell "
          f"-- expect ~15% of 23,418 = 3,512.7)")
    if patch_cells:
        scored = patch_cells & cells
        print(f"  patch-file cells present in evaluation_ids : {len(scored):,}/{len(patch_cells):,}"
              f"  -> {'SCORED' if scored else 'NOT SCORED (optional extra input data)'}")
    print(f"  eval cells subset of test cells             : "
          f"{'OK' if cells <= (cell_sets['cite_test'] | cell_sets['multi_test']) else 'MISMATCH!!'}")
    cite_genes = {g for g in genes if g in gene_sets["cite"]}
    print(f"  cite  eval genes subset of cite targets     : "
          f"{'OK' if cite_genes and cite_genes <= gene_sets['cite'] else 'MISMATCH!!'}")
    multi_genes = {g for g in genes if g in gene_sets["multi"]}
    print(f"  multi eval genes subset of multi targets    : "
          f"{'OK' if multi_genes and multi_genes <= gene_sets['multi'] else 'MISMATCH!!'}")
    print(f"  head:\n{pd.read_csv(path, nrows=3).to_string(index=False)}")
    return n, rmin, rmax


def inspect_submission(path, eval_info):
    hdr(f"sample_submission.csv  ({human(os.path.getsize(path))})")
    n, head = 0, None
    for ch in pd.read_csv(path, usecols=["row_id", "target"], chunksize=5_000_000):
        if head is None:
            head = ch.head(3)
        n += len(ch)
    print(f"  rows : {n:,}")
    if eval_info:
        n_ev, _, _ = eval_info
        print(f"  aligns with evaluation_ids ({n_ev:,} rows): "
              f"{'OK' if n == n_ev else 'MISMATCH!!'}")
    print(f"  head:\n{head.to_string(index=False)}")


# --------------------------------------------------------------------------- #
# Tiny synthetic mock (identical layout, ~1 MB) for testing the checker
# --------------------------------------------------------------------------- #
def make_mock(out):
    import hashlib
    rng = np.random.default_rng(0)
    os.makedirs(out, exist_ok=True)
    print(f"writing mock dataset to {out}/ ...")
    use_blosc = HAVE_TABLES
    if use_blosc:
        print("  (tables detected: files written as blosc-compressed pandas HDFStore, "
              "exactly like the real dataset)")

    def cid(seed):
        return hashlib.md5(seed.encode()).hexdigest()[:8]

    rows, cells = [], {}
    def add(tag, tech, donor, day, n, hidden=True):
        ids = []
        for i in range(n):
            c = cid(f"{tag}-{tech}-{donor}-{day}-{i}")
            ids.append(c)
            ct = "hidden" if (hidden and tag.startswith("test")) else \
                 rng.choice(["HSC", "MasP", "MkP", "NeuP", "MoP", "EryP", "BP"])
            rows.append((c, donor, day, tech, ct))
        cells[(tag, tech)] = cells.get((tag, tech), []) + ids

    for d in TRAIN_DONORS:
        for day in [2, 3, 4]:
            add("train", "citeseq", d, day, 4)
        for day in [2, 3, 4, 7]:
            add("train", "multiome", d, day, 3)
    for day in [2, 3, 4]:                                   # cite public
        add("test", "citeseq", PUBLIC_DONOR, day, 4, hidden=False)   # CITE test types VISIBLE
    for d in ALL_DONORS:                                    # cite private (day 7)
        add("test", "citeseq", d, 7, 2, hidden=False)
    for day in [2, 3, 7]:                                   # multi public
        add("test", "multiome", PUBLIC_DONOR, day, 3)
    for d in ALL_DONORS:                                    # multi private (day 10)
        add("test", "multiome", d, 10, 2)

    meta = pd.DataFrame(rows, columns=["cell_id", "donor", "day", "technology", "cell_type"])
    meta = meta.sample(frac=1, random_state=0).reset_index(drop=True)
    meta.to_csv(os.path.join(out, "metadata.csv"), index=False)

    genes_cite = [f"{g}_ENSG{i:011d}" for i, g in
                  enumerate(["A1BG", "NANOG", "POU5F1", "GATA1", "GATA2", "SPI1"] * 10)]
    genes_multi = [f"{g}_ENSG{i:011d}" for i, g in
                   enumerate(["A1BG", "NANOG", "KLF1", "MPO", "ITGA2B"] * 8)]
    peaks = [f"chr{c}:{10000*i}-{10000*i+500}" for i, c in
             enumerate((list("123456789") + ["10", "11", "12"]) * 5)]
    adts = ["CD4", "CD8a", "CD11b", "CD14", "CD19", "CD34", "CD38", "CD45",
            "CD90", "CD117", "HLA-A,B,C", "PECAM-1"]

    def matrix(n_cells, n_feat, neg=False):
        X = (rng.random((n_cells, n_feat)) < 0.12) * rng.gamma(2., 1.5, (n_cells, n_feat))
        if neg:  # dsb-normalised targets have negative values
            X = X - rng.normal(1.0, 0.8, X.shape) * (rng.random(X.shape) < 0.4)
        return X.astype(np.float32)

    def write_h5(name, key, cell_ids, feats, X):
        p = os.path.join(out, name)
        if use_blosc:                       # faithful replica of the real files
            df = pd.DataFrame(X, index=pd.Index(cell_ids, name="cell_id"),
                              columns=pd.Index(feats, name="gene_id"))
            df.to_hdf(p, key=key, complib="blosc", complevel=5)
        else:                               # plain h5py layout (fallback for testing)
            with h5py.File(p, "w") as f:
                g = f.create_group(key)
                g.create_dataset("axis0", data=np.array(feats, dtype="S32"))
                g.create_dataset("axis1", data=np.array(cell_ids, dtype="S32"))
                g.create_dataset("block0_items", data=np.array(feats, dtype="S32"))
                g.create_dataset("block0_values", data=X)

    write_h5("train_cite_inputs.h5", "train_cite_inputs",
             cells[("train", "citeseq")], genes_cite,
             matrix(len(cells[("train", "citeseq")]), len(genes_cite)))
    write_h5("train_cite_targets.h5", "train_cite_targets",
             cells[("train", "citeseq")], adts,
             matrix(len(cells[("train", "citeseq")]), len(adts), neg=True))
    write_h5("test_cite_inputs.h5", "test_cite_inputs",
             cells[("test", "citeseq")], genes_cite,
             matrix(len(cells[("test", "citeseq")]), len(genes_cite)))
    write_h5("train_multi_inputs.h5", "train_multi_inputs",
             cells[("train", "multiome")], peaks,
             matrix(len(cells[("train", "multiome")]), len(peaks)))
    write_h5("train_multi_targets.h5", "train_multi_targets",
             cells[("train", "multiome")], genes_multi,
             matrix(len(cells[("train", "multiome")]), len(genes_multi)))
    write_h5("test_multi_inputs.h5", "test_multi_inputs",
             cells[("test", "multiome")], peaks,
             matrix(len(cells[("test", "multiome")]), len(peaks)))

    # mimic the REAL patch file (verified on the actual dataset):
    #  - brand-NEW cell ids, absent from main h5 files and main metadata
    #  - its own metadata csv, NOT in evaluation_ids (unscored)
    #  - slightly LARGER feature set (like 22,085 vs 22,050)
    #  - columns split across MULTIPLE blocks (like the real file)
    patch_cells = [cid(f"reprocessed-{i}") for i in range(3)]
    patch_feats = genes_cite + ["EXTRA1_ENSG99999999999", "EXTRA2_ENSG99999999998"]
    Xp = matrix(len(patch_cells), len(patch_feats))
    with h5py.File(os.path.join(out, PATCH_H5), "w") as f:
        g = f.create_group("test_cite_inputs_day_2_donor_27678")
        g.create_dataset("axis0", data=np.array(patch_feats, dtype="S32"))
        g.create_dataset("axis1", data=np.array(patch_cells, dtype="S32"))
        g.create_dataset("block0_items", data=np.array(patch_feats[:60], dtype="S32"))
        g.create_dataset("block0_values", data=Xp[:, :60].astype(np.float32))
        g.create_dataset("block1_items", data=np.array(patch_feats[60:], dtype="S32"))
        g.create_dataset("block1_values", data=Xp[:, 60:].astype(np.float64))
    pd.DataFrame({"cell_id": patch_cells, "day": 2, "donor": PUBLIC_DONOR,
                  "cell_type": ["HSC", "EryP", "NeuP"],
                  "technology": "citeseq"}).to_csv(os.path.join(out, PATCH_META), index=False)

    # evaluation_ids: cite test = ALL targets; multi test = 15% of columns per row
    rows_ev = []
    rid = 0
    for c in cells[("test", "citeseq")]:
        for g in adts:
            rows_ev.append((rid, c, g)); rid += 1
    for c in cells[("test", "multiome")]:
        k = max(1, int(0.15 * len(genes_multi)))
        for g in rng.choice(genes_multi, k, replace=False):
            rows_ev.append((rid, c, g)); rid += 1
    pd.DataFrame(rows_ev, columns=["row_id", "cell_id", "gene_id"]).to_csv(
        os.path.join(out, "evaluation_ids.csv"), index=False)
    pd.DataFrame({"row_id": [r[0] for r in rows_ev], "target": 0.0}).to_csv(
        os.path.join(out, "sample_submission.csv"), index=False)
    kind = "blosc pandas-HDFStore" if use_blosc else "plain h5py"
    print(f"mock written ({kind}): metadata.csv, 7x h5 (incl. patch), "
          "evaluation_ids.csv, sample_submission.csv")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=None,
                    help="folder with the competition files "
                         "(default: auto-detect env OPM_DATA_DIR / Kaggle / kagglehub cache "
                         "/ kagglehub download)")
    ap.add_argument("--make-mock", metavar="DIR", help="generate a tiny synthetic mock dataset")
    ap.add_argument("--sample-rows", type=int, default=4096,
                    help="rows sampled per h5 for value stats (default 4096)")
    ap.add_argument("--full-scan", action="store_true",
                    help="scan ALL rows of every h5 (slow)")
    ap.add_argument("--no-stats", action="store_true", help="skip value statistics")
    args = ap.parse_args()

    if args.make_mock:
        make_mock(args.make_mock)
        if args.data_dir is None:
            return

    d = args.data_dir if args.data_dir else resolve_data_dir()
    print(f"DATA DIR: {os.path.abspath(d)}")

    hdr("1. FILE INVENTORY")
    if not os.path.isdir(d):
        sys.exit(f"  !! directory not found: {d}")
    files = sorted(os.listdir(d))
    total = 0
    for fn in files:
        p = os.path.join(d, fn)
        if os.path.isfile(p):
            sz = os.path.getsize(p); total += sz
            print(f"  {human(sz):>10}  {fn}")
    print(f"  {'-'*40}\n  {human(total):>10}  TOTAL ({len(files)} files)")

    if not HAVE_HDF5PLUGIN and not HAVE_TABLES:
        print("\n  NOTE: neither hdf5plugin nor tables is installed; h5py cannot read")
        print("  blosc-compressed competition files.  pip install tables  (or hdf5plugin)")

    # ---- h5 files ----
    hdr("2. HDF5 MATRICES")
    h5info = {}
    for tech in ["cite", "multi"]:
        h5info[tech] = {}
        for role, fn in H5_FILES[tech].items():
            h5info[tech][role] = (inspect_h5_file(f"{tech}.{role}", os.path.join(d, fn),
                                                  args.sample_rows, args.full_scan,
                                                  do_stats=not args.no_stats) or {})

    # ---- metadata + splits ----
    hdr("3. metadata.csv + SPLITS")
    meta = pd.read_csv(os.path.join(d, "metadata.csv"),
                       dtype={"cell_id": str, "donor": int, "day": int,
                              "technology": str, "cell_type": str})
    print(f"  shape: {meta.shape}   unique cell_ids: {meta.cell_id.nunique():,}")
    print(f"  head:\n{meta.head(4).to_string(index=False)}")
    crosstab(meta, "technology", "day", "ALL cells: technology x day")
    print("\n  cell_type counts (all):")
    print("  " + meta.cell_type.value_counts().to_string().replace("\n", "\n  "))

    for tech in ["cite", "multi"]:
        verify_split(meta, h5info, tech)

    # ---- organizer patch file ----
    patch_cells = inspect_patch(d, h5info, meta)

    # ---- overlap checks ----
    cell_sets = {
        "cite_train": set(h5info["cite"]["train_inputs"].get("cells", [])),
        "cite_test": set(h5info["cite"]["test_inputs"].get("cells", [])),
        "multi_train": set(h5info["multi"]["train_inputs"].get("cells", [])),
        "multi_test": set(h5info["multi"]["test_inputs"].get("cells", [])),
    }
    hdr("5. OVERLAP CHECKS")
    disjoint = not ((cell_sets["cite_train"] | cell_sets["cite_test"])
                    & (cell_sets["multi_train"] | cell_sets["multi_test"]))
    print(f"  cite cells vs multi cells disjoint          : "
          f"{'OK' if disjoint else 'OVERLAP!!'}")
    print(f"  cite train cells == cite target cells       : "
          f"{'OK' if cell_sets['cite_train'] == set(h5info['cite']['train_targets'].get('cells', [])) else 'MISMATCH'}")
    print(f"  multi train cells == multi target cells     : "
          f"{'OK' if cell_sets['multi_train'] == set(h5info['multi']['train_targets'].get('cells', [])) else 'MISMATCH'}")
    print(f"  cite test features == cite train features   : "
          f"{'OK' if h5info['cite']['test_inputs'].get('features') == h5info['cite']['train_inputs'].get('features') else 'MISMATCH'}")
    print(f"  multi test features == multi train features : "
          f"{'OK' if h5info['multi']['test_inputs'].get('features') == h5info['multi']['train_inputs'].get('features') else 'MISMATCH'}")

    ev_path = os.path.join(d, "evaluation_ids.csv")
    if os.path.exists(ev_path):
        gene_sets = {"cite": set(h5info["cite"]["train_targets"].get("features", [])),
                     "multi": set(h5info["multi"]["train_targets"].get("features", []))}
        eval_info = inspect_evaluation_ids(ev_path, cell_sets, gene_sets, patch_cells)
        sub_path = os.path.join(d, "sample_submission.csv")
        if os.path.exists(sub_path):
            inspect_submission(sub_path, eval_info)

    hdr("DONE -- full report above")


if __name__ == "__main__":
    main()
