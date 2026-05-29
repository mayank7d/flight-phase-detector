import json, math, os
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────

FILES = [
    # add more files here...
]

ALL_SHEETS  = False     # True = export every sheet | False = first sheet only
NA_STRATEGY = "null"    # "null" = JSON null | "empty" = "" | "drop" = omit key

# ─────────────────────────────────────────────────────────────────────────────

def clean(val):
    if val is None: return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)): return None
    try:
        if pd.isna(val): return None
    except: pass
    if hasattr(val, "item"):      return val.item()
    if hasattr(val, "isoformat"): return val.isoformat()
    return val

def convert(df):
    rows = df.map(clean).to_dict(orient="records")
    if NA_STRATEGY == "drop":
        rows = [{k: v for k, v in r.items() if v is not None} for r in rows]
    elif NA_STRATEGY == "empty":
        rows = [{k: ("" if v is None else v) for k, v in r.items()} for r in rows]
    return rows

def process(filepath):
    if not os.path.isfile(filepath):
        print(f"[skip]  Not found: '{filepath}'"); return

    print(f"[read]  {filepath}")
    sheets = pd.read_excel(filepath, sheet_name=None)

    if ALL_SHEETS:
        result = {}
        for name, df in sheets.items():
            print(f"  sheet '{name}': {len(df):,} rows")
            result[str(name)] = convert(df)
    else:
        name, df = next(iter(sheets.items()))
        print(f"  sheet '{name}': {len(df):,} rows")
        result = convert(df)

    out = os.path.splitext(filepath)[0] + ".json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"[done]  {out}  ({os.path.getsize(out)/1024:.1f} KB)\n")

for f in FILES:
    process(f)
