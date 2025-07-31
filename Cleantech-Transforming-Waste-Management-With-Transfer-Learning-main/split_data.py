## split_data.py
"""
SmartInternz CleanTech – dataset splitter
----------------------------------------
• Expects class folders under `raw_images/`
• Creates 70 % train, 15 % val, 15 % test
• Guarantees ≥1 image per class in every split
"""

import os, shutil, random, math, textwrap

# ── config ──────────────────────────────────────────────────────
SRC = "raw_images"  # root folder with class sub-dirs
DST      = "dataset"      # will create train/val/test here
SPLIT    = (0.70, 0.15, 0.15)   # train, val, test ratios

assert sum(SPLIT) == 1.0, "Split ratios must add up to 1.0"

# ── make sure target dirs are fresh ─────────────────────────────
if os.path.exists(DST):
    shutil.rmtree(DST)          # start clean
for split in ("train", "val", "test"):
    os.makedirs(os.path.join(DST, split), exist_ok=True)

# ── iterate over each class folder ──────────────────────────────
for cls in sorted(os.listdir(SRC)):
    src_cls = os.path.join(SRC, cls)
    if not os.path.isdir(src_cls):
        continue  # skip stray files

    # normalise class name (remove " Images", trim spaces)
    clean_cls = cls.replace(" Images", "").strip()

    # list image files
    files = [os.path.join(src_cls, f) for f in os.listdir(src_cls)
             if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]

    if not files:
        print(f"⚠️  No images found in '{cls}' – skipping.")
        continue

    random.shuffle(files)
    n = len(files)

    # compute split indices ensuring at least 1 image per split
    n_train = max(1, math.floor(SPLIT[0] * n))
    n_val   = max(1, math.floor(SPLIT[1] * n))
    # put the remainder in test so totals line up
    n_test  = n - n_train - n_val
    if n_test == 0:           # guarantee at least one test sample
        n_test, n_val = 1, n_val - 1

    splits = [
        ("train", files[:n_train]),
        ("val",   files[n_train:n_train + n_val]),
        ("test",  files[n_train + n_val:]),
    ]

    # copy to new structure
    for split_name, batch in splits:
        dest_dir = os.path.join(DST, split_name, clean_cls)
        os.makedirs(dest_dir, exist_ok=True)
        for f in batch:
            shutil.copy2(f, dest_dir)

    print(f"✓ {clean_cls:<15} --> "
          f"{n_train:3d} train | {n_val:3d} val | {n_test:3d} test")

# ── pretty summary ──────────────────────────────────────────────
print("\n" + textwrap.dedent(f"""
    ✅ Done splitting!
    New structure:
      {DST}/train/<class>/
      {DST}/val/<class>/
      {DST}/test/<class>/

   
"""))
