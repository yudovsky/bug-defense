"""
Batch Regenerate All Campaign GIFs:
Iterates over all simulation raw arrays (.npz) in knowledge/beetle/raw_arrays/
and re-renders the 4-panel native animations with:
  1) Unmasked optical power density Q in void voxels (plume absorption/scattering visible)
  2) Recalculated dual-metric temperature inset:
     - Core (Peak): Cyan dashed line (np.nanmax) showing 100°C latent plateau
     - Core (Mean): Light blue solid line (np.nanmean) showing true volume-averaged cooling
     - Ventral (Peak & Mean): Rose dashed / coral solid lines
  3) Panel 2 title displaying both Peak and Mean intact tissue temperatures:
     "Core: Peak X.X°C (Mean Y.Y°C) | Ventral: Peak X.X°C (Mean Y.Y°C)"
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import glob
import time

sys.path.insert(0, REPO_ROOT)
from knowledge.beetle.scripts.render_native_highres_animation import render_native_animation

KNOWLEDGE_DIR = os.path.join(REPO_ROOT, "knowledge/beetle")
RAW_DIR = os.path.join(KNOWLEDGE_DIR, "raw_arrays")
PLOTS_DIR = os.path.join(KNOWLEDGE_DIR, "plots")

import argparse
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Batch Regenerate All Campaign GIFs at Native 501-Frame Resolution")
    parser.add_argument("--force", action="store_true", help="Force re-rendering even if GIF already has 501 frames")
    args = parser.parse_args()

    npz_files = [f for f in sorted(glob.glob(os.path.join(RAW_DIR, "matrix_*.npz"))) if "_1ms_2x" not in f]
    total = len(npz_files)
    print(f"\n{'='*75}")
    print(f"  BATCH GIF REGENERATION: UNMASKED PLUME OPTICS & DUAL TEMPERATURE INSET")
    print(f"  Standard: 501 Native Frames (active_stride=2, cooling_stride=2)")
    print(f"  Target: {total} simulation raw arrays (.npz)")
    print(f"{'='*75}\n")

    t_start = time.time()
    rendered_count = 0
    skipped_count = 0

    for idx, npz_path in enumerate(npz_files, 1):
        base = os.path.basename(npz_path).replace("matrix_", "").replace(".npz", "")
        # Determine output gif name
        if "no_plume" in base:
            gif_name = f"beetle_{base}.gif"
        elif "dual" in base:
            gif_name = f"beetle_{base}.gif"
        else:
            gif_name = f"beetle_native_{base}.gif"
            
        out_gif = os.path.join(PLOTS_DIR, gif_name)

        # Check if already 501 frames
        if not args.force and os.path.exists(out_gif):
            try:
                with Image.open(out_gif) as im:
                    if getattr(im, 'n_frames', 0) == 501:
                        print(f"[{idx}/{total}] Skipping {base} -> {gif_name} (already 501 native frames).")
                        skipped_count += 1
                        continue
            except Exception:
                pass

        print(f"[{idx}/{total}] Re-rendering {base} -> {gif_name} (501 frames)...")
        t0 = time.time()
        try:
            render_native_animation(
                npz_file=npz_path,
                out_gif=out_gif,
                active_stride=2,
                cooling_stride=2
            )
            print(f"  -> Completed in {time.time() - t0:.1f}s")
            rendered_count += 1
        except Exception as e:
            print(f"  [ERROR] Failed to render {base}: {e}")

    print(f"\n{'='*75}")
    print(f"  BATCH REGENERATION FINISHED: {rendered_count} rendered, {skipped_count} skipped in {time.time() - t_start:.1f}s")
    print(f"{'='*75}\n")

if __name__ == "__main__":
    main()
