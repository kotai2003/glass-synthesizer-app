# glass-synthesizer-app

A self-contained Tkinter desktop application that synthesizes **NG (defect / anomaly) images** from
your own OK images and an arbitrary texture set, for use as **training data for AI anomaly-detection models**
(PatchCore, EfficientAD, custom models, etc.).

The synthesis algorithm is the **LAS (Local Anomaly Synthesis)** branch from
[cqylunlun/GLASS](https://github.com/cqylunlun/GLASS) (ECCV 2024) — Perlin-noise mask × random texture
blended with a beta-distributed mixing coefficient. The discriminator / training loop / GAS branch
of GLASS are **out of scope** for this app; only the image-space synthesis is reused (and vendored,
see `LICENSE_GLASS`).

## What it does

1. You point the app at: a folder of OK images, a folder of textures (DTD or in-house defect crops), and an output folder.
2. **Preview 1 sample** picks a random OK + texture and shows Original / Synthetic NG / Mask side-by-side.
3. **Generate batch** writes `N` synthetic NG images per OK image, plus per-pixel binary masks, in **MVTec-compatible layout**:
   ```
   <output>/<class>/
   ├── train/good/                 (copies of source OK images)
   ├── test/synthetic/0000.png ... (synthesized NG images)
   ├── ground_truth/synthetic/0000_mask.png ...
   └── run.json                    (params + per-sample log)
   ```

## Why it exists

GLASS's discriminator never persists the synthesized images — they live in memory for one minibatch
and are discarded. This app extracts that synthesis step and exposes it as a reproducible image-export
tool, so operators can grow a small "OK only" dataset into a balanced OK/NG dataset for downstream
training without running GLASS itself.

## Install

Tested on Windows 11 with Python 3.9.15 (from the GLASS-style conda env). Should also work on Linux/macOS
provided imgaug 0.4.0 installs (which means NumPy < 2.0).

```bash
pip install -r requirements.txt
# or, if you already have a GLASS-equivalent env:
#   conda env create -f environment.yml ; conda activate glass_env
```

Required packages: `torch`, `torchvision`, `Pillow`, `opencv-python`, `imgaug==0.4.0`, `numpy<2`,
plus standard library `tkinter` (bundled with most CPython builds).

## Run

```bash
python -m synthesizer_app.gui_main
```

Or on Windows, double-click `run.bat` (you may need to edit it to point at your Python).

The window opens with a 3-up Preview area on the left (Original | Synthetic | Mask), a thumbnail strip
of generated samples, a Status frame with a progress bar, and a Notebook on the right with **Control**
(folders, class name, N per OK, action buttons) and **Configure** (Perlin scale, beta distribution,
random texture aug, foreground mask, seed) tabs.

## Configure-tab knobs

| Knob | Range | Effect |
|---|---|---|
| Working size | 256/288/320/384/512 | Internal LAS resolution; output is resized back to source aspect |
| Perlin scale min/max | 0–7 | Granularity range of the Perlin mask (`2**min` × `2**max`) |
| Beta mean | 0.20–0.80 | Center of the texture/source mixing coefficient `β ~ N(mean, std)`, clipped to `[0.2, 0.8]` |
| Beta std | 0.00–0.30 | Spread of `β` |
| Random texture augmentation | bool | Apply 3 random color/flip/affine ops to the texture before blending |
| Seed | int / blank | Blank → random per call. Integer → `_seed_all(seed + sample_idx)` for batch reproducibility |
| Use foreground mask | bool + folder | Match per-image fg mask by basename; warn but proceed if missing |

Slider changes after the first preview trigger a 200 ms debounced live re-render, so the visual
effect of each knob is immediate.

## Reproducibility

The app seeds `numpy.random`, `torch`, **and `imgaug`** before each sample (the latter is required
because `perlin.py` rotates the noise via `imgaug.iaa.Affine`, whose RNG is independent of np/torch).

## License

MIT (this app), MIT (the vendored `core/_vendored/perlin.py` from cqylunlun/GLASS).
See `LICENSE` and `LICENSE_GLASS`.

## Credits

LAS algorithm and `perlin.py`: Chen et al., **"A Unified Anomaly Synthesis Strategy with Gradient
Ascent for Industrial Anomaly Detection and Localization"**, ECCV 2024 — [paper](https://arxiv.org/abs/2407.09359),
[code](https://github.com/cqylunlun/GLASS).

GUI design system: TOMOMI RESEARCH, INC. internal style guide.
