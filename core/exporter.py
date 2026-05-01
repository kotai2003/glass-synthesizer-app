"""
MVTec-compatible writer for the GUI app.

Output layout (Q2 = MVTec only):

    <output_root>/<class>/
      train/good/                   <stem>.png         OK source images (copied)
      test/synthetic/               <id4>.png          NG synthetic images
      ground_truth/synthetic/       <id4>_mask.png     binary masks (0/255)
      run.json                      run metadata + per-sample log

`id4` is a zero-padded 4-digit running counter so PatchCore / EfficientAD
can ingest the directory without further renaming.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import PIL.Image


@dataclass
class SampleRecord:
    sample_id: int
    ng_filename: str       # e.g. test/synthetic/0000.png
    mask_filename: str     # e.g. ground_truth/synthetic/0000_mask.png
    source_image: str      # absolute path of OK image
    texture_path: str      # absolute path of texture
    beta: float
    seed: Optional[int]


@dataclass
class RunMetadata:
    classname: str
    output_root: str
    params: dict
    samples: List[SampleRecord] = field(default_factory=list)


class MvtecExporter:
    """Writes one run's output. Stateful: hold across many `add_sample` calls."""

    DEFECT_NAME = "synthetic"

    def __init__(self, output_root: str, classname: str, params_dict: dict):
        self.output_root = os.path.abspath(output_root)
        self.classname = classname
        self.class_root = os.path.join(self.output_root, classname)
        self.dir_train_good = os.path.join(self.class_root, "train", "good")
        self.dir_test_synth = os.path.join(self.class_root, "test", self.DEFECT_NAME)
        self.dir_gt_synth = os.path.join(self.class_root, "ground_truth", self.DEFECT_NAME)
        for p in (self.dir_train_good, self.dir_test_synth, self.dir_gt_synth):
            os.makedirs(p, exist_ok=True)
        self.metadata = RunMetadata(
            classname=classname,
            output_root=self.output_root,
            params=params_dict,
        )
        self._counter = self._resume_counter()

    def _resume_counter(self) -> int:
        """If output dir already has 0000.png ... NNNN.png, continue from NNNN+1."""
        existing = [
            f for f in os.listdir(self.dir_test_synth)
            if f.endswith(".png") and len(f) == 8 and f[:4].isdigit()
        ]
        if not existing:
            return 0
        return max(int(f[:4]) for f in existing) + 1

    def copy_ok_image(self, source_path: str) -> str:
        """Copy an OK image into train/good/. Returns destination filename."""
        stem = os.path.splitext(os.path.basename(source_path))[0]
        dst_name = f"{stem}.png"
        dst = os.path.join(self.dir_train_good, dst_name)
        if not os.path.exists(dst):
            try:
                # Re-encode as PNG to normalize
                img = PIL.Image.open(source_path).convert("RGB")
                img.save(dst, "PNG")
            except Exception:
                shutil.copy2(source_path, dst)
        return dst_name

    def add_sample(
        self,
        ng_image_bgr: np.ndarray,
        mask_uint8: np.ndarray,
        source_image_path: str,
        texture_path: str,
        beta: float,
        seed: Optional[int] = None,
    ) -> SampleRecord:
        sample_id = self._counter
        self._counter += 1
        id4 = f"{sample_id:04d}"

        ng_name = f"{id4}.png"
        mask_name = f"{id4}_mask.png"
        cv2.imwrite(os.path.join(self.dir_test_synth, ng_name), ng_image_bgr)
        cv2.imwrite(os.path.join(self.dir_gt_synth, mask_name), mask_uint8)

        rec = SampleRecord(
            sample_id=sample_id,
            ng_filename=os.path.join("test", self.DEFECT_NAME, ng_name).replace("\\", "/"),
            mask_filename=os.path.join("ground_truth", self.DEFECT_NAME, mask_name).replace("\\", "/"),
            source_image=source_image_path,
            texture_path=texture_path,
            beta=beta,
            seed=seed,
        )
        self.metadata.samples.append(rec)
        return rec

    def finalize(self) -> str:
        """Flush run.json. Returns its absolute path."""
        path = os.path.join(self.class_root, "run.json")
        payload = {
            "classname": self.metadata.classname,
            "params": self.metadata.params,
            "samples": [s.__dict__ for s in self.metadata.samples],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path
