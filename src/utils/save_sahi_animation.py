import os
import re
import cv2
import glob
import pandas as pd
from typing import Dict, Optional, List
from src.utils.config_ops import load_config

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

class save_sahi_animation:
    def __init__(self, conf_yaml_path: str, csv_path: Optional[str] = None):
        self.cfg = load_config(conf_yaml_path)

        base_out = self.cfg["output_dir"]
        task_out = os.path.join(base_out, self.cfg.get("model", {}).get("task", ""))
        task_csv = os.path.join(task_out, "results.csv")
        base_csv = os.path.join(base_out, "results.csv")
        self.csv_path = csv_path or (task_csv if os.path.isfile(task_csv) else base_csv)

        self.output_root = os.path.dirname(self.csv_path)
        os.makedirs(self.output_root, exist_ok=True)

        self.class_names = self.cfg.get("model", {}).get("names", None)
        self.frame_pattern = self.cfg.get("model", {}).get("frame_pattern", None)
        self.default_fps = int(self.cfg.get("model", {}).get("fps", 25))
        self.vid_stride = int(self.cfg.get("model", {}).get("vid_stride", 1))
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.color = (255, 0, 0)  # blue
        self.images_dir = self.cfg["images_dir"]

        self.videos_by_stem, self.images_by_stem = self._inventory_sources(self.images_dir)

    def _inventory_sources(self, path: str):
        vids: dict[str, List[str]] = {}
        imgs: dict[str, List[str]] = {}

        def img_base_stem(stem: str) -> str:
            m = re.search(r"^(.*)_(\d+)$", stem)
            return m.group(1) if m else stem

        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for fn in files:
                    ext = os.path.splitext(fn)[1].lower()
                    stem = os.path.splitext(fn)[0]
                    full = os.path.join(root, fn)
                    if ext in VIDEO_EXTS:
                        vids.setdefault(stem, []).append(full)
                    elif ext in IMG_EXTS:
                        base = img_base_stem(stem)
                        imgs.setdefault(base, []).append(full)
        elif os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            stem = os.path.splitext(os.path.basename(path))[0]
            if ext in VIDEO_EXTS:
                vids.setdefault(stem, []).append(path)
            elif ext in IMG_EXTS:
                base = img_base_stem(stem)
                imgs.setdefault(base, []).append(path)

        for k in list(imgs.keys()):
            imgs[k].sort()
        return vids, imgs

    def _label_text(self, label_val, conf_val, track_id) -> str:
        if isinstance(label_val, (float, int)):
            idx = int(label_val)
            name = None
            if self.class_names is not None:
                if isinstance(self.class_names, dict):
                    name = self.class_names.get(idx, str(idx))
                elif isinstance(self.class_names, (list, tuple)) and 0 <= idx < len(self.class_names):
                    name = self.class_names[idx]
            if name is None:
                name = str(idx)
        else:
            name = str(label_val)
        conf_txt = f"{float(conf_val):.2f}"
        if track_id is not None:
            try:
                import math
                if not (isinstance(track_id, float) and math.isnan(track_id)):
                    return f"{name} {conf_txt} id:{int(track_id)}"
            except Exception:
                return f"{name} {conf_txt} id:{track_id}"
        return f"{name} {conf_txt}"

    def _draw_box(self, img, x, y, w, h, text):
        thickness = max(2, int(round(0.0025 * (img.shape[0] + img.shape[1]))))
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))
        cv2.rectangle(img, (x1, y1), (x2, y2), self.color, thickness)
        txt_scale = max(0.5, (img.shape[0] + img.shape[1]) / 1600.0)
        (tw, th), _ = cv2.getTextSize(text, self.font, txt_scale, max(1, thickness - 1))
        th = int(th * 1.2)
        bx2 = x1 + tw + 6
        by2 = y1 - th - 6
        by1 = y1
        if by2 < 0:
            by2 = y1 + th + 6
            by1 = y1 + 0
        cv2.rectangle(img, (x1, by1), (bx2, by2), self.color, -1)
        ty = by2 + th if by2 < y1 else by2 - 4
        cv2.putText(img, text, (x1 + 3, ty), self.font, txt_scale, (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)

    def _groups(self, df: pd.DataFrame):
        df.sort_values(["image_name", "image_idx", "box_idx"], inplace=True)
        return df.groupby(["image_name", "image_idx"], sort=True)

    def _frames_map_for_stem(self, df: pd.DataFrame, stem: str):
        sub = df[df["image_name"].astype(str) == stem]
        if sub.empty:
            return [], {}, None

        wh = None
        if "img_w" in sub.columns and "img_h" in sub.columns:
            first = sub.iloc[0]
            wh = (int(first["img_w"]), int(first["img_h"]))

        if "image_idx" in sub.columns:
            idxs = sorted(sub["image_idx"].astype(int).unique())
            frames = [i * self.vid_stride for i in idxs]  # multiply idx by stride
            groups = {f: sub[sub["image_idx"].astype(int) == i] for f, i in zip(frames, idxs)}
        else:
            # fallback if no image_idx — use frame numbers directly if available
            if "frame" in sub.columns:
                frames = sorted(sub["frame"].astype(int).unique())
                groups = {f: sub[sub["frame"].astype(int) == f] for f in frames}
            else:
                return [], {}, wh

        return frames, groups, wh




    def _resolve_video_path_exact(self, stem: str, wh_expected: Optional[tuple[int, int]]):
        candidates = self.videos_by_stem.get(stem, [])
        if not candidates:
            return None
        if wh_expected:
            for p in candidates:
                cap = cv2.VideoCapture(p)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    if (w, h) == wh_expected:
                        return p
                else:
                    cap.release()
        return candidates[0]

    def _find_image_file(self, stem: str, idx: int) -> Optional[str]:
        files = self.images_by_stem.get(stem, [])

        if self.frame_pattern:
            for candidate in (str(idx), f"{idx:06d}"):
                name = self.frame_pattern.format(image_name=stem, image_idx=candidate)
                base_dir = os.path.dirname(files[0]) if files else self.images_dir
                p = os.path.join(base_dir, name)
                if os.path.isfile(p):
                    return p

        for f in files:
            base = os.path.basename(f)
            if base.startswith(f"{stem}_{idx:06d}") or base.startswith(f"{stem}_{idx}."):
                return f

        for ext in IMG_EXTS:
            pat1 = os.path.join(self.images_dir, "**", f"{stem}_{idx:06d}{ext}")
            pat2 = os.path.join(self.images_dir, "**", f"{stem}_{idx}{ext}")
            pat3 = os.path.join(self.images_dir, "**", f"{stem}*{idx}*{ext}")
            for pat in (pat1, pat2, pat3):
                m = glob.glob(pat, recursive=True)
                if m:
                    m.sort(key=len)
                    return m[0]
        return None

    def _write_from_video(self, stem: str, df: pd.DataFrame):
        frames, groups, wh_expected = self._frames_map_for_stem(df, stem)
        if not frames:
            return

        video_path = self._resolve_video_path_exact(stem, wh_expected)
        if not video_path:
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or self.default_fps
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = os.path.join(self.output_root, f"{stem}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # seek to the exact (0-based) frames we computed
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, img = cap.read()
            if not ok:
                continue
            g = groups.get(f)
            if g is None:
                continue
            for _, r in g.iterrows():
                x, y, ww, hh = float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])
                text = self._label_text(r["label"], r["confidence"], r.get("track_id", None))
                self._draw_box(img, x, y, ww, hh, text)
            writer.write(img)

        writer.release()
        cap.release()


    def _write_from_images(self, stem: str, groups):
        keys = sorted([k for k in groups.keys() if k[0] == stem], key=lambda x: x[1])
        if not keys:
            return

        first_path = self._find_image_file(stem, keys[0][1])
        if first_path is None:
            return
        img0 = cv2.imread(first_path)
        if img0 is None:
            return
        h, w = img0.shape[:2]

        out_path = os.path.join(self.output_root, f"{stem}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(self.default_fps), (w, h))

        for _, idx in keys:
            fp = self._find_image_file(stem, idx)
            if fp is None:
                continue
            img = cv2.imread(fp)
            if img is None:
                continue
            g = groups.get((stem, idx))
            if g is None:
                continue
            for _, r in g.iterrows():
                x, y, ww, hh = float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])
                label = r["label"]; conf = r["confidence"]; track_id = r.get("track_id", None)
                text = self._label_text(label, conf, track_id)
                self._draw_box(img, x, y, ww, hh, text)
            writer.write(img)

        writer.release()

    def run(self):
        if not os.path.isfile(self.csv_path):
            return
        df = pd.read_csv(self.csv_path)
        grouped = self._groups(df)
        groups = {(k[0], int(k[1])): v for k, v in grouped}

        stems = sorted(set(df["image_name"].astype(str)))
        for stem in stems:
            if stem in self.videos_by_stem:
                self._write_from_video(stem, df)
            elif stem in self.images_by_stem:
                self._write_from_images(stem, groups)
            else:
                # no exact file match → skip to prevent wrong cross-assignment
                continue
