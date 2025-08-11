import os
import cv2
import glob
import math
import pandas as pd
from typing import Dict, Optional

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

class save_sahi_animation:
    def __init__(self, config: Dict, csv_path: Optional[str] = None):
        self.cfg = config
        base_out = self.cfg['output_dir']
        task_out = os.path.join(base_out, self.cfg.get('model', {}).get('task', ''))
        self.csv_path = csv_path or (os.path.join(task_out, "results.csv") if os.path.isfile(os.path.join(task_out, "results.csv")) else os.path.join(base_out, "results.csv"))
        self.output_root = os.path.dirname(self.csv_path)
        os.makedirs(self.output_root, exist_ok=True)

        self.class_names = self.cfg.get('model', {}).get('names', '1')
        self.frame_pattern = self.cfg.get('model', {}).get('frame_pattern', None)
        self.default_fps = int(self.cfg.get('model', {}).get('fps', 25))
        self.vid_stride = int(self.cfg.get('model', {}).get('vid_stride', 1))
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.images_dir = self.cfg['images_dir']
        self.videos_by_stem, self.images_by_stem = self._inventory_sources(self.images_dir)
        self.color = (255, 0, 0)

    def _inventory_sources(self, path: str):
        vids, imgs = {}, {}
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for fn in files:
                    ext = os.path.splitext(fn)[1].lower()
                    stem = os.path.splitext(fn)[0]
                    full = os.path.join(root, fn)
                    if ext in VIDEO_EXTS:
                        vids[stem] = full
                    elif ext in IMG_EXTS:
                        imgs.setdefault(stem, []).append(full)
        elif os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            stem = os.path.splitext(os.path.basename(path))[0]
            if ext in VIDEO_EXTS:
                vids[stem] = path
            elif ext in IMG_EXTS:
                imgs.setdefault(stem, []).append(path)
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
        if track_id is not None and not (isinstance(track_id, float) and math.isnan(track_id)):
            return f"{name} {conf_txt} id:{int(track_id)}"
        return f"{name} {conf_txt}"

    def _draw_box(self, img, x, y, w, h, text):
        color = self.color
        thickness = max(2, int(round(0.0025 * (img.shape[0] + img.shape[1]))))
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        txt_scale = max(0.5, (img.shape[0] + img.shape[1]) / 1600.0)
        (tw, th), _ = cv2.getTextSize(text, self.font, txt_scale, max(1, thickness - 1))
        th = int(th * 1.2)
        bx2 = x1 + tw + 6
        by2 = y1 - th - 6
        by1 = y1
        if by2 < 0:
            by2 = y1 + th + 6
            by1 = y1 + 0
        cv2.rectangle(img, (x1, by1), (bx2, by2), color, -1)
        ty = by2 + th if by2 < y1 else by2 - 4
        cv2.putText(img, text, (x1 + 3, ty), self.font, txt_scale, (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)

    def _groups(self, df: pd.DataFrame):
        df.sort_values(["image_name", "image_idx", "box_idx"], inplace=True)
        return df.groupby(["image_name", "image_idx"], sort=True)

    def _find_image_file(self, stem: str, idx: int) -> Optional[str]:
        files = self.images_by_stem.get(stem, [])
        if self.frame_pattern:
            for candidate in (str(idx), f"{idx:06d}"):
                name = self.frame_pattern.format(image_name=stem, image_idx=candidate)
                p = os.path.join(os.path.dirname(files[0]) if files else self.images_dir, name)
                if os.path.isfile(p):
                    return p
        for f in files:
            base = os.path.basename(f)
            if base.startswith(f"{stem}_{idx:06d}") or base.startswith(f"{stem}_{idx}."):
                return f
        for ext in IMG_EXTS:
            pat = os.path.join(self.images_dir, "**", f"{stem}*{idx}*{ext}")
            m = glob.glob(pat, recursive=True)
            if m:
                m.sort(key=len)
                return m[0]
        return None

    def _resolve_video_path(self, stem: str):
        p = self.videos_by_stem.get(stem)
        if p:
            return p, stem
        for s, path in self.videos_by_stem.items():
            if s.startswith(stem) or stem.startswith(s):
                return path, s
        return None, None

    def _frames_map_for_stem(self, df: pd.DataFrame, stem: str):
        sub = df[df["image_name"].astype(str) == stem]
        if sub.empty:
            sub = df[df["image_name"].astype(str).str.startswith(stem)]
            if sub.empty:
                sub = df[df["image_name"].astype(str).apply(lambda x: stem.startswith(x))]
        if sub.empty:
            return [], {}
        if "frame" in sub.columns:
            frames = sorted(sub["frame"].astype(int).unique())
            groups = {f: sub[sub["frame"].astype(int) == f] for f in frames}
        else:
            idxs = sorted(sub["image_idx"].astype(int).unique())
            frames = [i * self.vid_stride for i in idxs]
            groups = {i * self.vid_stride: sub[sub["image_idx"].astype(int) == i] for i in idxs}
        return frames, groups

    def _write_from_video(self, stem: str, df: pd.DataFrame):
        video_path, real_stem = self._resolve_video_path(stem)
        if not video_path:
            return
        frames, groups = self._frames_map_for_stem(df, real_stem or stem)
        if not frames:
            return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or self.default_fps
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = os.path.join(self.output_root, f"{stem}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
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
                label = r["label"]; conf = r["confidence"]; track_id = r.get("track_id", None)
                text = self._label_text(label, conf, track_id)
                self._draw_box(img, x, y, ww, hh, text)
            writer.write(img)
        writer.release()
        cap.release()

    def _write_from_images(self, stem: str, groups):
        keys = sorted([k for k in groups.keys() if k[0] == stem], key=lambda x: x[1])
        if not keys:
            alt = sorted([k for k in groups.keys() if str(k[0]).startswith(stem)], key=lambda x: (x[0], x[1]))
            keys = alt
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
            if idx % self.vid_stride != 0:
                continue
            fp = self._find_image_file(stem, idx)
            if fp is None:
                continue
            img = cv2.imread(fp)
            if img is None:
                continue
            g = groups.get((stem, idx))
            if g is None:
                k2 = next((k for k in groups.keys() if (k[0] == stem or str(k[0]).startswith(stem)) and k[1] == idx), None)
                if k2 is None:
                    continue
                g = groups[k2]
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
            if stem in self.videos_by_stem or any(s.startswith(stem) or stem.startswith(s) for s in self.videos_by_stem):
                self._write_from_video(stem, df)
            elif stem in self.images_by_stem or any(s.startswith(stem) or stem.startswith(s) for s in self.images_by_stem):
                self._write_from_images(stem, groups)
