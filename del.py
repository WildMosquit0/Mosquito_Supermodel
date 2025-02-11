import pandas as pd
import numpy as np
import os

# Ultralytics BoT-SORT module
# (We will manually instantiate a BoTSORT tracker with minimal optical flow.)
from ultralytics.trackers.bot_sort import BOTrack


def run_bot_sort(
    csv_path_in: str,
    csv_path_out: str = None,
    # BoT-SORT tracker settings (defaults from your snippet)
    track_high_thresh=0.25,
    track_low_thresh=0.1,
    new_track_thresh=0.25,
    track_buffer=30,
    match_thresh=0.8,
    fuse_score=True,
    gmc_method='none',        # set to 'sparseOptFlow' if you have frames and want motion compensation
    proximity_thresh=0.5,
    appearance_thresh=0.25,
    with_reid=False
):
    """
    Reads 'csv_path_in' with columns:
      [image_idx, box_idx, x, y, w, h, confidence, label, track_id, image_name, img_h, img_w]
    Applies BoT-SORT tracking to assign stable IDs. Saves a new CSV with track_id filled in.
    """

    # 1) Load CSV
    df = pd.read_csv(csv_path_in)

    # Make sure track_id is in the dataframe, even if empty
    if 'track_id' not in df.columns:
        df['track_id'] = np.nan

    # 2) Sort by (image_idx, box_idx) so we process in correct order
    df.sort_values(by=['image_idx', 'box_idx'], inplace=True)

    # 3) Initialize BoT-SORT
    #    NOTE: For real optical flow, you must pass actual frames to 'tracker.update()'.
    tracker = BOTrack(
        track_high_thresh=track_high_thresh,
        track_low_thresh=track_low_thresh,
        new_track_thresh=new_track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        fuse_score=fuse_score,
        proximity_thresh=proximity_thresh,
        appearance_thresh=appearance_thresh,
        with_reid=with_reid,
        gmc_method=gmc_method   # 'none' to skip optical flow
    )

    # We will collect the assigned track IDs in a list
    assigned_ids = []

    # 4) Group detections by image_idx (i.e. by frame)
    grouped = df.groupby('image_idx')

    for frame_idx, group in grouped:
        # Convert the group's bounding boxes into the tracker input format
        # BoT-SORT expects [x1, y1, x2, y2, confidence, class] by default
        # or sometimes [cx, cy, w, h, conf, class], depending on code version.
        # We'll do [x1, y1, x2, y2, confidence, class].
        # So, x1 = x, y1 = y, x2 = x + w, y2 = y + h.

        dets = []
        record_indices = group.index.tolist()  # We'll need to store track IDs back by row index

        for i in record_indices:
            row = df.loc[i]
            x1 = row['x']
            y1 = row['y']
            x2 = x1 + row['w']
            y2 = y1 + row['h']
            conf = row['confidence']
            cls_id = row['label']
            dets.append([x1, y1, x2, y2, conf, cls_id])

        dets = np.array(dets) if len(dets) > 0 else np.empty((0, 6), dtype=float)

        # Usually you'd pass the actual frame image to do optical flow:
        # e.g. track_output = tracker.update(dets, orig_img=frame_image)
        # Here we have no real image, so we pass None
        track_output = tracker.update(dets, orig_img=None)

        # 'track_output' is typically an array of shape [N, 10],
        # something like [x1, y1, x2, y2, track_id, ...]
        # We'll parse out x1,y1,x2,y2,track_id in the same order the tracker processed them.

        # But the returned order might differ from 'dets' order, because
        # the tracker might filter out low-conf or unmatched boxes. We have to match them up by IoU or by index:
        #
        # For a direct (naive) approach, we can assume the track_output is in the same order as 'dets' if
        # the tracker doesn't reorder. (But in practice, you often have to do an extra matching step.)
        #
        # For simplicity, let's just do a naive approach: if track_output has the same number of rows as dets,
        # we assume row i matches. If they differ, we might have to do a small iou-based assignment
        # (omitted here for brevity).

        if len(track_output) == len(dets):
            # straightforward case
            for i, row_idx in enumerate(record_indices):
                # track_id is in column 4 of track_output if the format is
                # [x1, y1, x2, y2, track_id, class(?), conf(?), etc.]
                track_id = int(track_output[i, 4])
                df.at[row_idx, 'track_id'] = track_id
        else:
            # If the numbers differ, you need a matching step.
            # For now, we'll just do an IoU-based approach or skip unmatched.
            # We'll do a minimal example that tries to find the best match for each det.
            # (In real usage, you'd handle edge cases more thoroughly.)
            for i, row_idx in enumerate(record_indices):
                x1, y1, x2, y2, conf, cls_id = dets[i]
                # Try to find a row in track_output with the same box
                best_iou = 0.0
                best_tid = None
                for t in track_output:
                    tx1, ty1, tx2, ty2, tid = t[0], t[1], t[2], t[3], t[4]
                    iou = box_iou((x1,y1,x2,y2), (tx1,ty1,tx2,ty2))
                    if iou > best_iou:
                        best_iou = iou
                        best_tid = int(tid)
                if best_tid is not None and best_iou > 0.5:
                    df.at[row_idx, 'track_id'] = best_tid
                else:
                    df.at[row_idx, 'track_id'] = -1  # or some "untracked" ID

    # 5) Save the updated CSV
    out_path = csv_path_out if csv_path_out else csv_path_in
    df.to_csv(out_path, index=False)
    print(f"Tracking complete. Results saved to: {out_path}")


def box_iou(boxA, boxB):
    """
    Compute IoU of two boxes in [x1, y1, x2, y2] format.
    """
    (xA1, yA1, xA2, yA2) = boxA
    (xB1, yB1, xB2, yB2) = boxB

    interX1 = max(xA1, xB1)
    interY1 = max(yA1, yB1)
    interX2 = min(xA2, xB2)
    interY2 = min(yA2, yB2)

    interArea = max(0, interX2 - interX1) * max(0, interY2 - interY1)
    boxAArea = (xA2 - xA1) * (yA2 - yA1)
    boxBArea = (xB2 - xB1) * (yB2 - yB1)
    if boxAArea < 1e-6 or boxBArea < 1e-6:
        return 0.0
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


if __name__ == "__main__":
    input_csv = "src/ddataset/slice/results.csv"
    output_csv = "src/ddataset/slice/results_1.csv"
    run_bot_sort(input_csv, output_csv)
