import cv2
import os
from pathlib import Path
from joblib import Parallel, delayed
import argparse
from tqdm import tqdm

N_CPU = os.cpu_count()


def extract_frames_from_video(video_path: Path, out_root: Path):
    """Extract frames from one video"""
    if isinstance(out_root, str):
        out_root = Path(out_root)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # get video name and create output folder
    base_name = video_path.stem
    out_dir_path = out_root / base_name
    out_dir_path.mkdir(exist_ok=True)  # make directory for video

    ret, frame = cap.read()
    frame_id = 1  # frame index starts from 1
    while ret:
        frame_path = out_dir_path / "{}_{}.png".format(
            base_name, str(frame_id).zfill(5)
        )
        _ = cv2.imwrite(str(frame_path), frame)
        ret, frame = cap.read()
        frame_id += 1

    # sanity check to make sure all frames have been converted
    assert n_frames == frame_id - 1, f"{n_frames} != {frame_id}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", required=True, type=str, help="Path to video files"
    )
    parser.add_argument(
        "--out_dir", required=True, type=str, help="Path to output png files"
    )
    args = parser.parse_args()

    # make output dir if it doesn't exist
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    video_paths = list(Path(args.input_dir).rglob("*.mp4"))

    # parallel call using joblib
    Parallel(n_jobs=N_CPU)(
        delayed(extract_frames_from_video)(video_path, args.out_dir)
        for video_path in tqdm(video_paths)
    )
