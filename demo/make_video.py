"""Edit the raw OBS recording into the final demo video using the timeline from record_demo.py.

Model "think" time (world building, recall, waiting for the chronicler) is compressed to a few seconds,
streaming narration is sped up gently, and everything the viewer should follow (typing, clicks, reading,
the tab tour) plays at real speed. Audio is dropped (the take records none).

  python demo/make_video.py                      # uses demo/timeline.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
THINK_TARGET = 3.0      # seconds a wait lasts in the final cut
STREAM_TARGET = 6.0     # streaming narration lasts at most this long


def segments(marks: list[dict], total: float) -> list[tuple[float, float, float]]:
    """(start, end, speed) covering the whole recording."""
    out, t = [], 0.0
    for m in sorted(marks, key=lambda m: m["start"]):
        a, b = max(m["start"], t), min(m["end"], total)
        if b - a < 0.2:
            continue
        if a > t:
            out.append((t, a, 1.0))
        target = THINK_TARGET if m["kind"] == "think" else STREAM_TARGET
        out.append((a, b, max(1.0, (b - a) / target)))
        t = b
    if total - t > 0.05:
        out.append((t, total, 1.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", default=str(HERE / "timeline.json"))
    ap.add_argument("--out", default=str(HERE / "loreweaver_demo.mp4"))
    a = ap.parse_args()
    tl = json.loads(Path(a.timeline).read_text())
    src = tl["recording"]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    _, total = imageio_ffmpeg.count_frames_and_secs(src)
    segs = segments(tl["marks"], total)
    final = sum((b - a) / s for a, b, s in segs)
    print(f"raw {total / 60:.1f} min -> final {final / 60:.1f} min in {len(segs)} segments")

    parts = [f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=(PTS-STARTPTS)/{sp:.4f}[v{i}]"
             for i, (s, e, sp) in enumerate(segs)]
    chain = "".join(f"[v{i}]" for i in range(len(segs)))
    fade_out = max(0.0, final - 1.0)
    graph = ";".join(parts) + f";{chain}concat=n={len(segs)}:v=1:a=0,fps=30," \
                              f"fade=t=in:st=0:d=0.8,fade=t=out:st={fade_out:.2f}:d=1[out]"
    script = HERE / "filter.txt"
    script.write_text(graph)
    subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "warning", "-stats", "-i", src,
                    "-/filter_complex", str(script), "-map", "[out]", "-an",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", a.out], check=True)
    script.unlink()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
