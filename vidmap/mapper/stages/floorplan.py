"""Synchronous ZfLOC pipeline before rotation averaging; no solver mutation."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

from vidmap.mapper.inputs.snapshot import FileProvenance
from vidmap.mapper.options.mapper import FloorplanGP1Options, FloorplanShadowOptions

logger = logging.getLogger(__name__)


def run_floorplan_gp1(options: FloorplanGP1Options, *, reconstruction, output_dir: Path) -> Path | None:
    """Export a copy of GP1 and synchronously reselect cached candidates before GP2."""
    if not options.enabled:
        return None
    import numpy as np

    script = Path(options.zfloc_root).expanduser() / "vidmap/utils/consensus_gp1.py"
    candidates = Path(options.candidates_dir).expanduser()
    if not script.is_file() or not candidates.is_dir():
        raise FileNotFoundError(f"GP1 consensus inputs: {script}, {candidates}")
    images = sorted(reconstruction.reg_image_ids())
    if len(images) < 2:
        raise ValueError("GP1 consensus requires at least two registered cameras")
    poses = np.tile(np.eye(4), (len(images), 1, 1))
    names = []
    for index, image_id in enumerate(images):
        camera = reconstruction.images[image_id]
        names.append(camera.name)
        poses[index, :3] = camera.cam_from_world().inverse().matrix()
    out = Path(output_dir) / "floorplan_gp1"
    out.mkdir(parents=True, exist_ok=False)
    # Tiny pose arrays need no compression; native COLMAP may bring its own zlib.
    np.savez(out / "trajectory_gp1.npz", names=names, image_ids=images, camera_poses=poses)
    command = [options.python_executable or sys.executable, str(script),
               "--trajectory", str(out / "trajectory_gp1.npz"), "--candidates-dir", str(candidates),
               "--out", str(out), "--trans-thresh", str(options.trans_thresh),
               "--max-interpolation-gap-s", str(options.max_interpolation_gap_s)]
    (out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONUNBUFFERED"] = "1"
    logger.info("ZFLOC_GP1_CONSENSUS_START cameras=%d output=%s", len(images), out)
    subprocess.run(command, env=env, check=True)
    report = json.loads((out / "summary.json").read_text())
    if report["scope"] != "whole_gp1_trajectory" or report["n_gp1_cameras"] != len(images):
        raise ValueError("GP1 consensus did not cover the exported trajectory")
    logger.info("ZFLOC_GP1_CONSENSUS_READY selected=%d output=%s", report["global"]["accepted"], out)
    return out


def run_floorplan_shadow(
    options: FloorplanShadowOptions,
    *,
    image_names: dict[int, str],
    image_dir: Path | None,
    output_dir: Path,
) -> Path | None:
    """Finish matching and verify exact input identity before returning to RA.

    The existing ZfLOC entry currently supports 40 LaMAR frames (two windows).
    Candidate yaw is radians; selected-prior yaw is degrees. Neither is applied
    to the reconstruction in this stage.
    """
    if not options.enabled:
        return None
    if image_dir is None or not Path(image_dir).is_dir():
        raise ValueError("floorplan_shadow requires local mapper-input image provenance")
    image_dir = Path(image_dir)
    if not image_names or len(set(image_names.values())) != len(image_names):
        raise ValueError("floorplan_shadow requires nonempty, unique mapper image names")
    if any(Path(name).name != name for name in image_names.values()):
        raise ValueError("floorplan_shadow expects the sampled LaMAR filenames")

    script = Path(options.zfloc_root).expanduser() / "vidmap/main.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    for name in ("raw_images", "engine", "f3loc", "meta_dir"):
        path = Path(getattr(options, name)).expanduser()
        if not (path.is_file() if name == "engine" else path.is_dir()):
            raise FileNotFoundError(f"floorplan_shadow {name}: {path}")
    out = Path(output_dir) / "floorplan"
    if out.exists():
        raise FileExistsError(f"Preserving existing ZfLOC output: {out}; choose a new run directory")
    command = [str(Path(options.python_executable).expanduser()) if options.python_executable else sys.executable,
               str(script)]
    for name in ("raw_images", "engine", "f3loc", "meta_dir"):
        command += ["--" + name.replace("_", "-"), str(Path(getattr(options, name)).expanduser())]
    command += ["--out", str(out), "--start-us", str(options.start_us),
                "--trans-thresh", str(options.trans_thresh), "--window-size", str(options.window_size)]
    # The SfM and TRT environments may use different Python ABIs. Do not pass
    # the parent's native-extension search path into the TRT interpreter.
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    da3_source = Path(__file__).resolve().parents[3] / "third_party/Depth-Anything-3/src"
    if da3_source.is_dir():
        env["PYTHONPATH"] = str(da3_source)
    env["PYTHONUNBUFFERED"] = "1"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "floorplan_command.json").write_text(json.dumps(command, indent=2) + "\n")
    logger.info("ZFLOC_PIPELINE_START output=%s", out)
    started = time.monotonic()
    # Frontend tensors have been released, but their cached GPU allocations may
    # still occupy the memory needed by the separate TRT process.
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    subprocess.run(command, env=env, check=True)

    selection = json.loads((out / "prep/selection.json").read_text())
    names, stamps = selection["names"], selection["selected_us"]
    if len(names) != 40 or len(set(names)) != 40 or len(stamps) != 40:
        raise ValueError("ZfLOC selection must contain 40 distinct frames and timestamps")
    if any(type(stamp) is not int for stamp in stamps) or any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("ZfLOC timestamps must be strictly increasing integer microseconds")
    if selection["start_us"] != options.start_us:
        raise ValueError("ZfLOC selection start timestamp differs from requested input")
    if any(Path(name).name != name for name in names):
        raise ValueError("Invalid ZfLOC sampled image name")
    inputs = {p.name for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    if not set(names).issubset(inputs) or not set(image_names.values()).issubset(inputs):
        raise ValueError("ZfLOC frames or mapper keyframes are missing from VidMap input images")
    windows = json.loads((out / "reconstruction/windows.json").read_text())["windows"]
    if len(windows) != 40 // options.window_size or any(len(window) != options.window_size for window in windows) or sum(windows, []) != names:
        raise ValueError("ZfLOC window membership does not match the 40-frame input")
    matching = json.loads((out / "matching_summary.json").read_text())["records"]
    expected = {name: stamp * 1000 for name, stamp in zip(names, stamps)}
    if len(matching) != 40 or {row["frame"]: row["timestamp_ns"] for row in matching} != expected:
        raise ValueError("ZfLOC matching timestamps do not match sampled raw timestamps")
    summary = json.loads((out / "consensus/summary.json").read_text())
    if (summary["scope"] != f"independent_{options.window_size}_view_windows" or summary["cross_window_alignment"]
            or summary["prior_columns"] != ["x_m", "y_m", "yaw_deg"]):
        raise ValueError("Unexpected ZfLOC consensus coordinate contract")
    for kind in ("local", "global"):
        if not (out / f"consensus/{kind}_priors.npy").is_file():
            raise FileNotFoundError(out / f"consensus/{kind}_priors.npy")
    ids = {name: image_id for image_id, name in image_names.items()}
    membership = {name: index for index, window in enumerate(windows) for name in window}
    records = []
    for name in names:
        proof = FileProvenance.from_path(image_dir / name)
        proof.validate(out / "prep/frames_ts" / name)
        records.append({"name": name, "image_id": ids.get(name), "timestamp_ns": expected[name],
                        "window": membership[name], "input_sha256": proof.sha256,
                        "excluded_by_frontend": name not in ids})
    report = {"solver_fusion": False, "candidate_yaw_unit": "rad", "prior_yaw_unit": "deg",
              "seconds": time.monotonic() - started, "images": records}
    (out / "vidmap_images.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("ZFLOC_OUTPUT_READY mapped=%d sampled=40 output=%s", len(ids), out)
    return out
