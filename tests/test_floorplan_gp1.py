"""Run: python -m unittest discover -s tests -p test_floorplan_gp1.py."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from vidmap.configuration.build import build_mapping_config
from vidmap.mapper.options.mapper import FloorplanGP1Options
from vidmap.mapper.options.positioning import GPOptions
from vidmap.mapper.stages.floorplan import run_floorplan_gp1
from vidmap.mapper.stages.global_positioning.positioner import GlobalPositioner


class GP1HookTest(unittest.TestCase):
    def test_config_and_disabled_noop(self):
        self.assertIsNone(run_floorplan_gp1(FloorplanGP1Options(), reconstruction=None, output_dir=None))
        with self.assertRaises(ValueError):
            FloorplanGP1Options(enabled=True)
        config = build_mapping_config(None, source_name="uncalib/base", override_tokens=[
            "mapper.floorplan_gp1.enabled=true", "mapper.floorplan_gp1.zfloc_root=/tmp/zfloc",
            "mapper.floorplan_gp1.candidates_dir=/tmp/candidates"])
        self.assertTrue(config.pipeline.mapper.floorplan_gp1.enabled)

    def test_real_position_order_and_failure(self):
        for enabled, fail in ((False, False), (True, False), (True, True)):
            events = []
            def callback():
                events.append("consensus")
                if fail:
                    raise RuntimeError("consensus failed")
            gp = GlobalPositioner(solve_state=MagicMock(), tracks={}, consecutive_pair_ids=[],
                sequence_id_to_index={}, inlier_thresholds=MagicMock(), boundary_depth_outliers_marked=False,
                options=GPOptions(), output_dir=Path('/tmp'), replay=MagicMock(),
                after_first_pass=callback if enabled else None)
            gp.replay.write_enabled.return_value = False
            with ExitStack() as stack:
                stack.enter_context(patch("vidmap.mapper.stages.global_positioning.positioner.build_first_global_positioning_options",
                                          return_value=(MagicMock(), MagicMock())))
                stack.enter_context(patch("vidmap.mapper.stages.global_positioning.positioner.configure_temporal_acceleration_options"))
                for name in ("first_pass", "second_pass", "filter_tracks", "save_checkpoint_before_ba"):
                    stack.enter_context(patch.object(gp, name, side_effect=lambda *a, n=name: events.append(n)))
                for name in ("sync_time", "record_timing", "log_memory"):
                    stack.enter_context(patch("vidmap.mapper.stages.global_positioning.positioner." + name, return_value=0))
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "consensus failed"):
                        gp.position()
                    self.assertEqual(events, ["first_pass", "consensus"])
                else:
                    gp.position()
                    self.assertEqual(events, ["first_pass"] + (["consensus"] if enabled else [])
                                     + ["second_pass", "filter_tracks", "save_checkpoint_before_ba"])

    def test_bridge_copies_poses_without_mutation(self):
        poses = np.column_stack((np.eye(3), [1., 2., 3.]))
        rec = MagicMock()
        rec.reg_image_ids.return_value = [5, 3]
        rec.images.__getitem__.return_value.name = "0000.000000.jpg"
        rec.images.__getitem__.return_value.cam_from_world.return_value.inverse.return_value.matrix.return_value = poses
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "vidmap/utils").mkdir(parents=True)
            (root / "vidmap/utils/consensus_gp1.py").touch()
            options = FloorplanGP1Options(enabled=True, zfloc_root=tmp, candidates_dir=tmp)
            def child(command, **kwargs):
                import json
                out = root / "floorplan_gp1"
                with np.load(out / "trajectory_gp1.npz") as data:
                    np.testing.assert_array_equal(data["camera_poses"][:, :3], np.tile(poses, (2, 1, 1)))
                    self.assertEqual(data["image_ids"].tolist(), [3, 5])
                (out / "summary.json").write_text(json.dumps({"scope": "whole_gp1_trajectory", "n_gp1_cameras": 2, "global": {"accepted": 1}}))
            before = poses.copy()
            with patch("vidmap.mapper.stages.floorplan.subprocess.run", side_effect=child):
                run_floorplan_gp1(options, reconstruction=rec, output_dir=root)
                with self.assertRaises(FileExistsError):
                    run_floorplan_gp1(options, reconstruction=rec, output_dir=root)
            np.testing.assert_array_equal(before, poses)


if __name__ == "__main__":
    unittest.main()
