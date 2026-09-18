"""CPU-only integration check: python -m unittest discover -s tests -p 'test_floorplan_shadow.py'."""

from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from vidmap.configuration.build import build_mapping_config
from vidmap.mapper.mapper import Mapper
from vidmap.mapper.options.mapper import FloorplanShadowOptions, MapperOptions
from vidmap.mapper.stages.floorplan import run_floorplan_shadow


class FloorplanShadowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / 'images'
        self.images.mkdir()
        (self.root / 'vidmap').mkdir()
        (self.root / 'vidmap/main.py').write_text('# pipeline entry\n')
        (self.root / 'engine').write_bytes(b'engine')
        self.names = [f'{i:011.6f}.jpg' for i in range(40)]
        for name in self.names:
            (self.images / name).write_bytes(name.encode())
        self.options = FloorplanShadowOptions(
            enabled=True, zfloc_root=str(self.root), raw_images=str(self.images),
            engine=str(self.root / 'engine'), f3loc=str(self.root), meta_dir=str(self.root),
        )
        self.ids = {i + 1: name for i, name in enumerate(self.names[:-1])}
        self.out = self.root / 'run'
        self.events = []

    def complete_pipeline(self, command, *, env, check):
        self.assertTrue(check)
        self.assertEqual(env['PYTHONUNBUFFERED'], '1')
        out = Path(command[command.index('--out') + 1])
        for directory in ('prep/frames_ts', 'reconstruction', 'consensus'):
            (out / directory).mkdir(parents=True, exist_ok=True)
        stamps = [self.options.start_us + i * 1_000_000 for i in range(40)]
        (out / 'prep/selection.json').write_text(json.dumps({
            'names': self.names, 'selected_us': stamps, 'start_us': self.options.start_us,
        }))
        for name in self.names:
            (out / 'prep/frames_ts' / name).write_bytes(name.encode())
        (out / 'reconstruction/windows.json').write_text(json.dumps({'windows': [self.names[i:i+self.options.window_size] for i in range(0,40,self.options.window_size)]}))
        (out / 'matching_summary.json').write_text(json.dumps({'records': [
            {'frame': name, 'timestamp_ns': stamp * 1000} for name, stamp in zip(self.names, stamps)
        ]}))
        (out / 'consensus/summary.json').write_text(json.dumps({
            'scope': f'independent_{self.options.window_size}_view_windows', 'cross_window_alignment': False,
            'prior_columns': ['x_m', 'y_m', 'yaw_deg'],
        }))
        for kind in ('local', 'global'):
            (out / f'consensus/{kind}_priors.npy').write_bytes(b'fixture')
        self.events.extend(['matching', 'consensus'])

    def solve(self, options, pipeline):
        mapper = Mapper.__new__(Mapper)
        mapper.conf = MapperOptions(floorplan_shadow=options)
        mapper.sfm_outputs_dir = self.out
        mapper.persist_intermediate_reconstructions = False
        mapper.mapper_inputs = SimpleNamespace(full_depth_maps_path=None, boundary_option=lambda _: False)
        rec = SimpleNamespace(images={i: SimpleNamespace(name=n) for i, n in self.ids.items()})
        inputs = SimpleNamespace(solve_state=SimpleNamespace(reconstruction=rec), vgc_exclusion_ids=set(),
                                 consecutive_pair_ids=[], sequence_id_to_index={}, focal_uncertainty=None)
        with ExitStack() as stack:
            classes = {}
            for name in ('ViewGraphFilter', 'ViewGraphCalibrator', 'GlobalPositioner', 'RelativePoseEstimator',
                         'RotationAverager', 'DepthConsistencyFilter', 'TrackBuilder', 'BundleAdjuster'):
                classes[name] = stack.enter_context(patch('vidmap.mapper.mapper.' + name))
            for name in ('sync_time', 'record_timing', 'log_memory'):
                stack.enter_context(patch('vidmap.mapper.mapper.' + name, return_value=0))
            stack.enter_context(patch('vidmap.reconstruction.local_input_image_dir', return_value=self.images))
            child = stack.enter_context(patch('vidmap.mapper.stages.floorplan.subprocess.run', side_effect=pipeline))
            classes['RotationAverager'].return_value.average.side_effect = lambda: self.events.append('ra')
            classes['GlobalPositioner'].return_value.position.side_effect = lambda: self.events.append('gp')
            classes['BundleAdjuster'].return_value.adjust.side_effect = lambda: self.events.append('ba')
            self.assertIs(mapper._solve(inputs, MagicMock(), None), rec)
            return child.call_count

    def test_enabled_runs_before_ra_and_records_ids(self):
        self.assertEqual(self.solve(self.options, self.complete_pipeline), 1)
        self.assertEqual(self.events, ['matching', 'consensus', 'ra', 'gp', 'ba'])
        report = json.loads((self.out / 'floorplan/vidmap_images.json').read_text())
        self.assertFalse(report['solver_fusion'])
        self.assertEqual(report['images'][0]['image_id'], 1)
        self.assertEqual(report['images'][0]['timestamp_ns'], self.options.start_us * 1000)
        self.assertIsNone(report['images'][-1]['image_id'])
        self.assertTrue(report['images'][-1]['excluded_by_frontend'])
        self.assertEqual(report['images'][-1]['window'], 1)

    def test_ten_view_windows(self):
        self.options = replace(self.options, window_size=10)
        self.assertEqual(self.solve(self.options, self.complete_pipeline), 1)
        report = json.loads((self.out / 'floorplan/vidmap_images.json').read_text())
        self.assertEqual([r['window'] for r in report['images']], [i//10 for i in range(40)])
        self.assertEqual(self.events, ['matching', 'consensus', 'ra', 'gp', 'ba'])

    def test_full_rate_vidmap_and_one_hz_zfloc(self):
        for i in range(40):
            name = f'{i + 0.5:011.6f}.jpg'
            (self.images / name).write_bytes(name.encode())
            self.ids[100 + i] = name
        self.assertEqual(self.solve(self.options, self.complete_pipeline), 1)
        self.assertEqual(self.events, ['matching', 'consensus', 'ra', 'gp', 'ba'])
        report = json.loads((self.out / 'floorplan/vidmap_images.json').read_text())
        self.assertEqual(len(report['images']), 40)
        self.assertEqual(len(list(self.images.iterdir())), 80)

    def test_missing_sample_stops_before_ra(self):
        (self.images / self.names[-1]).unlink()
        with self.assertRaisesRegex(ValueError, 'missing from VidMap'):
            self.solve(self.options, self.complete_pipeline)
        self.assertNotIn('ra', self.events)

    def test_disabled_does_not_run_zfloc(self):
        self.assertEqual(self.solve(FloorplanShadowOptions(), AssertionError('must not launch')), 0)
        self.assertEqual(self.events, ['ra', 'gp', 'ba'])
        self.assertFalse(self.out.exists())

    def test_child_failure_stops_before_ra(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.solve(self.options, subprocess.CalledProcessError(2, ['zfloc']))
        self.assertNotIn('ra', self.events)

    def test_mismatched_input_bytes_stop_before_ra(self):
        (self.images / self.names[0]).write_bytes(b'different image with same filename')
        with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
            self.solve(self.options, self.complete_pipeline)
        self.assertNotIn('ra', self.events)

    def test_mismatched_timestamps_stop_before_ra(self):
        def wrong_timestamp(*args, **kwargs):
            self.complete_pipeline(*args, **kwargs)
            path = self.out / 'floorplan/matching_summary.json'
            value = json.loads(path.read_text())
            value['records'][0]['timestamp_ns'] += 1
            path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'matching timestamps'):
            self.solve(self.options, wrong_timestamp)
        self.assertNotIn('ra', self.events)

    def test_existing_output_is_preserved(self):
        output = self.out / 'floorplan'
        output.mkdir(parents=True)
        sentinel = output / 'keep.txt'
        sentinel.write_text('keep')
        with self.assertRaises(FileExistsError):
            self.solve(self.options, AssertionError('must not launch'))
        self.assertEqual(sentinel.read_text(), 'keep')

    def test_config_path_and_validation(self):
        config = build_mapping_config(None, source_name='uncalib/base', override_tokens=[
            'mapper.floorplan_shadow.enabled=true',
            *[f'mapper.floorplan_shadow.{key}={getattr(self.options, key)}'
              for key in ('zfloc_root', 'raw_images', 'engine', 'f3loc', 'meta_dir')],
        ])
        self.assertTrue(config.pipeline.mapper.floorplan_shadow.enabled)
        with self.assertRaises(ValueError):
            FloorplanShadowOptions(enabled=True)
        for value in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                replace(self.options, trans_thresh=value)


if __name__ == '__main__':
    unittest.main()
