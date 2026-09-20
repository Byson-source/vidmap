from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from vidmap.configuration.build import build_mapping_config
from vidmap.mapper.options.positioning import GPOptions, GPFloorplanOptions
from vidmap.mapper.stages.global_positioning.floorplan import warmup_floorplan, prepare_floorplan, prepare_initial_position_prior


class WallOptionsTest(unittest.TestCase):
    def test_config_validation_and_zero_noop(self):
        self.assertFalse(GPFloorplanOptions().active)
        self.assertFalse(GPFloorplanOptions(enabled=True,first_pass_weight=0.,second_pass_weight=0.).active)
        for kw in ({'enabled':True},{'sigma_m':0.},{'huber_m':float('nan')},{'first_pass_weight':-1.}):
            with self.assertRaises(ValueError):GPFloorplanOptions(**kw)
        config=build_mapping_config(None,source_name='uncalib/base',override_tokens=[
            'mapper.gp.floorplan.enabled=true','mapper.gp.floorplan.zfloc_root=/tmp/zfloc',
            'mapper.gp.floorplan.candidates_dir=/tmp/candidates'])
        self.assertTrue(config.pipeline.mapper.gp.floorplan.active)
        gp=SimpleNamespace(options=GPOptions())
        options=SimpleNamespace(floorplan_wall_priors=['old'])
        prepare_floorplan(gp,options,'gp2')
        self.assertEqual(options.floorplan_wall_priors,[])

    def test_initial_position_reference_retained_across_passes(self):
        import numpy as np
        camera=MagicMock(name='first_camera')
        camera.name='0000.000000.jpg'
        camera.projection_center.return_value=np.array([1.,2.,3.])
        rec=SimpleNamespace(reg_image_ids=lambda:[7],images={7:camera})
        gp=SimpleNamespace(options=GPOptions(floorplan=GPFloorplanOptions(initial_position_sigma_m=.5)),reconstruction=rec)
        options=SimpleNamespace()
        prepare_initial_position_prior(gp,options)
        camera.projection_center.return_value=np.array([4.,5.,6.])
        prepare_initial_position_prior(gp,options)
        self.assertEqual(options.floorplan_anchor_image_id,7)
        np.testing.assert_array_equal(options.floorplan_anchor_reference,[1.,2.,3.])
        self.assertEqual(options.floorplan_anchor_sigma,.5)
        for sigma in (0.,-1.,float('inf'),float('nan')):
            with self.assertRaises(ValueError):GPFloorplanOptions(initial_position_sigma_m=sigma)
        config=build_mapping_config(None,source_name='uncalib/base',override_tokens=[
            'mapper.gp.floorplan.initial_position_sigma_m=0.5'])
        self.assertEqual(config.pipeline.mapper.gp.floorplan.initial_position_sigma_m,.5)

    def test_warmup_keeps_gp1_loss_budget_and_pose(self):
        gp=MagicMock()
        options=SimpleNamespace(max_num_iterations=100,sequential_support_warmup_rounds=16,
             sequential_support_observations_per_track=16,sequential_support_image_timeline=[1,2],
             loss_normal_depth=object(),loss_normal_geometry=object())
        depth,geometry=options.loss_normal_depth,options.loss_normal_geometry
        result=SimpleNamespace(depth_map_scales={1:1.2})
        with patch('vidmap.mapper.stages.global_positioning.floorplan.native', new=MagicMock()) as native, \
             patch('vidmap.mapper.stages.global_positioning.floorplan.prepare_floorplan') as prepare:
            def solve(opts,problem):
                self.assertEqual(opts.max_num_iterations,1)
                self.assertEqual(opts.sequential_support_warmup_rounds,16)
                return result
            native.run_global_positioning.side_effect=solve
            warmup_floorplan(gp,options)
            prepare.assert_called_once_with(gp,options,'gp1')
        self.assertEqual(options.max_num_iterations,99)
        self.assertTrue(options.use_initial_positions)
        self.assertFalse(options.generate_scales)
        self.assertEqual(options.initial_depth_map_scales,{1:1.2})
        self.assertEqual(options.sequential_support_warmup_rounds,0)
        self.assertIs(options.loss_normal_depth,depth)
        self.assertIs(options.loss_normal_geometry,geometry)
        gp.playback_trace.attach_global_positioning.assert_called_once_with(options,'gp1')


if __name__=='__main__':unittest.main()
