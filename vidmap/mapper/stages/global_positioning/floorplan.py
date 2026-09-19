"""Offline wall association bridge; native GP owns the actual residuals."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from vidmap.mapper.native.extension import native

logger = logging.getLogger(__name__)


def export_state(reconstruction, path, candidate_names):
    images = sorted(reconstruction.reg_image_ids(), key=lambda i: reconstruction.images[i].name)
    poses = np.tile(np.eye(4), (len(images), 1, 1))
    names, sizes, observation_images, observation_tracks, xy = [], [], [], [], []
    for index, image_id in enumerate(images):
        image = reconstruction.images[image_id]
        poses[index, :3] = image.cam_from_world().inverse().matrix()
        names.append(image.name)
        camera = reconstruction.cameras[image.camera_id]
        sizes.append([camera.width, camera.height])
        if image.name in candidate_names:
            for point in image.points2D:
                if point.has_point3D():
                    observation_images.append(index)
                    observation_tracks.append(point.point3D_id)
                    xy.append(point.xy)
    ids = sorted(reconstruction.point3D_ids())
    # Avoid zlib ABI collisions in the native SfM interpreter.
    np.savez(path, names=names, image_ids=images, camera_poses=poses, camera_sizes=sizes,
             observation_image_indices=np.asarray(observation_images, dtype=np.int64),
             observation_track_ids=np.asarray(observation_tracks, dtype=np.uint64),
             observation_xy=np.asarray(xy, dtype=float).reshape(-1, 2),
             point_ids=np.asarray(ids, dtype=np.uint64),
             point_xyz=np.asarray([reconstruction.points3D[i].xyz for i in ids]))


def prepare_floorplan(positioner, native_options, stage):
    options = positioner.options.floorplan
    native_options.floorplan_wall_priors = []  # replace, never append GP1's factors
    weight = options.first_pass_weight if stage == 'gp1' else options.second_pass_weight
    if not options.active or weight == 0:
        return
    candidates = Path(options.candidates_dir).expanduser()
    script = Path(options.zfloc_root).expanduser() / 'vidmap/utils/wall_factors.py'
    if not script.is_file():
        raise FileNotFoundError(script)
    summary = json.loads((candidates / 'matching_summary.json').read_text())
    root = positioner.output_dir / 'floorplan_wall'
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / f'{stage}_input.npz'
    export_state(positioner.reconstruction, state_path, {r['frame'] for r in summary['records']})
    out = root / stage
    command = [options.python_executable or sys.executable, str(script), '--state', str(state_path),
               '--candidates-dir', str(candidates), '--out', str(out), '--stage', stage,
               '--sigma', str(options.sigma_m), '--huber', str(options.huber_m), '--weight', str(weight),
               '--threshold', str(options.trans_thresh_m), '--max-gap', str(options.max_interpolation_gap_s),
               '--association-m', str(options.association_m)]
    consensus = positioner.output_dir / 'floorplan_gp1'
    if stage == 'gp2' and (consensus / 'summary.json').is_file():
        command += ['--consensus', str(consensus)]
    (root / f'{stage}_command.json').write_text(json.dumps(command, indent=2) + '\n')
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    subprocess.run(command, env=env, check=True)
    report = json.loads((out / 'factors.json').read_text())
    if report['stage'] != stage or report['trajectory_gt_used']:
        raise ValueError('Invalid floorplan factor provenance')
    if not report['factors']:
        logger.warning('Floorplan %s abstained: %s', stage, report['reason'])
        return
    priors = []
    for row in report['factors']:
        prior = native.FloorplanWallPrior()
        prior.point3D_id = row['point3D_id']
        prior.start, prior.end, prior.weight = row['start'], row['end'], row['weight']
        priors.append(prior)
    native_options.floorplan_wall_priors = priors
    native_options.floorplan_projection = np.asarray(report['projection'])
    native_options.floorplan_offset = np.asarray(report['offset'])
    native_options.floorplan_sigma = options.sigma_m
    loss = native.LossConfig()
    loss.type = native.LossFunctionType.HUBER
    loss.scale, loss.weight = options.huber_m / options.sigma_m, weight
    native_options.floorplan_loss = loss
    logger.info('Floorplan %s: %d unique wall tracks, Huber=%g m, sigma=%g m',
                stage, len(priors), options.huber_m, options.sigma_m)


def warmup_floorplan(positioner, native_options):
    budget = native_options.max_num_iterations
    if budget < 2:
        raise ValueError('GP1 floorplan requires at least two normal iterations for warm-up and fusion')
    native_options.max_num_iterations = 1
    if positioner.playback_trace is not None:
        positioner.playback_trace.attach_global_positioning(native_options, 'gp1')
    result = native.run_global_positioning(native_options, positioner.solve_state.native_problem)
    positioner.export_solved_scene()
    positioner.save_pass_result('gp_warmup', result)
    positioner.require_success(result, 'Floorplan visual warm-up')
    native_options.max_num_iterations = budget - 1
    native_options.use_initial_positions = True
    native_options.generate_scales = False
    native_options.initial_depth_map_scales = result.depth_map_scales
    native_options.sequential_support_warmup_rounds = 0
    native_options.sequential_support_observations_per_track = 0
    native_options.sequential_support_image_timeline = []
    prepare_floorplan(positioner, native_options, 'gp1')


def save_floorplan_result(positioner, stage, result):
    root = positioner.output_dir / 'floorplan_wall'
    path = root / stage / 'factors.json'
    if not path.is_file():
        return
    report = json.loads(path.read_text())
    rows = report['factors']
    diagnostics = {'success': bool(result.success),
                   'native_factor_count': result.diagnostics.num_floorplan_wall_residuals,
                   'factors': []}
    if rows:
        with np.load(root / f'{stage}_input.npz', allow_pickle=False) as data:
            initial = dict(zip(map(int, data['point_ids']), data['point_xyz']))
        projection, offset = np.asarray(report['projection']), np.asarray(report['offset'])
        for row in rows:
            point_id = row['point3D_id']
            a, b = np.asarray(row['start']), np.asarray(row['end'])
            direction = b - a
            distances = []
            for xyz in (initial[point_id], positioner.reconstruction.points3D[point_id].xyz):
                delta = projection @ xyz + offset - a
                t = np.clip(delta @ direction / (direction @ direction), 0, 1)
                distances.append(float(np.linalg.norm(delta - t * direction)))
            diagnostics['factors'].append({'point3D_id': point_id, 'wall_id': row['wall_id'],
                'initial_m': distances[0], 'final_m': distances[1],
                'huber_derivative': min(1., report['huber_m'] / max(distances[1], 1e-30)),
                'weight': row['weight'] * report['loss_weight']})
    if 'projection' in report:
        with np.load(root / f'{stage}_input.npz', allow_pickle=False) as data:
            names, image_ids = data['names'], data['image_ids']
            initial_centers = data['camera_poses'][:, :3, 3]
        centers = np.array([positioner.reconstruction.images[int(i)].projection_center() for i in image_ids])
        projection, offset = np.asarray(report['projection']), np.asarray(report['offset'])
        trajectory = path.parent / 'trajectory_floorplan.npz'
        np.savez(trajectory, names=names, image_ids=image_ids, camera_centers_world=centers,
                 initial_centers_world=initial_centers, floorplan_xy=centers @ projection.T + offset,
                 initial_floorplan_xy=initial_centers @ projection.T + offset,
                 projection=projection, offset=offset)
        options = positioner.options.floorplan
        command = [options.python_executable or sys.executable,
                   str(Path(options.zfloc_root) / 'vidmap/utils/wall_factors.py'),
                   '--state', str(trajectory), '--candidates-dir', options.candidates_dir,
                   '--out', str(path.parent), '--stage', stage, '--plot-result']
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        subprocess.run(command, env=env, check=True)
    (path.parent / 'result.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
