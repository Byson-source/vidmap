"""Actual native joint GP checks (no mocked solver)."""
import numpy as np
import vidmap_native._core as native


def scene():
    problem = native.MappingProblem()
    camera = native.CameraRecord()
    camera.camera_id, camera.model_id, camera.width, camera.height = 1, 1, 640, 480
    camera.params = np.array([500., 500., 320., 240.]); camera.has_prior_focal_length = True
    problem.add_camera(camera)
    points = np.array([[x,y,z] for x in (-1.,1.) for z in (4.,6.) for y in (-.5,0.,.5)])
    centers = np.array([[0.,0.,0.],[.6,0.,.2],[-.5,.2,.3],[.1,-.1,.7]])
    rng = np.random.default_rng(10)
    for i, center in enumerate(centers, 1):
        q = points-center
        image = native.ImageRecord(); image.image_id=i; image.camera_id=1; image.frame_id=i; image.name=f'{i:.6f}.jpg'
        image.keypoints = q[:,:2]/q[:,2:]*500 + [320,240]
        image.bearings = q/np.linalg.norm(q,axis=1)[:,None]
        image.angular_stddevs=np.full((len(q),2),.01)
        image.depth_values=q[:,2];image.depth_stddevs=np.full(len(q),.1)
        image.depth_validity=np.ones(len(q),np.uint8)
        pose=native.PoseRecord();pose.has_pose=True;pose.translation=-center-rng.normal(0,.15,3)
        image.pose=pose;problem.add_image(image)
    for k,xyz in enumerate(points,1):
        track=native.TrackRecord();track.point3D_id=k;track.xyz=xyz+rng.normal(0,.2,3)
        track.observations=np.array([[i,k-1] for i in range(1,5)],np.uint32)
        problem.add_track(track)
    return problem,points,centers


def solve(*,walls=True,bad=False,width=2.,weight=1.,anchor_sigma=0.):
    problem,points,centers=scene()
    o=native.GlobalPositioningOptions();o.use_initial_positions=True;o.generate_scales=False
    o.min_num_views_per_track=2;o.num_threads=1;o.random_seed=1;o.max_num_iterations=100
    o.use_metric_depth_constraint=True;o.scale_prior_stddev=.05
    o.use_log_depth_map_scales=False
    priors=[]
    if walls:
        for k,p in enumerate(points,1):
            wall=native.FloorplanWallPrior();wall.point3D_id=k
            if k%2:
                wall.start=np.array([p[0],3.]);wall.end=np.array([p[0],7.])
            else:
                wall.start=np.array([-2.,p[2]]);wall.end=np.array([2.,p[2]])
            if bad == 'correlated' or (bad and k==1):
                wall.start=wall.start+np.array([30.,0.]);wall.end=wall.end+np.array([30.,0.])
            priors.append(wall)
    o.floorplan_anchor_sigma=anchor_sigma
    o.floorplan_anchor_image_id=1
    o.floorplan_anchor_reference=np.array([0.,123.,0.])  # Height is deliberately unrelated.
    o.floorplan_wall_priors=priors
    loss=native.LossConfig();loss.type=native.LossFunctionType.HUBER;loss.scale=width;loss.weight=weight
    o.floorplan_loss=loss
    result=native.run_global_positioning(o,problem)
    assert result.success
    solved=np.array([problem.track(i).xyz for i in range(1,len(points)+1)])
    cam=np.array([-problem.image(i).pose.translation for i in range(1,5)])
    return result,solved,cam,points,centers


def test_native_wall_joint_and_outlier():
    clean=solve();robust=solve(bad=True);quadratic=solve(bad=True,width=1e6)
    assert clean[0].diagnostics.num_floorplan_wall_residuals==12
    assert set(clean[0].final_residual_costs)=={'bearing','depth','scale','wall'}
    np.testing.assert_allclose(sum(clean[0].final_residual_costs.values()),clean[0].diagnostics.final_cost,atol=1e-8)
    clean_error=np.linalg.norm(clean[1][:,[0,2]]-clean[3][:,[0,2]])
    robust_error=np.linalg.norm(robust[2][:,[0,2]]-robust[4][:,[0,2]])
    quadratic_error=np.linalg.norm(quadratic[2][:,[0,2]]-quadratic[4][:,[0,2]])
    assert clean_error < .02,clean_error
    assert robust_error < quadratic_error*.5,(robust_error,quadratic_error)
    assert np.linalg.norm(clean[2][:,[0,2]]-clean[4][:,[0,2]])<.02
    print('joint clean XY point error',clean_error,'camera corrupted robust/quadratic',robust_error,quadratic_error)


def test_native_zero_weight_and_validation():
    disabled=solve(walls=False);zero=solve(weight=0.)
    np.testing.assert_allclose(disabled[1],zero[1],rtol=0,atol=1e-10);np.testing.assert_allclose(disabled[2],zero[2],rtol=0,atol=1e-10)
    problem,_,_=scene();o=native.GlobalPositioningOptions();wall=native.FloorplanWallPrior();wall.point3D_id=1
    o.floorplan_wall_priors=[wall,wall]
    with np.testing.assert_raises(ValueError):native.run_global_positioning(o,problem)
    o.floorplan_wall_priors=[wall];loss=native.LossConfig();o.floorplan_loss=loss
    with np.testing.assert_raises(ValueError):native.run_global_positioning(o,problem)


def test_correlated_wrong_floorplan_can_move_the_whole_scene():
    result, points, centers, truth, truth_centers = solve(bad="correlated")
    assert np.mean(centers[:,0]-truth_centers[:,0])>25
    np.testing.assert_allclose(np.diff(centers,axis=0),np.diff(truth_centers,axis=0),atol=.02)


def test_soft_initial_position_sigma_and_disabled_regression():
    weak=solve(bad='correlated',anchor_sigma=10.)
    strong=solve(bad='correlated',anchor_sigma=.01)
    movement=lambda x: np.linalg.norm(x[2][0,[0,2]])
    assert movement(weak)>10
    assert 1e-8<movement(strong)<.02, movement(strong)
    np.testing.assert_allclose(weak[2][0,1],strong[2][0,1],atol=1e-10)
    assert 'initial_position' in strong[0].final_residual_costs
    expected=.5*movement(strong)**2/.01**2
    np.testing.assert_allclose(strong[0].final_residual_costs['initial_position'],expected,rtol=1e-8)
    baseline=solve(walls=False)
    for disabled in (solve(walls=False,anchor_sigma=.01),solve(weight=0.,anchor_sigma=.01)):
        np.testing.assert_allclose(disabled[2],baseline[2],rtol=0,atol=1e-10)
    for sigma in (-1.,float('nan'),float('inf')):
        with np.testing.assert_raises(ValueError):solve(anchor_sigma=sigma)
    problem,_,_=scene();options=native.GlobalPositioningOptions()
    wall=native.FloorplanWallPrior();wall.point3D_id=1
    options.floorplan_wall_priors=[wall];options.floorplan_anchor_sigma=1.
    options.floorplan_anchor_image_id=999
    with np.testing.assert_raises(ValueError):native.run_global_positioning(options,problem)
    options.floorplan_anchor_image_id=1;options.floorplan_anchor_reference=np.full(3,np.nan)
    with np.testing.assert_raises(ValueError):native.run_global_positioning(options,problem)
    print('soft initial position weak/strong displacement',movement(weak),movement(strong))
