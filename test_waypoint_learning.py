"""Numerical and semantic regressions for the multi-demonstration model."""
import copy
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from waypoint_learning import (ANCHOR_FRAME, Config, combine, detect_events, dtw_path, fit_gmm,
    fit_tpgmm, fuse_world, gmr, learn, load_model, predict_segment, predict_world, relative_pose,
    save_model, select_reference, select_references, tolerance, tpgmr, world_pose)
from waypoint_replay import reconstruct_fused, reconstruct_target
from waypoint_validate import evaluate


def demo(shift=0.,bend=0.,speed=1.):
    count=int(61*speed);t=np.linspace(0,1,count)
    keys=np.array([[0,-.12,.2],[0,0,.10],[0,0,.10],[0,0,.25],[.2,0,.25],[.2,0,.1],[.2,0,.16]])
    p=np.array([np.interp(t,np.linspace(0,1,len(keys)),keys[:,j]) for j in range(3)]).T
    p[:,0]+=shift;p[:,1]+=bend*np.sin(np.pi*t)
    poses={'source':[[shift,0,0],np.eye(3).tolist()],'destination':[[.2+shift,0,0],np.eye(3).tolist()]}
    ss=[]
    for i,tt in enumerate(t):
        held=.32<=tt<.84;closing=.27<=tt<.8
        current=copy.deepcopy(poses)
        if held:current['source'][0][2]=max(0,float(p[i,2]-.1))
        ss.append(dict(position=p[i].tolist(),orientation=Rotation.from_rotvec([0,0,.15*np.sin(tt*np.pi)]).as_matrix().tolist(),
                       timestamp=i*.05,gripper='close' if closing else 'open',grasped=['source'] if held else [],
                       object_poses=current,contact_state={'source':dict(left=True,right=held,both=held,kind='object')} if .30<=tt<.84 else {},
                       task_success=tt>=.98))
    return dict(version=2,suite='synthetic',task_id=0,language='place source on destination',instruction='place source on destination',
                initial_gripper='open',poses=poses,initial_object_poses=poses,objects=['source','destination'],fixtures=[],
                goals=[['on','source','destination']],samples=ss[:-1],terminal_sample=ss[-1],
                end_position=ss[-1]['position'],end_orientation=ss[-1]['orientation'],success=True)


class LearningTest(unittest.TestCase):
    def config(self):return Config(samples_per_interval=8,components=2,max_waypoints=200)

    def test_frame_and_orientation_roundtrip(self):
        rng=np.random.default_rng(1);p=rng.normal(size=(12,3));r=Rotation.random(12,random_state=rng).as_matrix()
        frame=(rng.normal(size=3),Rotation.from_rotvec([.4,.7,-.2]).as_matrix())
        pp,rr=relative_pose(p,r,frame);a,b=world_pose(pp,rr,frame)
        np.testing.assert_allclose(a,p,atol=1e-12);np.testing.assert_allclose(b,r,atol=1e-12)
        np.testing.assert_allclose(rr[0],frame[1].T@r[0])

    def test_events_and_semantic_references(self):
        d=demo();ev=detect_events(d);kinds=[e['kind'] for e in ev]
        for kind in ('close','contact','stable_grasp','lift','open','release','success'):self.assertIn(kind,kinds)
        refs=[select_reference(d,ev,j) for j in range(len(ev)-1)]
        self.assertEqual(refs[0]['name'],'source');self.assertIn('destination',[r['name'] for r in refs])
        d['goals']=[];d['samples']=[dict(s,grasped=[],contact_state={}) for s in d['samples']];d['terminal_sample']=dict(d['terminal_sample'],grasped=[],contact_state={})
        ev=detect_events(d)
        with self.assertRaisesRegex(ValueError,'Ambiguous'):select_reference(d,ev,0)
        self.assertEqual(select_reference(d,ev,0,{'target':'source'})['name'],'source')

    def test_dtw_endpoints_and_monotonicity(self):
        t=np.linspace(0,1,30);p=np.c_[t,t*0,t*0];q=np.c_[t**1.4,t*0,t*0];r=np.tile(np.eye(3),(30,1,1));g=(t>.5).astype(float)
        path=dtw_path(p,r,g,q,r,g)
        np.testing.assert_array_equal(path[0],[0,0]);np.testing.assert_array_equal(path[-1],[29,29]);self.assertTrue(np.all(np.diff(path,axis=0)>=0))
        self.assertLess(np.mean(abs(p[path[:,0],0]-q[path[:,1],0])),np.mean(abs(t-t**1.4)))

    def test_gmm_conditional_moments(self):
        rng=np.random.default_rng(4);t=np.linspace(0,1,500);y=2*t+rng.normal(0,.03,len(t));m=fit_gmm(np.c_[t,y],3)
        mu,cov=gmr(m,[.2,.5,.8]);np.testing.assert_allclose(mu[:,0],[.4,1.,1.6],atol=.04)
        self.assertTrue(np.all(cov[:,0,0]>0));self.assertLess(cov[:,0,0].max(),.005)

    def test_covariance_and_event_preservation(self):
        c=self.config();same=learn([demo(),demo(),demo()],c)
        noisy=learn([demo(bend=-.02),demo(),demo(bend=.02)],c)
        sig=lambda m:np.mean([np.trace(np.array(s['covariance'])[:,:3,:3],axis1=1,axis2=2).mean() for s in m['segments']])
        self.assertLess(sig(same),1e-7);self.assertGreater(sig(noisy),sig(same)*10)
        self.assertGreater(np.mean([w['position_tolerance'] for w in noisy['waypoints']]),np.mean([w['position_tolerance'] for w in same['waypoints']]))
        selected={w['semantic_event']['key'] for w in noisy['waypoints'] if w['semantic_event']}
        self.assertEqual(selected,{e['key'] for e in noisy['events']})
        for seg in same['segments']:
            p=np.array(seg['aligned_position']);np.testing.assert_allclose(p[0,[0,-1]],p[1,[0,-1]])
        with self.assertRaisesRegex(ValueError,'safety bound'):learn([demo(),demo()],Config(max_waypoints=2))

    def test_object_shift_invariance_and_reconstruction(self):
        model=learn([demo(),demo(.1)],self.config());self.assertEqual(len(model['used_demonstrations']),2)
        node=model['waypoints'][0];a=reconstruct_target(node,(np.zeros(3),np.eye(3)))
        rotation=Rotation.from_rotvec([0,0,.8]).as_matrix();translation=np.array([.2,.1,.3])
        b=reconstruct_target(node,(translation,rotation))
        np.testing.assert_allclose(b[0],rotation@a[0]+translation);np.testing.assert_allclose(b[1],rotation@a[1])

    def test_continuous_gmr_and_orientation_wrap(self):
        first,second=demo(),demo()
        for d,angle in ((first,np.pi-.02),(second,-np.pi+.02)):
            for sample in d['samples']+[d['terminal_sample']]:
                sample['orientation']=Rotation.from_rotvec([0,0,angle]).as_matrix().tolist()
        model=learn([first,second],self.config());seg=model['segments'][0]
        prediction=predict_segment(seg,np.linspace(0,1,8))
        np.testing.assert_allclose(prediction['position'],seg['position'],atol=1e-10)
        angles=Rotation.from_matrix(prediction['orientation']).magnitude()
        self.assertTrue(np.all(angles>3.1))
        self.assertTrue(np.all(np.linalg.eigvalsh(prediction['covariance'])>=-1e-10))
        with self.assertRaisesRegex(ValueError,'phase'):predict_segment(seg,[-.1])

    def test_temporal_variation_keeps_hard_anchors(self):
        model=learn([demo(speed=.85),demo(speed=1.2)],self.config())
        self.assertEqual(len(model['used_demonstrations']),2)
        for seg in model['segments']:
            for warp in seg['warps']:
                self.assertEqual(warp[0],0);self.assertEqual(warp[-1],7)
        self.assertGreater(max(model['event_episode_fraction_std']),0)

    def test_invalid_data_and_trivial_success(self):
        bad=demo();bad['samples'][3]['position'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):learn([demo(),bad],self.config())
        tiny=demo();tiny['samples']=tiny['samples'][:1]
        model=learn([demo(),demo(),tiny],self.config())
        self.assertEqual(model['excluded_demonstrations'][0]['reason'],'trivial/already-solved episode')

    def test_serialization(self):
        model=learn([demo(),demo(.1)],self.config())
        with tempfile.TemporaryDirectory() as folder:
            save_model(model,folder);self.assertEqual(load_model(folder),model)

    def test_heldout_and_incompatible_modes(self):
        demos=[demo(bend=b) for b in (-.01,0,.01)]
        result=evaluate(demos,self.config());self.assertEqual(result['evaluated_folds'],3,result)
        for fold in result['folds']:
            self.assertTrue(fold['gripper_event_consistency']);self.assertIn('mean_relative_position_error_m',fold['single_demonstration'])
        # Longer than gripper_debounce, so this is a real extra grasp, not a blip.
        broken=demo()
        for i in range(5,5+Config().gripper_debounce+2):broken['samples'][i]['gripper']='close'
        model=learn(demos+[broken],self.config());self.assertEqual(len(model['excluded_demonstrations']),1)

    def test_articulation_without_grasp(self):
        d=demo();d['objects']=[];d['fixtures']=['cabinet'];d['goals']=[['open','handle']]
        d['poses']['handle']=[[0,0,0],np.eye(3).tolist()];d['joints']={'hinge':{}}
        d['poses']['hinge']=[[0,0,0],np.eye(3).tolist()]
        for i,s in enumerate(d['samples']+[d['terminal_sample']]):
            s['object_poses']=copy.deepcopy(d['poses']);s['grasped']=[]
            s['contact_state']={'hinge':dict(kind='fixture_link')} if 10<=i<55 else {}
            s['joint_positions']={'hinge':max(0,(i-15)*.01)};s['goal_predicates']=[i>=50]
        events=detect_events(d);kinds=[e['kind'] for e in events]
        for kind in ('fixture_contact','joint_motion','goal'):self.assertIn(kind,kinds)
        ref=select_reference(d,events,0);self.assertEqual(ref['name'],'hinge');self.assertEqual(ref['policy'],'interval_start')


def placed(source=(0.,0.),destination=(.2,0.),bend=0.,speed=1.):
    """Pick and place with independently positioned source and destination.

    The robot always starts at the same home pose, so no single object frame can
    explain both the start of the motion and its end.
    """
    count=int(61*speed);t=np.linspace(0,1,count);sx,sy=source;dx,dy=destination
    keys=np.array([[0,-.12,.2],[sx,sy,.10],[sx,sy,.10],[sx,sy,.25],[dx,dy,.25],[dx,dy,.1],[dx,dy,.16]])
    p=np.array([np.interp(t,np.linspace(0,1,len(keys)),keys[:,j]) for j in range(3)]).T
    p[:,1]+=bend*np.sin(np.pi*t)
    poses={'source':[[sx,sy,0],np.eye(3).tolist()],'destination':[[dx,dy,0],np.eye(3).tolist()]}
    ss=[]
    for i,tt in enumerate(t):
        held=.32<=tt<.84;closing=.27<=tt<.8
        current=copy.deepcopy(poses)
        if held:current['source'][0]=[float(p[i,0]),float(p[i,1]),max(0.,float(p[i,2]-.1))]
        ss.append(dict(position=p[i].tolist(),orientation=Rotation.from_rotvec([0,0,.15*np.sin(tt*np.pi)]).as_matrix().tolist(),
                       timestamp=i*.05,gripper='close' if closing else 'open',grasped=['source'] if held else [],
                       object_poses=current,contact_state={'source':dict(left=True,right=held,both=held,kind='object')} if .30<=tt<.84 else {},
                       task_success=tt>=.98))
    return dict(version=2,suite='synthetic',task_id=0,language='place source on destination',instruction='place source on destination',
                initial_gripper='open',poses=poses,initial_object_poses=poses,objects=['source','destination'],fixtures=[],
                goals=[['on','source','destination']],samples=ss[:-1],terminal_sample=ss[-1],
                end_position=ss[-1]['position'],end_orientation=ss[-1]['orientation'],success=True)


def rebuild(model,layout,home):
    """Reconstruct every waypoint in world coordinates for an unseen object layout.

    Mirrors waypoint_replay: the single task frame supplies the offset and the
    reached previous waypoint supplies the connector start.
    """
    poses={name:[list(value),np.eye(3)] for name,value in layout.items()}
    out=[];anchor=[np.asarray(home,dtype=float),np.eye(3)];segment=None;last=anchor
    for node in model['waypoints']:
        if node['segment']!=segment:segment,anchor=node['segment'],last
        frames=[anchor if b['name']==ANCHOR_FRAME else poses[b['name']] for b in node['frames']]
        p,r=reconstruct_fused(node,frames,anchor);out.append((node,p,r));last=[p,r]
    return out


class TaskParameterizedTest(unittest.TestCase):
    def config(self):return Config(samples_per_interval=8,components=2,max_waypoints=200)

    def demos(self):
        return [placed(source=(0.,0.),destination=(.2,0.)),
                placed(source=(.06,.05),destination=(.16,-.06),bend=.01),
                placed(source=(-.05,-.04),destination=(.26,.05),bend=-.01)]

    def test_transient_gripper_toggles_do_not_fragment_signatures(self):
        from waypoint_learning import event_signature, gripper_states
        clean=placed();reference=event_signature(detect_events(clean))
        span=range(5,5+Config().gripper_debounce-1)
        self.assertTrue(all(placed()['samples'][i]['gripper']=='open' for i in span),'blip must start from open')
        blipped=placed()
        for i in span:blipped['samples'][i]['gripper']='close'
        self.assertEqual(event_signature(detect_events(blipped)),reference,
                         'a toggle shorter than the debounce must not create events')
        self.assertNotIn('close',gripper_states(blipped)[span.start:span.stop])
        # A toggle at or beyond the debounce is a real event and must still register.
        real=placed()
        for i in range(5,5+Config().gripper_debounce+2):real['samples'][i]['gripper']='close'
        self.assertNotEqual(event_signature(detect_events(real)),reference)

    def test_short_contact_dropout_does_not_create_a_new_interaction(self):
        from waypoint_learning import event_signature
        clean = placed()
        reference = event_signature(detect_events(clean))
        short = copy.deepcopy(clean)
        for i in (40, 41):
            self.assertIn('source', short['samples'][i]['contact_state'])
            short['samples'][i]['contact_state'] = {}
        self.assertEqual(event_signature(detect_events(short)), reference)
        long = copy.deepcopy(clean)
        for i in range(40, 44):
            long['samples'][i]['contact_state'] = {}
        self.assertNotEqual(event_signature(detect_events(long)), reference)

    def test_each_interval_uses_one_semantic_reference(self):
        d=self.demos()[0];ev=detect_events(d)
        refs=[select_references(d,ev,j) for j in range(len(ev)-1)]
        self.assertTrue(all(len(r)==1 for r in refs))
        self.assertTrue(any(r[0]['role']=='target' for r in refs))
        self.assertTrue(any(r[0]['role']=='destination' for r in refs))
        self.assertFalse(any(r[0]['name']==ANCHOR_FRAME for r in refs))

    def test_shared_responsibilities_and_block_widths(self):
        rng=np.random.default_rng(0);t=np.linspace(0,1,200)
        a=np.c_[t,2*t+rng.normal(0,.01,len(t))];b=np.c_[t,-3*t+rng.normal(0,.01,len(t))]
        mixture=fit_tpgmm([a,b,np.c_[t,t*0]],3)
        self.assertEqual(len(mixture['frames']),3)
        np.testing.assert_allclose(sum(mixture['weights']),1,atol=1e-9)
        # One shared phase column: every block agrees on where each component sits.
        for block in mixture['frames'][1:]:
            np.testing.assert_allclose(np.array(block['means'])[:,0],np.array(mixture['frames'][0]['means'])[:,0],atol=1e-9)
        means,_=tpgmr(mixture,[.25,.75])
        np.testing.assert_allclose(means[0][:,0],[.5,1.5],atol=.05)
        np.testing.assert_allclose(means[1][:,0],[-.75,-2.25],atol=.05)

    def test_product_of_gaussians_follows_precision(self):
        tight=np.eye(3)*1e-8;loose=np.eye(3)*1.
        mean,_=combine([np.zeros(3),np.ones(3)],[tight,loose],[tight,loose])
        np.testing.assert_allclose(mean,np.zeros(3),atol=1e-6)
        mean,_=combine([np.zeros(3),np.ones(3)],[loose,tight],[loose,tight])
        np.testing.assert_allclose(mean,np.ones(3),atol=1e-6)
        # Equal precision is an even blend.
        mean,_=combine([np.zeros(3),np.ones(3)],[loose,loose],[loose,loose])
        np.testing.assert_allclose(mean,np.full(3,.5),atol=1e-6)

    def test_start_pose_is_connected_to_the_previous_waypoint(self):
        model=learn(self.demos(),self.config());home=[0,-.12,.2]
        far={'source':[.30,.22,0],'destination':[-.18,-.25,0]}
        node,p,_=rebuild(model,far,home)[0]
        self.assertLess(np.linalg.norm(p-np.array(home)),.01)
        # The single-frame reconstruction follows the object instead of the robot.
        single,_=reconstruct_target(node,[np.array(far[node['reference']['name']],dtype=float),np.eye(3)])
        self.assertGreater(np.linalg.norm(single-np.array(home)),.05)

    def test_shuffled_layout_reaches_both_objects(self):
        model=learn(self.demos(),self.config());home=[0,-.12,.2]
        for layout in ({'source':[.30,.22,0],'destination':[-.18,-.25,0]},
                       {'source':[-.22,.18,0],'destination':[.28,-.10,0]}):
            built=rebuild(model,layout,home)
            grasp=[p for node,p,_ in built if node['semantic_phase'] in ('close','contact','stable_grasp')]
            place=[p for node,p,_ in built if node['semantic_phase'] in ('open','release')]
            self.assertTrue(grasp and place)
            near_source=min(np.linalg.norm(np.array(p[:2])-np.array(layout['source'][:2])) for p in grasp)
            near_destination=min(np.linalg.norm(np.array(p[:2])-np.array(layout['destination'][:2])) for p in place)
            self.assertLess(near_source,.03,'grasp must track the shuffled source')
            self.assertLess(near_destination,.03,'release must track the shuffled destination')

    def test_dense_world_prediction_matches_waypoints(self):
        model=learn(self.demos(),self.config())
        segment=model['segments'][0];frames=segment['frame_poses']
        dense=predict_world(segment,[0.,1.],frames)
        np.testing.assert_allclose(dense['position'][0],np.array(segment['world_position'])[0],atol=2e-3)
        np.testing.assert_allclose(dense['position'][1],np.array(segment['world_position'])[-1],atol=2e-3)
        self.assertEqual(dense['orientation'].shape,(2,3,3))

    def test_model_round_trip_and_version(self):
        model=learn(self.demos(),self.config())
        self.assertEqual((model['version'],model['model_type']),(2,'task_parameterized_gmm_gmr'))
        with tempfile.TemporaryDirectory() as folder:
            save_model(model,folder);again=load_model(folder)
        self.assertEqual(len(again['waypoints']),len(model['waypoints']))
        self.assertTrue(all(w['frames'] for w in again['waypoints']))


if __name__=='__main__':unittest.main()
