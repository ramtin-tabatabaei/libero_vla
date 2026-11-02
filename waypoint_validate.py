"""Leave-one-demonstration-out trajectory and legacy waypoint evaluation."""
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from waypoint_learning import (Config, detect_events, event_signature, frame_for, fused_trajectory,
    interval_data, learn, load_demos, relative_pose, resample_pose, samples, warp_to, write_json)
from waypoint_learn import arguments,settings


def errors(pred_p,pred_r,true_p,true_r):
    pos=np.linalg.norm(pred_p-true_p,axis=1)
    ori=Rotation.from_matrix(np.einsum('tji,tjk->tik',pred_r,true_r)).magnitude()
    return dict(mean_relative_position_error_m=float(pos.mean()),max_relative_position_error_m=float(pos.max()),
                mean_orientation_error_rad=float(ori.mean()),max_orientation_error_rad=float(ori.max()))


def baseline_curve(train,held):
    """Unchanged legacy extractor, adapted to held-out episode-start scene frames."""
    from waypoint_library import demonstration_plan
    from waypoint_chain import resolve
    chain=demonstration_plan(train)
    train_poses={n:(np.array(p),np.array(r)) for n,(p,r) in train['poses'].items()}
    test_poses={n:(np.array(p),np.array(r)) for n,(p,r) in held['poses'].items()}
    ss=samples(train); recorded=np.array([s['position'] for s in ss]);orient=np.array([s['orientation'] for s in ss])
    previous=(recorded[0],orient[0]);test_previous=(np.array(samples(held)[0]['position']),np.array(samples(held)[0]['orientation']))
    points=[test_previous[0]];rotations=[test_previous[1]];indices=[0];grip=[float(ss[0]['gripper']=='close')]
    last=0
    for w in chain.waypoints:
        previous=resolve(w,train_poses,previous);test_previous=resolve(w,test_poses,test_previous)
        distance=np.linalg.norm(recorded-previous[0],axis=1)+.01*np.linalg.norm(orient-previous[1],axis=(1,2));distance[:last+1]=np.inf
        idx=int(np.argmin(distance))
        if not np.isfinite(distance[idx]):continue
        indices.append(idx);points.append(test_previous[0]);rotations.append(test_previous[1]);last=idx
        grip.append(grip[-1] if w.gripper_after=='keep' else float(w.gripper_after=='close'))
    return np.array(points),np.array(rotations),np.array(grip),np.array(indices),chain


def world_interval(demo,events,j,count):
    """One held-out interval in world coordinates, resampled like interval_data."""
    ss=samples(demo);a,b=events[j]['index'],events[j+1]['index'];raw=ss[a:b+1]
    p=np.array([s['position'] for s in raw]);r=np.array([s['orientation'] for s in raw])
    g=np.array([1. if s['gripper']=='close' else 0. for s in raw])
    return resample_pose(p,r,g,np.arange(len(raw)),np.linspace(0,len(raw)-1,count))


def evaluate(demos,config=None,overrides=None):
    c=config or Config();folds=[]
    successful=[d for d in demos if d.get('success') and len(d['samples'])>=c.min_episode_steps and not d['samples'][0].get('task_success',False)]
    if len(successful)<3:raise ValueError('Leave-one-out requires at least 3 successful demonstrations')
    for k,held in enumerate(successful):
        fold=dict(held_out=k,file=held.get('_path'),demonstration_success=True)
        train=[d for i,d in enumerate(successful) if i!=k]
        try:
            model=learn(train,c,overrides);events=detect_events(held,c)
            train_events=model['events']
            grip_signature=lambda es:[e['kind'] for e in es if e['kind'] in ('close','open')]
            fold['gripper_event_consistency']=grip_signature(events)==grip_signature(train_events)
            if event_signature(events)!=event_signature(train_events):
                fold.update(status='incompatible_events',held_out_events=[e['key'] for e in events],training_events=[e['key'] for e in train_events]);folds.append(fold);continue
            fractions=np.array([e['index']/max(len(samples(held))-1,1) for e in events])
            phase_error=np.abs(fractions-np.array(model['event_episode_fraction_mean']))
            fold.update(mean_event_phase_error=float(phase_error.mean()),max_event_phase_error=float(phase_error.max()),
                        event_phase_errors=[dict(event=e['key'],error=float(v)) for e,v in zip(events,phase_error)])
            actual_p=[];actual_r=[];pred_p=[];pred_r=[];coverage=[]
            world_true_p=[];world_true_r=[];world_pred_p=[];world_pred_r=[];world_coverage=[]
            for j,seg in enumerate(model['segments']):
                raw=interval_data(held,events,j,seg['reference'],c.samples_per_interval)
                template=tuple(np.array(seg['template'][key]) for key in ('position','orientation','gripper'))
                aligned,indices=warp_to(template,raw,c)
                actual_p.extend(aligned[0]);actual_r.extend(aligned[1]);pred_p.extend(seg['position']);pred_r.extend(seg['orientation'])
                covariance=np.array(seg['covariance'])[:,:3,:3]
                delta=aligned[0]-np.array(seg['position'])
                mahal=np.einsum('ti,tij,tj->t',delta,np.linalg.pinv(covariance),delta)
                coverage.extend((mahal<=7.815).tolist())
                if not seg.get('references'):continue
                # The task-parameterized prediction is only meaningful in world
                # coordinates: it is the held-out scene's own frames that move.
                frames=[frame_for(held,events[j],ref) for ref in seg['references']]
                wp,wr,wpc,_=fused_trajectory(seg,frames)
                truth=resample_pose(*world_interval(held,events,j,c.samples_per_interval)[:3],
                                    np.arange(c.samples_per_interval),indices)
                world_pred_p.extend(wp);world_pred_r.extend(wr)
                world_true_p.extend(truth[0]);world_true_r.extend(truth[1])
                wdelta=truth[0]-wp
                world_coverage.extend((np.einsum('ti,tij,tj->t',wdelta,np.linalg.pinv(wpc),wdelta)<=7.815).tolist())
            fold['multi_demonstration']=errors(np.array(pred_p),np.array(pred_r),np.array(actual_p),np.array(actual_r))
            fold['multi_demonstration']['position_95_percent_ellipsoid_coverage']=float(np.mean(coverage))
            if world_pred_p:
                fold['task_parameterized_world']=errors(np.array(world_pred_p),np.array(world_pred_r),
                                                        np.array(world_true_p),np.array(world_true_r))
                fold['task_parameterized_world']['position_95_percent_ellipsoid_coverage']=float(np.mean(world_coverage))
            # Baseline selection uses training data only: first compatible successful run.
            base=train[model['used_demonstrations'][0]];base_events=detect_events(base,c)
            try:
                bp,br,bg,bi,chain=baseline_curve(base,held)
                pred_p=[];pred_r=[];true_p=[];true_r=[]
                for j,seg in enumerate(model['segments']):
                    query=np.linspace(base_events[j]['index'],base_events[j+1]['index'],c.samples_per_interval)
                    pp,rr,gg=resample_pose(bp,br,bg,bi,query)
                    pp,rr=relative_pose(pp,rr,frame_for(held,events[j],seg['reference']))
                    raw=interval_data(held,events,j,seg['reference'],c.samples_per_interval)
                    template=(pp,rr,gg);aligned,_=warp_to(template,raw,c)
                    pred_p.extend(pp);pred_r.extend(rr);true_p.extend(aligned[0]);true_r.extend(aligned[1])
                fold['single_demonstration']=errors(np.array(pred_p),np.array(pred_r),np.array(true_p),np.array(true_r))
                fold['single_demonstration']['waypoints']=len(chain.waypoints)
                base_fractions=np.array([e['index']/max(len(samples(base))-1,1) for e in base_events])
                fold['single_demonstration']['mean_event_phase_error']=float(np.mean(abs(base_fractions-fractions)))
                fold['single_demonstration']['max_event_phase_error']=float(np.max(abs(base_fractions-fractions)))
                fold['single_demonstration']['gripper_event_consistency']=grip_signature(base_events)==grip_signature(events)
            except ValueError as exc:fold['single_demonstration']=dict(status='extraction_failed',error=str(exc))
            fold.update(status='evaluated',training_demonstrations=len(model['used_demonstrations']),waypoints=len(model['waypoints']))
        except ValueError as exc:fold.update(status='training_failed',error=str(exc))
        folds.append(fold)
    return dict(version=2,folds=folds,successful_demonstrations=len(successful),
                evaluated_folds=sum(f['status']=='evaluated' for f in folds),
                protocol='Training-only frames/template/model. task_parameterized_world fuses every frame at the held-out scene poses and is scored in world coordinates. Held-out DTW is shape registration, not predictive timing. Timing errors use unwarped episode fractions. Baseline uses identical event intervals and its own DTW shape registration.',
                replay_validation='not_run')


def main():
    parser=argparse.ArgumentParser(description=__doc__);arguments(parser)
    parser.add_argument('--output',default='waypoint_validation.json')
    args=parser.parse_args();c,refs=settings(args)
    result=evaluate(load_demos(args.input),c,refs);write_json(args.output,result)
    print('%d/%d folds evaluated; report: %s'%(result['evaluated_folds'],len(result['folds']),args.output))


if __name__=='__main__':main()
