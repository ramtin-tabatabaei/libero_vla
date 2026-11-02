"""Save offline trajectory, uncertainty, event, and importance figures."""
import argparse
from pathlib import Path
import numpy as np
from waypoint_learning import load_model


def visualize(model,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fig=plt.figure(figsize=(14,5));raw=fig.add_subplot(121,projection='3d');aligned=fig.add_subplot(122,projection='3d')
    import json
    from waypoint_learning import samples
    for i,path in enumerate(model['demonstration_files']):
        if path and Path(path).exists():
            points=np.array([s['position'] for s in samples(json.loads(Path(path).read_text()))])
            raw.plot(*points.T,alpha=.65,label='Demo %d'%(i+1))
    raw.set_title('Executed demonstrations (world, metres)')
    if raw.lines:raw.legend(fontsize=7)
    colors=plt.cm.tab20(np.linspace(0,1,len(model['segments'])))
    for j,(seg,color) in enumerate(zip(model['segments'],colors)):
        for demo in seg['aligned_position']:aligned.plot(*np.array(demo).T,color=color,alpha=.18)
        p=np.array(seg['position']);aligned.plot(*p.T,color=color,lw=2,label='%d: %s'%(j,seg['reference']['name']))
        # Three principal 2-sigma axes expose anisotropy in the frame coordinates.
        for i in range(0,len(p),max(1,len(p)//3)):
            vals,vecs=np.linalg.eigh(np.array(seg['covariance'])[i,:3,:3])
            for k in range(3):
                delta=2*np.sqrt(max(vals[k],0))*vecs[:,k]
                aligned.plot(*np.array([p[i]-delta,p[i]+delta]).T,color=color,alpha=.6)
    for w in model['waypoints']:
        aligned.scatter(*w['relative_position_mean'],c='black',s=13)
    aligned.set_title('Aligned local frames: demos, GMR mean, 2σ axes\nDifferent frame coordinates are overlaid; compare within each interval')
    for ax in (raw,aligned):ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.set_zlabel('z (m)')
    fig.tight_layout();fig.savefig(output/'trajectories.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(14,11),sharex=True)
    for seg,color in zip(model['segments'],colors):
        phase=seg['phase']
        for ax,key in zip(axes,('position_uncertainty','orientation_uncertainty','gripper_probability','importance')):
            ax.plot(phase,seg[key],color=color)
            ax.axvline(phase[0],color='gray',lw=.4,alpha=.6)
    labels=[]
    for phase,event in [(s['phase'][0],s['start_event']['kind']) for s in model['segments']]+[(1.,model['segments'][-1]['end_event']['kind'])]:
        if labels and phase-labels[-1][0]<.005:
            labels[-1][1].append(event)
        else:labels.append([phase,[event]])
    for phase,kinds in labels:
        axes[0].annotate('/'.join(dict.fromkeys(kinds)),(phase,.97),xycoords=('data','axes fraction'),
                         rotation=90,fontsize=8,va='top',ha='right',backgroundcolor='white')
    for w in model['waypoints']:axes[3].scatter(w['phase'],w['importance'],c='black',s=18)
    for ax,label in zip(axes,('Position √trace(Σ), m','Orientation √trace(Σ), rad','P(gripper closed)','Waypoint importance')):ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[-1].set_xlabel('Normalized event-aligned phase φ');fig.tight_layout();fig.savefig(output/'phase_diagnostics.png',dpi=160);plt.close(fig)
    # A row-per-interval view avoids visually connecting unrelated frames.
    fig,axes=plt.subplots(len(model['segments']),1,figsize=(12,max(3,2*len(model['segments']))),squeeze=False)
    for j,(seg,ax) in enumerate(zip(model['segments'],axes[:,0])):
        p=np.array(seg['position']);cov=np.array(seg['covariance'])
        for k,label in enumerate('xyz'):
            line,=ax.plot(seg['phase'],p[:,k],label=label);sigma=np.sqrt(np.maximum(cov[:,k,k],0))
            ax.fill_between(seg['phase'],p[:,k]-2*sigma,p[:,k]+2*sigma,color=line.get_color(),alpha=.15)
        ax.set_title('%d · %s → %s · frame %s'%(j,seg['start_event']['kind'],seg['end_event']['kind'],seg['reference']['name']),fontsize=9);ax.legend(loc='upper right',ncol=3,fontsize=7)
    fig.tight_layout();fig.savefig(output/'frame_intervals.png',dpi=130);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',required=True,type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();visualize(load_model(args.model),args.output or args.model/'plots')


if __name__=='__main__':main()
