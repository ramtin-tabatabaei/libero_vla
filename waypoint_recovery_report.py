"""Export timing matrices and a local paired-video viewer for the recovery study."""
import argparse
import json
from pathlib import Path

import numpy as np

from waypoint_recovery_benchmark import TASKS, FAILURES, INTENSITIES, summarize
from waypoint_recovery_metrics import derive_results, timing_comparisons


def distribution(values):
    values = [v for v in values if v is not None]
    return dict(count=len(values), median=None if not values else float(np.median(values)),
                minimum=None if not values else float(min(values)),
                maximum=None if not values else float(max(values)))


def analysis(folder, results):
    valid = [r for r in results if r['status'] in ('recovered','recovery_failed')]
    recovery = [r for r in results if r['mode']=='recovery']
    not_held = [r for r in valid if not r['motion_metrics']['source_already_grasped']]
    def times(rows, field):
        return distribution([r.get('motion_metrics',{}).get(field) for r in rows])
    report = dict(completed_conditions=len(results), planned_conditions=200,
        normal_policy=times([r for r in results if r['mode']=='normal_policy'],'completion_sim_seconds'),
        normal_waypoints=times([r for r in results if r['mode']=='normal_waypoints'],'completion_sim_seconds'),
        recovery_completion=times(valid,'completion_sim_seconds'),
        failure_to_completion=distribution([r.get('failure_to_completion_sim_seconds') for r in valid]),
        total_task_completion=distribution([r.get('total_task_sim_seconds') for r in valid]),
        correct_grasp_when_empty_at_takeover=times(not_held,'first_correct_grasp_seconds'),
        wrong_object_contact_before_correct_grasp_trials=sum(bool(r['motion_metrics']['wrong_object_contacts_before_correct_grasp']) for r in not_held),
        wrong_object_grasp_trials=sum(bool(r['motion_metrics']['wrong_objects_grasped']) for r in valid),
        observed_recovery_trials=len(valid), recovered=sum(r['success'] for r in valid),
        unrealized_fault_conditions=[r['trial_id'] for r in recovery if r['status']=='failure_not_observed'],
        additional_attempts=sum(max(0,len(r.get('attempts',[]))-1) for r in recovery),
        response_cutoff_seconds=None,
        completion_statistics='Only completed episodes; unfinished episodes are counted separately, not assigned a completion time',
        groups={})
    for scene in ('original','shuffled'):
        for kind in FAILURES:
            for intensity in INTENSITIES:
                subset=[r for r in valid if (r['scene'],r['kind'],r['intensity'])==(scene,kind,intensity)]
                report['groups'][scene+'/'+kind+'/'+intensity]=dict(
                    observed_trials=len(subset), recovered=sum(r['success'] for r in subset),
                    completion=times(subset,'completion_sim_seconds'),
                    correct_grasp=times([r for r in subset if not r['motion_metrics']['source_already_grasped']],
                                        'first_correct_grasp_seconds'))
    (folder/'timing_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def number(v):return '—' if v is None else '%.2f'%v
    lines=['# Pi 0.5 recovery timing study','',
        '%d of 200 conditions completed: 10 tasks, four failure types, Low/High, original/shuffled layouts, plus normal waypoint and Pi 0.5 baselines.'%len(results),'',
        'The policy is the RLinf Pi 0.5 checkpoint serving on port 8002. One failure is armed per recovery episode. Pi 0.5 takes over the existing scene two simulation seconds after a measured physical failure, without a reset. One saved simulation state per task and layout makes all starts identical. The recovery budget is 400 control steps, or 20 simulation seconds. There is one selected episode per condition, so these results describe this run, not a robust estimate of policy performance.','',
        '| Failure | Low | High | Measured onset |',
        '| --- | --- | --- | --- |',
        '| Collision | Drive 1 cm beyond the obstacle centre | Drive 3 cm beyond the centre | First physical robot contact with the selected obstacle |',
        '| Transition deviation | 3 cm sideways target offset | 12 cm sideways target offset | Actual sideways displacement reaches 80% of the requested offset |',
        '| Orientation deviation | 20–40° wrist tilt about a drawn roll, pitch or yaw axis | 45–70° about a drawn axis, requested in full | Wrist is within 4 cm of the target and within 35% of the requested angular deviation, with at least half the tilt measured |',
        '| Grasp loss | Prevent grasp in Grasp | Drop an acquired object in Transport | Suppressed close with an empty hand; or loss of finger contact while opening a previously measured grasp |','',
        'No response cutoff is used. Approach timing measures sustained robot translation toward the live source object before the correct grasp. If the source is already held at takeover, the target is the destination. An approach time of zero also applies when the hand already touches the source or starts within 5 cm of its target. Contact times use finger-pad contacts at 0.05-second control intervals; grasp uses LIBERO’s two-finger contact check. Wrong-object contacts, wrong grasps, and the path to the correct grasp are kept separate from eventual task success. Videos provide the full behavior.','',
        'Table and plot times are simulation seconds. The CSV also records observed wall times, which include policy inference and recording. Completion time is the start of the final continuously successful LIBERO goal-predicate interval. Unfinished episodes have no completion time; reported completion medians are conditional on finishing. Recovery times start at takeover, after some waypoint execution has already happened. Total task time includes that prefix; failure-to-completion time includes the two-second handoff delay.','',
        'Physically unrealized injections are excluded from recovery success rates. Retries use the same stored start and another eligible waypoint, and stop at the first measured injection regardless of its recovery outcome. Earlier attempts are preserved under attempts/. Move has no failures. Transport permits only High grasp loss; Low no-grasp occurs in Grasp.','',
        '| Measurement | Completed events | Median seconds | Range seconds |',
        '| --- | ---: | ---: | --- |']
    for key,label in [('normal_waypoints','Normal waypoints, task start → done'),('normal_policy','Normal Pi 0.5, task start → done'),
                      ('recovery_completion','Recovery, takeover → done'),('failure_to_completion','Failure → done'),
                      ('total_task_completion','Recovery, task start → done'),
                      ('correct_grasp_when_empty_at_takeover','Correct grasp, initially empty hand')]:
        stats=report[key]
        lines.append('| %s | %d | %s | %s–%s |'%(label,stats['count'],number(stats['median']),number(stats['minimum']),number(stats['maximum'])))
    lines += ['', '%d of %d measured-failure episodes recovered. %d measured-failure episodes grasped a wrong object.'%
              (report['recovered'],len(valid),report['wrong_object_grasp_trials']), '',
              'Additional attempts: %d. Unrealized fault conditions: %d.'%(report['additional_attempts'],len(report['unrealized_fault_conditions'])), '',
              'Open [the paired-video viewer](report.html), [all timings](timing.csv), or [timing summary](timing_summary.json) for each task and failure condition.','']
    (folder/'ANALYSIS.md').write_text('\n'.join(lines))
    return report


def matrix(folder, results, field, title, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    columns = [('normal_policy', None, None)] + [('recovery', k, i) for k in FAILURES for i in INTENSITIES]
    labels = ['Normal\nPi 0.5'] + [FAILURES[k].replace(' deviation', '')+'\n'+INTENSITIES[i]
                                  for _, k, i in columns[1:]]
    names = {key: next((r.get('language', '') for r in results if (r['suite'],r['task_id'])==key),
                      '%s:%s'%key) for key in TASKS}
    lookup = {(r['suite'],r['task_id'],r['scene'],r['mode'],r.get('kind'),r.get('intensity')):r for r in results}
    fig, axes = plt.subplots(1,2,figsize=(19,8.5),sharey=True)
    cmap, norm = plt.get_cmap('Blues'), Normalize(0,20)
    for ax, scene in zip(axes, ('original','shuffled')):
        for y, (suite, task) in enumerate(TASKS):
            for x, (mode,kind,intensity) in enumerate(columns):
                row = lookup.get((suite,task,scene,mode,kind,intensity), {})
                value = row.get('motion_metrics',{}).get(field)
                status = row.get('status')
                if not row:
                    colour, text = '#eeeeee','pending'
                elif status == 'failure_not_observed':
                    colour, text = '#d4d4d4','no fault'
                elif value is None:
                    colour, text = ('#f2c1bd','unfinished') if field=='completion_sim_seconds' else ('#e6e6e6','—')
                else:
                    colour, text = cmap(norm(value)), '%.2f'%value
                    if field=='first_correct_grasp_seconds' and row.get('motion_metrics',{}).get('source_already_grasped'):
                        text = 'held'
                    if status=='already_successful':
                        text = 'pre-done'
                ax.add_patch(Rectangle((x-.5,y-.5),1,1,facecolor=colour,edgecolor='white',linewidth=2))
                ax.text(x,y,text,ha='center',va='center',fontsize=8.5,
                        color='white' if value is not None and value>12 else '#183044')
        ax.set(xlim=(-.5,len(columns)-.5),ylim=(len(TASKS)-.5,-.5),title=scene.title()+' layout')
        ax.set_xticks(range(len(columns)));ax.set_xticklabels(labels,fontsize=8,rotation=25,ha='right')
        ax.set_yticks(range(len(TASKS)))
        ax.set_yticklabels([names[k].replace('pick up the ','').replace(' and place it','') for k in TASKS],fontsize=9)
        for spine in ax.spines.values():spine.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle(title+' — simulation seconds',fontsize=16)
    fig.text(.35,.025,'One trial per condition. “held” means the correct object was held at takeover. No response cutoff.',fontsize=10)
    fig.subplots_adjust(left=.30,right=.90,bottom=.16,top=.90,wspace=.07)
    colourbar = fig.add_axes([.925,.22,.012,.58])
    fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=colourbar,label='Seconds')
    fig.savefig(folder/output,dpi=160)
    plt.close(fig)


PAGE = r'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Pi 0.5 recovery timing</title>
<style>body{margin:28px;background:#f5f7fa;color:#203040;font:15px system-ui,sans-serif}h1{font-size:26px}p{max-width:1100px;line-height:1.5}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}select,button{padding:8px;border:1px solid #ccd4dd;border-radius:6px;background:white}.players{display:grid;grid-template-columns:1fr 1fr;gap:20px;max-width:1080px}video{width:100%;background:#101820}.table{overflow:auto;max-height:530px;background:white;margin:22px 0}table{border-collapse:collapse;white-space:nowrap;width:100%}th,td{padding:9px 12px;border-bottom:1px solid #e3e8ed;text-align:left}th{position:sticky;top:0;background:#eaf0f5;cursor:pointer}tr:hover{background:#eef5fa;cursor:pointer}.selected{background:#dceaf5}small{color:#526475}img{max-width:100%;background:white}a{color:#1b6692}.details{max-width:1080px;padding:12px;background:white;line-height:1.6}</style>
<h1>Pi 0.5 recovery timing</h1><p id="summary"></p>
<p>Times in the table are simulation seconds. The CSV also includes observed wall time with inference and recording. Pi 0.5 takes control two simulation seconds after a measured failure. Recoveries have a 400-step limit (20 simulation seconds). A dash means no measured event or no completion; it is not zero. Contact times use finger-pad contacts sampled every control step (0.05 seconds); grasp uses LIBERO’s two-finger contact check. No response cutoff is applied.</p>
<p>The approach measure detects at least 1 mm of sustained robot translation toward the target over a 0.25-second window. Zero means the hand starts within 5 cm of the target or already touches it. If the correct object is already held, the approach target is the destination. Use the videos and wrong-object contacts to judge how direct the recovery is.</p>
<p><a href="timing.csv">Download all timings (CSV)</a> · <a href="results.json">Complete results (JSON)</a></p>
<div class="controls"><select id="task"></select><select id="scene"><option value="">Both layouts</option><option value="original">Original</option><option value="shuffled">Shuffled</option></select><select id="kind"><option value="">All cases</option><option value="collision">Collision</option><option value="transition">Transition deviation</option><option value="orientation">Orientation deviation</option><option value="grasp_loss">Grasp loss</option><option value="normal_policy">Normal Pi 0.5</option><option value="normal_waypoints">Normal waypoints</option></select><select id="intensity"><option value="">Both intensities</option><option value="low">Low</option><option value="high">High</option></select><button id="play">Play paired videos</button></div>
<div class="players"><div><h3>Normal Pi 0.5, same task and layout</h3><video id="normal" controls preload="none"></video></div><div><h3 id="case-title">Selected recovery</h3><video id="recovery" controls preload="none"></video></div></div>
<p class="details" id="detail">Select a row to compare videos and inspect object contacts.</p>
<details><summary>Failure to takeover clip</summary><video id="fault" controls preload="none" style="max-width:530px"></video></details>
<div class="table"><table><thead id="head"></thead><tbody id="body"></tbody></table></div>
<h2>Completion after Pi 0.5 starts</h2><img src="completion_matrix.png" alt="Completion timing matrix">
<h2>Time to the correct grasp after Pi 0.5 starts</h2><img src="grasp_matrix.png" alt="Grasp timing matrix">
<script>
const DATA=__DATA__,RAW=__RAW__,SUMMARY=__SUMMARY__;
const headers=[['language','Task'],['scene','Layout'],['case','Case'],['status','Outcome'],['normal_waypoints_completion_sim_seconds','Normal waypoints'],['normal_policy_completion_sim_seconds','Normal Pi 0.5'],['total_task_sim_seconds','Total task'],['failure_to_completion_sim_seconds','Failure → done'],['completion_sim_seconds','Takeover → done'],['first_goalward_motion_seconds','First goalward translation'],['first_correct_contact_seconds','Correct contact'],['first_correct_grasp_seconds','Correct grasp'],['correct_grasp_to_completion_seconds','Correct grasp → done'],['focus','Approach target'],['wrong_objects_grasped','Wrong grasps'],['wrong_contacts_before_correct_grasp','Other contacts before correct grasp'],['path_to_grasp_detour_ratio','Path / direct distance to grasp']];
const raw=new Map(RAW.map(r=>[r.trial_id,r]));let sortKey='trial_id',ascending=true,selected=null;
const task=document.getElementById('task');task.add(new Option('All tasks',''));
for(const [key,name] of new Map(RAW.map(r=>[r.suite+':'+r.task_id,r.language])))task.add(new Option(name,key));
document.getElementById('summary').textContent=SUMMARY.total+' of 200 trials complete. '+SUMMARY.successes+' recovered out of '+SUMMARY.recovery_trials+' observed-failure recovery trials.';
const head=document.getElementById('head'),hr=document.createElement('tr');
for(const [key,name] of headers){const th=document.createElement('th');th.textContent=name;th.onclick=()=>{ascending=sortKey===key?!ascending:true;sortKey=key;render()};hr.append(th)}head.append(hr);
function fmt(v){return v===null||v===undefined?'—':typeof v==='number'?v.toFixed(2):String(v)}
function select(r){selected=r.trial_id;const original=RAW.find(x=>x.suite===r.suite&&x.task_id===r.task_id&&x.scene===r.scene&&x.mode==='normal_policy');
const info=raw.get(r.trial_id),m=info.motion_metrics||{};const n=document.getElementById('normal'),v=document.getElementById('recovery'),f=document.getElementById('fault');
n.src=original?'trials/'+original.trial_id+'/policy.mp4':'';
v.src=info.policy_result?'trials/'+r.trial_id+'/policy.mp4':'';f.src=info.observed_failure?'trials/'+r.trial_id+'/failure_to_handoff.mp4':'';
document.getElementById('case-title').textContent=r.case+' · '+r.scene+(info.policy_result?'':' · timing data only');
document.getElementById('detail').textContent='First object contact: '+(m.first_object_contact_names||[]).join(', ')+' at '+fmt(m.first_object_contact_seconds)+' s. Correct object already grasped at rollout start: '+(m.source_already_grasped?'yes':'no')+'. Contacts before the correct grasp with other objects: '+((m.wrong_object_contacts_before_correct_grasp||[]).join(', ')||'none')+'. Focus: '+(m.focus||'—')+'. First control command (observed wall time): '+fmt(m.first_action_wall_seconds)+' s. Correct grasp (observed wall time): '+fmt(m.first_correct_grasp_wall_seconds)+' s.';render()}
function render(){let rows=DATA.filter(r=>(!task.value||r.suite+':'+r.task_id===task.value)&&(!document.getElementById('scene').value||r.scene===document.getElementById('scene').value)&&(!document.getElementById('kind').value||r.kind===document.getElementById('kind').value||r.mode===document.getElementById('kind').value)&&(!document.getElementById('intensity').value||r.intensity===document.getElementById('intensity').value));
rows.sort((a,b)=>{const x=a[sortKey],y=b[sortKey];if(x===null||x===undefined)return 1;if(y===null||y===undefined)return-1;return(ascending?1:-1)*(typeof x==='number'?x-y:String(x).localeCompare(String(y)))});
const body=document.getElementById('body');body.replaceChildren();for(const r of rows){const tr=document.createElement('tr');if(r.trial_id===selected)tr.className='selected';for(const [key] of headers){const td=document.createElement('td');td.textContent=fmt(r[key]);tr.append(td)}tr.onclick=()=>select(r);body.append(tr)}}
for(const id of ['task','scene','kind','intensity'])document.getElementById(id).onchange=render;
document.getElementById('play').onclick=()=>{for(const id of ['normal','recovery']){const v=document.getElementById(id);if(v.getAttribute('src')){v.currentTime=0;v.play().catch(()=>{})}}};render();
const first=DATA.find(r=>r.mode==='recovery'&&raw.get(r.trial_id).observed_failure);if(first)select(first);
</script></html>'''


def generate(folder):
    folder = Path(folder)
    results = derive_results(folder)
    analysis(folder, results)
    rows = timing_comparisons(results)
    for row, result in zip(rows, results):
        row['language'] = result.get('language', row['suite']+':'+str(row['task_id']))
        row['case'] = ({'normal_policy':'Normal Pi 0.5','normal_waypoints':'Normal waypoints'}.get(row['mode']) or
                       FAILURES[row['kind']]+' / '+INTENSITIES[row['intensity']])
    encode = lambda value:json.dumps(value, allow_nan=False).replace('<','\\u003c')
    page = PAGE.replace('__DATA__',encode(rows)).replace('__RAW__',encode(results)).replace('__SUMMARY__',encode(summarize(results)))
    (folder/'report.html').write_text(page)
    matrix(folder, results, 'completion_sim_seconds','Completion time from Pi 0.5 start','completion_matrix.png')
    matrix(folder, results, 'first_correct_grasp_seconds','Correct grasp time from Pi 0.5 start','grasp_matrix.png')
    print('Report generated:',folder/'report.html')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    generate(parser.parse_args().folder)
