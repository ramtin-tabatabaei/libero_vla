"""Train a multi-demonstration object-relative trajectory model."""
import argparse
import json
from pathlib import Path
from waypoint_learning import Config, learn, load_demos, save_model, write_json


def arguments(parser):
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--config', type=Path, help='JSON Config overrides')
    parser.add_argument('--reference-map', type=Path, help='JSON mapping target/destination/fixture or event keys to frame names')
    parser.add_argument('--max-waypoints', type=int, default=None)


def settings(args):
    cfg=Config(**(json.loads(args.config.read_text()) if args.config else {}))
    if args.max_waypoints is not None:cfg.max_waypoints=args.max_waypoints
    refs=json.loads(args.reference_map.read_text()) if args.reference_map else {}
    return cfg,refs


def main():
    parser=argparse.ArgumentParser(description=__doc__);arguments(parser)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--method',choices=('auto','gmm','constrained'),default='auto')
    args=parser.parse_args();config,refs=settings(args)
    demos=load_demos(args.input)
    try:
        if args.method == 'gmm':
            model=learn(demos,config,refs)
        elif args.method == 'constrained':
            from waypoint_skill import learn_skill
            from waypoint_skill_editor import model_from_skill
            model=model_from_skill(learn_skill(demos),demos,config)
        else:
            from waypoint_skill_editor import fit_editor_model
            model=fit_editor_model(demos,config,refs)
    except ValueError as exc:
        write_json(args.output/'training_error.json',dict(error=str(exc),demonstration_files=[d.get('_path') for d in demos],reference_candidates=[d.get('reference_candidates',list(d.get('poses',{}))) for d in demos]))
        raise SystemExit(str(exc))
    save_model(model,args.output)
    print('Saved %d adaptive waypoints from %d successful compatible demonstrations to %s' % (len(model['waypoints']),len(model['used_demonstrations']),args.output))
    if model.get('excluded_demonstrations'):print('Excluded:',model['excluded_demonstrations'])


if __name__=='__main__':main()
