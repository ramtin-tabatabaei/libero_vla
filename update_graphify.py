"""Refresh indexed code from this workspace, optionally adding named code files.

Run with the Python interpreter from the installed graphifyy environment.
Non-code graph content is preserved; semantic updates use the Graphify skill.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

from graphify.build import build_merge
from graphify.cluster import cluster
from graphify.detect import save_manifest
from graphify.export import to_json
from graphify.extract import collect_files, extract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="Additional project code files to index")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = root / "graphify-out"
    graph_path = output / "graph.json"
    data = json.loads(graph_path.read_text())
    sources = {node["source_file"] for node in data["nodes"] if node.get("source_file")}
    sources.update(args.files)
    code = set()
    deleted = []
    for source in sorted(sources):
        path = (root / source).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Source outside project: {source}")
        if not path.exists():
            if source in args.files:
                raise FileNotFoundError(path)
            deleted.append(source)
        elif path.is_file():
            found = collect_files(path)
            if source in args.files and not found:
                raise ValueError(f"Use the semantic update workflow for {source}")
            code.update(found)
        else:
            raise ValueError(f"Pass individual code files, not directories: {source}")
    paths = sorted(code)
    extraction = extract(paths, root=root, cache_root=root)
    if not extraction["nodes"]:
        raise RuntimeError("Empty extraction; keeping the existing graph")
    graph = build_merge([extraction], graph_path=graph_path, root=root,
                        prune_sources=deleted, ast_sources=paths)
    communities = cluster(graph)
    shutil.copy2(graph_path, output / "graph.previous.json")
    if not to_json(graph, communities, str(graph_path)):
        raise RuntimeError("Graphify declined the update; existing graph retained")
    save_manifest({"code": [str(p) for p in paths]},
                  manifest_path=str(output / "manifest.json"), root=root, kind="ast")
    (output / ".graphify_root").write_text(str(root))
    (output / ".graphify_python").write_text(sys.executable)
    print(f"Graphify updated: {len(paths)} code files, "
          f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")


if __name__ == "__main__":
    main()
