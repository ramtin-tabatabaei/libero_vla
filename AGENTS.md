# Project workflow

Always keep Graphify current after making project changes, before the final
handoff. This is a standing user preference; no additional confirmation is needed.

- Refresh the existing graph from the actual workspace files, never temporary
  copies. Preserve its indexed sources and add code files changed or created
  during the task.
- Run from the project root:
  `/home/ramtin/.local/share/uv/tools/graphifyy/bin/python update_graphify.py <changed-code-files>`
  Omit the file arguments to refresh all currently indexed code.
- For changed documentation or other non-code content, use the Graphify skill's
  semantic update workflow and merge it into the existing graph.
- Keep updates scoped to indexed and task-related files. Do not rescan vendored
  repositories or large asset directories merely to refresh this graph.
- Verify the update succeeded. If blocked, report the reason and that the graph
  is stale; do not silently skip the update or claim it is current.

This is an agent workflow requirement, not a background filesystem watcher.
