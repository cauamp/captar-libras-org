# Working notes


## Communication

- Be brief, direct and clear when explaining anything. Lead with the answer or the result;
  skip preamble, recaps and long lists of caveats.
- Avoid spontaneous abbreviations, acronyms and shorthand. Write terms out; if a project
  term is needed, say what it means the first time.

## Operating Principles & Tone

- Math & logic rigor: When deriving or implementing custom loss functions, attention mechanisms, or geometric projections, explain the underlying tensor shapes and mathematical rationale clearly.
- Shape annotations: Always add inline comments for high-dimensional tensor shapes (e.g., `B x C x H x W` or `B x N x D`).
- Don't mention claude as a author in commit messages or code comments.
- Don't create new git branches unless I ask for it or allow it; commit on the current branch.

## Implementation: delegate to Sonnet agents

Default for any implementation task beyond a few-line edit: the main session designs and
reviews; Sonnet subagents write the code (`Agent` tool, `model: "sonnet"`,
`subagent_type: "general-purpose"`). No branch and no commit per step unless asked.

1. **Design first, in the main session.** Read the code the change touches, decide the
   interfaces (module, function signatures, tensor shapes, batch keys, config fields,
   CLI flags, defaults) and what "done" means. Agents implement a spec; they don't
   choose the design.
2. **Split by files, not by layers.** One agent per disjoint set of files. Run agents in
   parallel only when their files don't overlap and one doesn't need the other's API;
   otherwise run them in sequence (e.g. the dataset agent after the renderer agent).
3. **Each prompt is self-contained** (the agent has none of this conversation):
   - repo path, `uv run`, `export TORCH_HOME=$HOME/.cache/torch` if it loads models,
     which GPU (`CUDA_VISIBLE_DEVICES`), scratch dir for outputs;
   - files to read first (this file, the relevant plan, the code it plugs into);
   - the why, in two or three sentences, and the exact spec: signatures, shapes,
     conventions (pixel centres, cell grid, NaN = no label), defaults (new behaviour off
     by default so existing runs are unchanged);
   - rules: touch only the listed files, re-read a file right before editing it (other
     sessions may edit the repo concurrently), surgical changes, match the surrounding
     style, ruff clean, no branches or commits;
   - verification it must run: new tests (CPU, synthetic, fast), `uv run ruff check .`,
     `uv run pytest -q`, plus an equivalence check when moving code (same outputs before
     and after) and a small real-data smoke run; never the full expensive run;
   - what to report: files changed, public API, test results, per-item timings,
     decisions beyond the spec, anything surprising.
4. **Mid-course changes** go to the running agent with `SendMessage`, not a new agent.
5. **Review before reporting.** Read the diff of every agent (not only its report), run
   the tests and a smoke run yourself, fix or send back what is wrong, then update the
   Layout section here and the relevant plan.
