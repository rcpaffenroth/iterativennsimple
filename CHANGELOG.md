## [v0.5.0] - TBD
- Version bump from 0.4.1 to 0.5.0
- Added `iterativennsimple/Sequential2DRNN.py`: runs a `Sequential2D` block map as a discrete dynamical system,
  `z_{t+1} = (A . b . M)^K . Inject_{t+1} . z_t`.  The state is a list of slots; `M` is the block map, `b` a
  per-slot bias, `A` a per-slot activation, and `Inject` overwrites the input slot with the next token.  `K` is
  the number of internal iterations per token, which decouples the network's own dynamics from the sequence's
  clock -- the point of the module.  Based on Hershey, Paffenroth, Pathak & Tavener, arXiv:2404.00880.
- `Sequential2DRNN.from_rnn` copies the weights of a single-layer `torch.nn.RNN` and reproduces its output
  exactly (`tanh` and `relu`, both `batch_first` settings, any sequence length).  This is a regression test
  rather than the intended use case.  `from_3x3` exposes the six free blocks of the general three-slot map.
- **Fixed a dtype bug in `Sequential2D.forward_vector`** (`Sequential2D.py:127`): the accumulator was built with
  `torch.zeros(..., device=...)` and no `dtype=`, pinning every result to float32.  A float64 input was silently
  *demoted*, losing precision, and a float16 input promoted, losing the reason for using it.  Added dtype tests
  covering every forward method in the package -- `forward_vector`, `forward_list`, `MaskedLinear`,
  `MonarchLinear` (both the view and copy paths), and `Identity` -- across float64/32/16.  Only
  `forward_vector` was affected; the others were already correct.
- `Sequential2DRNN.from_3x3` takes its six blocks as **keyword-only** arguments and validates them: declared
  in/out features, actual weight shape where the block has a `.weight`, and absence of a per-block bias.  Blocks
  without a `.weight` (`MonarchLinear`, nested `Sequential2D`) skip the shape check.  All of this exists to make
  transposition errors loud, since the source-first/target-first naming was got wrong once during design.
- Documentation:
	- `README_Sequential2DRNN.md` -- user-facing entry point, quickstart, and the five things that bite.
	- `OVERVIEW_RNN_SEQUENTIAL_2D.md` -- the design record: every decision with its reasoning, the decisions
	  deliberately *not* taken (multi-layer stacking, the lifted formulation, a learnable `M_xx`), and the open
	  questions.  Includes a standing instruction that orientation claims in RCP's prose be re-derived rather
	  than trusted, which caught one real error during authoring.
	- `TODO_Sequential2DRNN.md` -- deferred work and the experiment queue.
	- `README_RCP.md` -- review checklist, now annotated with RCP's responses.
- Notebooks:
	- `notebooks/7-rcp-RNN-as-Sequential2D.ipynb` -- builds the block map by hand, checks it against
	  `torch.nn.RNN`, then raises `K`.  Symlinked into `tests/` and run by nbmake, so it is simultaneously the
	  tutorial and the equivalence test.
	- `notebooks/advanced/12-claude-fixed-points-and-bistability.ipynb` -- the dynamical-systems view at
	  `hidden_size = 1`, where cobwebs, bistability and an exactly-located saddle-node fold are all drawable.
	  Shows memory stored as the identity of an attractor: 120 applications of the map with no decay.
- `examples/rnn_internal_iterations.py` -- trains at several `K` on a memory task.  A worked negative result:
  `K > 1` collapses to chance, the memory horizon is measured and found to shrink geometrically in `K`,
  orthogonal initialisation of `W_hh` repairs it (chance to 0.89), and `K = 1` still wins on a task that is pure
  memory.  The transferable finding is that comparisons across `K` are meaningless unless initialisation is
  controlled for.
- All 95 tests pass.

## [v0.4.0] - TBD
- Version bump from 0.3.1 to 0.4.0
- Widened all dependency pins from exact/minor-locked (`==2.5.1`, `==2.1.*`) to major-bounded ranges (`>=2.5,<3`, `>=2.1,<3`) so the package can be installed alongside others without version conflicts.  The dev group (`pytest`, `pylint`, `nbmake`) was widened the same way.  All 75 tests pass against the newly resolved versions.
- Updated `generatedata` source from branch `release/v0.4.0` to `release/v0.4.1`.  This was the change that actually mattered: v0.4.0 hard-pinned the whole scientific stack transitively (`torch==2.5.1`, `numpy<2.2`, `plotly<5.25`, ...), so widening our own pins had no effect on its own.
- Note: `torch`/`torchvision` are still effectively pinned to 2.5.1 because `generatedata` routes them to the `download.pytorch.org/whl/cu121` index, which published nothing past 2.5.1.  That wheel has no cp313 build, which is why `requires-python` remains `<3.13`.  Lifting either limit requires dropping the `cu121` index from `generatedata` (it cannot be overridden downstream — uv rejects conflicting indexes).
- Implemented sparse monarch matrix
- Removed torch-sparse and torch-scatter dependencies.  At the end of the day, on a GPU, these were significantly slower that MaskedLinear, and they were causing a lot of installation issues.  The new implementation is still sparse, but it is based on the Monarch matrix format, which is a custom sparse format that we developed for this project.  
- Moved to uv
- Protect main with github actions

## [v0.3.1] - 3-2-2026
- Version bump from 0.3.0 to 0.3.1
- Added `scikit-learn = 1.6.*` dependency
- Updated `generatedata` source from tag `v0.3.0` to branch `release/v0.3.1`
- Updated `notebooks/6-rcp-Sequential-vs-Sequential2D.ipynb`
- Added advanced notebooks for spectra and mother/child examples:
	- `notebooks/advanced/6-rcp-Sequential2D-training-spectra.ipynb`
	- `notebooks/advanced/7-rcp-basic-mother-and-child.ipynb`
	- `notebooks/advanced/prompts.txt`
- Updated `.gitignore`

## [v0.3.0] - 8-23-2025
- Major refactor: Sequential2D is now optimized
- New version of generatedata