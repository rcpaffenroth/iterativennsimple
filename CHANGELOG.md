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