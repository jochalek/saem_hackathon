# Project Script Conventions

1. **Environment (always use mise + uv)**
   - `mise use python@3.12`
   - `mise use uv`
   - Keep dependencies in `scripts/pyproject.toml`.
   - Run scripts via `uv run python <script>.py` (never system python).

2. **Script location and structure**
   - Put runnable code in `scripts/`.
   - Use `if __name__ == "__main__": main()`.
   - Resolve paths from repo root with `Path(__file__).resolve().parents[1]`.

3. **Data and outputs**
   - Read raw inputs from `data/raw/` only.
   - Write outputs to `outputs/<task_name>/`.
   - Never overwrite raw data.

4. **GPU support plan (required)**
   - Prefer GPU-capable libraries when available (`torch` CUDA, `cudf`, `cuml`).
   - Implement safe import checks and automatic CPU fallback.
   - Log/report whether GPU path was used.

5. **Reproducibility**
   - Set fixed random seeds for numpy/sklearn/torch.
   - Save run metadata (params, library/device info, input/output paths).

6. **Code quality**
   - Keep scripts modular (small pure functions).
   - Add type hints for public/helper functions.
   - Fail fast with clear errors for missing files/columns.

7. **Project hygiene**
   - Keep scripts concise and task-focused.
   - Update docs when behavior/output changes.
   - Commit all code changes to git.
