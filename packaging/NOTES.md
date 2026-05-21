# Packaging notes (not yet validated end-to-end)

## `pyproject.toml` build-system mismatch

`pyproject.toml` declares both `setuptools.build_meta` (active backend) and
`scikit-build-core` / `pybind11` in `requires`, plus a `[tool.scikit-build]`
section. These are effectively unused by the current build path.

**If/when** the backend is switched to `scikit_build_core.build`, revisit:
- `[build-system].requires` — drop the unused entries or keep them intentional
- `[tool.setuptools.*]` sections — may conflict with scikit-build-core's own
  configuration

This is tracked here (in `packaging/`) rather than as an in-source comment so
the upstream `pyproject.toml` stays untouched by Debian/RPM packaging changes.
