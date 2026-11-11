# Contributing

Contributions that improve the implementation, scientific protocol, documentation, or usability are welcome. For substantial changes, open an issue first to discuss the proposed scope.

## Development setup

```bash
git clone https://github.com/ranajahanzaib/industrial-visual-defect-detection.git
cd industrial-visual-defect-detection
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The web workbench has a separate dependency installation:

```bash
cd apps/web
pnpm install
```

## Changes and pull requests

- Keep changes focused and describe the problem they solve.
- Add or update tests for behavior changes.
- Run `python -m pytest`, `python -m ruff check src tests`, and `python -m mypy --ignore-missing-imports src/industrial_visual_defect_detection` for Python changes.
- Run `pnpm test` and `pnpm run build` for web changes.
- Preserve the documented separation between normal training/calibration data and the official test split.
- Do not commit datasets, model weights, local environment files, or generated reports.
- Describe validation performed and identify checks that could not be run.

Open a pull request with a concise summary, relevant context, and test results. Report security issues through the private process in [SECURITY.md](.github/SECURITY.md), not public issues.
