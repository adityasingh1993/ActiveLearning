#!/usr/bin/env python
"""
Pre-commit hook script for HASSL.

Runs import smoke tests and verifies zero SyntaxError or NameError regressions before committing.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib
import pkgutil
import hassl


def main():
    print("[HASSL Pre-Commit Hook] Running static analysis (ruff F821/F401)...")
    import subprocess
    try:
        res = subprocess.run(["ruff", "check", "hassl", "--select", "F821,F401"], capture_output=True, text=True)
        if res.returncode != 0:
            print("[HASSL Pre-Commit Hook] FAILED! Ruff static analysis found errors:")
            print(res.stdout)
            sys.exit(1)
        else:
            print("  [OK] Ruff F821/F401 passed cleanly")
    except FileNotFoundError:
        print("  [INFO] Ruff not installed in environment, skipping ruff check")

    print("[HASSL Pre-Commit Hook] Running module verification (SyntaxError, NameError, & Internal Imports)...")
    failed = []
    external_missing = []

    for m in pkgutil.walk_packages(hassl.__path__, prefix="hassl."):
        try:
            importlib.import_module(m.name)
            print(f"  [OK] {m.name}")
        except (NameError, SyntaxError, AttributeError) as e:
            failed.append(f"{m.name}: {type(e).__name__}: {e}")
            print(f"  [FAIL - Code Error] {m.name}: {type(e).__name__}: {e}")
        except (ModuleNotFoundError, ImportError) as e:
            msg = str(e)
            # If missing external 3rd party package, log info; if internal module failure, fail pre-commit
            if any(pkg in msg for pkg in ["monai", "nrrd", "scipy", "matplotlib", "SimpleITK", "fastapi", "uvicorn", "PIL", "wandb", "mlflow"]):
                external_missing.append(f"{m.name}: {e}")
                print(f"  [SKIP - Missing External Pkg] {m.name} ({e})")
            else:
                failed.append(f"{m.name}: {type(e).__name__}: {e}")
                print(f"  [FAIL - Internal Import Error] {m.name}: {e}")
        except Exception as e:
            failed.append(f"{m.name}: {type(e).__name__}: {e}")
            print(f"  [FAIL] {m.name}: {e}")

    if failed:
        print("\n[HASSL Pre-Commit Hook] FAILED! Internal code errors detected:")
        for err in failed:
            print(f"  - {err}")
        sys.exit(1)

    print("[HASSL Pre-Commit Hook] Running CI test suite (pytest tests/test_pipeline_ci.py -q)...")
    pytest_res = subprocess.run([sys.executable, "-m", "pytest", "tests/test_pipeline_ci.py", "-q"], capture_output=True, text=True)
    if pytest_res.returncode != 0:
        print("[HASSL Pre-Commit Hook] FAILED! Pytest suite reported failures:")
        print(pytest_res.stdout)
        print(pytest_res.stderr)
        sys.exit(1)
    else:
        print("  [OK] Pytest suite passed cleanly")

    print(f"\n[HASSL Pre-Commit Hook] PASSED! All modules syntactically clean and all tests passed ({len(external_missing)} skipped due to uninstalled external packages).")
    sys.exit(0)


if __name__ == "__main__":
    main()
