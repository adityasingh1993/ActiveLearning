import importlib
import pkgutil
import pytest
import hassl


def test_all_modules_import():
    """Import smoke test (R-5 fix): guarantees zero SyntaxError or NameError across all modules."""
    failed = []
    for m in pkgutil.walk_packages(hassl.__path__, prefix="hassl."):
        try:
            importlib.import_module(m.name)
        except Exception as e:
            failed.append(f"{m.name}: {type(e).__name__}: {e}")
    assert not failed, "Modules failed to import:\n" + "\n".join(failed)
