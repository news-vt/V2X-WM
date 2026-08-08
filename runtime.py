"""Runtime helpers that must execute before importing Mitsuba/Sionna RT."""
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Optional


def _brew_prefix(formula: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["brew", "--prefix", formula], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return out or None
    except Exception:
        return None


def configure_drjit_llvm(strict_on_macos: bool = True) -> Optional[str]:
    """Configure ``DRJIT_LIBLLVM_PATH`` before importing Mitsuba/Sionna.

    Dr.Jit's CPU backend on macOS requires an LLVM shared library.  The
    environment variable has to be set *before* importing ``mitsuba``,
    ``drjit`` or ``sionna.rt``.

    Returns
    -------
    Optional[str]
        The selected LLVM library path, or ``None`` if no explicit path is
        required/detected.
    """
    existing = os.environ.get("DRJIT_LIBLLVM_PATH")
    if existing and Path(existing).is_file():
        return existing

    if platform.system() != "Darwin":
        # Linux/CUDA installations generally do not need an explicit path.
        return existing

    candidates: list[Path] = []
    for formula in ("llvm", "llvm@18", "llvm@19", "llvm@20"):
        prefix = _brew_prefix(formula)
        if prefix:
            candidates.append(Path(prefix) / "lib" / "libLLVM.dylib")

    candidates += [
        Path("/opt/homebrew/opt/llvm/lib/libLLVM.dylib"),
        Path("/opt/homebrew/opt/llvm@18/lib/libLLVM.dylib"),
        Path("/usr/local/opt/llvm/lib/libLLVM.dylib"),
        Path("/usr/local/opt/llvm@18/lib/libLLVM.dylib"),
    ]

    for path in candidates:
        if path.is_file():
            os.environ["DRJIT_LIBLLVM_PATH"] = str(path)
            return str(path)

    if strict_on_macos:
        raise RuntimeError(
            "Dr.Jit could not find libLLVM.dylib. Install LLVM with "
            "`brew install llvm`, restart the Python/Jupyter process, and "
            "run this program again. You can also set DRJIT_LIBLLVM_PATH "
            "manually before importing Mitsuba/Sionna RT."
        )
    return None
