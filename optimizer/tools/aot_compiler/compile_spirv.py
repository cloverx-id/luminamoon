import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
SHADER_DIR = ROOT / "crates" / "luminav-vulkan" / "shaders"


def find_glslang() -> Optional[str]:
    which_bin = shutil.which("glslang") or shutil.which("glslangValidator")
    if which_bin:
        return which_bin

    candidates = [
        ROOT / "tools" / "aot_compiler" / "glslang" / "bin" / ("glslang.exe" if sys.platform == "win32" else "glslang"),
        ROOT / "tools" / "aot_compiler" / "glslang" / "bin" / "glslang.exe",
        ROOT / "tools" / "aot_compiler" / "glslang" / "bin" / "glslangValidator",
        ROOT / "tools" / "aot_compiler" / "glslang" / "bin" / "glslang",
    ]
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK if sys.platform != "win32" else os.F_OK):
            return str(cand)
        elif cand.is_file():
            return str(cand)

    return None


def main() -> None:
    glslang_bin = find_glslang()
    if not glslang_bin:
        print(
            "Error: glslang / glslangValidator not found in system PATH or local tools directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    shaders = list(SHADER_DIR.glob("*.comp"))
    if not shaders:
        print(f"No .comp shaders found in {SHADER_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Using glslang compiler: {glslang_bin}")
    for comp in shaders:
        spv = comp.with_suffix(".spv")
        cmd = [
            glslang_bin,
            "-V",
            "--target-env",
            "vulkan1.3",
            "-o",
            str(spv),
            str(comp),
        ]
        print(f"Compiling {comp.name} -> {spv.name}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Failed to compile {comp.name}:\n{res.stdout}\n{res.stderr}", file=sys.stderr)
            sys.exit(1)
        print(f"Compiled {spv.name} ({spv.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
