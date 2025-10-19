from __future__ import annotations
import subprocess, os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)

def ls_py():
    out = subprocess.check_output(["git","ls-files","*.py"], text=True)
    return [Path(p) for p in out.splitlines() if p.strip()]

def is_entry(p: Path)->bool:
    try:
        t = p.read_text(encoding="utf-8", errors="ignore")
        return "__name__" in t and "__main__" in t
    except Exception:
        return False

def inject(p: Path)->bool:
    txt = p.read_text(encoding="utf-8")
    if "init_backend_shims()" in txt: return False
    lines = txt.splitlines()
    i=0
    while i<len(lines) and lines[i].startswith("from __future__"): i+=1
    if i<len(lines) and lines[i].strip()=="": i+=1
    new = lines[:i] + ["from fuka5.io.compat_shim import init_backend_shims","init_backend_shims()",""] + lines[i:]
    p.write_text("\n".join(new)+("\n" if not new[-1].endswith("\n") else ""), encoding="utf-8")
    return True

changed=[]
for f in ls_py():
    if is_entry(f):
        if inject(f): changed.append(str(f))
print("\n".join(changed))
