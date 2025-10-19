from __future__ import annotations
import re, subprocess, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
ok = True
issues = []
SELF_ALLOW = {'scripts/verify_busbar.py'}
ALLOW = {'fuka5/io/compat_shim.py','fuka5/io/gcs.py'}

# 1) grep for live GCS references in code (ignore comments & docs)
pat = re.compile(r'gs://|from\s+google\.cloud|import\s+google\.cloud')
for f in subprocess.check_output(["git","ls-files","*.py"], text=True).splitlines():
    if f in SELF_ALLOW: continue
    if f in ALLOW:
        continue
    p = root / f
    t = p.read_text(encoding="utf-8", errors="ignore")
    for i, line in enumerate(t.splitlines(), 1):
        s = line.strip()
        if s.startswith("#"): continue
        if '"""' in s or "'''" in s: 
            # rough skip docstring lines
            pass
        if pat.search(s):
            issues.append(f"{f}:{i}:{s[:200]}")
            ok = False

# 2) compile all python files (syntax check)
bad = []
for f in subprocess.check_output(["git","ls-files","*.py"], text=True).splitlines():
    if f in SELF_ALLOW: continue
    if f in ALLOW:
        continue
    p = root / f
    try:
        compile(p.read_text(encoding="utf-8", errors="ignore"), f, "exec")
    except SyntaxError as e:
        bad.append(f"{f}:{e.lineno}:{e.msg}")
        ok = False

# 3) ensure Streamlit header present
app = root / "app" / "streamlit_app.py"
req = ["from fuka5.io.storage_facade import", "init_backend_shims()", "CFG_PATH ="]
head_ok = all(x in app.read_text(encoding="utf-8") for x in req) if app.exists() else True
if not head_ok:
    issues.append("streamlit_app.py: missing canonical header imports")
    ok = False

if not ok:
    print("VERIFY: FAIL")
    print("---- Issues ----")
    for i in issues:
        print(i)
    for b in bad:
        print("SYNTAX:", b)
    sys.exit(1)

print("VERIFY: PASS")
