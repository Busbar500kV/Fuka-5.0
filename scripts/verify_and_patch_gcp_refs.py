from __future__ import annotations
import re, sys, subprocess, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOW = {
    "scripts/verify_busbar.py",
    "fuka5/io/gcs.py",
    "fuka5/io/compat_shim.py",
}
SKIP_PREFIXES = (".venv/", "venv/",)

os.chdir(ROOT)

# Patterns we consider "live" (executable) GCP usage
PAT_LINE = re.compile(r"""
    (?:\bfrom\s+google\.cloud\b) |
    (?:\bimport\s+google\.cloud\b) |
    (?:\bgcs_path\s*\() |
    (?:\bload_gcp_config\s*\() |
    (?:["']gs://)    # literal gs:// inside code
""", re.VERBOSE)

def ls_py():
    out = subprocess.check_output(["git","ls-files","*.py"], text=True)
    files = [p for p in out.splitlines() if p.strip()]
    # skip venv noise just in case
    files = [p for p in files if not p.startswith(SKIP_PREFIXES)]
    return [Path(p) for p in files]

def is_comment_or_doc(s: str) -> bool:
    ss = s.strip()
    if ss.startswith("#"): return True
    if ss.startswith('"""') or ss.startswith("'''"): return True
    return False

def scan():
    issues = []
    for p in ls_py():
        rel = str(p).replace("\\","/")
        if rel in ALLOW: 
            continue
        if not p.exists():    # defensive
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for ln, line in enumerate(text.splitlines(), 1):
            if is_comment_or_doc(line): 
                continue
            if PAT_LINE.search(line):
                issues.append((rel, ln, line.rstrip()))
    return issues

def patch():
    changed = []
    for p in ls_py():
        rel = str(p).replace("\\","/")
        if rel in ALLOW: 
            continue
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        orig = txt

        # 1) gcs_path(...) -> storage_path(...)
        txt = re.sub(r'\bgcs_path\s*\(', 'storage_path(', txt)

        # 2) load_gcp_config(...) -> load_config(...)
        txt = re.sub(r'\bload_gcp_config\s*\(', 'load_config(', txt)

        # 3) wrap literal gs:// strings with smart_path(...)
        txt2 = re.sub(r'(\w+\s*\()\s*("gs://[^"]+"|\'gs://[^\']+\')', r'\1smart_path(\2)', txt)

        if txt2 != orig:
            txt = txt2
            if 'smart_path(' in txt and 'smart_path' not in orig:
                lines = txt.splitlines()
                i = 0
                while i < len(lines) and lines[i].startswith("from __future__"):
                    i += 1
                if i < len(lines) and lines[i].strip() == "":
                    i += 1
                lines[i:i] = ["from fuka5.io.compat_shim import smart_path",""]
                txt = "\n".join(lines) + ("\n" if not lines[-1].endswith("\n") else "")

        if txt != orig:
            p.write_text(txt, encoding="utf-8")
            changed.append(rel)

    return changed

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    issues = scan()
    if mode == "verify":
        if issues:
            print("VERIFY: FAIL")
            for f, ln, s in issues:
                print(f"{f}:{ln}:{s}")
            sys.exit(1)
        print("VERIFY: PASS")
        return

    if mode == "patch":
        changed = patch()
        if changed:
            print("PATCHED FILES:")
            for c in changed:
                print(" -", c)
        else:
            print("No changes needed.")
        issues = scan()
        if issues:
            print("\nAfter patch, still found live GCP refs:")
            for f, ln, s in issues:
                print(f"{f}:{ln}:{s}")
            sys.exit(1)
        # Syntax compile all .py
        bad=[]
        for p in ls_py():
            t = p.read_text(encoding="utf-8", errors="ignore")
            try:
                compile(t, str(p), "exec")
            except SyntaxError as e:
                bad.append(f"{p}:{e.lineno}:{e.msg}")
        if bad:
            print("\nSyntax errors after patch:")
            print("\n".join(bad))
            sys.exit(1)
        print("\nVERIFY: PASS")
        return

    print("Usage: python scripts/verify_and_patch_gcp_refs.py [verify|patch]")
    sys.exit(2)

if __name__ == "__main__":
    main()
