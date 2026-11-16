"""Check ?> transcripts in fenced blocks of the given markdown files.
Each block runs in one fresh REPL interpreter (extras on the path)."""
import contextlib, io, re, sys, warnings
warnings.simplefilter("ignore")
from klongpy.repl import create_repl, cleanup_repl
from klongpy.writer import kg_write

def blocks(text):
    for m in re.finditer(r"```[a-z]*\n(.*?)```", text, re.S):
        body = m.group(1)
        if body.lstrip().startswith("?> ") and not re.search(r"\.clid?\(|\.srv\(88|\.web\(|\.ws\(", body):
            yield body

def parse(body):
    steps = []
    for line in body.splitlines():
        if line.startswith("?> "):
            steps.append([line[3:], []])
        elif steps:
            steps[-1][1].append(line)
    return steps

total = bad = 0
for path in sys.argv[1:]:
    text = open(path).read()
    for body in blocks(text):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            klong, loops = create_repl()
            try:
                for expr, exp in parse(body):
                    buf.seek(0); buf.truncate()
                    try:
                        r = klong(expr)
                        got = (buf.getvalue() + kg_write(r, klong._backend, display=False)).split("\n")
                    except Exception as ex:
                        got = (buf.getvalue() + "error: " + str(ex).split("\n")[0]).split("\n")
                    total += 1
                    if got != exp:
                        bad += 1
                        sys.__stdout__.write(f"MISMATCH {path}: {expr}\n  expected {exp}\n  got      {got}\n")
            finally:
                cleanup_repl(loops)
print(f"{total} examples checked, {bad} mismatches")
