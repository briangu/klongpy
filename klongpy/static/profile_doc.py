"""
docs/static-profile.md: the rule tables the checker and the compiler decide by, as the text of the page.

    generated_block()      the block of the page that is generated from the tables
    render_page()          the whole page
    check_text(text)       True if the generated block of a page text is up to date
    main(argv)             python -m klongpy.static.profile_doc [--check [PATH] | --write [PATH]]

The tables are the native engine's (klong_static::profile_doc); this module calls into the extension and does the file handling
around it. Between

    <!-- BEGIN GENERATED -->  ...  <!-- END GENERATED -->

the page is never edited by hand.
"""
import sys

from ..native_ext import native

__all__ = ["BEGIN", "END", "DEFAULT_PATH", "generated_block", "render_page", "check_text", "main"]

BEGIN = "<!-- BEGIN GENERATED -->"
END = "<!-- END GENERATED -->"
DEFAULT_PATH = "docs/static-profile.md"


def generated_block():
    """The block of the page that is generated from the rule tables."""
    return native().profile_doc_block()


def render_page():
    """The whole page."""
    return native().profile_doc_page()


def _split(text):
    a = text.find(BEGIN)
    b = text.find(END)
    if a < 0 or b < a:
        return None
    return text[:a + len(BEGIN)], text[a + len(BEGIN):b], text[b:]


def check_text(text):
    """True if the generated block of a page text is up to date."""
    parts = _split(text)
    return parts is not None and parts[1].strip("\n") == generated_block().strip("\n")


def main(argv=None, stdout=None, stderr=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    mode = None
    if argv and argv[0] in ("--check", "--write"):
        mode = argv.pop(0)
    if (mode is None and argv) or len(argv) > 1:
        print("usage: python -m klongpy.static.profile_doc [--check [PATH] | --write [PATH]]", file=err)
        return 2
    if mode is None:
        out.write(render_page())
        return 0
    path = argv[0] if argv else DEFAULT_PATH
    try:
        with open(path, "r") as f:
            text = f.read()
    except OSError:
        text = None
    if mode == "--check":
        if text is None:
            print(f"{path}: not found; create it with: python -m klongpy.static.profile_doc --write {path}", file=err)
            return 2
        if check_text(text):
            return 0
        print(f"{path}: the generated block is out of date; run: python -m klongpy.static.profile_doc --write {path}", file=err)
        return 1
    parts = _split(text) if text is not None else None
    new = render_page() if parts is None else parts[0] + "\n" + generated_block() + parts[2]
    with open(path, "w") as f:
        f.write(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
