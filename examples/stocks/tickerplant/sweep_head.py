"""Render a temporary benchmark variant of the ticker plant head to stdout."""
import argparse
from pathlib import Path


def substitute(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"expected one benchmark insertion point {old!r}")
    return source.replace(old, new)


def render(source, batch_limit, memory, journal=None, password=""):
    if batch_limit < 65536 or batch_limit > 524288 or batch_limit & (batch_limit - 1):
        raise ValueError("the staging limit must be a power of two between 65,536 and 524,288")
    source = substitute(source, "widths::2^!17", f"widths::2^!{batch_limit.bit_length()}")
    source = substitute(source, ":[n>65536;insertLarge", f":[n>{batch_limit};insertLarge")
    if memory:
        source = substitute(source, '''live("ATTACH 'rdb.duckdb' AS rdb")''',
                            '''live("SET temp_directory=''");live("ATTACH ':memory:' AS rdb")''')
    if journal:
        if not memory:
            raise ValueError("the separate journal fixture requires an in-memory RDB")
        if not journal.isdecimal() or not 1 <= int(journal) <= 65535:
            raise ValueError("supply the loopback journal port")
        if password and not (password.isascii() and password.isalnum()):
            raise ValueError("use an alphanumeric journal test password")
        credentials = f":bench:{password}" if password else ""
        source = substitute(source, "histAddr::0;",
                            f'wal::.cli("127.0.0.1:{journal}{credentials}");replaying::0;histAddr::0;')
        source = substitute(source, "    recentFirst::first;nextSeq::first+n;",
                            '    :[replaying;0;need((first+n)~wal([;:append;dayNo;first;ts;sym;px;qty]);'
                            '"journal watermark")];\n    recentFirst::first;nextSeq::first+n;')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("head.kg"))
    parser.add_argument("--max-batch", type=int, default=524288)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--journal-port")
    parser.add_argument("--journal-password-file", type=Path)
    args = parser.parse_args()
    try:
        password = args.journal_password_file.read_text().strip() if args.journal_password_file else ""
        print(render(args.source.read_text(), args.max_batch, args.memory, args.journal_port, password), end="")
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
