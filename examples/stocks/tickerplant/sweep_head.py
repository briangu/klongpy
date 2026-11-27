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
    if journal:
        if not journal.isdecimal() or not 1 <= int(journal) <= 65535:
            raise ValueError("supply the loopback journal port")
        if password and not (password.isascii() and password.isalnum()):
            raise ValueError("use an alphanumeric journal test password")
        credentials = f":bench:{password}" if password else ""
        address = f"127.0.0.1:{journal}{credentials}"
        source = substitute(source, 'journalAddr::""', f'journalAddr::"{address}"')
    elif memory:
        source = substitute(source, 'journalDir::"log"', 'journalDir::""')
    return source


def profiled_head(source):
    source = substitute(source, "journalProfile::0", "journalProfile::1")
    def around(text, stage):
        return f'tpMark("{stage}";0);{text};tpMark("{stage}";1)'

    source = 'tpMark(s;e)::{.p(.json([;"stage";s;e;.pc();.now()]));1}\n' + source
    copy = next(line.strip().rstrip(";") for line in source.splitlines() if line.strip().startswith('live("COPY rdb.ticks'))
    for expression, stage in (
        ("{buf,x}'data", "table_columns"),
        ('live(:[n=width;sql;sql," WHERE seq<",number(first+n)])', "insert"),
        ('logBatch(first;ts;sym;px;qty);1]', "wal"),
        ('live("COMMIT")', "commit"),
        ('reply::.try({handle(msg)};[])', "subscriber"),
        (copy, "export"),
        ('reply::.try({peer([;:seal;dayNo;dayFirst;nextSeq])};[])', "seal"),
        ('live("DELETE FROM rdb.ticks")', "reset"),
    ):
        if stage == "wal":
            source = substitute(source, expression, '{' + around("logBatch(first;ts;sym;px;qty)", stage) + '}();1]')
        else:
            source = substitute(source, expression, around(expression, stage))
    metadata = 'live("SELECT file_size_bytes,format_version FROM parquet_file_metadata(\'",path,"\')")'
    source = substitute(source, 'tpMark("export";1)', 'tpMark("export";1);.p(.json([;"parquet";dayNo;' + metadata + ']))')
    source = substitute(source, "publish(first;ts;sym;px;qty)::{", "tpPublish(first;ts;sym;px;qty)::{")
    source = substitute(source, "latest(sym)::{", "tpLatest(sym)::{")
    wrappers = ('publish(first;ts;sym;px;qty)::{[r];tpMark("publish";0);r::tpPublish(first;ts;sym;px;qty);'
                'tpMark("publish";1);r}\nlatest(sym)::{[r];tpMark("quote";0);r::tpLatest(sym);tpMark("quote";1);r}\n')
    return substitute(source, "setJournal(path)::{", wrappers + "setJournal(path)::{")


def profiled_subscriber(source):
    expression = "cursor::first+n;cursor}"
    return substitute(source, expression, 'cursor::first+n;.p(.json([;"applied";cursor;.now()]));cursor}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("head.kg"))
    parser.add_argument("--max-batch", type=int, default=524288)
    parser.add_argument("--memory", action="store_true", help="legacy sweep option: disable the WAL unless --journal-port is given")
    parser.add_argument("--profile-subscriber", type=Path, metavar="SCRIPT", help="render a subscriber with applied-watermark timestamps")
    parser.add_argument("--profile", action="store_true", help="write timed stage markers as JSON lines")
    logging = parser.add_mutually_exclusive_group()
    logging.add_argument("--no-wal", action="store_true", help="benchmark without open-day durability")
    logging.add_argument("--journal-port")
    parser.add_argument("--journal-password-file", type=Path)
    args = parser.parse_args()
    try:
        if args.profile_subscriber:
            print(profiled_subscriber(args.profile_subscriber.read_text()), end="")
            return
        password = args.journal_password_file.read_text().strip() if args.journal_password_file else ""
        if password and not args.journal_port:
            parser.error("--journal-password-file requires --journal-port")
        source = render(args.source.read_text(), args.max_batch, args.memory or args.no_wal, args.journal_port, password)
        print(profiled_head(source) if args.profile else source, end="")
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
