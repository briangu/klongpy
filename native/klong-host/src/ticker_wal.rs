//! Synchronous ticker-plant batch log. KGT1/KGT2 files and the session checkpoint
//! are shared with the original benchmark writer. A handle owns the exclusive writer lock.
use std::{
    cell::RefCell,
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    rc::Rc,
    time::Duration,
};

use klong_core::{error::Result, value::Func, Error, Int, Interp, Value};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};

const HEADER: usize = 28;
const MAX_ROWS: usize = (256 * 1024 * 1024 - 128) / 32;

fn storage(e: impl std::fmt::Display) -> Error {
    Error::system(format!("ticker WAL: {e}"))
}

fn day_file(dir: &Path, day: Int) -> PathBuf {
    dir.join(format!("day-{day}.wal"))
}

fn options() -> OpenOptions {
    let mut options = OpenOptions::new();
    options.read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
}

fn sync_dir(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    File::open(if path.as_os_str().is_empty() { Path::new(".") } else { path })?.sync_all()?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

fn ensure_day(dir: &Path, day: Int) -> io::Result<File> {
    let path = day_file(dir, day);
    match options().create_new(true).open(&path) {
        Ok(file) => {
            file.sync_all()?;
            sync_dir(dir)?;
            Ok(file)
        }
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => options().open(path),
        Err(e) => Err(e),
    }
}

#[derive(Copy, Clone, PartialEq)]
enum Phase {
    NeedsOpen,
    Replaying,
    Ready,
}

struct Journal {
    dir: PathBuf,
    db: Connection,
    file: File,
    day: Int,
    first: Int,
    closing_end: Option<Int>,
    next: Int,
    phase: Phase,
}

impl Journal {
    fn new(dir: &Path, initial: Option<(Int, Int)>) -> Result<Self> {
        if dir.as_os_str().is_empty() || initial.is_some_and(|(day, first)| day < 0 || first < 0) {
            return Err(Error::range("expected a journal directory and nonnegative initial coordinates"));
        }
        let mut dirs = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            dirs.mode(0o700);
        }
        dirs.recursive(true).create(dir).map_err(storage)?;
        if let Some(parent) = dir.parent() {
            sync_dir(parent).map_err(storage)?;
        }
        let checkpoint = dir.join("session.sqlite3");
        // Only precreate a new checkpoint. Closing an independent descriptor to
        // an existing SQLite file would release this process's POSIX file locks.
        match options().create_new(true).open(&checkpoint) {
            Ok(file) => {
                file.sync_all().map_err(storage)?;
                sync_dir(dir).map_err(storage)?;
            }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(storage(e)),
        }
        let mut db = Connection::open(checkpoint).map_err(storage)?;
        db.busy_timeout(Duration::ZERO).map_err(storage)?;
        db.execute_batch("PRAGMA locking_mode=EXCLUSIVE; PRAGMA synchronous=FULL")
            .map_err(storage)?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Exclusive).map_err(storage)?;
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS session(id INTEGER PRIMARY KEY CHECK(id=1),
                          day INTEGER NOT NULL, first_seq INTEGER NOT NULL, closing_end INTEGER)",
        )
        .map_err(storage)?;
        let row: Option<(Int, Int, Option<Int>)> = tx
            .query_row("SELECT day, first_seq, closing_end FROM session WHERE id=1", [], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })
            .optional()
            .map_err(storage)?;
        let (day, first, closing_end, file) = if let Some((day, first, end)) = row {
            if day < 0 || first < 0 || end.is_some_and(|n| n <= first) {
                return Err(storage("invalid session checkpoint"));
            }
            (day, first, end, options().open(day_file(dir, day)).map_err(storage)?)
        } else {
            let (day, first) = initial.unwrap_or((0, 0));
            for entry in fs::read_dir(dir).map_err(storage)? {
                let entry = entry.map_err(storage)?;
                let name = entry.file_name();
                let Some(number) = name
                    .to_str()
                    .and_then(|n| n.strip_prefix("day-"))
                    .and_then(|n| n.strip_suffix(".wal"))
                else {
                    continue;
                };
                let n: u64 = number.parse().map_err(|_| storage("invalid daily WAL filename"))?;
                if n > day as u64 || (initial.is_none() && entry.metadata().map_err(storage)?.len() != 0) {
                    return Err(storage("existing WAL needs the initial day and first sequence from the HDB"));
                }
            }
            let file = ensure_day(dir, day).map_err(storage)?;
            tx.execute("INSERT INTO session VALUES(1, ?, ?, NULL)", params![day, first])
                .map_err(storage)?;
            (day, first, None, file)
        };
        tx.commit().map_err(storage)?;
        Ok(Self {
            dir: dir.into(),
            db,
            file,
            day,
            first,
            closing_end,
            next: first,
            phase: Phase::NeedsOpen,
        })
    }

    fn rewind(&mut self) -> Result<Value> {
        self.phase = Phase::NeedsOpen;
        self.file.rewind().map_err(storage)?;
        self.next = self.first;
        self.phase = Phase::Replaying;
        Ok(Value::ints(vec![self.day, self.first, self.closing_end.is_some() as Int]))
    }

    fn replay(&mut self, cursor: Int) -> Result<Value> {
        if self.phase != Phase::Replaying || cursor != self.next {
            return Err(storage("unexpected replay cursor"));
        }
        self.phase = Phase::NeedsOpen;
        let offset = self.file.stream_position().map_err(storage)?;
        match read_record(&mut self.file, self.day, self.next)? {
            Record::Batch(cols, end) => {
                let first = self.next;
                self.next = end;
                let mut reply = vec![Value::Int(first)];
                reply.extend(cols);
                let batch = Value::raw_list(reply)?;
                self.phase = Phase::Replaying;
                return Ok(batch);
            }
            Record::Torn if self.closing_end.is_some() => return Err(storage("incomplete closing journal")),
            Record::Torn => {
                self.file.set_len(offset).map_err(storage)?;
                self.file.seek(SeekFrom::Start(offset)).map_err(storage)?;
            }
            Record::End => {}
        }
        if self.closing_end.is_some_and(|n| n != self.next) {
            return Err(storage("journal does not reach the closing watermark"));
        }
        // A dead writer may have completed a record immediately before its sync.
        self.file.sync_all().map_err(storage)?;
        self.phase = Phase::Ready;
        Ok(Value::empty())
    }

    fn append(&mut self, day: Int, first: Int, columns: &[Value]) -> Result<Value> {
        if self.phase != Phase::Ready || self.closing_end.is_some() || (day, first) != (self.day, self.next) {
            return Err(storage("journal is closing, awaiting replay or at a different sequence"));
        }
        let n = columns[0].len().unwrap_or(0);
        if !(1..=MAX_ROWS).contains(&n)
            || !columns
                .iter()
                .all(|c| matches!(c, Value::Ints(a) if a.rank() == 1 && a.data.len() == n))
        {
            return Ok(Value::Int(-1)); // Definite input rejection before any write.
        }
        let end = first
            .checked_add(n as Int)
            .ok_or_else(|| Error::range("journal sequence overflow"))?;
        let mut record = Vec::new();
        record.try_reserve_exact(32 + n * 32).map_err(|_| Error::memory())?;
        record.extend_from_slice(b"KGT2");
        record.extend_from_slice(&(day as u64).to_le_bytes());
        record.extend_from_slice(&(first as u64).to_le_bytes());
        record.extend_from_slice(&(n as u32).to_le_bytes());
        record.resize(32, 0);
        for col in columns {
            if let Value::Ints(a) = col {
                for &v in &a.data {
                    record.extend_from_slice(&(v as i64).to_le_bytes());
                }
            }
        }
        let payload_crc = crc32fast::hash(&record[32..]);
        record[24..28].copy_from_slice(&payload_crc.to_le_bytes());
        let header_crc = crc32fast::hash(&record[..HEADER]);
        record[28..32].copy_from_slice(&header_crc.to_le_bytes());
        self.phase = Phase::NeedsOpen;
        self.file.write_all(&record).map_err(storage)?;
        self.file.sync_data().map_err(storage)?;
        self.next = end;
        self.phase = Phase::Ready;
        Ok(Value::Int(end))
    }

    fn roll(&mut self, day: Int, end: Int, sealed: bool) -> Result<Value> {
        if self.phase != Phase::Ready || (day, end) != (self.day, self.next) || end <= self.first {
            return Err(storage("unexpected rollover watermark"));
        }
        self.phase = Phase::NeedsOpen;
        if sealed {
            if self.closing_end != Some(end) {
                return Err(storage("no matching close intent"));
            }
            let next_day = day.checked_add(1).ok_or_else(|| Error::range("trading day overflow"))?;
            let file = ensure_day(&self.dir, next_day).map_err(storage)?;
            if file.metadata().map_err(storage)?.len() != 0 {
                return Err(storage("next day already contains trades"));
            }
            self.db
                .execute(
                    "UPDATE session SET day=?, first_seq=?, closing_end=NULL WHERE id=1",
                    params![next_day, end],
                )
                .map_err(storage)?;
            self.day = next_day;
            self.first = end;
            self.closing_end = None;
            self.file = file;
        } else {
            self.db
                .execute("UPDATE session SET closing_end=? WHERE id=1", [end])
                .map_err(storage)?;
            self.closing_end = Some(end);
        }
        self.phase = Phase::Ready;
        Ok(Value::Int(end))
    }
}

enum Record {
    End,
    Torn,
    Batch(Vec<Value>, Int),
}

fn read_until_eof(file: &mut File, buf: &mut [u8]) -> io::Result<usize> {
    let mut count = 0;
    while count < buf.len() {
        match file.read(&mut buf[count..]) {
            Ok(0) => break,
            Ok(n) => count += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(count)
}

fn read_record(file: &mut File, day: Int, first: Int) -> Result<Record> {
    let mut header = [0u8; HEADER];
    let got = read_until_eof(file, &mut header).map_err(storage)?;
    if got == 0 {
        return Ok(Record::End);
    }
    let prefix = &header[..got.min(4)];
    if !b"KGT1".starts_with(prefix) && !b"KGT2".starts_with(prefix) {
        return Err(storage("invalid journal header"));
    }
    if got != HEADER {
        return Ok(Record::Torn);
    }
    let legacy = &header[..4] == b"KGT1";
    if !legacy {
        let mut crc = [0; 4];
        if read_until_eof(file, &mut crc).map_err(storage)? != 4 {
            return Ok(Record::Torn);
        }
        if u32::from_le_bytes(crc) != crc32fast::hash(&header) {
            return Err(storage("damaged journal header"));
        }
    }
    let n = u32::from_le_bytes(header[20..24].try_into().unwrap()) as usize;
    if header[4..12] != (day as u64).to_le_bytes() || header[12..20] != (first as u64).to_le_bytes() || !(1..=MAX_ROWS).contains(&n) {
        return Err(storage("invalid or discontinuous journal record"));
    }
    let end = first
        .checked_add(n as Int)
        .ok_or_else(|| Error::range("journal sequence overflow"))?;
    let mut data = Vec::new();
    data.try_reserve_exact(n * 32).map_err(|_| Error::memory())?;
    data.resize(n * 32, 0);
    if read_until_eof(file, &mut data).map_err(storage)? != data.len() {
        return if legacy {
            Err(storage("incomplete legacy journal record with unchecked length"))
        } else {
            Ok(Record::Torn)
        };
    }
    if header[24..28] != crc32fast::hash(&data).to_le_bytes() {
        return Err(storage("damaged journal record"));
    }
    let mut cols = Vec::with_capacity(4);
    for block in data.chunks_exact(n * 8) {
        let mut col = Vec::new();
        col.try_reserve_exact(n).map_err(|_| Error::memory())?;
        for item in block.chunks_exact(8) {
            let v = i64::from_le_bytes(item.try_into().unwrap());
            col.push(Int::try_from(v).map_err(|_| Error::range("journal value exceeds integer width"))?);
        }
        cols.push(Value::ints(col));
    }
    Ok(Record::Batch(cols, end))
}

pub(crate) fn register(ip: &mut Interp) {
    ip.register(
        ".tickerwal",
        1,
        Rc::new(|ip, args| {
            let (dir, initial) = match &args[0] {
                Value::Str(dir) => (dir, None),
                Value::List(list) => match &list[..] {
                    [Value::Str(dir), Value::Int(day), Value::Int(first)] => (dir, Some((*day, *first))),
                    _ => return Err(Error::typ("expected [directory day first-sequence]")),
                },
                _ => return Err(Error::typ(".tickerwal expects a directory or initial checkpoint")),
            };
            let journal = RefCell::new(Journal::new(Path::new(dir.as_str()), initial)?);
            let call = Rc::new(move |ip: &mut Interp, args: &[Value]| {
                let Value::List(request) = &args[0] else {
                    return Err(Error::typ("expected a WAL request list"));
                };
                let Some((Value::Sym(op), args)) = request.split_first() else {
                    return Err(Error::typ("expected a WAL operation symbol"));
                };
                let mut j = journal.try_borrow_mut().map_err(|_| Error::system("ticker WAL is busy"))?;
                match (ip.syms.name(*op), args) {
                    ("open", [Value::Int(0)]) => j.rewind(),
                    ("read", [Value::Int(cursor)]) => j.replay(*cursor),
                    ("append", [Value::Int(day), Value::Int(first), cols @ ..]) if cols.len() == 4 => j.append(*day, *first, cols),
                    (op @ ("beginClose" | "finishClose"), [Value::Int(day), Value::Int(end)]) => {
                        j.roll(*day, *end, op == "finishClose")
                    }
                    _ => Err(Error::typ("invalid ticker WAL request")),
                }
            });
            Ok(Value::Fn(Rc::new(Func::native(ip.syms.intern(".tickerwal.request"), 1, call))))
        }),
    );
}
