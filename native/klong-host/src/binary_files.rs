//! Binary file handles and little-endian integer vectors on the desktop filesystem.
use klong_core::{
    error::Result,
    value::{try_vec, Func},
    Error, Int, Interp, Value,
};
use std::{
    borrow::Cow,
    cell::RefCell,
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom, Write},
    path::Path,
    rc::Rc,
};

use crate::local_file::{create_file, io_error, options, sync_parent};

fn count(v: u64) -> Result<Value> {
    Int::try_from(v)
        .map(Value::Int)
        .map_err(|_| Error::range("file offset exceeds integer width"))
}

fn width(n: Int) -> Result<usize> {
    match n {
        1 | 4 | 8 => Ok(n as usize),
        _ => Err(Error::range("integer width must be 1, 4 or 8 bytes")),
    }
}

fn ints(v: &Value) -> Result<&[Int]> {
    match v {
        Value::Ints(a) if a.rank() == 1 => Ok(&a.data),
        Value::List(a) if a.is_empty() => Ok(&[]),
        _ => Err(Error::typ("expected an integer vector")),
    }
}

fn encode(data: &[Int], width: usize) -> Result<Cow<'_, [u8]>> {
    if cfg!(target_endian = "little") && width == std::mem::size_of::<Int>() {
        // Int has no padding. Borrow bytes of the immutable slice for this call only.
        return Ok(Cow::Borrowed(unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
        }));
    }
    let mut bytes = try_vec(data.len().checked_mul(width).ok_or_else(Error::memory)?)?;
    for &v in data {
        match width {
            1 => bytes.push(u8::try_from(v).map_err(|_| Error::range("byte out of range"))?),
            4 => bytes.extend_from_slice(
                &i32::try_from(v)
                    .map_err(|_| Error::range("integer exceeds 32 bits"))?
                    .to_le_bytes(),
            ),
            _ => bytes.extend_from_slice(&(v as i64).to_le_bytes()),
        }
    }
    Ok(Cow::Owned(bytes))
}

fn decode(data: &[u8], width: usize) -> Result<Value> {
    if data.len() % width != 0 {
        return Err(Error::range("incomplete binary integer"));
    }
    let mut values = try_vec(data.len() / width)?;
    for chunk in data.chunks_exact(width) {
        let v = match width {
            1 => chunk[0] as i64,
            4 => i32::from_le_bytes(chunk.try_into().unwrap()) as i64,
            _ => i64::from_le_bytes(chunk.try_into().unwrap()),
        };
        values.push(Int::try_from(v).map_err(|_| Error::range("value exceeds integer width"))?);
    }
    Ok(Value::ints(values))
}

fn chunks(
    value: &Value,
    width: usize,
    mut use_bytes: impl FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    if let Value::Ints(a) = value {
        use_bytes(&encode(&a.data, width)?)?;
    } else if let Value::List(items) = value {
        for item in items.iter() {
            use_bytes(&encode(ints(item)?, width)?)?;
        }
    } else {
        use_bytes(&encode(ints(value)?, width)?)?;
    }
    Ok(())
}

fn mkdir(path: &Path) -> io::Result<()> {
    if path.is_dir() {
        return sync_parent(path);
    }
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        if !parent.is_dir() {
            mkdir(parent)?;
        }
    }
    let mut dirs = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        dirs.mode(0o700);
    }
    match dirs.create(path) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists && path.is_dir() => {}
        Err(e) => return Err(e),
    }
    sync_parent(path)
}

fn request(file: &mut File, ip: &Interp, req: &Value) -> Result<Value> {
    let (op, args): (_, &[Value]) = match req {
        Value::Sym(op) => (*op, &[]),
        Value::List(list) => match &list[..] {
            [Value::Sym(op), args @ ..] => (*op, args),
            _ => return Err(Error::typ("expected a binary file operation")),
        },
        _ => return Err(Error::typ("expected a binary file operation")),
    };
    match (ip.syms.name(op), args) {
        ("pos", []) => count(file.stream_position().map_err(io_error)?),
        ("size", []) => count(file.metadata().map_err(io_error)?.len()),
        ("sync", []) => {
            file.sync_data().map_err(io_error)?;
            Ok(Value::Int(1))
        }
        ("seek", [Value::Int(n)]) if *n >= 0 => {
            count(file.seek(SeekFrom::Start(*n as u64)).map_err(io_error)?)
        }
        ("truncate", [Value::Int(n)]) if *n >= 0 => {
            file.set_len(*n as u64).map_err(io_error)?;
            Ok(Value::Int(*n))
        }
        ("write", [Value::Int(n), values]) => {
            chunks(values, width(*n)?, |bytes| {
                file.write_all(bytes).map_err(io_error)
            })?;
            Ok(Value::Int(1))
        }
        ("read", [Value::Int(n), Value::Int(items)]) if *items >= 0 => {
            let width = width(*n)?;
            let length = usize::try_from(*items)
                .ok()
                .and_then(|v| v.checked_mul(width))
                .ok_or_else(Error::memory)?;
            // A short read (including EOF) should allocate only the bytes available.
            let available = file
                .metadata()
                .map_err(io_error)?
                .len()
                .saturating_sub(file.stream_position().map_err(io_error)?);
            let length = length.min(usize::try_from(available).unwrap_or(usize::MAX));
            let mut buf = try_vec(length)?;
            buf.resize(length, 0u8);
            let mut got = 0;
            while got < length {
                match file.read(&mut buf[got..]) {
                    Ok(0) => break,
                    Ok(n) => got += n,
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(io_error(e)),
                }
            }
            decode(&buf[..got], width)
        }
        _ => Err(Error::typ("invalid binary file operation or arguments")),
    }
}

pub(super) fn register(ip: &mut Interp) {
    ip.register(
        ".bpack",
        2,
        Rc::new(|_, a| {
            let [Value::Int(w), v] = a else {
                return Err(Error::typ("expected width and integer vector"));
            };
            decode(&encode(ints(v)?, width(*w)?)?, 1)
        }),
    );
    ip.register(
        ".bunpack",
        2,
        Rc::new(|_, a| {
            let [Value::Int(w), v] = a else {
                return Err(Error::typ("expected width and bytes"));
            };
            decode(&encode(ints(v)?, 1)?, width(*w)?)
        }),
    );
    ip.register(
        ".crc32",
        2,
        Rc::new(|_, a| {
            let [Value::Int(w), v] = a else {
                return Err(Error::typ("expected width and integer vectors"));
            };
            let mut crc = crc32fast::Hasher::new();
            chunks(v, width(*w)?, |bytes| {
                crc.update(bytes);
                Ok(())
            })?;
            Ok(Value::Int(crc.finalize() as i32 as Int))
        }),
    );
    ip.register(
        ".mkdir",
        1,
        Rc::new(|_, a| {
            let [Value::Str(path)] = a else {
                return Err(Error::typ("expected a directory path"));
            };
            mkdir(Path::new(path.as_str())).map_err(io_error)?;
            Ok(Value::Int(1))
        }),
    );
    ip.register(
        ".files",
        1,
        Rc::new(|_, a| {
            let [Value::Str(path)] = a else {
                return Err(Error::typ("expected a directory path"));
            };
            let mut rows = Vec::new();
            for item in fs::read_dir(path.as_str()).map_err(io_error)? {
                let item = item.map_err(io_error)?;
                let meta = item.metadata().map_err(io_error)?;
                if meta.is_file() {
                    rows.push(Value::raw_list(vec![
                        Value::text(&item.file_name().to_string_lossy())?,
                        count(meta.len())?,
                    ])?);
                }
            }
            Value::raw_list(rows)
        }),
    );
    ip.register(
        ".bfile",
        2,
        Rc::new(|ip, a| {
            let [Value::Str(name), Value::Sym(mode)] = a else {
                return Err(Error::typ("expected a path and file mode"));
            };
            let path = Path::new(name.as_str());
            let file = match ip.syms.name(*mode) {
                "read" => File::open(path),
                "update" => options().open(path),
                "create" => match create_file(path).map_err(io_error)? {
                    Some(file) => Ok(file),
                    None => options().open(path),
                },
                _ => return Err(Error::typ("expected :read, :update or :create")),
            }
            .map_err(io_error)?;
            let file = RefCell::new(file);
            let f = Rc::new(move |ip: &mut Interp, args: &[Value]| {
                let mut writer = file
                    .try_borrow_mut()
                    .map_err(|_| Error::system("binary file is busy"))?;
                request(&mut writer, ip, &args[0])
            });
            Ok(Value::Fn(Rc::new(Func::native(
                ip.syms.intern(".bfile.handle"),
                1,
                f,
            ))))
        }),
    );
}
