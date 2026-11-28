//! Integer word XOR, with the leading-dimension agreement of Klong atomic dyads.
use alloc::rc::Rc;
use alloc::vec::Vec;

use klong_core::error::Result;
use klong_core::value::{try_vec, Arr, MAX_RANK};
use klong_core::{Error, Int, Interp, Value};

fn with_shape(data: Vec<Int>, like: &Arr<Int>) -> Value {
    let mut dims = [0; MAX_RANK];
    dims[0] = like.len() as u32;
    dims[1..like.rank()].copy_from_slice(like.row_dims());
    Value::Ints(Rc::new(Arr::shaped(data, &dims[..like.rank()])))
}

fn arrays(full: &Arr<Int>, prefix: &Arr<Int>) -> Result<Value> {
    if full.len() != prefix.len() || !full.row_dims().starts_with(prefix.row_dims()) {
        return Err(Error::shape(".bitxor: array dimensions must agree from the leading dimension"));
    }
    let mut data = try_vec(full.data.len())?;
    if !full.data.is_empty() {
        // Each integer of the lower-rank operand pairs with one whole trailing subarray.
        let stride = full.data.len() / prefix.data.len();
        for (row, &word) in full.data.chunks_exact(stride).zip(&prefix.data) {
            data.extend(row.iter().map(|x| x ^ word));
        }
    }
    Ok(with_shape(data, full))
}

fn bitxor(_: &mut Interp, args: &[Value]) -> Result<Value> {
    let [left, right] = args else {
        return Err(Error::arity(".bitxor requires two arguments"));
    };
    match (left.unhost(".bitxor")?, right.unhost(".bitxor")?) {
        (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x ^ y)),
        (Value::Int(word), Value::Ints(a)) | (Value::Ints(a), Value::Int(word)) => {
            let mut data = try_vec(a.data.len())?;
            data.extend(a.data.iter().map(|x| x ^ word));
            Ok(with_shape(data, &a))
        }
        (Value::Ints(a), Value::Ints(b)) if a.rank() >= b.rank() => arrays(&a, &b),
        (Value::Ints(a), Value::Ints(b)) => arrays(&b, &a),
        #[cfg(feature = "bigint")]
        (Value::Big(_), _) | (_, Value::Big(_)) => Err(Error::range(".bitxor: integer exceeds the machine word width")),
        _ => Err(Error::typ(".bitxor needs machine integers or packed integer arrays")),
    }
}

pub(crate) fn register(ip: &mut Interp) {
    ip.register(".bitxor", 2, Rc::new(bitxor));
}
