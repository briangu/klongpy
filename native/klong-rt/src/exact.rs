//! Opt-in scalar integer arithmetic. Ordinary Klong operators keep their machine-width semantics.
use alloc::rc::Rc;

#[cfg(feature = "bigint")]
use klong_core::big;
use klong_core::error::Result;
use klong_core::{Error, Int, Interp, Value};

#[derive(Clone, Copy)]
enum Operation {
    Add,
    Multiply,
    Subtract,
    Divide,
}

impl Operation {
    /// None means that both operands fit the machine width, but their result does not.
    fn machine(self, a: Int, b: Int) -> Result<Option<Int>> {
        match self {
            Self::Add => Ok(a.checked_add(b)),
            Self::Multiply => Ok(a.checked_mul(b)),
            Self::Subtract => Ok(a.checked_sub(b)),
            Self::Divide if b == 0 => Err(Error::range("exact division by zero")),
            Self::Divide => match a.checked_div(b) {
                Some(_) if a % b != 0 => {
                    Err(Error::range("exact division requires an integral quotient"))
                }
                quotient => Ok(quotient),
            },
        }
    }

    fn apply(self, args: &[Value]) -> Result<Value> {
        let [a, b] = args else {
            return Err(Error::arity("exact arithmetic takes two arguments"));
        };
        if let (Value::Int(left), Value::Int(right)) = (a, b) {
            if let Some(n) = self.machine(*left, *right)? {
                return Ok(Value::Int(n));
            }
            #[cfg(not(feature = "bigint"))]
            return Err(Error::range(
                "exact result exceeds the integer width of this build",
            ));
        }

        let type_error = || Error::typ("exact arithmetic requires two scalar integers");
        #[cfg(not(feature = "bigint"))]
        return Err(type_error());

        #[cfg(feature = "bigint")]
        {
            let a = big::int_of(a).ok_or_else(type_error)?;
            let b = big::int_of(b).ok_or_else(type_error)?;
            let result = match self {
                Self::Add => a + b,
                Self::Multiply => a * b,
                Self::Subtract => a - b,
                Self::Divide => {
                    let quotient = a
                        .checked_div(&b)
                        .ok_or_else(|| Error::range("exact division by zero"))?;
                    if &quotient * b != a {
                        return Err(Error::range("exact division requires an integral quotient"));
                    }
                    quotient
                }
            };
            Ok(big::norm(result))
        }
    }
}

pub(crate) fn register(ip: &mut Interp) {
    for (name, operation) in [
        (".exactadd", Operation::Add),
        (".exactmul", Operation::Multiply),
        (".exactsub", Operation::Subtract),
        (".exactdiv", Operation::Divide),
    ] {
        ip.register(
            name,
            2,
            Rc::new(move |_: &mut Interp, args: &[Value]| operation.apply(args)),
        );
    }
}

#[cfg(all(test, not(feature = "bigint")))]
mod tests {
    use super::*;
    use klong_core::Kind;

    #[test]
    fn narrow_builds_report_overflow_but_keep_fitting_results() {
        for (operation, a, b) in [
            (Operation::Add, Int::MAX, 1),
            (Operation::Add, Int::MIN, -1),
            (Operation::Multiply, Int::MAX, 2),
            (Operation::Subtract, Int::MIN, 1),
            (Operation::Divide, Int::MIN, -1),
        ] {
            assert!(
                matches!(operation.apply(&[Value::Int(a), Value::Int(b)]), Err(e) if e.kind == Kind::Range)
            );
        }
        for (operation, a, b, expected) in [
            (Operation::Add, Int::MAX, Int::MIN, -1),
            (Operation::Add, Int::MIN, 0, Int::MIN),
            (Operation::Multiply, Int::MIN, 1, Int::MIN),
            (Operation::Subtract, Int::MAX, Int::MAX, 0),
            (Operation::Divide, Int::MIN, 2, Int::MIN / 2),
        ] {
            assert!(
                matches!(operation.apply(&[Value::Int(a), Value::Int(b)]), Ok(Value::Int(n)) if n == expected)
            );
        }
    }
}
