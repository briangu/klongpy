//! Metal numeric arrays, including a reverse-mode tape for the supported arithmetic.
#[path = "gpu_bridge.rs"]
mod bridge;

use bridge::{Kernel, Object};
use klong_core::error::Result;
use klong_core::value::{try_vec, Arr, Ext, ExtBox};
use klong_core::{Error, Flt, Int, Interp, Value};
use std::cell::{Cell, OnceCell, RefCell};
use std::collections::BTreeSet;
use std::rc::Rc;

pub struct Session {
    name: String,
    device: Rc<Device>,
}

impl Session {
    pub fn open(name: &str, ordinal: i64) -> std::result::Result<Self, String> {
        let metal = bridge::Device::open(ordinal).map_err(|e| e.message.to_string())?;
        let device = Device {
            metal,
            stats: RefCell::default(),
            session: Cell::new(0),
            tape: RefCell::new(None),
        };
        Ok(Self {
            name: name.to_owned(),
            device: Rc::new(device),
        })
    }

    pub fn install(&self, ip: &mut Interp) {
        let device = self.device.clone();
        ip.host_numeric = Some(Rc::new(move |glyph, x, y| device.numeric(glyph, x, y)));
        let device = self.device.clone();
        ip.host_op = Some(Rc::new(move |ip, glyph, args| match (glyph, args) {
            (":>", [f, x, _]) => device.gradient(ip, f, x),
            _ => Err(Error::typ(format!("Metal does not implement {glyph}"))),
        }));
    }

    pub fn trace(&self) -> String {
        let stats = self.device.stats.borrow();
        // Selected device names and operator glyphs are restricted to JSON-safe characters.
        let ops = stats
            .operations
            .iter()
            .map(|op| format!("\"{op}\""))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"device\":\"{}\",\"kernel_count\":{},\"device_operations\":[{}],\"host_upload_bytes\":{},\"host_readback_bytes\":{}}}\n",
            self.name, stats.kernels, ops, stats.upload, stats.readback,
        )
    }
}

#[derive(Default)]
struct Stats {
    kernels: usize,
    operations: BTreeSet<&'static str>,
    upload: usize,
    readback: usize,
}

struct Device {
    metal: bridge::Device,
    stats: RefCell<Stats>,
    session: Cell<u64>,
    tape: RefCell<Option<Tape>>,
}

#[derive(Clone)]
struct Shape {
    dims: Rc<[u32]>,
    count: u32,
}

impl Shape {
    fn scalar() -> Self {
        Self {
            dims: Rc::new([]),
            count: 1,
        }
    }

    fn array<T: Copy>(arr: &Arr<T>) -> Option<Self> {
        let count = u32::try_from(arr.data.len()).ok()?;
        let mut dims = Vec::with_capacity(arr.rank());
        dims.push(u32::try_from(arr.len()).ok()?);
        dims.extend_from_slice(arr.row_dims());
        Some(Self {
            dims: dims.into(),
            count,
        })
    }

    fn rows(&self) -> Option<Self> {
        let (&rows, suffix) = self.dims.split_first()?;
        if self.count == 0 {
            return None;
        }
        Some(Self {
            dims: Rc::from(suffix),
            count: self.count / rows,
        })
    }

    /// Atomic dyads pair common leading dimensions and repeat along trailing dimensions.
    fn pair(&self, other: &Self) -> Option<Self> {
        if self.count == 0 || other.count == 0 {
            return None;
        }
        if self.dims.starts_with(&other.dims) {
            return Some(self.clone());
        }
        if other.dims.starts_with(&self.dims) {
            return Some(other.clone());
        }
        None
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Float = 0,
    Int64 = 1,
    Int32 = 2,
}

impl Kind {
    const INT: Self = if Int::BITS == 32 {
        Self::Int32
    } else {
        Self::Int64
    };

    fn bytes(self, count: u32) -> usize {
        count as usize * if self == Self::Int64 { 8 } else { 4 }
    }
}

#[derive(Clone)]
struct Tensor {
    buf: Rc<Object>,
    kind: Kind,
    shape: Shape,
    track: Option<Track>,
}

enum Input<'a> {
    Device(&'a Tensor),
    Ints(&'a [Int], Shape),
    Floats(&'a [Flt], Shape),
}

impl<'a> Input<'a> {
    fn of(value: &'a Value, device: &Device) -> Option<Self> {
        match value {
            Value::Int(n) => Some(Self::Ints(std::slice::from_ref(n), Shape::scalar())),
            Value::Flt(n) => Some(Self::Floats(std::slice::from_ref(n), Shape::scalar())),
            Value::Ints(a) => Some(Self::Ints(&a.data, Shape::array(a)?)),
            Value::Flts(a) => Some(Self::Floats(&a.data, Shape::array(a)?)),
            Value::Ext(e) => {
                let array = e.0.as_any().downcast_ref::<Array>()?;
                std::ptr::eq(array.device.as_ref(), device).then_some(Self::Device(&array.tensor))
            }
            _ => None,
        }
    }

    fn shape(&self) -> &Shape {
        match self {
            Self::Device(a) => &a.shape,
            Self::Ints(_, shape) | Self::Floats(_, shape) => shape,
        }
    }

    fn on_device(&self) -> bool {
        matches!(self, Self::Device(_))
    }
}

#[derive(Clone, Copy)]
enum Op {
    Add = 0,
    Sub = 1,
    Mul = 2,
}

impl Op {
    fn parse(glyph: &str) -> Option<Self> {
        match glyph {
            "+" => Some(Self::Add),
            "-" => Some(Self::Sub),
            "*" => Some(Self::Mul),
            _ => None,
        }
    }

    fn glyph(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
        }
    }
}

#[derive(Clone, Copy)]
struct Track {
    session: u64,
    slot: usize,
}

struct Tape {
    session: u64,
    nodes: Vec<Node>,
}

enum Node {
    Input,
    Binary(Op, Tensor, Tensor),
    Sum(Tensor),
}

impl Device {
    fn upload<T: bridge::Scalar>(&self, kind: Kind, data: &[T], shape: Shape) -> Result<Tensor> {
        let buf = self.metal.upload(data)?;
        self.stats.borrow_mut().upload += std::mem::size_of_val(data);
        Ok(Tensor {
            buf: Rc::new(buf),
            kind,
            shape,
            track: None,
        })
    }

    fn upload_floats(&self, data: &[Flt], shape: Shape) -> Result<Tensor> {
        let mut floats = try_vec(data.len())?;
        floats.extend(data.iter().map(|&n| n as f32));
        self.upload(Kind::Float, &floats, shape)
    }

    fn tensor(&self, input: Input<'_>) -> Result<Tensor> {
        match input {
            Input::Device(tensor) => Ok(tensor.clone()),
            Input::Ints(data, shape) => self.upload(Kind::INT, data, shape),
            Input::Floats(data, shape) => self.upload_floats(data, shape),
        }
    }

    fn readback<T: bridge::Scalar + Default>(&self, tensor: &Tensor) -> Result<Vec<T>> {
        let n = tensor.shape.count as usize;
        let mut data = try_vec(n)?;
        data.resize(n, T::default());
        self.metal.download(&tensor.buf, &mut data)?;
        self.stats.borrow_mut().readback += std::mem::size_of_val(data.as_slice());
        Ok(data)
    }

    fn dispatch(
        &self,
        kernel: Kernel,
        a: &Tensor,
        b: Option<&Tensor>,
        kind: Kind,
        shape: Shape,
        threads: u32,
        params: &[u32],
    ) -> Result<Tensor> {
        let buf = self.metal.dispatch(
            kernel,
            &a.buf,
            b.map(|b| b.buf.as_ref()),
            kind.bytes(shape.count),
            threads,
            params,
        )?;
        self.stats.borrow_mut().kernels += 1;
        Ok(Tensor {
            buf: Rc::new(buf),
            kind,
            shape,
            track: None,
        })
    }

    fn record(&self, node: Node) -> Option<Track> {
        let mut tape = self.tape.borrow_mut();
        let tape = tape.as_mut()?;
        let track = Track {
            session: tape.session,
            slot: tape.nodes.len(),
        };
        tape.nodes.push(node);
        Some(track)
    }

    fn is_tracked(&self, tensor: &Tensor) -> bool {
        let Some(track) = tensor.track else {
            return false;
        };
        self.tape
            .borrow()
            .as_ref()
            .is_some_and(|tape| track.session == tape.session)
    }

    fn binary(&self, op: Op, a: &Tensor, b: &Tensor, shape: Shape) -> Result<Tensor> {
        let kind = if a.kind == Kind::Float || b.kind == Kind::Float {
            Kind::Float
        } else {
            Kind::INT
        };
        let n = shape.count;
        let params = [
            n,
            n / a.shape.count,
            n / b.shape.count,
            op as u32,
            a.kind as u32,
            b.kind as u32,
        ];
        let mut result = self.dispatch(Kernel::Elementwise, a, Some(b), kind, shape, n, &params)?;
        self.stats.borrow_mut().operations.insert(op.glyph());
        if self.is_tracked(a) || self.is_tracked(b) {
            result.track = self.record(Node::Binary(op, a.clone(), b.clone()));
        }
        Ok(result)
    }

    fn sum(&self, a: &Tensor, shape: Shape) -> Result<Tensor> {
        if a.shape.dims[0] == 1 {
            let mut value = a.clone();
            value.shape = shape;
            if self.is_tracked(a) {
                value.track = self.record(Node::Sum(a.clone()));
            }
            return Ok(value);
        }
        let mut result = if shape.dims.is_empty() {
            let mut values = a.clone();
            let mut n = a.shape.count;
            while n > 1 {
                let blocks = n.div_ceil(bridge::BLOCK_WIDTH);
                let partial = Shape {
                    dims: Rc::from([blocks]),
                    count: blocks,
                };
                let params = [n, a.kind as u32];
                values = self.dispatch(
                    Kernel::SumBlocks,
                    &values,
                    None,
                    a.kind,
                    partial,
                    n,
                    &params,
                )?;
                n = blocks;
            }
            values.shape = shape;
            values
        } else {
            let cols = shape.count;
            let params = [a.shape.dims[0], cols, a.kind as u32];
            self.dispatch(Kernel::SumColumns, a, None, a.kind, shape, cols, &params)?
        };
        self.stats.borrow_mut().operations.insert("+/");
        if self.is_tracked(a) {
            result.track = self.record(Node::Sum(a.clone()));
        }
        Ok(result)
    }

    fn value(self: &Rc<Self>, tensor: Tensor) -> Value {
        Value::Ext(Rc::new(ExtBox(Box::new(Array {
            device: self.clone(),
            tensor,
            native: OnceCell::new(),
        }))))
    }

    fn numeric(
        self: &Rc<Self>,
        glyph: &str,
        a: &Value,
        b: Option<&Value>,
    ) -> Option<Result<Value>> {
        if let Some(b) = b {
            let op = Op::parse(glyph)?;
            let (left, right) = (Input::of(a, self)?, Input::of(b, self)?);
            let shape = left.shape().pair(right.shape())?;
            if shape.dims.is_empty() && !left.on_device() && !right.on_device() {
                return None;
            }
            let calc = || {
                self.binary(op, &self.tensor(left)?, &self.tensor(right)?, shape)
                    .map(|t| self.value(t))
            };
            return Some(calc());
        }
        if glyph != "+/" {
            return None;
        }
        let input = Input::of(a, self)?;
        if let Input::Device(tensor) = &input {
            if tensor.shape.dims.is_empty() {
                return Some(Ok(a.clone()));
            }
        }
        let shape = input.shape().rows()?;
        let calc = || self.sum(&self.tensor(input)?, shape).map(|t| self.value(t));
        Some(calc())
    }

    fn gradient(self: &Rc<Self>, ip: &mut Interp, f: &Value, point: &Value) -> Result<Value> {
        if self.tape.borrow().is_some() {
            return Err(Error::typ("Metal does not support nested gradients"));
        }
        let point =
            Input::of(point, self).ok_or_else(|| Error::typ(":> expects a numeric point"))?;
        let shape = point.shape().clone();
        if shape.count == 0 {
            return Err(Error::shape(":> expects a nonempty point"));
        }
        let mut input = match point {
            Input::Ints(data, _) => {
                let mut floats = try_vec(data.len())?;
                floats.extend(data.iter().map(|&n| n as f32));
                self.upload(Kind::Float, &floats, shape)?
            }
            Input::Device(t) if t.kind != Kind::Float => {
                let zero = self.upload(Kind::Float, &[0.0f32], Shape::scalar())?;
                self.binary(Op::Add, t, &zero, shape)?
            }
            _ => self.tensor(point)?,
        };
        let session = self.session.get().wrapping_add(1);
        self.session.set(session);
        *self.tape.borrow_mut() = Some(Tape {
            session,
            nodes: vec![Node::Input],
        });
        input.track = Some(Track { session, slot: 0 });
        let result = ip.apply(f, vec![self.value(input)], None);
        let tape = self.tape.borrow_mut().take().expect("active Metal tape");
        let value = result?;
        let Some(Input::Device(output)) = Input::of(&value, self) else {
            return Err(Error::typ(
                ":> function result does not depend on the differentiable input",
            ));
        };
        if output.shape.count != 1 {
            return Err(Error::shape(":> requires a single-element loss"));
        }
        let track = output
            .track
            .filter(|t| t.session == session)
            .ok_or_else(|| {
                Error::typ(":> function result does not depend on the differentiable input")
            })?;
        let seed = self.upload(Kind::Float, &[1.0f32], output.shape.clone())?;
        self.backward(tape, track.slot, seed).map(|t| self.value(t))
    }

    fn backward(&self, tape: Tape, output: usize, seed: Tensor) -> Result<Tensor> {
        let mut gradients = try_vec(tape.nodes.len())?;
        gradients.resize_with(tape.nodes.len(), || None);
        gradients[output] = Some(seed);
        for (slot, node) in tape.nodes.into_iter().enumerate().rev() {
            let Some(grad) = gradients[slot].take() else {
                continue;
            };
            match node {
                Node::Input => return Ok(grad),
                Node::Sum(input) => {
                    let mut expanded = grad;
                    if input.shape.count == expanded.shape.count {
                        expanded.shape = input.shape.clone();
                    } else {
                        let n = input.shape.count;
                        let params = [n, expanded.shape.count];
                        expanded = self.dispatch(
                            Kernel::GradientTile,
                            &expanded,
                            None,
                            Kind::Float,
                            input.shape.clone(),
                            n,
                            &params,
                        )?;
                    }
                    self.accumulate(&mut gradients, tape.session, &input, expanded)?;
                }
                Node::Binary(op, a, b) => {
                    for (input, other, left) in [(&a, &b, true), (&b, &a, false)] {
                        if !input.track.is_some_and(|t| t.session == tape.session) {
                            continue;
                        }
                        let contribution = match op {
                            Op::Mul => self.binary(Op::Mul, &grad, other, grad.shape.clone())?,
                            Op::Sub if !left => {
                                let zero = self.upload(Kind::Float, &[0.0f32], Shape::scalar())?;
                                self.binary(Op::Sub, &zero, &grad, grad.shape.clone())?
                            }
                            _ => grad.clone(),
                        };
                        self.accumulate(&mut gradients, tape.session, input, contribution)?;
                    }
                }
            }
        }
        Err(Error::internal(
            "Metal gradient tape has no input contribution",
        ))
    }

    fn accumulate(
        &self,
        gradients: &mut [Option<Tensor>],
        session: u64,
        input: &Tensor,
        mut grad: Tensor,
    ) -> Result<()> {
        let track = input
            .track
            .filter(|t| t.session == session)
            .ok_or_else(|| Error::internal("invalid Metal gradient edge"))?;
        let n = input.shape.count;
        if n < grad.shape.count {
            let params = [n, grad.shape.count / n];
            grad = self.dispatch(
                Kernel::GradientCollapse,
                &grad,
                None,
                Kind::Float,
                input.shape.clone(),
                n,
                &params,
            )?;
        } else {
            grad.shape = input.shape.clone();
        }
        if let Some(old) = gradients[track.slot].take() {
            grad = self.binary(Op::Add, &old, &grad, input.shape.clone())?;
        }
        gradients[track.slot] = Some(grad);
        Ok(())
    }
}

struct Array {
    device: Rc<Device>,
    tensor: Tensor,
    native: OnceCell<Value>,
}

impl Array {
    fn native(&self, what: &str) -> Result<Value> {
        if self.device.is_tracked(&self.tensor) {
            return Err(Error::typ(format!(
                "{what}: cannot read back a value while differentiating"
            )));
        }
        self.materialize()
    }

    fn materialize(&self) -> Result<Value> {
        if let Some(value) = self.native.get() {
            return Ok(value.clone());
        }
        let shape = &self.tensor.shape.dims;
        let value = if self.tensor.kind == Kind::Float {
            let data: Vec<f32> = self.device.readback(&self.tensor)?;
            if shape.is_empty() {
                Value::Flt(data[0] as Flt)
            } else {
                let mut out = try_vec(data.len())?;
                out.extend(data.into_iter().map(|n| n as Flt));
                Value::Flts(Rc::new(Arr::shaped(out, shape)))
            }
        } else {
            let data: Vec<Int> = self.device.readback(&self.tensor)?;
            if shape.is_empty() {
                Value::Int(data[0])
            } else {
                Value::Ints(Rc::new(Arr::shaped(data, shape)))
            }
        };
        let _ = self.native.set(value.clone());
        Ok(value)
    }
}

impl Ext for Array {
    fn type_name(&self) -> &str {
        "Metal array"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn host_shape(&self) -> Option<Vec<u32>> {
        Some(self.tensor.shape.dims.to_vec())
    }

    fn host_native(&self, what: &str) -> Option<Result<Value>> {
        Some(self.native(what))
    }

    fn host_items(&self) -> Option<Result<Value>> {
        Some(self.native("destructuring"))
    }

    fn host_truthy(&self) -> Option<bool> {
        if self.tensor.shape.dims.is_empty() {
            // The interpreter must choose a branch from the computed value.
            self.materialize().ok().map(|v| v.truthy())
        } else {
            Some(true)
        }
    }

    fn host_collect(&self, items: &[Value]) -> Option<Result<Value>> {
        let native = || {
            let mut values = try_vec(items.len())?;
            for value in items {
                let materialized = match Input::of(value, &self.device) {
                    Some(Input::Device(_)) => value.unhost("list construction")?,
                    _ => value.clone(),
                };
                values.push(materialized);
            }
            Value::from_items(values)
        };
        Some(native())
    }

    fn host_format(&self, display: bool) -> Option<String> {
        let value = self.native("formatting").ok()?;
        klong_core::fmt::format_value(&klong_core::sym::Interner::new(), &value, display).ok()
    }

    fn host_monad(&self, ip: &mut Interp, glyph: &str, x: &Value) -> Option<Result<Value>> {
        let id = ip.syms.lookup(glyph)?;
        if glyph == "-" {
            let zero = if self.tensor.kind == Kind::Float {
                Value::Flt(-0.0)
            } else {
                Value::Int(0)
            };
            return self.device.numeric("-", &zero, Some(x));
        }
        Some(self.native(glyph).and_then(|value| ip.monad(id, value)))
    }

    fn host_dyad(
        &self,
        ip: &mut Interp,
        glyph: &str,
        x: &Value,
        y: &Value,
    ) -> Option<Result<Value>> {
        let id = ip.syms.lookup(glyph)?;
        Some(
            x.unhost(glyph)
                .and_then(|a| ip.dyad(id, a, y.unhost(glyph)?)),
        )
    }
}
