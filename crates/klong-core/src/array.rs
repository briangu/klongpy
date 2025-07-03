#[derive(Clone)]
pub struct Array<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl<T: Clone> Array<T> {
    pub fn from_slice(slice: &[T], shape: &[usize]) -> Self {
        let data = slice.to_vec();
        let shape = shape.to_vec();
        let mut strides = vec![1; shape.len()];
        for i in (1..shape.len()).rev() {
            strides[i - 1] = strides[i] * shape[i];
        }
        Self { data, shape, strides }
    }
}
