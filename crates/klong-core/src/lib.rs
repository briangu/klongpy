pub mod array;
pub mod kernels;

pub use array::Array;
pub use kernels::{
    par_add_inplace, par_sub_inplace, par_mul_inplace, par_div_inplace,
    par_add_inplace_f32, par_sub_inplace_f32, par_mul_inplace_f32,
    par_div_inplace_f32, map_inplace_f64, map_inplace_f32,
};

#[cfg(feature = "duckdb")]
use arrow::record_batch::RecordBatch;
#[cfg(feature = "duckdb")]
use duckdb::Result as DuckResult;

#[cfg(feature = "duckdb")]
pub fn duckdb_query_arrow(_sql: &str, batches: &[RecordBatch]) -> DuckResult<Vec<RecordBatch>> {
    // Placeholder: real implementation would execute SQL against DuckDB
    Ok(batches.to_vec())
}
