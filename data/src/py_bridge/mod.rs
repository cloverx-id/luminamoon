mod pipeline;
mod state;
mod text;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::error::DataLoaderError;

pub use pipeline::{
    clean_batch_records, clean_single_record, dataloader_cpu_isa, extract_py_columns_table,
    fused_pipeline_process_batch, fused_pipeline_process_batch_indexed, serialize_table_to_bytes,
    suggest_closest_matches,
};
pub use state::{PyStateTracker, PyStreamCoordinate, PyStreamDeduplicator};
pub use text::{PyMinHashDeduplicator, PyQualityFilter, PyStringTensor};

pub(super) fn to_py_err(err: DataLoaderError) -> PyErr {
    match err {
        DataLoaderError::InvalidConfig(msg) | DataLoaderError::ColumnNotFound(msg) => {
            PyValueError::new_err(msg)
        }
        DataLoaderError::InvalidCoordinate => PyValueError::new_err(err.to_string()),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

pub fn register_dataloader_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    parent_module.add_class::<PyStreamCoordinate>()?;
    parent_module.add_class::<PyStreamDeduplicator>()?;
    parent_module.add_class::<PyStateTracker>()?;
    parent_module.add_class::<PyQualityFilter>()?;
    parent_module.add_class::<PyMinHashDeduplicator>()?;
    parent_module.add_class::<PyStringTensor>()?;
    parent_module.add_function(wrap_pyfunction!(dataloader_cpu_isa, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(suggest_closest_matches, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(serialize_table_to_bytes, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(clean_single_record, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(clean_batch_records, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(
        fused_pipeline_process_batch,
        parent_module
    )?)?;
    parent_module.add_function(wrap_pyfunction!(
        fused_pipeline_process_batch_indexed,
        parent_module
    )?)?;
    Ok(())
}
