use pyo3::prelude::*;

#[pyfunction]
fn encode(text: &str) -> Vec<u8> {
    text.as_bytes().to_vec()
}

#[pyfunction]
fn decode(tokens: Vec<u8>) -> String {
    String::from_utf8_lossy(&tokens).into_owned()
}

#[pymodule]
fn croba_tokenizer_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(encode, module)?)?;
    module.add_function(wrap_pyfunction!(decode, module)?)?;
    Ok(())
}

