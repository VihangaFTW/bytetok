//! Fast BPE (Byte-Pair Encoding) trainer using Algorithm 2
//! from "Byte Pair Encoding is Suboptimal for Language Model Pretraining"
//!
//! This is a PyO3 extension module providing Python bindings.

#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(unused_must_use)]

use pyo3::prelude::*;

mod types;
mod bpe_encoder;

mod encoder;

mod bpe_trainer;
use bpe_trainer::BPETrainer;



/// Python wrapper for BPE trainer.
#[pyclass]
pub struct RustBPETrainer {
    trainer: BPETrainer,
}

#[pymethods]
impl RustBPETrainer {
    #[new]
    fn new(tokens: Vec<usize>, next_token_id: usize) -> Self {
        RustBPETrainer {
            trainer: BPETrainer::new(tokens, next_token_id),
        }
    }

    fn train(&mut self, num_merges: usize) {
        self.trainer.train(num_merges);
    }

    fn merge_step(&mut self) -> bool {
        self.trainer.merge_step()
    }

    fn get_tokens(&self) -> Vec<usize> {
        self.trainer.get_encodings()
    }

    fn get_merge_history(&self) -> Vec<((usize, usize), usize)> {
        self.trainer.get_merge_history()
    }

    fn print_state(&self) {
        self.trainer.print_state();
    }
}

#[pymodule]
fn _bpe_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustBPETrainer>()?;
    Ok(())
}
