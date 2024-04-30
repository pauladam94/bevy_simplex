mod error;
mod constraint;
mod linear_function;
mod linear_programm;
mod simplex;
pub mod gui_simplex;

pub use crate::error::SimplexError;
pub use crate::linear_function::{Coefficient, Variable};
pub use crate::simplex::Simplex;
pub use crate::gui_simplex::UiState;
pub use constraint::Constraints;
pub use linear_function::LinearFunction;
pub use linear_programm::LinearProgram;
pub use std::collections::HashSet;
