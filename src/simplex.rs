use crate::linear_programm::LinearProgram;
use crate::Variable;
use crate::Coefficient;
use crate::error::SimplexError;

#[derive(Debug, Clone)]
pub struct Simplex {
    index: usize,
    historic: Vec<LinearProgram>,
}

impl Simplex {
    pub fn first_simplex(&mut self) -> Option<LinearProgram> {
        if self.historic.len() > 0 {
            Some(self.historic[0].clone())
        } else {
            None
        }
    } 
    fn is_first_step(&self) -> bool {
        self.index == 0
    }

    pub fn next_step(&mut self, use_bland_rule: bool) -> Result<(), SimplexError> {
        if let Some(var) = self
            .current_state()
            .linear_function
            .first_positive_coefficient(use_bland_rule)
        {
            if self.index == self.historic.len() - 1 {
                let mut new = self.current_state().clone();
                new.pivot(var)?;
                self.historic.push(new);
            }
            self.index += 1;
            Ok(())
        } else {
            Err(SimplexError::AlreadyOptimal)
        }
    }

    pub fn previous_step(&mut self) {
        if !self.is_first_step() {
            self.index -= 1;
        }
    }

    /// Returns a reference to the current state of the algorithm
    pub fn current_state(&self) -> &LinearProgram {
        &self.historic[self.index]
    }

    pub fn current_point(&self) -> Vec<f32> {
        self.current_state().point()
    }

    pub fn current_values(&self) -> Vec<(Variable, Coefficient)> {
        self.current_state().values()
    }

    pub fn draw(&self) {}

    pub fn every_points(&self) -> Vec<Vec<f32>> {
        self.current_state().constraints.every_points()
    }
}

impl From<LinearProgram> for Simplex {
    fn from(value: LinearProgram) -> Self {
        Simplex {
            index: 0,
            historic: vec![value],
        }
    }
}
