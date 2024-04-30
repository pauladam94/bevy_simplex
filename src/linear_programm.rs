use crate::constraint::Constraints;
use crate::error::SimplexError;
use crate::linear_function::LinearFunction;
use crate::linear_function::{Coefficient, Variable};
use itertools::Itertools;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct LinearProgram {
    pub linear_function: LinearFunction,
    pub constraints: Constraints,
}

impl LinearProgram {
    /// Do a Pivot operation on the linear program according to the variable given
    pub fn pivot(&mut self, var: String) -> Result<(), SimplexError> {
        let max_constraint_index = self
            .constraints
            .most_restrictive(&var)
            .ok_or(SimplexError::Unbounded)?;
        self.constraints.pivot(max_constraint_index, &var);
        self.linear_function
            .replace(&var, &self.constraints[max_constraint_index].right);
        Ok(())
    }

    pub fn is_valid(&self) -> bool {
        self.constraints.is_valid()
    }

    pub fn is_unbounded(&self) -> bool {
        self.linear_function
            .var_iter()
            .any(|v| self.constraints.most_restrictive(v).is_none())
    }

    /// only works on a proper linear program which is verif by is_valid function
    pub fn point(&self) -> Vec<f32> {
        if !self.is_valid() {
            panic!("Linear program is not valid");
        }
        let variables = self.non_gap_variables();
        let mut point = vec![0.0; variables.len()];

        for constraint in self.constraints.iter() {
            if let Some(left_variable) = constraint.left.name_single_variable() {
                if let Some(index) = variables.iter().position(|v| *v == left_variable) {
                    point[index] = constraint.right.constant;
                }
            }
        }
        point
    }

    pub fn values(&self) -> Vec<(Variable, Coefficient)> {
        let variables = self.non_gap_variables();
        let values = self.point();

        variables.into_iter().zip(values).collect()
    }

    /// Give every non gap variables of a linear program sorted by alphabetical order
    pub fn non_gap_variables(&self) -> Vec<String> {
        let mut var_set: HashSet<Variable> =
            HashSet::from_iter(self.linear_function.non_gap_variables());
        for v in self.constraints.non_gap_variables() {
            var_set.insert(v);
        }
        var_set.into_iter().sorted().collect()
    }

    pub fn out_of_base_variables(&self) -> Vec<Variable> {
        let mut variables = HashSet::new();
        for constraint in self.constraints.iter() {
            for var in constraint.right.var_iter() {
                variables.insert(var);
            }
        }
        variables.into_iter().cloned().collect()
    }
}

impl std::fmt::Display for LinearProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "max {}", self.linear_function)?;
        write!(f, "{}", self.constraints)
    }
}
