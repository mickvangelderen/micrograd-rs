use crate::engine::{Insertable, NodeId, Operations};

pub struct FullyConnectedLayer {
    pub input_count: usize,
    pub output_count: usize,
    vars: Box<[NodeId]>,
}

impl FullyConnectedLayer {
    pub fn new<AF: Fn(NodeId) -> A, A: Insertable<Output = NodeId>>(inputs: &[NodeId], output_count: usize, ops: &mut Operations, activation_fn: AF) -> Self {
        let input_count = inputs.len();
        let weight_count = input_count * output_count;
        let bias_count = output_count;
        let mut vars = ops.vars_vec(weight_count + bias_count);

        for output_index in 0..output_count {
            let weight = |input_index| vars[output_index * input_count + input_index];
            let bias = vars[weight_count + output_index];
            let output = inputs
                .iter()
                .copied()
                .zip((0..input_count).map(weight))
                .fold(bias, |sum, (a, b)| {
                    let node = ops.insert(sum + a * b);
                    ops.insert(activation_fn(node))
                });
            vars.push(output);
        }

        Self {
            input_count,
            output_count,
            vars: vars.into_boxed_slice(),
        }
    }

    fn weight_count(&self) -> usize {
        self.input_count * self.output_count
    }

    pub fn weights(&self) -> &[NodeId] {
        &self.vars[0..self.weight_count()]
    }

    pub fn biases(&self) -> &[NodeId] {
        let weight_count = self.weight_count();
        &self.vars[weight_count..weight_count + self.output_count]
    }

    pub fn outputs(&self) -> &[NodeId] {
        &self.vars[self.weight_count() + self.output_count..]
    }
}
