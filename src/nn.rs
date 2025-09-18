use std::ops::Range;

use crate::engine::{Insertable, NodeId, Operations};
use crate::impl_index;
use crate::view::{Index as _, IndexTuple, View};

impl_index!(I);
impl_index!(B);
impl_index!(O);

pub struct FullyConnectedLayer {
    pub batch_count: B,
    pub input_count: I,
    pub output_count: O,
    vars: Box<[NodeId]>,
}

#[inline]
pub fn batched_output_to_input((b, o): (B, O)) -> (B, I) {
    (b, o.reindex())
}

impl FullyConnectedLayer {
    pub fn new<AF: Fn(NodeId) -> A, A: Insertable<Output = NodeId>>(
        inputs: View<&[NodeId], (B, I)>,
        output_count: O,
        ops: &mut Operations,
        activation_fn: AF,
    ) -> Self {
        let (batch_count, input_count) = *inputs.len();
        let weight_dims = (input_count, output_count);
        let bias_dims = (output_count,);

        // We're going to write all nodes into a single vec so we need to
        // compute some offsets.
        let bias_offset = weight_dims.product();
        let output_offset = bias_offset + bias_dims.product();

        let mut vars = ops.vars_vec(output_offset);

        vars.reserve((batch_count, output_count).product());
        for (batch_index, output_index) in (batch_count, output_count).indices() {
            let weights = View::new(&vars[0..bias_offset], weight_dims);
            let biases = View::new(&vars[bias_offset..output_offset], bias_dims);

            let input_iter = input_count.indices().map(|i| inputs[(batch_index, i)]);
            let weight_iter = input_count.indices().map(|i| weights[(i, output_index)]);
            let output = std::iter::zip(input_iter, weight_iter).fold(
                biases[(output_index,)],
                |sum, (a, b)| {
                    let node = ops.insert(sum + a * b);
                    ops.insert(activation_fn(node))
                },
            );

            vars.push(output);
        }

        Self {
            batch_count,
            input_count,
            output_count,
            vars: vars.into_boxed_slice(),
        }
    }

    #[inline]
    fn bias_offset(&self) -> usize {
        (self.input_count, self.output_count).product()
    }

    #[inline]
    fn output_offset(&self) -> usize {
        self.bias_offset() + usize::from(self.output_count)
    }

    #[inline]
    fn weight_indices(&self) -> Range<usize> {
        0..self.bias_offset()
    }

    #[inline]
    fn bias_indices(&self) -> Range<usize> {
        self.bias_offset()..self.output_offset()
    }

    #[inline]
    fn output_indices(&self) -> std::ops::RangeFrom<usize> {
        self.output_offset()..
    }

    #[inline]
    pub fn weights(&self) -> View<&[NodeId], (I, O)> {
        let range = self.weight_indices();
        View::new(&self.vars[range], (self.input_count, self.output_count))
    }

    #[inline]
    pub fn weights_mut(&mut self) -> View<&mut [NodeId], (I, O)> {
        let range = self.weight_indices();
        View::new(&mut self.vars[range], (self.input_count, self.output_count))
    }

    #[inline]
    pub fn biases(&self) -> View<&[NodeId], (O,)> {
        let range = self.bias_indices();
        View::new(&self.vars[range], (self.output_count,))
    }

    #[inline]
    pub fn biases_mut(&mut self) -> View<&mut [NodeId], (O,)> {
        let range = self.bias_indices();
        View::new(&mut self.vars[range], (self.output_count,))
    }

    #[inline]
    pub fn outputs(&self) -> View<&[NodeId], (B, O)> {
        let range = self.output_indices();
        View::new(&self.vars[range], (self.batch_count, self.output_count))
    }

    #[inline]
    pub fn outputs_mut(&mut self) -> View<&mut [NodeId], (B, O)> {
        let range = self.output_indices();
        View::new(&mut self.vars[range], (self.batch_count, self.output_count))
    }
}

pub fn input_layer_vec(len: (B, O), ops: &mut Operations) -> View<Vec<NodeId>, (B, O)> {
    let data = ops.vars_vec(len.product());
    View::new(data, len)
}
