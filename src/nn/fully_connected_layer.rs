use std::ops::Range;

use split_spare::SplitSpare;

use crate::{
    engine::{Gradients, Insertable, NodeId, Operations, Values},
    nn::{B, I, O},
    view::{Index as _, IndexTuple, View},
};

pub struct FullyConnectedLayer {
    pub batch_size: B,
    pub input_size: I,
    pub output_size: O,
    vars: Box<[NodeId]>,
}

#[inline]
pub fn batched_output_to_input((b, o): (B, O)) -> (B, I) {
    (b, o.reindex())
}

impl FullyConnectedLayer {
    pub fn new<AF: Fn(NodeId) -> A, A: Insertable<Output = NodeId>>(
        inputs: View<&[NodeId], (B, I)>,
        output_size: O,
        ops: &mut Operations,
        activation_fn: AF,
    ) -> Self {
        let (batch_size, input_size) = inputs.shape();
        let weight_dims = (input_size, output_size);
        let bias_dims = (output_size,);

        // We're going to write all nodes into a single vec so we need to
        // compute some offsets.
        let bias_offset = weight_dims.product();
        let output_offset = bias_offset + bias_dims.product();

        let mut vars = ops.vars_vec(output_offset);

        let (init, mut spare) = vars.reserve_split_spare((batch_size, output_size).product());
        let weights = View::new(&init[0..bias_offset], weight_dims);
        let biases = View::new(&init[bias_offset..output_offset], bias_dims);
        spare.extend((batch_size, output_size).indices().map(|(batch_index, output_index)| {
            let input_iter = input_size.indices().map(|i| inputs[(batch_index, i)]);
            let weight_iter = input_size.indices().map(|i| weights[(i, output_index)]);
            std::iter::zip(input_iter, weight_iter).fold(biases[(output_index,)], |sum, (a, b)| {
                let node = ops.insert(sum + a * b);
                ops.insert(activation_fn(node))
            })
        }));

        Self {
            batch_size,
            input_size,
            output_size,
            vars: vars.into_boxed_slice(),
        }
    }

    #[inline]
    fn bias_offset(&self) -> usize {
        (self.input_size, self.output_size).product()
    }

    #[inline]
    fn output_offset(&self) -> usize {
        self.bias_offset() + usize::from(self.output_size)
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
        View::new(&self.vars[range], (self.input_size, self.output_size))
    }

    #[inline]
    pub fn biases(&self) -> View<&[NodeId], (O,)> {
        let range = self.bias_indices();
        View::new(&self.vars[range], (self.output_size,))
    }

    #[inline]
    pub fn outputs(&self) -> View<&[NodeId], (B, O)> {
        let range = self.output_indices();
        View::new(&self.vars[range], (self.batch_size, self.output_size))
    }

    #[inline]
    pub fn parameters(&self) -> impl Iterator<Item = NodeId> {
        self.weights()
            .into_iter()
            .copied()
            .chain(self.biases().into_iter().copied())
    }

    #[inline]
    pub fn init_parameters(&self, values: &mut Values, rng: &mut impl rand::Rng) {
        use rand::distr::Distribution;

        let dist = rand::distr::Uniform::new(-0.05, 0.05).unwrap();

        for (weight, value) in self.weights().iter().copied().zip(dist.sample_iter(rng)) {
            values[weight] = value;
        }
        for bias in self.biases().iter().copied() {
            values[bias] = 0.0;
        }
    }

    #[inline]
    pub fn update_weights(&self, values: &mut Values, gradients: &Gradients) {
        for node in self.parameters() {
            values[node] -= gradients[node];
        }
    }
}

pub fn input_layer_vec(len: (B, O), ops: &mut Operations) -> View<Vec<NodeId>, (B, O)> {
    let data = ops.vars_vec(len.product());
    View::new(data, len)
}
