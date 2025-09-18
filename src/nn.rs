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

impl FullyConnectedLayer {
    pub fn new<AF: Fn(NodeId) -> A, A: Insertable<Output = NodeId>>(
        inputs: View<&[NodeId], (B, I)>,
        output_count: O,
        ops: &mut Operations,
        activation_fn: AF,
    ) -> Self {
        let (batch_count, input_count) = inputs.len();
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

    fn bias_offset(&self) -> usize {
        (self.input_count, self.output_count).product()
    }

    fn output_offset(&self) -> usize {
        self.bias_offset() + usize::from(self.output_count)
    }

    pub fn weights(&self) -> View<&[NodeId], (I, O)> {
        View::new(
            &self.vars[0..self.bias_offset()],
            (self.input_count, self.output_count),
        )
    }

    pub fn biases(&self) -> View<&[NodeId], (O,)> {
        View::new(
            &self.vars[self.bias_offset()..self.output_offset()],
            (self.output_count,),
        )
    }

    pub fn outputs(&self) -> View<&[NodeId], (B, O)> {
        View::new(
            &self.vars[self.output_offset()..],
            (self.batch_count, self.output_count),
        )
    }
}
