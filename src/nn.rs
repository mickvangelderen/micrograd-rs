use crate::engine::{NodeId, Operations};

pub fn fully_connected_layer(
    ops: &mut Operations,
    inputs: &[NodeId],
    output_len: usize,
) -> Vec<(Vec<NodeId>, NodeId, NodeId)> {
    (0..output_len)
        .map(|_| {
            let weights: Vec<_> = ops.vars_vec(inputs.len());
            let bias = ops.var();
            let output = inputs
                .iter()
                .copied()
                .zip(weights.iter().copied())
                .fold(bias, |sum, (a, b)| ops.insert(sum + a * b));
            (weights, bias, output)
        })
        .collect()
}
