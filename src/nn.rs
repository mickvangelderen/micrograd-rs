// use crate::engine::{NodeId, Operations};
// use crate::syntax::Expr;

// fn add_n<I: IntoIterator<Item = NodeId>>(args: I, ops: &mut Operations) -> Option<NodeId> {
//     let mut args = args.into_iter();
//     let mut acc = args.next()?;
//     for arg in args {
//         acc = ops.add(acc, arg)
//     }
//     Some(acc)
// }

// fn lit_n(ops: &mut Operations, len: usize) -> Vec<NodeId> {
//     (0..len).map(|_| ops.var()).collect()
// }

// fn fully_connected_layer(
//     ops: &mut Operations,
//     inputs: &[NodeId],
//     output_len: usize,
// ) -> Vec<(Vec<NodeId>, NodeId, NodeId)> {
//     (0..output_len)
//         .map(|_| {
//             let weights = lit_n(ops, inputs.len());
//             let bias = ops.var();
//             let output =
//                 inputs
//                     .iter()
//                     .copied()
//                     .zip(weights.iter().copied())
//                     .fold(bias, |sum, (a, b)| {
//                         (sum + a * b).insert(ops)
//                         // ops.expr(sum + a * b) // TODO: Make this work.
//                     });
//             (weights, bias, output)
//         })
//         .collect()
// }
