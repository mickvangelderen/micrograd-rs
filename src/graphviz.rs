use crate::engine::{Binary, NodeId, Op, Operations, Unary};
use std::io::Write;

pub fn export_to_dot<
    'l,
    W: Write,
    L: Fn(NodeId) -> &'l str,
    R: Fn(NodeId) -> Option<usize>,
>(
    ops: &Operations,
    labels: L,
    ranks: R,
    writer: &mut W,
) -> std::io::Result<()> {
    writeln!(writer, "digraph ComputationGraph {{")?;
    writeln!(writer, "    rankdir=LR;")?;
    writeln!(writer, "    node [shape=record, style=\"rounded,filled\"];")?;

    let should_emit_value_node = |node: NodeId| -> bool {
        match ops[node] {
            Op::Nullary(crate::engine::Nullary::Var) => true, // Always emit variables
            _ => !labels(node).is_empty(), // Only emit value nodes if they have a label
        }
    };

    // Collect rank groups from emitted nodes
    let rank_groups: std::collections::HashMap<usize, Vec<NodeId>> = ops
        .nodes()
        .fold(Default::default(), |mut map, node| {
            if let Some(rank) = ranks(node) {
                map.entry(rank).or_default().push(node);
            }
            map
        });

    // Emit value nodes
    for node in ops.nodes() {
        if should_emit_value_node(node) {
            let index = usize::from(node);
            let label = labels(node);
            let fillcolor = match ops[node] {
                Op::Nullary(crate::engine::Nullary::Var) => "lightblue",
                _ => "lightyellow",
            };

            writeln!(
                writer,
                "    n{index} [label=\"{label}\", shape=box, fillcolor={fillcolor}];",
            )?;
        }
    }

    // Emit operation nodes
    for node in ops.nodes() {
        let index = usize::from(node);

        match ops[node] {
            Op::Nullary(crate::engine::Nullary::Var) => {
                // Nothing to do.
            }
            Op::Unary(unary_op, _) => {
                let label = unary_op_to_str(unary_op);
                writeln!(
                    writer,
                    "    op{index} [label=\"{label}\", shape=diamond, regular=true, fillcolor=lightgreen, width=0.5, height=0.5, fixedsize=true];"
                )?;
            }
            Op::Binary(binary_op, _) => {
                let label = binary_op_to_str(binary_op);
                writeln!(
                    writer,
                    "    op{index} [label=\"{label}\", shape=diamond, regular=true, fillcolor=lightgreen, width=0.5, height=0.5, fixedsize=true];"
                )?;
            }
        }
    }

    // Emit rank constraints
    for (_, nodes) in rank_groups {
        if nodes.len() > 1 {
            write!(writer, "    {{ rank=same; ")?;
            for node in nodes {
                write!(writer, "n{}; ", usize::from(node))?;
            }
            writeln!(writer, "}}")?;
        }
    }

    // Emit connections
    for node in ops.nodes() {
        let index = usize::from(node);

        match ops[node] {
            Op::Nullary(crate::engine::Nullary::Var) => {
                // Nothing to do.
            }
            Op::Unary(_, input) => {
                let input_source = if should_emit_value_node(input) {
                    "n"
                } else {
                    "op"
                };
                writeln!(
                    writer,
                    "    {input_source}{} -> op{index};",
                    usize::from(input)
                )?;

                if should_emit_value_node(node) {
                    writeln!(writer, "    op{index} -> n{index};")?;
                }
            }
            Op::Binary(_, (a, b)) => {
                let a_source = if should_emit_value_node(a) {
                    "n"
                } else {
                    "op"
                };
                let b_source = if should_emit_value_node(b) {
                    "n"
                } else {
                    "op"
                };
                writeln!(writer, "    {a_source}{} -> op{index};", usize::from(a))?;
                writeln!(writer, "    {b_source}{} -> op{index};", usize::from(b))?;

                if should_emit_value_node(node) {
                    writeln!(writer, "    op{index} -> n{index};")?;
                }
            }
        }
    }

    writeln!(writer, "}}")?;
    Ok(())
}

fn binary_op_to_str(op: Binary) -> &'static str {
    match op {
        Binary::Add => "+",
        Binary::Sub => "-",
        Binary::Mul => "*",
        Binary::Div => "/",
        Binary::Pow => "^",
    }
}

fn unary_op_to_str(op: Unary) -> &'static str {
    match op {
        Unary::Neg => "-",
        Unary::Recip => "1/x",
        Unary::Pow2 => "x²",
        Unary::Ln => "ln",
        Unary::Ln1P => "ln(1+x)",
        Unary::Exp => "exp",
        Unary::Exp2 => "2^x",
        Unary::ExpM1 => "exp(x)-1",
    }
}
