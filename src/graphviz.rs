use crate::engine::{Binary, NodeId, Op, Operations, Unary};
use std::io::Write;

pub fn export_to_dot<W: Write, L: FnMut(NodeId) -> LO, LO: std::fmt::Display>(
    ops: &Operations,
    mut labels: L,
    writer: &mut W,
) -> std::io::Result<()> {
    writeln!(writer, "digraph ComputationGraph {{")?;
    writeln!(writer, "    rankdir=LR;")?;
    writeln!(writer, "    node [shape=record, style=\"rounded,filled\"];")?;

    for node in ops.nodes() {
        let index = usize::from(node);
        let label = labels(node);

        let fillcolor = match ops[node] {
            Op::Nullary(crate::engine::Nullary::Var) => "lightblue",
            _ => "lightyellow",
        };

        writeln!(
            writer,
            "    n{index} [label=\"{label}\", shape=box, fillcolor={fillcolor}];"
        )?;
    }

    for node in ops.nodes() {
        let index = usize::from(node);

        match ops[node] {
            Op::Nullary(crate::engine::Nullary::Var) => {
                // Nothing to do.
            }
            Op::Unary(unary_op, input) => {
                let label = unary_op_to_str(unary_op);
                writeln!(
                    writer,
                    "    op{index} [label=\"{label}\", shape=diamond, regular=true, fillcolor=lightgreen];"
                )?;

                // Connect input -> operator -> output
                writeln!(writer, "    n{} -> op{index};", usize::from(input))?;
                writeln!(writer, "    op{index} -> n{index};")?;
            }
            Op::Binary(binary_op, (a, b)) => {
                let label = binary_op_to_str(binary_op);
                writeln!(
                    writer,
                    "    op{index} [label=\"{label}\", shape=diamond, regular=true, fillcolor=lightgreen];"
                )?;

                // Connect inputs -> operator -> output
                writeln!(writer, "    n{} -> op{index};", usize::from(a))?;
                writeln!(writer, "    n{} -> op{index};", usize::from(b))?;
                writeln!(writer, "    op{index} -> n{index};")?;
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
