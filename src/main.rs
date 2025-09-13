use micrograd_rs::engine::{Operations, Values};
use std::io::Write;

fn export_to_dot<W: Write>(ops: &Operations, values: Option<&Values>, labels: &[Option<&'static str>], writer: &mut W) -> std::io::Result<()> {
    writeln!(writer, "digraph ComputationGraph {{")?;
    writeln!(writer, "    rankdir=LR;")?;
    writeln!(writer, "    node [shape=record, style=\"rounded,filled\"];")?;
    
    // Create value nodes and operator nodes separately
    for node in ops.nodes() {
        let index = usize::from(node);
        
        // Create value node
        let value_str = if let Some(vals) = values {
            match labels[index] {
                Some(label) => format!("{label} = {:.2}", vals[node]),
                None => format!("{:.2}", vals[node]),
            }
        } else {
            labels[index].unwrap_or_default().to_string()
        };
        
        match ops[node] {
            micrograd_rs::engine::Op::Nullary(micrograd_rs::engine::Nullary::Var) => {
                // Variables are just value nodes
                writeln!(writer, "    n{} [label=\"{}\", shape=box, fillcolor=lightblue];", index, value_str)?;
            }
            micrograd_rs::engine::Op::Unary(unary_op, input) => {
                // Create operator node
                let op_name = match unary_op {
                    micrograd_rs::engine::Unary::Neg => "-",
                    micrograd_rs::engine::Unary::Recip => "1/x",
                    micrograd_rs::engine::Unary::Pow2 => "x²",
                    micrograd_rs::engine::Unary::Ln => "ln",
                    micrograd_rs::engine::Unary::Ln1P => "ln(1+x)",
                    micrograd_rs::engine::Unary::Exp => "exp",
                    micrograd_rs::engine::Unary::Exp2 => "2^x",
                    micrograd_rs::engine::Unary::ExpM1 => "exp(x)-1",
                };
                writeln!(writer, "    op{} [label=\"{}\", shape=diamond, regular=true, fillcolor=lightgreen];", index, op_name)?;
                writeln!(writer, "    n{} [label=\"{}\", shape=box, fillcolor=lightyellow];", index, value_str)?;
                
                // Connect input -> operator -> output
                writeln!(writer, "    n{} -> op{};", usize::from(input), index)?;
                writeln!(writer, "    op{} -> n{};", index, index)?;
            }
            micrograd_rs::engine::Op::Binary(binary_op, (a, b)) => {
                // Create operator node
                let op_name = match binary_op {
                    micrograd_rs::engine::Binary::Add => "+",
                    micrograd_rs::engine::Binary::Sub => "-",
                    micrograd_rs::engine::Binary::Mul => "*",
                    micrograd_rs::engine::Binary::Div => "/",
                    micrograd_rs::engine::Binary::Pow => "^",
                };
                writeln!(writer, "    op{} [label=\"{}\", shape=diamond, regular=true, fillcolor=lightgreen];", index, op_name)?;
                writeln!(writer, "    n{} [label=\"{}\", shape=box, fillcolor=lightyellow];", index, value_str)?;
                
                // Connect inputs -> operator -> output
                writeln!(writer, "    n{} -> op{};", usize::from(a), index)?;
                writeln!(writer, "    n{} -> op{};", usize::from(b), index)?;
                writeln!(writer, "    op{} -> n{};", index, index)?;
            }
        }
    }
    
    writeln!(writer, "}}")?;
    Ok(())
}

fn nn() {
    // Create a sample computation graph
    let mut ops = Operations::default();
    let [x0, x1] = ops.vars();
    let l1 = micrograd_rs::nn::fully_connected_layer(&mut ops, &[x0, x1], 2);
    let _l2 = micrograd_rs::nn::fully_connected_layer(&mut ops, &l1[..], 1);

    let labels: Vec<_> = ops.nodes().map(|_| None).collect();

    export_to_dot(&ops, None, &labels, &mut std::io::stdout()).unwrap();
}

fn sample() {
    // Create a sample computation graph
    let mut ops = Operations::default();
    let [a, x, b, y] = ops.vars();
    let y_pred = ops.insert(a * x + b);
    let loss = ops.insert((y - y_pred).pow_2());
    let ops = ops;

    let mut labels: Vec<_> = ops.nodes().map(|_| None).collect();
    labels[usize::from(a)] = Some("a");
    labels[usize::from(x)] = Some("x");
    labels[usize::from(b)] = Some("b");
    labels[usize::from(y)] = Some("y");
    labels[usize::from(y_pred)] = Some("y_pred");
    labels[usize::from(loss)] = Some("loss");

    // Create values and run forward pass
    let mut values = Values::new(ops.len());
    values[a] = 2.0;
    values[b] = 3.0;
    values[x] = 10.0;
    values[y] = -2.0;
    ops.forward(&mut values);
    
    let mut writer = std::io::stdout();
    export_to_dot(&ops, Some(&values), &labels, &mut writer).expect("Failed to export to DOT");
}

fn main() {
    nn();
}
