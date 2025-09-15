use micrograd_rs::engine::{NodeId, Operations, Values};
use micrograd_rs::graphviz::export_to_dot;

fn main() {
    // Create a sample computation graph
    let mut ops = Operations::default();
    let [a, x, b, y] = ops.vars();
    let y_pred = ops.insert(a * x + b);
    let loss = ops.insert((y - y_pred).pow_2());
    let ops = ops;

    // Create values and run forward pass
    let mut values = Values::new(ops.len());
    values[a] = 2.0;
    values[b] = 3.0;
    values[x] = 10.0;
    values[y] = -2.0;
    ops.forward(&mut values);

    let labels = (0..ops.len())
        .map(NodeId::from)
        .map(|node| {
            let name = match node {
                node if node == a => Some("a"),
                node if node == x => Some("x"),
                node if node == b => Some("b"),
                node if node == y => Some("y"),
                node if node == y_pred => Some("y_pred"),
                node if node == loss => Some("loss"),
                _ => None,
            };
            let value = values[node];
            match name {
                Some(name) => format!("{name} = {value}"),
                None => format!("{value}"),
            }
        })
        .collect::<Vec<_>>();

    let mut writer = std::io::stdout();
    export_to_dot(
        &ops,
        |node| labels[usize::from(node)].as_str(),
        |_node| None::<usize>,
        &mut writer,
    )
    .expect("Failed to export to DOT");
}
