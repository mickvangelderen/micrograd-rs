use micrograd_rs::engine::Operations;
use micrograd_rs::graphviz::export_to_dot;
use micrograd_rs::nn::FullyConnectedLayer;

fn main() {
    // Create a sample computation graph
    let mut ops = Operations::default();
    let l0 = ops.vars::<2>();
    let l1 = FullyConnectedLayer::new(&l0, 5, &mut ops);
    let l2 = FullyConnectedLayer::new(l1.outputs(), 3, &mut ops);

    let mut labels: Vec<_> = ops.nodes().map(|_| String::default()).collect();

    labels[usize::from(l0[0])] = "l0 a_0".to_string();
    labels[usize::from(l0[1])] = "l0 a_1".to_string();

    label_layer(l1, "l1", &mut labels);
    label_layer(l2, "l2", &mut labels);

    export_to_dot(
        &ops,
        |node| &labels[usize::from(node)],
        &mut std::io::stdout(),
    )
    .unwrap();
}

fn label_layer(layer: FullyConnectedLayer, layer_label: &'static str, labels: &mut Vec<String>) {
    for (index, &node) in layer.weights().iter().enumerate() {
        let input_index = index / layer.input_count;
        let output_index = index % layer.input_count;
        labels[usize::from(node)] = format!("{layer_label} w_{input_index},{output_index}");
    }
    for (index, &node) in layer.biases().iter().enumerate() {
        labels[usize::from(node)] = format!("{layer_label} b_{index}");
    }
    for (index, &node) in layer.outputs().iter().enumerate() {
        labels[usize::from(node)] = format!("{layer_label}a_{index}");
    }
}
