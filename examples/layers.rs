use micrograd_rs::engine::Operations;
use micrograd_rs::graphviz::export_to_dot;
use micrograd_rs::nn::FullyConnectedLayer;

fn main() {
    // Create a sample computation graph
    let mut ops = Operations::default();
    let l0 = ops.vars::<2>();
    let l1 = FullyConnectedLayer::new(&l0, 5, &mut ops, |x| x);
    let l2 = FullyConnectedLayer::new(l1.outputs(), 3, &mut ops, |x| x);

    let mut labels: Vec<_> = ops.nodes().map(|_| String::default()).collect();
    let mut ranks: Vec<Option<usize>> = ops.nodes().map(|_| None).collect();

    labels[usize::from(l0[0])] = "l0 a_0".to_string();
    labels[usize::from(l0[1])] = "l0 a_1".to_string();
    ranks[usize::from(l0[0])] = Some(0);
    ranks[usize::from(l0[1])] = Some(0);

    label_layer(l1, 1, &mut labels, &mut ranks);
    label_layer(l2, 2, &mut labels, &mut ranks);

    export_to_dot(
        &ops,
        |node| &labels[usize::from(node)],
        |node| ranks[usize::from(node)],
        &mut std::io::stdout(),
    )
    .unwrap();
}

fn label_layer(
    layer: FullyConnectedLayer,
    layer_index: usize,
    labels: &mut Vec<String>,
    ranks: &mut Vec<Option<usize>>,
) {
    for (index, &node) in layer.weights().iter().enumerate() {
        let input_index = index / layer.input_count;
        let output_index = index % layer.input_count;
        labels[usize::from(node)] = format!("l{layer_index} w_{input_index},{output_index}");
        ranks[usize::from(node)] = Some(layer_index * 2);
    }
    for (index, &node) in layer.biases().iter().enumerate() {
        labels[usize::from(node)] = format!("l{layer_index} b_{index}");
        ranks[usize::from(node)] = Some(layer_index * 2);
    }
    for (index, &node) in layer.outputs().iter().enumerate() {
        labels[usize::from(node)] = format!("l{layer_index} a_{index}");
        ranks[usize::from(node)] = Some(layer_index * 2 + 1);
    }
}
