use micrograd_rs::{
    engine::{NodeId, Operations},
    graphviz::export_to_dot,
    nn::{self},
    view::{Index as _, View},
};

fn main() {
    struct ModelParams {
        batch_size: usize,
        l0_size: usize,
        l1_size: usize,
        l2_size: usize,
    }

    let params = ModelParams {
        batch_size: 2,
        l0_size: 1,
        l1_size: 2,
        l2_size: 1,
    };

    // Create a sample computation graph
    let mut ops = Operations::default();

    let l0 = nn::input_layer_vec((nn::B(params.batch_size), nn::O(params.l0_size)), &mut ops);
    let l1 = nn::FullyConnectedLayer::new(
        l0.as_deref().reindex(nn::batched_output_to_input),
        nn::O::from(params.l1_size),
        &mut ops,
        |x| x,
    );
    let l2 = nn::FullyConnectedLayer::new(
        l1.outputs().reindex(nn::batched_output_to_input),
        nn::O::from(params.l2_size),
        &mut ops,
        |x| x,
    );

    let mut labels = View::new(
        ops.nodes()
            .map(|_| String::default())
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        (NodeId::from(ops.len()),),
    );
    let mut ranks = View::new(
        ops.nodes().map(|_| None).collect::<Vec<_>>().into_boxed_slice(),
        (NodeId::from(ops.len()),),
    );

    for b in l0.len().0.indices() {
        for o in l0.len().1.indices() {
            labels[(l0[(b, o)],)] = format!("layer0 batch{b} activation{o}", b = usize::from(b), o = usize::from(o));
            ranks[(l0[(b, o)],)] = Some(0);
        }
    }

    label_layer(l1, 1, labels.as_deref_mut(), ranks.as_deref_mut());
    label_layer(l2, 2, labels.as_deref_mut(), ranks.as_deref_mut());

    export_to_dot(
        &ops,
        |node| &labels[(node,)],
        |node| ranks[(node,)],
        &mut std::io::stdout(),
    )
    .unwrap();
}

fn label_layer<L, R>(layer: nn::FullyConnectedLayer, layer_index: usize, mut labels: L, mut ranks: R)
where
    L: std::ops::IndexMut<(NodeId,), Output = String>,
    R: std::ops::IndexMut<(NodeId,), Output = Option<usize>>,
{
    for ((i, o), &node) in layer.weights().iter_enumerate() {
        labels[(node,)] = format!(
            "layer{layer_index} weight{i},{o}",
            i = usize::from(i),
            o = usize::from(o)
        );
        ranks[(node,)] = Some(layer_index * 2);
    }
    for ((o,), &node) in layer.biases().iter_enumerate() {
        labels[(node,)] = format!("layer{layer_index} bias{o}", o = usize::from(o));
        ranks[(node,)] = Some(layer_index * 2);
    }
    for ((b, o), &node) in layer.outputs().iter_enumerate() {
        labels[(node,)] = format!(
            "layer{layer_index} batch{b}, activation{o}",
            b = usize::from(b),
            o = usize::from(o)
        );
        ranks[(node,)] = Some(layer_index * 2 + 1);
    }
}
