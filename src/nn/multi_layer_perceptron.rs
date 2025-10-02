use crate::{
    engine::{Expr, Gradients, NodeId, Operations, Values},
    nn::{self, FullyConnectedLayer, FullyConnectedLayerParams},
    view::View,
};

#[derive(Debug, Clone)]
pub struct MultiLayerPerceptronParams<TLayers> {
    pub layers: TLayers,
}

pub struct MultiLayerPerceptron {
    pub layers: Vec<FullyConnectedLayer>,
}

impl MultiLayerPerceptron {
    pub fn new<'a, TLayers>(
        input_layer: View<&[NodeId], (nn::B, nn::O)>,
        params: MultiLayerPerceptronParams<TLayers>,
        ops: &mut Operations,
    ) -> Self
    where
        TLayers: IntoIterator<Item = &'a FullyConnectedLayerParams>,
    {
        let layer_params = params.layers.into_iter();
        let min_layer_count = layer_params.size_hint().0;
        let layers = layer_params.fold(Vec::with_capacity(min_layer_count), |mut layers, params| {
            let prev_output = layers
                .last()
                .map_or(input_layer, |layer: &FullyConnectedLayer| layer.outputs());
            let layer = FullyConnectedLayer::new(
                prev_output.as_deref().reindex(nn::batched_output_to_input),
                nn::O(params.size),
                ops,
                Expr::relu,
            );
            layers.push(layer);
            layers
        });

        Self { layers }
    }

    #[inline]
    pub fn init_parameters(&self, values: &mut Values, rng: &mut impl rand::Rng) {
        for layer in &self.layers {
            layer.init_parameters(values, rng);
        }
    }

    #[inline]
    pub fn update_weights(&self, values: &mut Values, gradients: &Gradients) {
        for layer in &self.layers {
            layer.update_weights(values, gradients);
        }
    }

    #[inline]
    pub fn outputs(&self) -> View<&[NodeId], (nn::B, nn::O)> {
        self.layers
            .last()
            .expect("Network must have at least one layer")
            .outputs()
    }

    #[inline]
    pub fn parameters(&self) -> impl Iterator<Item = NodeId> {
        self.layers.iter().flat_map(|layer| {
            layer
                .weights()
                .into_iter()
                .copied()
                .chain(layer.biases().into_iter().copied())
        })
    }
}
