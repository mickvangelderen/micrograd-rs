pub use std::io::Result;
use std::io::{Read, Write};

use byteorder::{LE, ReadBytesExt, WriteBytesExt};

use crate::{
    engine::Values,
    nn::{self, FullyConnectedLayer, MultiLayerPerceptron},
};

pub trait Serialize {
    fn serialize(&self, values: &Values, writer: &mut impl Write) -> Result<()>;
}

pub trait Deserialize {
    fn deserialize(&self, values: &mut Values, reader: &mut impl Read) -> Result<()>;
}

impl Serialize for FullyConnectedLayer {
    fn serialize(&self, values: &Values, writer: &mut impl Write) -> Result<()> {
        writer.write_u64::<LE>(usize::from(self.input_size) as u64)?;
        writer.write_u64::<LE>(usize::from(self.output_size) as u64)?;

        for &node in self.weights().iter() {
            writer.write_f64::<LE>(values[node])?;
        }
        for &node in self.biases().iter() {
            writer.write_f64::<LE>(values[node])?;
        }
        Ok(())
    }
}

impl Deserialize for FullyConnectedLayer {
    fn deserialize(&self, values: &mut Values, reader: &mut impl Read) -> Result<()> {
        let input_count = nn::I(reader.read_u64::<LE>()? as usize);
        let output_count = nn::O(reader.read_u64::<LE>()? as usize);

        let expected = (self.input_size, self.output_size);
        let actual = (input_count, output_count);
        if expected != actual {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Layer shape mismatch: expected {expected:?} but got {actual:?}"),
            );
        }

        for &node in self.weights().iter() {
            values[node] = reader.read_f64::<LE>()?;
        }
        for &node in self.biases().iter() {
            values[node] = reader.read_f64::<LE>()?;
        }
        Ok(())
    }
}

impl Serialize for MultiLayerPerceptron {
    fn serialize(&self, values: &Values, writer: &mut impl Write) -> Result<()> {
        writer.write_u64::<LE>(self.layers.len() as u64)?;

        for layer in &self.layers {
            layer.serialize(values, writer)?;
        }
        Ok(())
    }
}

impl Deserialize for MultiLayerPerceptron {
    fn deserialize(&self, values: &mut Values, reader: &mut impl Read) -> Result<()> {
        let layer_count = reader.read_u64::<LE>()? as usize;

        let expected = self.layers.len();
        let actual = layer_count;
        if expected != actual {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Layer count mismatch: expected {expected:?} but got {actual:?}"),
            );
        }

        for layer in &self.layers {
            layer.deserialize(values, reader)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::engine::Operations;

    #[test]
    fn save_and_load_fully_connected_layer() {
        let mut ops = Operations::default();
        let input = nn::input_layer_vec((nn::B(2), nn::O(3)), &mut ops);
        let layer = FullyConnectedLayer::new(
            input.as_deref().reindex(nn::batched_output_to_input),
            nn::O(4),
            &mut ops,
            crate::engine::Expr::relu,
        );

        let mut values = crate::engine::Values::new(ops.len());

        // Set predictable values for weights and biases
        for (index, &node) in layer.weights().iter().enumerate() {
            values[node] = (index as f64) * 0.1;
        }
        for (index, &node) in layer.biases().iter().enumerate() {
            values[node] = (index as f64) * 0.01;
        }

        let mut serialized = Vec::new();
        Serialize::serialize(&layer, &values, &mut serialized).unwrap();

        // Reset values to NaN and load
        values.fill(f64::NAN);

        let mut cursor = Cursor::new(serialized);
        Deserialize::deserialize(&layer, &mut values, &mut cursor).unwrap();

        // Verify weights and biases match original values
        for (index, &node) in layer.weights().iter().enumerate() {
            assert_eq!(values[node], (index as f64) * 0.1);
        }
        for (index, &node) in layer.biases().iter().enumerate() {
            assert_eq!(values[node], (index as f64) * 0.01);
        }
    }
}
