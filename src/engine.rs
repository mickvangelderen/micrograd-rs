use core::f64;

use crate::syntax::Expr;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct NodeId(usize);

macro_rules! impl_index_node_id {
    ($T:ty, $O:ty) => {
        impl ::std::ops::Index<Expr<NodeId>> for $T {
            type Output = $O;
            #[inline]
            fn index(&self, index: Expr<NodeId>) -> &Self::Output {
                &self.0[index.0.0]
            }
        }
        impl ::std::ops::IndexMut<Expr<NodeId>> for $T {
            #[inline]
            fn index_mut(&mut self, index: Expr<NodeId>) -> &mut Self::Output {
                &mut self.0[index.0.0]
            }
        }
    };
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Unary {
    Neg,
}

impl Unary {
    fn forward(self, a: f64) -> f64 {
        match self {
            Unary::Neg => -a,
        }
    }

    /// Given the unary function b(a) represented by this operation, returns the
    /// partial derivative db/da.
    fn backward(self, _a: f64) -> f64 {
        match self {
            Unary::Neg => -1.0,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Binary {
    Add,
    Mul,
    Pow,
}

impl Binary {
    fn forward(self, a: f64, b: f64) -> f64 {
        match self {
            Binary::Add => a + b,
            Binary::Mul => a * b,
            Binary::Pow => a.powf(b),
        }
    }

    /// Given the binary function c(a, b) represented by this operation, returns
    /// the partial derivatives dc/da and dc/db.
    fn backward(self, a: f64, b: f64) -> (f64, f64) {
        match self {
            Binary::Add => (1.0, 1.0),
            Binary::Mul => (b, a),
            Binary::Pow => (b * a.powf(b - 1.0), a.ln() * a.powf(b)),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Op {
    Var,
    Unary(Unary, Expr<NodeId>),
    Binary(Binary, Expr<NodeId>, Expr<NodeId>),
}

/// A buffer storing values for the nodes in the computation graph respresented by `Operations`.
#[derive(Debug, Default)]
pub struct Values(Vec<f64>);

impl Values {
    /// Creates and returns a buffer of the specified size, with every element
    /// initialized to NaN.
    ///
    /// This buffer is intended to be re-used when doing multiple forward and
    /// backward passes over the same computation graph.
    pub fn new(len: usize) -> Self {
        Self(std::iter::repeat(f64::NAN).take(len).into_iter().collect())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn resize(&mut self, new_len: usize, value: f64) {
        self.0.resize(new_len, value);
    }
}

impl_index_node_id!(Values, f64);

/// A buffer storing gradients for the nodes in the computation graph respresented by `Operations`.
#[derive(Debug, Default)]
pub struct Gradients(Vec<f64>);

impl Gradients {
    /// Creates and returns a buffer of the specified size, with every element
    /// initialized to NaN.
    ///
    /// This buffer is intended to be re-used when doing multiple forward and
    /// backward passes over the same computation graph.
    pub fn new(len: usize) -> Self {
        Self(std::iter::repeat(f64::NAN).take(len).into_iter().collect())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn resize(&mut self, new_len: usize, value: f64) {
        self.0.resize(new_len, value);
    }
}

impl_index_node_id!(Gradients, f64);

#[derive(Debug, Default)]
pub struct Operations(Vec<Op>);

impl Operations {
    pub(crate) fn push(&mut self, operation: Op) -> NodeId {
        let id = NodeId(self.0.len());
        self.0.push(operation);
        id
    }

    pub fn insert<T: Insertable>(&mut self, insertable: T) -> T::Output {
        insertable.insert(self)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn forward(&self, values: &mut Values) {
        assert_eq!(self.0.len(), values.0.len());

        for (index, &op) in self.0.iter().enumerate() {
            let id = Expr(NodeId(index));
            match op {
                Op::Var => {
                    // Nothing to do.
                }
                Op::Unary(unary, a) => values[id] = unary.forward(values[a]),
                Op::Binary(binary, a, b) => values[id] = binary.forward(values[a], values[b]),
            }
        }
    }

    pub fn backward(&self, values: &Values, gradients: &mut Gradients, target: Expr<NodeId>) {
        assert_eq!(self.len(), values.len());
        assert_eq!(self.len(), gradients.len());

        gradients[target] = 1.0;
        let mut todo = vec![target];
        let mut new_todo = vec![];
        while !todo.is_empty() {
            for id in todo.iter().copied() {
                match self[target] {
                    Op::Var => {
                        // Nothing to do.
                    }
                    Op::Unary(unary, a) => {
                        let grad = gradients[id];
                        gradients[a] = unary.backward(values[a]) * grad;
                        new_todo.push(a);
                    }
                    Op::Binary(binary, a, b) => {
                        let grad = gradients[id];
                        let (da, db) = binary.backward(values[a], values[b]);
                        gradients[a] = da * grad;
                        gradients[b] = db * grad;
                        new_todo.push(a);
                        new_todo.push(b);
                    }
                }
            }
            std::mem::swap(&mut todo, &mut new_todo);
            new_todo.clear()
        }
    }
}

impl_index_node_id!(Operations, Op);

#[cfg(test)]
pub mod tests {
    use super::*;

    fn test_binary<
        T: Insertable<Output = Expr<NodeId>>,
        F: FnOnce(Expr<NodeId>, Expr<NodeId>) -> Expr<T>,
    >(
        f: F,
        vc: f64,
        dcda: f64,
        dcdb: f64,
    ) {
        // Construct computation graph.
        let mut operations = Operations::default();
        let [a, b] = operations.insert([crate::syntax::Var; _]);
        let c = operations.insert(f(a, b));

        // Initialize buffers.
        let mut values = Values::new(operations.len());
        let mut gradients = Gradients::new(operations.len());

        // Forward pass
        values[a] = 3.0;
        values[b] = 4.0;
        operations.forward(&mut values);
        assert_eq!(values[c], vc);

        // Backward pass
        operations.backward(&values, &mut gradients, c);
        assert_eq!(gradients[a], dcda);
        assert_eq!(gradients[b], dcdb);
    }

    #[test]
    fn test_add() {
        test_binary(|a, b| a + b, 7.0, 1.0, 1.0);
    }

    #[test]
    fn test_mul() {
        test_binary(|a, b| a * b, 12.0, 4.0, 3.0);
    }

    #[test]
    fn test_pow() {
        test_binary(|a, b| a.pow(b), 81.0, 108.0, 88.9875953821169);
    }
}

pub trait Insertable {
    type Output;

    fn insert(self, ops: &mut Operations) -> Self::Output;
}
