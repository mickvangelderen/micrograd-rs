use core::f64;

#[derive(Copy, Clone)]
pub struct Var;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Id(pub(crate) usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Expr<T>(pub(crate) T);

pub type NodeId = Expr<Id>;

impl NodeId {
    #[inline]
    pub(crate) fn new(index: usize) -> Self {
        Expr(Id(index))
    }

    #[inline]
    pub(crate) fn index(self) -> usize {
        self.0.0
    }
}

macro_rules! impl_index_node_id {
    ($T:ty, $O:ty) => {
        impl ::std::ops::Index<NodeId> for $T {
            type Output = $O;
            #[inline]
            fn index(&self, index: NodeId) -> &Self::Output {
                &self.0[index.index()]
            }
        }
        impl ::std::ops::IndexMut<NodeId> for $T {
            #[inline]
            fn index_mut(&mut self, index: NodeId) -> &mut Self::Output {
                &mut self.0[index.index()]
            }
        }
    };
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Nullary {
    Var,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Unary {
    Neg,
}

impl Unary {
    #[inline]
    pub fn forward(self, a: f64) -> f64 {
        match self {
            Unary::Neg => -a,
        }
    }

    /// Given the unary function b(a) represented by this operation, returns the
    /// partial derivative db/da.
    #[inline]
    pub fn backward(self, _a: f64) -> f64 {
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
    #[inline]
    pub fn forward(self, a: f64, b: f64) -> f64 {
        match self {
            Binary::Add => a + b,
            Binary::Mul => a * b,
            Binary::Pow => a.powf(b),
        }
    }

    /// Given the binary function c(a, b) represented by this operation, returns
    /// the partial derivatives dc/da and dc/db.
    #[inline]
    pub fn backward(self, a: f64, b: f64) -> (f64, f64) {
        match self {
            Binary::Add => (1.0, 1.0),
            Binary::Mul => (b, a),
            Binary::Pow => (b * a.powf(b - 1.0), a.ln() * a.powf(b)),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Op {
    Nullary(Nullary),
    Unary(Unary, NodeId),
    Binary(Binary, (NodeId, NodeId)),
}

pub trait Insertable {
    type Output;

    fn insert_into(self, ops: &mut Operations) -> Self::Output;
}

impl<A: Insertable, B: Insertable> Insertable for (A, B) {
    type Output = (A::Output, B::Output);

    #[inline]
    fn insert_into(self, ops: &mut Operations) -> Self::Output {
        (ops.insert(self.0), ops.insert(self.1))
    }
}

impl<T: Insertable, const N: usize> Insertable for [T; N] {
    type Output = [T::Output; N];

    #[inline]
    fn insert_into(self, ops: &mut Operations) -> Self::Output {
        self.map(|item| ops.insert(item))
    }
}

impl Insertable for Id {
    type Output = NodeId;

    #[inline]
    fn insert_into(self, ops: &mut Operations) -> NodeId {
        assert!(self.0 < ops.len(), "Are you using a node from another graph?");
        Expr(self)
    }
}

impl<I: Insertable> Insertable for Expr<I> {
    type Output = I::Output;

    #[inline]
    fn insert_into(self, ops: &mut Operations) -> Self::Output {
        self.0.insert_into(ops)
    }
}

impl Insertable for Var {
    type Output = NodeId;

    #[inline]
    fn insert_into(self, ops: &mut Operations) -> Self::Output {
        ops.insert(Op::Nullary(Nullary::Var))
    }
}

impl Insertable for Op {
    type Output = NodeId;

    #[inline]
    fn insert_into(self, ops: &mut Operations) -> NodeId {
        let id = NodeId::new(ops.0.len());
        ops.0.push(self);
        id
    }
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
    #[inline]
    pub fn new(len: usize) -> Self {
        Self(std::iter::repeat_n(f64::NAN, len).collect())
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
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
    /// initialized to zero.
    ///
    /// This buffer is intended to be re-used when doing multiple forward and
    /// backward passes over the same computation graph.
    #[inline]
    pub fn new(len: usize) -> Self {
        Self(std::iter::repeat_n(0.0, len).collect())
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    pub fn fill(&mut self, value: f64) {
        self.0.fill(value)
    }

    #[inline]
    pub fn resize(&mut self, new_len: usize, value: f64) {
        self.0.resize(new_len, value);
    }

    // NOTE: Decided against implementing AddAssign because it requires Add
    // which would alloc.
    #[inline]
    pub fn accumulate(&mut self, rhs: &Gradients) {
        assert_eq!(self.len(), rhs.len());
        for i in 0..self.len() {
            self.0[i] += rhs.0[i]
        }
    }
}

impl_index_node_id!(Gradients, f64);

#[derive(Debug, Default)]
pub struct Operations(Vec<Op>);

impl Operations {
    #[inline]
    pub fn insert<I: Insertable>(&mut self, insertable: I) -> I::Output {
        insertable.insert_into(self)
    }

    #[inline]
    pub fn extend<I>(
        &mut self,
        collection: I,
    ) -> impl Iterator<Item = <I::Item as Insertable>::Output>
    where
        I: IntoIterator,
        I::Item: Insertable,
    {
        collection.into_iter().map(|item| self.insert(item))
    }

    #[inline]
    pub fn var(&mut self) -> NodeId {
        self.insert(Var)
    }

    #[inline]
    pub fn vars<const N: usize>(&mut self) -> [NodeId; N] {
        self.insert([Var; N])
    }

    #[inline]
    pub fn vars_iter(&mut self, count: usize) -> impl Iterator<Item = NodeId> {
        self.extend(std::iter::repeat_n(Var, count))
    }

    #[inline]
    pub fn vars_vec(&mut self, count: usize) -> Vec<NodeId> {
        self.vars_iter(count).collect()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn forward(&self, values: &mut Values) {
        debug_assert_eq!(self.0.len(), values.0.len());

        for (index, &op) in self.0.iter().enumerate() {
            let id = NodeId::new(index);
            match op {
                Op::Nullary(Nullary::Var) => {
                    // Nothing to do.
                }
                Op::Unary(unary, a) => values[id] = unary.forward(values[a]),
                Op::Binary(binary, (a, b)) => values[id] = binary.forward(values[a], values[b]),
            }
        }
    }

    pub fn backward(
        &self,
        values: &Values,
        gradients: &mut Gradients,
        target: NodeId,
        gradient: f64,
    ) {
        debug_assert_eq!(self.len(), values.len());
        debug_assert_eq!(self.len(), gradients.len());

        gradients.fill(0.0);
        gradients[target] = gradient;

        for id in (0..=target.index()).map(NodeId::new).rev() {
            let grad = gradients[id];

            // If a node's gradient is zero, it can not change it's children and
            // so we can skip processing it.
            if grad == 0.0 {
                continue;
            }

            match self[id] {
                Op::Nullary(Nullary::Var) => {
                    // Nothing to do.
                }
                Op::Unary(unary, a) => {
                    gradients[a] += unary.backward(values[a]) * grad;
                }
                Op::Binary(binary, (a, b)) => {
                    let (da, db) = binary.backward(values[a], values[b]);
                    gradients[a] += da * grad;
                    gradients[b] += db * grad;
                }
            }
        }
    }
}

impl_index_node_id!(Operations, Op);

#[cfg(test)]
pub mod tests {
    use super::*;

    fn test_binary_op(op: Binary, vc: f64, dcda: f64, dcdb: f64) {
        // Construct computation graph.
        let mut ops = Operations::default();
        let [a, b] = ops.vars();
        let c = ops.insert(Op::Binary(op, (a, b)));

        // Forward pass
        let mut values = Values::new(ops.len());
        values[a] = 3.0;
        values[b] = 4.0;
        ops.forward(&mut values);
        assert_eq!(values[c], vc);

        // Backward pass
        let mut gradients = Gradients::new(ops.len());
        ops.backward(&values, &mut gradients, c, 1.0);
        assert_eq!(gradients[a], dcda);
        assert_eq!(gradients[b], dcdb);
    }

    #[test]
    fn add() {
        test_binary_op(Binary::Add, 7.0, 1.0, 1.0);
    }

    #[test]
    fn mul() {
        test_binary_op(Binary::Mul, 12.0, 4.0, 3.0);
    }

    #[test]
    fn pow() {
        test_binary_op(Binary::Pow, 81.0, 108.0, 88.9875953821169);
    }

    #[test]
    fn node_reuse() {
        // Construct computation graph.
        let mut ops = Operations::default();
        let a = ops.var();
        let b = ops.insert(a + a);

        // Forward pass
        let mut values = Values::new(ops.len());
        values[a] = 1.0;
        ops.forward(&mut values);
        assert_eq!(values[b], 2.0);

        // Backward pass
        let mut gradients = Gradients::new(ops.len());
        ops.backward(&values, &mut gradients, b, 1.0);
        assert_eq!(gradients[a], 2.0);
    }

    #[test]
    fn batching() {
        // Construct computation graph.
        let mut ops = Operations::default();
        let [a, x, b, y, const_minus_one, const_two] = ops.vars();
        let y_pred = ops.insert(a * x + b);
        let loss = ops.insert((y + const_minus_one * y_pred).pow(const_two));
        let ops = ops;

        // Create buffers.
        let mut values = Values::new(ops.len());
        let mut acc = Gradients::new(ops.len());
        let mut gradients = Gradients::new(ops.len());

        // Initialize constants.
        values[const_minus_one] = -1.0;
        values[const_two] = 2.0;

        // "Randomly" initialize parameters.
        values[a] = 0.5;
        values[b] = -0.5;

        // Function to learn: y = 2x + 3
        fn f(x: f64) -> f64 {
            2.0 * x + 3.0
        }

        const LR: f64 = 0.005;

        for _ in 0..50 {
            for batch in [
                [-3.6, 2.2, 1.0],
                [3.5, 20.1, 0.4],
                [-0.3, -0.10, 4.0],
                [-10.0, 5.1, 8.0],
                [4.6, 5.9, -6.7],
            ] {
                acc.fill(0.0);

                for vx in batch {
                    values[x] = vx;
                    values[y] = f(vx);
                    ops.forward(&mut values);
                    ops.backward(&values, &mut gradients, loss, LR);
                    acc.accumulate(&gradients);
                }

                // Apply gradients to weights.
                values[a] -= acc[a];
                values[b] -= acc[b];
            }
        }

        assert!((values[a] - 2.0).abs() < 0.2, "expected a to be close to 2.0 but got {}", values[a]);
        assert!((values[b] - 3.0).abs() < 0.2, "expected b to be close to 3.0 but got {}", values[b]);
    }

    #[test]
    #[should_panic]
    fn insert_node_from_future() {
        let mut ops1 = Operations::default();
        let [a1] = ops1.vars();

        let mut ops2 = Operations::default();
        let _a2 = ops2.insert(a1); // should panic becasue NodeId(0) doesn't exist in ops2.
    }
}
