use core::f64;

#[derive(Copy, Clone)]
pub struct Var;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Id(pub(crate) usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Expr<T>(pub(crate) T);

pub type NodeId = Expr<Id>;

impl From<usize> for NodeId {
    fn from(value: usize) -> Self {
        Expr(Id(value))
    }
}

impl From<NodeId> for usize {
    fn from(value: NodeId) -> Self {
        value.0.0
    }
}

macro_rules! impl_index_node_id {
    ($T:ty, $O:ty) => {
        impl ::std::ops::Index<NodeId> for $T {
            type Output = $O;
            #[inline]
            fn index(&self, index: NodeId) -> &Self::Output {
                &self.0[usize::from(index)]
            }
        }
        impl ::std::ops::IndexMut<NodeId> for $T {
            #[inline]
            fn index_mut(&mut self, index: NodeId) -> &mut Self::Output {
                &mut self.0[usize::from(index)]
            }
        }
    };
}

macro_rules! impl_buffer {
    ($T:ty, $I: ty) => {
        impl IntoIterator for $T {
            type Item = $I;
            type IntoIter = <Vec<$I> as IntoIterator>::IntoIter;

            #[inline]
            fn into_iter(self) -> Self::IntoIter {
                self.0.into_iter()
            }
        }

        impl<'a> IntoIterator for &'a $T {
            type Item = &'a $I;
            type IntoIter = std::slice::Iter<'a, $I>;

            #[inline]
            fn into_iter(self) -> Self::IntoIter {
                self.0.iter()
            }
        }

        impl<'a> IntoIterator for &'a mut $T {
            type Item = &'a mut $I;
            type IntoIter = std::slice::IterMut<'a, $I>;

            #[inline]
            fn into_iter(self) -> Self::IntoIter {
                self.0.iter_mut()
            }
        }

        impl $T {
            #[inline]
            pub fn iter(&self) -> <&Self as IntoIterator>::IntoIter {
                IntoIterator::into_iter(self)
            }

            #[inline]
            pub fn iter_mut(&mut self) -> <&mut Self as IntoIterator>::IntoIter {
                IntoIterator::into_iter(self)
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
            pub fn nodes(&self) -> impl ExactSizeIterator<Item = NodeId> + DoubleEndedIterator {
                (0..self.len()).map(NodeId::from)
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
    Recip,
    Pow2,
    Ln,
    Ln1P,
    Exp,
    Exp2,
    ExpM1,
    TanH,
}

impl Unary {
    #[inline]
    pub fn forward(self, a: f64) -> f64 {
        match self {
            Unary::Neg => -a,
            Unary::Recip => a.recip(),
            Unary::Pow2 => a.powi(2),
            Unary::Ln => a.ln(),
            Unary::Ln1P => a.ln_1p(),
            Unary::Exp => a.exp(),
            Unary::Exp2 => a.exp2(),
            Unary::ExpM1 => a.exp_m1(),
            Unary::TanH => a.tanh(),
        }
    }

    /// Given the unary function b(a) represented by this operation, returns the
    /// partial derivative db/da.
    #[inline]
    pub fn backward(self, a: f64, b: f64) -> f64 {
        match self {
            Unary::Neg => -1.0,
            Unary::Recip => -b.powi(2),
            Unary::Pow2 => 2.0 * a,
            Unary::Ln => a.recip(),
            Unary::Ln1P => (1.0 + a).recip(),
            Unary::Exp => b,
            Unary::Exp2 => f64::consts::LN_2 * b,
            Unary::ExpM1 => a.exp(), // or b + 1
            Unary::TanH => 1.0 - b.powi(2),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Binary {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

impl Binary {
    #[inline]
    pub fn forward(self, a: f64, b: f64) -> f64 {
        match self {
            Binary::Add => a + b,
            Binary::Sub => a - b,
            Binary::Mul => a * b,
            Binary::Div => a / b,
            Binary::Pow => a.powf(b),
        }
    }

    /// Given the binary function c(a, b) represented by this operation, returns
    /// the partial derivatives dc/da and dc/db.
    #[inline]
    pub fn backward(self, a: f64, b: f64, c: f64) -> (f64, f64) {
        match self {
            Binary::Add => (1.0, 1.0),
            Binary::Sub => (1.0, -1.0),
            Binary::Mul => (b, a),
            Binary::Div => {
                let b_inv = b.recip();
                (b_inv, -b_inv * c)
            }
            Binary::Pow => {
                // Not using b*a.powf(b)/a because it would return NaN for a ==
                // 0.0 intead of the correct 0.0.
                (b * a.powf(b - 1.0), a.ln() * c)
            }
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
        assert!(
            self.0 < ops.len(),
            "Are you using a node from another graph?"
        );
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
        let id = NodeId::from(ops.0.len());
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
    pub fn resize(&mut self, new_len: usize, value: f64) {
        self.0.resize(new_len, value);
    }
}

impl_index_node_id!(Values, f64);

impl_buffer!(Values, f64);

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

impl_buffer!(Gradients, f64);

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
    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn forward(&self, values: &mut Values) {
        debug_assert_eq!(self.0.len(), values.0.len());

        for output in self.nodes() {
            match self[output] {
                Op::Nullary(Nullary::Var) => {
                    // Nothing to do.
                }
                Op::Unary(unary, input) => values[output] = unary.forward(values[input]),
                Op::Binary(binary, input) => {
                    values[output] = binary.forward(values[input.0], values[input.1])
                }
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

        for o in self.nodes().rev() {
            let gradients_o = gradients[o];

            // If a node's gradient is zero, it can not change it's children and
            // so we can skip processing it.
            if gradients_o == 0.0 {
                continue;
            }

            match self[o] {
                Op::Nullary(Nullary::Var) => {
                    // Nothing to do.
                }
                Op::Unary(unary, i0) => {
                    gradients[i0] += unary.backward(values[i0], values[o]) * gradients_o;
                }
                Op::Binary(binary, (i0, i1)) => {
                    let (gradients_i0, gradients_i1) =
                        binary.backward(values[i0], values[i1], values[o]);
                    gradients[i0] += gradients_i0 * gradients_o;
                    gradients[i1] += gradients_i1 * gradients_o;
                }
            }
        }
    }
}

impl_index_node_id!(Operations, Op);

impl_buffer!(Operations, Op);

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
        let [a, x, b, y] = ops.vars();
        let y_pred = ops.insert(a * x + b);
        let loss = ops.insert((y - y_pred).pow_2());
        let ops = ops;

        // Create buffers.
        let mut values = Values::new(ops.len());
        let mut acc = Gradients::new(ops.len());
        let mut gradients = Gradients::new(ops.len());

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

        assert!(
            (values[a] - 2.0).abs() < 0.2,
            "expected a to be close to 2.0 but got {}",
            values[a]
        );
        assert!(
            (values[b] - 3.0).abs() < 0.2,
            "expected b to be close to 3.0 but got {}",
            values[b]
        );
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
