use crate::engine::{NodeId, Operations};

pub trait OperationsExt {
    fn expr<T: Resolve>(&mut self, expr: T) -> NodeId;
}

impl OperationsExt for Operations {
    fn expr<T: Resolve>(&mut self, expr: T) -> NodeId {
        expr.resolve(self)
    }
}

pub trait Resolve {
    fn resolve(self, ops: &mut Operations) -> NodeId;
}

pub struct Expr<T>(T);

impl<T> Resolve for Expr<T> where T: Resolve {
    fn resolve(self, ops: &mut Operations) -> NodeId {
        self.0.resolve(ops)
    }
}

impl Resolve for NodeId {
    fn resolve(self, _: &mut Operations) -> NodeId { self }
}

pub struct Add<L, R>(L, R);

impl<L: Resolve, R: Resolve> Resolve for Add<L, R>{
    fn resolve(self, ops: &mut Operations) -> NodeId {
        let a = self.0.resolve(ops);
        let b = self.1.resolve(ops);
        ops.add(a, b)
    }
}

pub fn add<L, R>(l: L, r: R) -> Expr<Add<L, R>> {
    Expr(Add(l, r))
}

impl<L, R> std::ops::Add<Expr<R>> for Expr<L> {
    type Output = Expr<Add<L, R>>;

    fn add(self, rhs: Expr<R>) -> Self::Output {
        add(self.0, rhs.0)
    }
}

pub struct Mul<L, R>(L, R);

impl<L: Resolve, R: Resolve> Resolve for Mul<L, R>{
    fn resolve(self, ops: &mut Operations) -> NodeId {
        let a = self.0.resolve(ops);
        let b = self.1.resolve(ops);
        ops.mul(a, b)
    }
}

fn add_n<I: IntoIterator<Item = NodeId>>(args: I, ops: &mut Operations) -> Option<NodeId> {
    let mut args = args.into_iter();
    let mut acc = args.next()?;
    for arg in args {
        acc = ops.add(acc, arg)
    }
    Some(acc)
}

fn lit_n(ops: &mut Operations, len: usize) -> Vec<NodeId> {
    (0..len).map(|_| ops.lit()).collect()
}

fn fully_connected_layer(
    ops: &mut Operations,
    inputs: &[NodeId],
    output_len: usize,
) -> Vec<(Vec<NodeId>, NodeId, NodeId)> {
    (0..output_len)
        .map(|_| {
            let weights = lit_n(ops, inputs.len());
            let bias = ops.lit();
            let output =
                inputs
                    .iter()
                    .copied()
                    .zip(weights.iter().copied())
                    .fold(bias, |sum, (a, b)| {
                        ops.expr(Add(sum, Mul(a, b)))
                        // ops.expr(sum + a * b) // TODO: Make this work.
                    });
            (weights, bias, output)
        })
        .collect()
}
