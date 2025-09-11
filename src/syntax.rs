use crate::engine::{Binary, NodeId, Op, Operations};

impl Operations {
    pub fn expr<T: Resolve>(&mut self, expr: T) -> NodeId {
        expr.resolve(self)
    }
}

pub trait Resolve {
    fn resolve(self, ops: &mut Operations) -> NodeId;
}

impl Resolve for NodeId {
    fn resolve(self, _: &mut Operations) -> NodeId {
        self
    }
}

pub struct Add<A, B>(A, B);

pub struct Mul<A, B>(A, B);

pub struct Pow<A, B>(A, B);

impl<L: Resolve, R: Resolve> Resolve for Add<L, R> {
    fn resolve(self, ops: &mut Operations) -> NodeId {
        let a = self.0.resolve(ops);
        let b = self.1.resolve(ops);
        ops.push(Op::Binary(Binary::Add, a, b))
    }
}

impl<L: Resolve, R: Resolve> Resolve for Mul<L, R> {
    fn resolve(self, ops: &mut Operations) -> NodeId {
        let a = self.0.resolve(ops);
        let b = self.1.resolve(ops);
        ops.push(Op::Binary(Binary::Add, a, b))
    }
}

impl<L: Resolve, R: Resolve> Resolve for Pow<L, R> {
    fn resolve(self, ops: &mut Operations) -> NodeId {
        let a = self.0.resolve(ops);
        let b = self.1.resolve(ops);
        ops.push(Op::Binary(Binary::Pow, a, b))
    }
}

pub struct Expr<T>(T);

impl<A> Expr<A> {
    pub fn add<B>(self, rhs: Expr<B>) -> Expr<Add<Self, Expr<B>>> {
        Expr(Add(self, rhs))
    }

    pub fn mul<B>(self, rhs: Expr<B>) -> Expr<Mul<Self, Expr<B>>> {
        Expr(Mul(self, rhs))
    }

    pub fn pow<B>(self, rhs: Expr<B>) -> Expr<Pow<Self, Expr<B>>> {
        Expr(Pow(self, rhs))
    }
}

impl<A, B> std::ops::Add<Expr<B>> for Expr<A> {
    type Output = Expr<Add<Self, Expr<B>>>;

    fn add(self, rhs: Expr<B>) -> Self::Output {
        self.add(rhs)
    }
}

impl<A, B> std::ops::Mul<Expr<B>> for Expr<A> {
    type Output = Expr<Mul<Self, Expr<B>>>;

    fn mul(self, rhs: Expr<B>) -> Self::Output {
        self.mul(rhs)
    }
}

impl<T> Resolve for Expr<T>
where
    T: Resolve,
{
    fn resolve(self, ops: &mut Operations) -> NodeId {
        self.0.resolve(ops)
    }
}
