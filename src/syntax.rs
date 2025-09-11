use crate::engine::{Insertable, NodeId, Operations};

#[derive(Copy, Clone)]
pub struct Var;

pub mod unary {
    pub struct Neg;
}

pub struct Unary<O, A>(O, A);

pub type Neg<A> = Unary<unary::Neg, A>;


pub mod binary {
    pub struct Add;

    pub struct Mul;

    pub struct Pow;

    impl From<Add> for crate::engine::Binary {
        fn from(_: Add) -> Self {
            crate::engine::Binary::Add
        }
    }

    impl From<Mul> for crate::engine::Binary {
        fn from(_: Mul) -> Self {
            crate::engine::Binary::Mul
        }
    }

    impl From<Pow> for crate::engine::Binary {
        fn from(_: Pow) -> Self {
            crate::engine::Binary::Pow
        }
    }
}

pub struct Binary<O, A, B>(O, A, B);

pub type Add<A, B> = Binary<binary::Add, A, B>;

pub type Mul<A, B> = Binary<binary::Mul, A, B>;

pub type Pow<A, B> = Binary<binary::Pow, A, B>;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Expr<T>(pub(crate) T);

impl<A> Expr<A> {
    pub fn add<B>(self, rhs: Expr<B>) -> Expr<Add<A, B>> {
        Expr(Binary(binary::Add, self.0, rhs.0))
    }

    pub fn mul<B>(self, rhs: Expr<B>) -> Expr<Mul<A, B>> {
        Expr(Binary(binary::Mul, self.0, rhs.0))
    }

    pub fn pow<B>(self, rhs: Expr<B>) -> Expr<Pow<A, B>> {
        Expr(Binary(binary::Pow, self.0, rhs.0))
    }
}

impl<A, B> std::ops::Add<Expr<B>> for Expr<A> {
    type Output = Expr<Add<A, B>>;

    fn add(self, rhs: Expr<B>) -> Self::Output {
        Expr::add(self, rhs)
    }
}

impl<A, B> std::ops::Mul<Expr<B>> for Expr<A> {
    type Output = Expr<Mul<A, B>>;

    fn mul(self, rhs: Expr<B>) -> Self::Output {
        Expr::mul(self, rhs)
    }
}

impl<A: Insertable, B: Insertable> Insertable for (A, B) {
    type Output = (A::Output, B::Output);

    fn insert(self, ops: &mut Operations) -> Self::Output {
        (self.0.insert(ops), self.1.insert(ops))
    }
}

impl<T: Insertable, const N: usize> Insertable for [T; N] {
    type Output = [T::Output; N];
    
    fn insert(self, ops: &mut Operations) -> Self::Output {
        self.map(|item| ops.insert(item))
    }
}

impl Insertable for Var {
    type Output = Expr<NodeId>;

    fn insert(self, ops: &mut Operations) -> Expr<NodeId> {
        Expr(ops.push(crate::engine::Op::Var))
    }
}

impl<O: Into<crate::engine::Unary>, A: Insertable<Output = Expr<NodeId>>> Insertable for Unary<O, A> {
    type Output = Expr<NodeId>;

    fn insert(self, ops: &mut Operations) -> Expr<NodeId> {
        let a= self.1.insert(ops);
        Expr(ops.push(crate::engine::Op::Unary(self.0.into(), a)))
    }
}

impl<O: Into<crate::engine::Binary>, A: Insertable<Output = Expr<NodeId>>, B: Insertable<Output = Expr<NodeId>>> Insertable for Binary<O, A, B> {
    type Output = Expr<NodeId>;

    fn insert(self, ops: &mut Operations) -> Expr<NodeId> {
        let (a, b) = (self.1, self.2).insert(ops);
        Expr(ops.push(crate::engine::Op::Binary(self.0.into(), a, b)))
    }
}

impl<T: Insertable> Insertable for Expr<T> {
    type Output = T::Output;

    fn insert(self, ops: &mut Operations) -> Self::Output {
        ops.insert(self.0)
    }
}

impl Insertable for NodeId {
    type Output = Expr<NodeId>;

    fn insert(self, _: &mut Operations) -> Self::Output {
        Expr(self)
    }
}
