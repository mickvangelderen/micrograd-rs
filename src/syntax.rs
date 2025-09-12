use crate::engine::{self, Expr, Insertable, NodeId, Op, Operations};

pub struct Unary<O, A>(O, A);

pub mod unary {
    use crate::engine;

    pub trait Dynamic {
        const DYNAMIC: engine::Unary;
    }

    pub struct Neg;

    impl Dynamic for Neg {
        const DYNAMIC: engine::Unary = engine::Unary::Neg;
    }
}

pub type Neg<A> = Unary<unary::Neg, A>;

trait UnaryStatic {
    const DYNAMIC: engine::Unary;
}

impl UnaryStatic for unary::Neg {
    const DYNAMIC: engine::Unary = engine::Unary::Neg;
}

impl<O: UnaryStatic, A: Insertable<Output = NodeId>> Insertable for Unary<O, A> {
    type Output = NodeId;

    fn insert_into(self, ops: &mut Operations) -> Self::Output {
        Op::Unary(O::DYNAMIC, self.1.insert_into(ops)).insert_into(ops)
    }
}

pub struct Binary<O, A>(O, A);

pub mod binary {
    pub struct Add;
    pub struct Mul;
    pub struct Pow;
}

pub type Add<A, B> = Binary<binary::Add, (A, B)>;
pub type Mul<A, B> = Binary<binary::Mul, (A, B)>;
pub type Pow<A, B> = Binary<binary::Pow, (A, B)>;

trait BinaryStatic {
    const DYNAMIC: engine::Binary;
}

impl BinaryStatic for binary::Add {
    const DYNAMIC: engine::Binary = engine::Binary::Add;
}

impl BinaryStatic for binary::Mul {
    const DYNAMIC: engine::Binary = engine::Binary::Mul;
}

impl BinaryStatic for binary::Pow {
    const DYNAMIC: engine::Binary = engine::Binary::Pow;
}

impl<O: BinaryStatic, A: Insertable<Output = (NodeId, NodeId)>> Insertable for Binary<O, A> {
    type Output = NodeId;

    fn insert_into(self, ops: &mut Operations) -> Self::Output {
        Op::Binary(O::DYNAMIC, self.1.insert_into(ops)).insert_into(ops)
    }
}

impl<A> Expr<A> {
    pub fn pow<B>(self, rhs: Expr<B>) -> Expr<Pow<A, B>> {
        Expr(Binary(binary::Pow, (self.0, rhs.0)))
    }
}

impl<A> std::ops::Neg for Expr<A> {
    type Output = Expr<Neg<A>>;

    fn neg(self) -> Self::Output {
        Expr(Unary(unary::Neg, self.0))
    }
}

impl<A, B> std::ops::Add<Expr<B>> for Expr<A> {
    type Output = Expr<Add<A, B>>;

    fn add(self, rhs: Expr<B>) -> Self::Output {
        Expr(Binary(binary::Add, (self.0, rhs.0)))
    }
}

impl<A, B> std::ops::Mul<Expr<B>> for Expr<A> {
    type Output = Expr<Mul<A, B>>;

    fn mul(self, rhs: Expr<B>) -> Self::Output {
        Expr(Binary(binary::Mul, (self.0, rhs.0)))
    }
}

#[cfg(test)]
pub mod tests {
    use crate::engine::Operations;

    #[allow(unused)]
    fn should_compile() {
        let ops = &mut Operations::default();
        let [a, b] = ops.vars();
        let _c = ops.insert(a + b);
        let _d = ops.insert(a * b);
        let _e = ops.insert(a.pow(b));
    }
}
