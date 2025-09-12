use crate::engine::{Expr, Insertable, NodeId, Op, Operations};

pub struct Unary<O, A>(O, A);

impl<O: unary::Variant, A: Insertable<Output = NodeId>> Insertable for Unary<O, A> {
    type Output = NodeId;

    fn insert_into(self, ops: &mut Operations) -> Self::Output {
        Op::Unary(O::OP, self.1.insert_into(ops)).insert_into(ops)
    }
}

macro_rules! call_with_unary_variants {
    ($macro:ident) => {
        $macro!(Neg, neg);
        $macro!(Recip, recip);
        $macro!(Pow2, pow_2);
        $macro!(Ln, ln);
        $macro!(Ln1P, ln_1p);
        $macro!(Exp, exp);
        $macro!(Exp2, exp_2);
        $macro!(ExpM1, exp_m1);
    };
}

pub mod unary {
    use crate::engine;

    pub trait Variant {
        const OP: engine::Unary;
    }

    macro_rules! impl_struct {
        ($V:ident, $v:ident) => {
            pub struct $V;
        };
    }
    call_with_unary_variants!(impl_struct);

    macro_rules! impl_trait {
        ($V:ident, $v:ident) => {
            impl Variant for $V {
                const OP: engine::Unary = engine::Unary::$V;
            }
        };
    }
    call_with_unary_variants!(impl_trait);
}

macro_rules! impl_unary_alias {
    ($V:ident, $v:ident) => {
        pub type $V<A> = Unary<unary::$V, A>;
    };
}
call_with_unary_variants!(impl_unary_alias);

macro_rules! impl_unary_op {
    (Neg, neg) => {
        impl_unary_op!(@trait Neg, neg);
    };
    ($V:ident, $v:ident) => {
        impl<A> Expr<A> {
            pub fn $v(self) -> Expr<$V<A>> {
                Expr(Unary(unary::$V, self.0))
            }
        }
    };
    (@trait $V:ident, $v:ident) => {
        impl<A> std::ops::$V for Expr<A> {
            type Output = Expr<$V<A>>;

            fn $v(self) -> Self::Output {
                Expr(Unary(unary::$V, self.0))
            }
        }
    }
}
call_with_unary_variants!(impl_unary_op);

pub struct Binary<O, A>(O, A);

impl<O: binary::Variant, A: Insertable<Output = (NodeId, NodeId)>> Insertable for Binary<O, A> {
    type Output = NodeId;

    fn insert_into(self, ops: &mut Operations) -> Self::Output {
        Op::Binary(O::OP, self.1.insert_into(ops)).insert_into(ops)
    }
}

macro_rules! call_with_binary_variants {
    ($macro:ident) => {
        $macro!(Add, add);
        $macro!(Sub, sub);
        $macro!(Mul, mul);
        $macro!(Div, div);
        $macro!(Pow, pow);
    };
}

pub mod binary {
    use crate::engine;

    pub trait Variant {
        const OP: engine::Binary;
    }

    macro_rules! impl_struct {
        ($V:ident, $v:ident) => {
            pub struct $V;
        };
    }
    call_with_binary_variants!(impl_struct);

    macro_rules! impl_trait {
        ($V:ident, $v:ident) => {
            impl Variant for $V {
                const OP: engine::Binary = engine::Binary::$V;
            }
        };
    }
    call_with_binary_variants!(impl_trait);
}

macro_rules! impl_binary_alias {
    ($V:ident, $v:ident) => {
        pub type $V<A, B> = Binary<binary::$V, (A, B)>;
    };
}
call_with_binary_variants!(impl_binary_alias);

macro_rules! impl_binary_op {
    (Add, add) => {
        impl_binary_op!(@trait Add, add);
    };
    (Sub, sub) => {
        impl_binary_op!(@trait Sub, sub);
    };
    (Mul, mul) => {
        impl_binary_op!(@trait Mul, mul);
    };
    (Div, div) => {
        impl_binary_op!(@trait Div, div);
    };
    ($V:ident, $v:ident) => {
        impl<A> Expr<A> {
            pub fn $v<B>(self, rhs: Expr<B>) -> Expr<$V<A, B>> {
                Expr(Binary(binary::$V, (self.0, rhs.0)))
            }
        }
    };
    (@trait $V:ident, $v:ident) => {
        impl<A, B> std::ops::$V<Expr<B>> for Expr<A> {
            type Output = Expr<$V<A, B>>;

            fn $v(self, rhs: Expr<B>) -> Self::Output {
                Expr(Binary(binary::$V, (self.0, rhs.0)))
            }
        }
    }
}
call_with_binary_variants!(impl_binary_op);

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
