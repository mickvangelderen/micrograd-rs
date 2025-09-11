use crate::engine::{NodeId, Operations};

#[macro_export]
macro_rules! compute {
    // Looks like a function call...
    ($ctx:expr, $f:ident ( $($args:tt)* )) => {
        $crate::compute!(@split ($ctx) $f { } { } $($args)* )
    };

    // Otherwise, return as-is.
    ($ctx:expr, $e:expr) => { $e };

    // Splitting - we have to parse groups of tokens that form expressions
    // ourselves, using :expr doesn't allow us to see if it is a function call
    // later, see
    // https://lukaswirth.dev/tlborm/decl-macros/minutiae/metavar-and-expansion.html. 

    // End of input: push the current (if any) and finish.
    (@split ($ctx:expr) $f:ident { $($args_acc:tt)* } { $($cur:tt)* }) => {
        $crate::compute!(@call ($ctx) $f { $($args_acc)* [ $($cur)* ] })
    };

    // Comma at top level: push current arg, reset it, continue.
    (@split ($ctx:expr) $f:ident { $($args_acc:tt)* } { $($cur:tt)* } , $($rest:tt)* ) => {
        $crate::compute!(@split ($ctx) $f { $($args_acc)* [ $($cur)* ] } { } $($rest)* )
    };

    // Balanced group: parens — append as a single token to current.
    (@split ($ctx:expr) $f:ident { $($args_acc:tt)* } { $($cur:tt)* } ( $($g:tt)* ) $($rest:tt)* ) => {
        $crate::compute!(@split ($ctx) $f { $($args_acc)* } { $($cur)* ( $($g)* ) } $($rest)* )
    };
    // Brackets.
    (@split ($ctx:expr) $f:ident { $($args_acc:tt)* } { $($cur:tt)* } [ $($g:tt)* ] $($rest:tt)* ) => {
        $crate::compute!(@split ($ctx) $f { $($args_acc)* } { $($cur)* [ $($g)* ] } $($rest)* )
    };
    // Braces.
    (@split ($ctx:expr) $f:ident { $($args_acc:tt)* } { $($cur:tt)* } { $($g:tt)* } $($rest:tt)* ) => {
        $crate::compute!(@split ($ctx) $f { $($args_acc)* } { $($cur)* { $($g)* } } $($rest)* )
    };

    // Any other single token: append to current and continue.
    (@split ($ctx:expr) $f:ident { $($args_acc:tt)* } { $($cur:tt)* } $t:tt $($rest:tt)* ) => {
        $crate::compute!(@split ($ctx) $f { $($args_acc)* } { $($cur)* $t } $($rest)* )
    };

    // Calling

    (@call ($ctx:expr) $f:ident { }) => { ($ctx).$f() };

    (@call ($ctx:expr) $f:ident { [ $($a0:tt)* ] }) => {{
        let a0 = $crate::compute!($ctx, $($a0)*);
        ($ctx).$f(a0)
    }};

    (@call ($ctx:expr) $f:ident { [ $($a0:tt)* ] [ $($a1:tt)* ] }) => {{
        let a0 = $crate::compute!($ctx, $($a0)*);
        let a1 = $crate::compute!($ctx, $($a1)*);
        ($ctx).$f(a0, a1)
    }};

    (@call ($ctx:expr) $f:ident { [ $($a0:tt)* ] [ $($a1:tt)* ] [ $($a2:tt)* ] }) => {{
        let a0 = $crate::compute!($ctx, $($a0)*);
        let a1 = $crate::compute!($ctx, $($a1)*);
        let a2 = $crate::compute!($ctx, $($a2)*);
        ($ctx).$f(a0, a1, a2)
    }};

    (@call ($ctx:expr) $f:ident { [ $($a0:tt)* ] [ $($a1:tt)* ] [ $($a2:tt)* ] [ $($a3:tt)* ] }) => {{
        let a0 = $crate::compute!($ctx, $($a0)*);
        let a1 = $crate::compute!($ctx, $($a1)*);
        let a2 = $crate::compute!($ctx, $($a2)*);
        let a3 = $crate::compute!($ctx, $($a3)*);
        ($ctx).$f(a0, a1, a2, a3)
    }};
}

fn add_n<I: IntoIterator<Item = NodeId>>(args: I, ops: &mut Operations) -> Option<NodeId> {
    let mut args = args.into_iter();
    let mut acc = args.next()?;
    for arg in args {
        acc = ops.add(acc, arg)
    }
    Some(acc)
}

fn layer() {
    let mut ops = Operations::default();

    // inputs
    let x00 = ops.lit();
    let x01 = ops.lit();

    // parameters
    let a00_10 = ops.lit();
    let a01_10 = ops.lit();
    let b10 = ops.lit();

    let a00_11 = ops.lit();
    let a01_11 = ops.lit();
    let b11 = ops.lit();

    let all = add_n([x00, x01, a00_10], &mut ops);
    
    // outputs
    let x10 = compute!(ops, add(add(mul(x00, a00_10), mul(x01, a01_10)), b10));
    let x11 = compute!(ops, add(add(mul(x00, a00_11), mul(x01, a01_11)), b11));

    
}
