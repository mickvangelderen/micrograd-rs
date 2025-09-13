mod sealed {
    pub trait Sealed {}
}

/// Map the first element in a tuple.
pub trait MapT0: sealed::Sealed {
    type T0;
    type Output<U0>;

    fn map_t0<U0, F>(self, f: F) -> Self::Output<U0>
    where
        F: FnOnce(Self::T0) -> U0;
}

macro_rules! impl_map_t0 {
    ($t0:ident : $T0:ident $(, $ti:ident : $Ti:ident )* $(,)?) => {
        impl<$T0, $($Ti, )*> sealed::Sealed for ($T0, $($Ti, )*) {}

        impl<$T0, $($Ti, )*> MapT0 for ($T0, $($Ti, )*) {
            type T0 = $T0;
            type Output<U0> = (U0, $($Ti, )* );

            #[inline]
            fn map_t0<U0, F>(self, f: F) -> Self::Output<U0>
            where
                F: FnOnce(Self::T0) -> U0,
            {
                let ($t0, $($ti, )*) = self;
                (f($t0), $($ti, )*)
            }
        }
    }
}

impl_map_t0!(t0: T0, t1: T1);

pub trait IteratorExt: Iterator + Sized {
    fn map_t0<F, U0>(self, f: F) -> impl Iterator<Item = <Self::Item as MapT0>::Output<U0>>
    where
        Self::Item: MapT0,
        F: FnMut(<Self::Item as MapT0>::T0) -> U0;

    fn enumerate_with<F, U0>(self, f: F) -> impl Iterator<Item = (U0, Self::Item)>
    where
        F: FnMut(usize) -> U0;
}

impl<I: Iterator> IteratorExt for I {
    fn map_t0<F, U0>(self, mut f: F) -> impl Iterator<Item = <Self::Item as MapT0>::Output<U0>>
    where
        Self::Item: MapT0,
        F: FnMut(<Self::Item as MapT0>::T0) -> U0,
    {
        self.map(move |item| item.map_t0(&mut f))
    }

    fn enumerate_with<F, U0>(self, f: F) -> impl Iterator<Item = (U0, Self::Item)>
    where
        F: FnMut(usize) -> U0,
    {
        self.enumerate().map_t0(f)
    }
}
