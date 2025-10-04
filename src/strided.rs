pub trait Index: Copy + From<usize> + Into<usize> {}

// pub trait StridelessShape {
//     fn flatten(&self, index: Self) -> usize;
// }

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Strided<X> {
    pub stride: X,
    pub count: X,
}

impl<X> Strided<X> {
    fn dense(count: X) -> Self
    where
        X: From<usize>,
    {
        Self {
            stride: 1.into(),
            count,
        }
    }

    fn untype(self) -> Strided<usize>
    where
        X: Into<usize>,
    {
        Strided {
            stride: self.stride.into(),
            count: self.count.into(),
        }
    }
}

impl Strided<usize> {
    fn retype<X>(self) -> Strided<X>
    where
        X: From<usize>,
    {
        Strided {
            stride: self.stride.into(),
            count: self.count.into(),
        }
    }
}

pub trait Shape {
    type Index;

    fn dense(counts: Self::Index) -> Self;

    fn flatten(self, index: Self::Index) -> usize;

    fn unflatten(self, index: usize) -> Self::Index;

    fn product(self) -> usize;
}

impl<const N: usize> Shape for [Strided<usize>; N] {
    type Index = [usize; N];

    #[inline]
    fn dense(counts: Self::Index) -> Self {
        std::array::from_fn(|i| Strided::dense(counts[i]))
    }

    #[inline]
    fn flatten(self, index: Self::Index) -> usize {
        for i in 0..N {
            debug_assert!(index[i] < self[i].count);
        }
        (0..N).map(|i| self[i].stride * index[i]).sum()
    }

    #[inline]
    fn unflatten(self, index: usize) -> Self::Index {
        let mut rem = index;
        std::array::from_fn(move |i| {
            let s = self[i].stride;
            let c = self[i].count;
            let i = (rem / s) % c;
            rem -= i * s;
            i
        })
    }

    #[inline]
    fn product(self) -> usize {
        (0..N).map(|i| self[i].count).product()
    }
}

impl<X> Shape for X
where
    X: StridedTuple,
    X::StridedArray: Shape<Index = X::Array>,
{
    type Index = X::Tuple;

    #[inline]
    fn dense(counts: Self::Index) -> Self {
        Self::from_strided_array(Shape::dense(counts.into_array()))
    }

    #[inline]
    fn flatten(self, index: Self::Index) -> usize {
        self.into_strided_array().flatten(index.into_array())
    }

    #[inline]
    fn unflatten(self, index: usize) -> Self::Index {
        Self::Index::from_array(self.into_strided_array().unflatten(index))
    }

    #[inline]
    fn product(self) -> usize {
        self.into_strided_array().product()
    }
}

pub trait Tuple {
    type Array;

    fn into_array(self) -> Self::Array;

    fn from_array(value: Self::Array) -> Self;
}

pub trait StridedArray {
    type Array;
}

impl<const N: usize, T> StridedArray for [Strided<T>; N] {
    type Array = [T; N];
}

pub trait StridedTuple {
    type StridedArray: StridedArray<Array = Self::Array>;
    type Tuple: Tuple<Array = Self::Array>;
    type Array;

    fn into_strided_array(self) -> Self::StridedArray;
    fn from_strided_array(value: Self::StridedArray) -> Self;
}

macro_rules! impl_tuple {
    ($N:literal => $($t:ident: $T:ident),*) => {
        impl<$($T),*> StridedTuple for ($(Strided<$T>,)*)
        where
            $(
                $T: Index,
            )*
        {
            type StridedArray = [Strided<usize>; $N];
            type Tuple = ($($T,)*);
            type Array = [usize; $N];

            fn into_strided_array(self) -> Self::StridedArray {
                let ($($t,)*) = self;
                [$($t.untype(),)*]
            }

            fn from_strided_array(value: Self::StridedArray) -> Self {
                let [$($t,)*] = value;
                ($($t.retype(),)*)
            }
        }

        impl<$($T,)*> Tuple for ($($T,)*)
        where
            $(
                $T: Index,
            )*
        {
            type Array = [usize; $N];

            fn into_array(self) -> Self::Array {
                let ($($t,)*) = self;
                [$($t.into(),)*]
            }

            fn from_array(value: Self::Array) -> Self {
                let [$($t,)*] = value;
                ($($t.into(),)*)
            }
        }
    };
}

impl_tuple!(1 => x0: X0);
impl_tuple!(2 => x0: X0, x1: X1);
impl_tuple!(3 => x0: X0, x1: X1, x2: X2);

pub trait TupleInto<T> {
    fn tuple_into(self) -> T;
}

macro_rules! impl_tuple_into {
    ($($t:ident: $T:ident),* $(,)?) => {
        impl<T0, T1, U0, U1> TupleInto<(U0, U1)> for (T0, T1)
        where
            T0: Into<U0>,
            T1: Into<U1>,
        {
            fn tuple_into(value: (T0, T1)) -> Self {
                (value.0.into(), value.1.into())
            }
        }
    };
}

pub trait ShapeTranspose {
    type Transposed;

    fn transpose(self) -> Self::Transposed;
}

impl ShapeTranspose for [Strided<usize>; 2] {
    type Transposed = Self;

    fn transpose(self) -> Self::Transposed {
        let [a, b] = self;
        [b, a]
    }
}

impl<X0, X1> ShapeTranspose for (Strided<X0>, Strided<X1>) {
    type Transposed = (Strided<X1>, Strided<X0>);

    fn transpose(self) -> Self::Transposed {
        (self.1, self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x() {
        let shape = StridelessShape::new((
            DimSpec {
                offset: 1,
                stride: NonZero::new(3).unwrap(),
                count: NonZero::new(2).unwrap(),
                size: NonZero::new(10).unwrap(),
            },
            DimSpec {
                offset: 2,
                stride: NonZero::new(2).unwrap(),
                count: NonZero::new(3).unwrap(),
                size: NonZero::new(9).unwrap(),
            },
        ));

        assert_eq!(
            shape,
            StridelessShape {
                offset: 11,
                dims: (
                    Strided {
                        stride: NonZero::new(27).unwrap(),
                        count: NonZero::new(2).unwrap(),
                    },
                    Strided {
                        stride: NonZero::new(2).unwrap(),
                        count: NonZero::new(3).unwrap()
                    },
                )
            }
        );

        assert_eq!(shape.flatten((0, 0)), 11);
        assert_eq!(shape.flatten((0, 1)), 13);
        assert_eq!(shape.flatten((0, 2)), 15);
        assert_eq!(shape.flatten((1, 0)), 38);
        assert_eq!(shape.flatten((1, 1)), 40);
        assert_eq!(shape.flatten((1, 2)), 42);

        let shape = shape.transpose();

        assert_eq!(
            shape,
            StridelessShape {
                offset: 11,
                dims: (Strided { stride: 2, count: 3 }, Strided { stride: 27, count: 2 })
            }
        );

        assert_eq!(shape.flatten((0, 0)), 11);
        assert_eq!(shape.flatten((0, 1)), 38);
        assert_eq!(shape.flatten((1, 0)), 13);
        assert_eq!(shape.flatten((1, 1)), 40);
        assert_eq!(shape.flatten((2, 0)), 15);
        assert_eq!(shape.flatten((2, 1)), 42);
    }
}

// struct S<T>(T, T);

// trait StridedArray {
//     type Unstrided;
// }

// impl<const N: usize> StridedArray for [S<usize>; N] {
//     type Unstrided = [usize; N];
// }

// trait Tuple {
//     type Array;
// }

// trait StridedTuple {
//     // type UnstridedArray;
//     // type Unstrided: Tuple<Array = Self::UnstridedArray>;
//     // type Array: StridedArray<Unstrided = Self::UnstridedArray>;
//     type Unstrided: Tuple;
//     type Array: StridedArray;
// }

// impl<X0, X1> Tuple for (X0, X1) {
//     type Array = [usize; 2];
// }

// impl<X0, X1> StridedTuple for (S<X0>, S<X1>) {
//     // type UnstridedArray = [usize; 2];

//     type Unstrided = (X0, X1);

//     type Array = [S<usize>; 2];
// }

// fn x<ST>()
// where
//     ST: StridedTuple,
//     ST::Unstrided: Tuple<Array = <<ST as StridedTuple>::Array as StridedArray>::Unstrided>,
// {
// }

// trait AB {
//     type A: A<B = <Self::B as B>::A>;
//     type B: B<A = <Self::A as A>::B>;
// }

// trait A {
//     type B;
// }

// trait B {
//     type A;
// }

// trait AB {
//     type A: A<B = Self::AB>;
//     type B: B<A = Self::AB>;
//     type AB;
// }

// trait A {
//     type B;
// }

// trait B {
//     type A;
// }
