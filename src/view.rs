use std::ops::{Deref, DerefMut};

pub trait Index: Into<usize> + From<usize> + Copy {
    fn indices(self) -> impl Iterator<Item = Self> {
        (0..self.into()).map(From::from)
    }

    fn reindex<U: Index>(self) -> U {
        U::from(Into::<usize>::into(self))
    }
}
impl<T> Index for T where T: Into<usize> + From<usize> + Copy {}

#[macro_export]
macro_rules! impl_index {
    ($T:ident) => {
        #[derive(Debug, Copy, Clone)]
        pub struct $T(pub usize);
        impl From<usize> for $T {
            fn from(value: usize) -> Self {
                $T(value)
            }
        }
        impl From<$T> for usize {
            fn from(value: $T) -> Self {
                value.0
            }
        }
    };
}

pub trait IndexTuple: Copy {
    fn flatten(&self, index: Self) -> usize;
    fn unflatten(&self, index: usize) -> Self;
    fn product(&self) -> usize;
    fn indices(&self) -> impl Iterator<Item = Self>;
}

impl<X0> IndexTuple for (X0,)
where
    X0: Index,
{
    fn flatten(&self, index: Self) -> usize {
        index.0.into()
    }

    fn unflatten(&self, index: usize) -> Self {
        (X0::from(index),)
    }

    fn product(&self) -> usize {
        self.0.into()
    }

    fn indices(&self) -> impl Iterator<Item = Self> {
        self.0.indices().map(|i0| (i0,))
    }
}

impl<X0, X1> IndexTuple for (X0, X1)
where
    X0: Index,
    X1: Index,
{
    fn flatten(&self, index: Self) -> usize {
        index.1.into() * self.0.into() + index.0.into()
    }

    fn unflatten(&self, index: usize) -> Self {
        (
            X0::from(index / self.0.into()),
            X1::from(index % self.0.into()),
        )
    }

    fn product(&self) -> usize {
        self.0.into() * self.1.into()
    }

    fn indices(&self) -> impl Iterator<Item = Self> {
        self.1
            .indices()
            .flat_map(|i1| self.0.indices().map(move |i0| (i0, i1)))
    }
}

// pub trait Zero {
//     const ZERO: Self;
// }

// pub struct Tensor<T, X> {
//     data: Box<[T]>,
//     len: X,
// }

// impl<T, X0, X1, X2> std::ops::Mul<&Tensor<T, (X1, X2)>> for &Tensor<T, (X0, X1)> where T: std::ops::Mul<Output = T> + std::ops::AddAssign + Zero + Copy, X0: Index, X1: Index, X2: Index {
//     type Output = Tensor<T, (X0, X2)>;

//     fn mul(self, rhs: &Tensor<T, (X1, X2)>) -> Self::Output {
//         assert_eq!(self.len.1.into(), rhs.len.0.into());
//         let len = (self.len.0, rhs.len.1);
//         let mut data = Vec::with_capacity(len.product());
//         for x2 in rhs.len.1.indices() {
//             for x0 in self.len.0.indices() {
//                 let mut sum = T::ZERO;
//                 for x1 in self.len.1.indices() {
//                     sum += self[(x0, x1)] * rhs[(x1, x2)];
//                 }
//                 data[len.flatten((x0, x2))] = sum;
//             }
//         }
//         Tensor {
//             data: data.into_boxed_slice(), len
//         }
//     }
// }

// impl<T, X> std::ops::Index<X> for Tensor<T, X> where X: IndexTuple {
//     type Output = T;

//     fn index(&self, index: X) -> &Self::Output {
//         &self.data[self.len.flatten(index)]
//     }
// }

// impl<T, X> std::ops::IndexMut<X> for Tensor<T, X> where X: IndexTuple {
//     fn index_mut(&mut self, index: X) -> &mut Self::Output {
//         &mut self.data[self.len.flatten(index)]
//     }
// }

// impl<T, X0, X1> Tensor<T, (X0, X1)> where T: Zero + Copy, X0: Index, X1: Index {
//     pub fn zeros(len: (X0, X1)) -> Self {
//         let data = vec![T::ZERO; len.product()].into_boxed_slice();
//         Self { data, len }
//     }
// }

#[derive(Debug, Copy, Clone)]
pub struct View<A, X> {
    data: A,
    len: X,
}

impl<A, X> View<A, X>
where
    X: IndexTuple,
{
    pub fn new<T>(data: A, len: X) -> Self
    where
        A: AsRef<[T]>,
    {
        assert_eq!(data.as_ref().len(), len.product());
        Self { data, len }
    }

    pub fn len(&self) -> X {
        self.len
    }

    pub fn data(&self) -> &A {
        &self.data
    }

    pub fn reindex<T, Y: IndexTuple, F: FnOnce(X) -> Y>(self, f: F) -> View<A, Y>
    where
        A: AsRef<[T]>,
    {
        let Self { data, len } = self;
        View::new(data, f(len))
    }

    pub fn as_ref<T>(&self) -> View<&A, X>
    where
        A: AsRef<[T]>,
    {
        View::new(&self.data, self.len)
    }

    pub fn as_mut<T>(&mut self) -> View<&mut A, X>
    where
        A: AsRef<[T]>,
    {
        View::new(&mut self.data, self.len)
    }

    pub fn as_deref<T>(&self) -> View<&<A as Deref>::Target, X>
    where
        A: Deref,
        <A as Deref>::Target: AsRef<[T]>,
    {
        View::new(self.data.deref(), self.len)
    }

    pub fn as_deref_mut<T>(&mut self) -> View<&mut <A as Deref>::Target, X>
    where
        A: DerefMut,
        <A as Deref>::Target: AsRef<[T]>,
    {
        View::new(self.data.deref_mut(), self.len)
    }
}

impl<A, X> View<A, X>
where
    Self: std::ops::Index<X>,
    X: IndexTuple,
{
    pub fn iter_enumerate(
        &self,
    ) -> impl Iterator<Item = (X, &<Self as std::ops::Index<X>>::Output)> {
        self.len.indices().map(|i| (i, &self[i]))
    }
}

impl<A, X> std::ops::Index<X> for View<A, X>
where
    A: std::ops::Deref,
    A::Target: std::ops::Index<usize>,
    X: IndexTuple,
{
    type Output = <<A as std::ops::Deref>::Target as std::ops::Index<usize>>::Output;

    fn index(&self, index: X) -> &Self::Output {
        &self.data.deref()[self.len.flatten(index)]
    }
}

impl<A, X> std::ops::IndexMut<X> for View<A, X>
where
    A: std::ops::DerefMut,
    A::Target: std::ops::IndexMut<usize>,
    X: IndexTuple,
{
    fn index_mut(&mut self, index: X) -> &mut Self::Output {
        &mut self.data.deref_mut()[self.len.flatten(index)]
    }
}
