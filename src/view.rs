use crate::{
    deref_slice::{DerefSlice, DerefSliceMut},
    iter_ext::IteratorExt as _,
};

pub trait Index: Into<usize> + From<usize> + Eq + PartialEq + Copy {
    fn indices(self) -> impl Iterator<Item = Self> {
        (0..self.into()).map(From::from)
    }

    fn reindex<U: Index>(self) -> U {
        U::from(Into::<usize>::into(self))
    }
}
impl<T> Index for T where T: Into<usize> + From<usize> + Eq + PartialEq + Copy {}

#[macro_export]
macro_rules! impl_index {
    ($T:ident) => {
        #[derive(Debug, Copy, Clone, Eq, PartialEq)]
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
        (X0::from(index / self.0.into()), X1::from(index % self.0.into()))
    }

    fn product(&self) -> usize {
        self.0.into() * self.1.into()
    }

    fn indices(&self) -> impl Iterator<Item = Self> {
        self.1.indices().flat_map(|i1| self.0.indices().map(move |i0| (i0, i1)))
    }
}

#[derive(Debug, Copy, Clone)]
pub struct View<A, X> {
    data: A,
    len: X,
}

impl<A, X> View<A, X> {
    pub fn new(data: A, len: X) -> Self
    where
        A: DerefSlice,
        X: IndexTuple,
    {
        assert_eq!(data.len(), len.product());
        Self { data, len }
    }

    pub fn len(&self) -> &X {
        &self.len
    }

    pub fn data(&self) -> &A {
        &self.data
    }

    pub fn reindex<Y, F>(self, f: F) -> View<A, Y>
    where
        A: DerefSlice,
        Y: IndexTuple,
        F: FnOnce(X) -> Y,
    {
        let Self { data, len } = self;
        View::new(data, f(len))
    }

    pub fn as_ref(&self) -> View<&A, X>
    where
        X: IndexTuple,
        for<'a> &'a A: DerefSlice,
    {
        View::new(&self.data, self.len)
    }

    pub fn as_mut(&mut self) -> View<&mut A, X>
    where
        X: IndexTuple,
        for<'a> &'a mut A: DerefSliceMut,
    {
        View::new(&mut self.data, self.len)
    }

    pub fn as_deref(&self) -> View<&[A::Item], X>
    where
        X: IndexTuple,
        A: DerefSlice,
    {
        View::new(self.data.deref(), self.len)
    }

    pub fn as_deref_mut(&mut self) -> View<&mut [A::Item], X>
    where
        X: IndexTuple,
        A: DerefSliceMut,
    {
        View::new(self.data.deref_mut(), self.len)
    }
}

impl<A, X> View<A, X>
where
    X: IndexTuple,
    A: DerefSlice,
{
    pub fn iter(&self) -> impl Iterator<Item = &<A as DerefSlice>::Item> {
        self.data.iter()
    }

    pub fn iter_enumerate(&self) -> impl Iterator<Item = (X, &<A as DerefSlice>::Item)> {
        // TODO: View assembly/benchmark computing the index vs zipping it.
        let len = self.len;
        self.data.iter().enumerate().map_t0(move |index| len.unflatten(index))
    }
}

impl<A, X> View<A, X>
where
    X: IndexTuple,
    A: DerefSliceMut,
{
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut <A as DerefSlice>::Item> {
        self.data.iter_mut()
    }

    pub fn iter_mut_enumerate(&mut self) -> impl Iterator<Item = (X, &mut <A as DerefSlice>::Item)> {
        let len = self.len;
        self.data.iter_mut().enumerate_with(move |index| len.unflatten(index))
    }
}

// NOTE(mickvangelderen): I am not sure we can abstract over borrowed and owned implementations so easily here.
impl<'a, T, X> IntoIterator for View<&'a [T], X>
where
    X: IndexTuple,
{
    type Item = &'a T;

    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T, X> IntoIterator for View<&'a mut [T], X>
where
    X: IndexTuple,
{
    type Item = &'a mut T;

    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl<A, X> std::ops::IndexMut<X> for View<A, X>
where
    A: DerefSliceMut,
    X: IndexTuple,
{
    fn index_mut(&mut self, index: X) -> &mut Self::Output {
        &mut self.data.deref_mut()[self.len.flatten(index)]
    }
}

impl<A, X> std::ops::Index<X> for View<A, X>
where
    A: DerefSlice,
    X: IndexTuple,
{
    type Output = <A as DerefSlice>::Item;

    fn index(&self, index: X) -> &Self::Output {
        &self.data.deref()[self.len.flatten(index)]
    }
}
