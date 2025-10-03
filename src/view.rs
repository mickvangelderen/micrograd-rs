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

#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct IndexTupleIter<X> {
    index: usize,
    shape: X,
}

impl<X> IndexTupleIter<X> {
    fn new(shape: X) -> Self {
        IndexTupleIter {
            index: Default::default(),
            shape,
        }
    }
}

impl<X> Iterator for IndexTupleIter<X>
where
    X: IndexTuple,
{
    type Item = X;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.shape.product() {
            let next = self.shape.unflatten(self.index);
            self.index += 1;
            Some(next)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remainder = self.shape.product().wrapping_sub(self.index);
        (remainder, Some(remainder))
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.shape.product().wrapping_sub(self.index)
    }

    fn last(mut self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        if self.index < self.shape.product() {
            self.index = self.shape.product() - 1;
        }
        self.next()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if n > 0 {
            let remainder = self.shape.product().wrapping_sub(self.index);
            self.index += std::cmp::min(n, remainder);
        }
        self.next()
    }
}

impl<X> std::iter::FusedIterator for IndexTupleIter<X> where X: IndexTuple {}
impl<X> ExactSizeIterator for IndexTupleIter<X> where X: IndexTuple {}

pub trait IndexTuple: Copy {
    fn flatten(&self, index: Self) -> usize;
    fn unflatten(&self, index: usize) -> Self;
    fn product(&self) -> usize;
    fn indices(self) -> impl Iterator<Item = Self> {
        IndexTupleIter::new(self)
    }
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
        (X0::from(index / self.1.into()), X1::from(index % self.1.into()))
    }

    fn product(&self) -> usize {
        self.0.into() * self.1.into()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct View<A, X> {
    data: A,
    shape: X,
}

impl<A, X> View<A, X> {
    pub fn new(data: A, shape: X) -> Self
    where
        A: DerefSlice,
        X: IndexTuple,
    {
        assert_eq!(data.len(), shape.product());
        Self { data, shape }
    }

    pub fn shape(&self) -> X
    where
        X: Copy,
    {
        self.shape
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
        let Self { data, shape } = self;
        View::new(data, f(shape))
    }

    pub fn as_ref(&self) -> View<&A, X>
    where
        X: IndexTuple,
        for<'a> &'a A: DerefSlice,
    {
        View::new(&self.data, self.shape)
    }

    pub fn as_mut(&mut self) -> View<&mut A, X>
    where
        X: IndexTuple,
        for<'a> &'a mut A: DerefSliceMut,
    {
        View::new(&mut self.data, self.shape)
    }

    pub fn as_deref(&self) -> View<&[A::Item], X>
    where
        X: IndexTuple,
        A: DerefSlice,
    {
        View::new(self.data.deref(), self.shape)
    }

    pub fn as_deref_mut(&mut self) -> View<&mut [A::Item], X>
    where
        X: IndexTuple,
        A: DerefSliceMut,
    {
        View::new(self.data.deref_mut(), self.shape)
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
        let shape = self.shape;
        self.data.iter().enumerate().map_t0(move |index| shape.unflatten(index))
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
        let shape = self.shape;
        self.data.iter_mut().enumerate_with(move |index| shape.unflatten(index))
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
        &mut self.data.deref_mut()[self.shape.flatten(index)]
    }
}

impl<A, X> std::ops::Index<X> for View<A, X>
where
    A: DerefSlice,
    X: IndexTuple,
{
    type Output = <A as DerefSlice>::Item;

    fn index(&self, index: X) -> &Self::Output {
        &self.data.deref()[self.shape.flatten(index)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl_index!(A);
    impl_index!(B);

    #[test]
    fn index_tuple_indices_iter_empty() {
        let shape = (A(0), B(0));
        assert!(shape.indices().next().is_none());
        assert_eq!(shape.indices().size_hint(), (0, Some(0)));
        assert_eq!(shape.indices().count(), 0);
        assert!(shape.indices().last().is_none());
        #[allow(clippy::iter_nth_zero)]
        {
            assert!(shape.indices().nth(0).is_none());
        }
        assert!(shape.indices().nth(1).is_none());
    }

    #[test]
    fn index_tuple_indices_iter() {
        let shape = (A(2), B(3));
        {
            let mut iter = shape.indices();
            assert_eq!(iter.next(), Some((A(0), B(0))));
            assert_eq!(iter.next(), Some((A(0), B(1))));
            assert_eq!(iter.next(), Some((A(0), B(2))));
            assert_eq!(iter.next(), Some((A(1), B(0))));
            assert_eq!(iter.next(), Some((A(1), B(1))));
            assert_eq!(iter.next(), Some((A(1), B(2))));
            assert_eq!(iter.next(), None);
        }
        assert_eq!(shape.indices().size_hint(), (6, Some(6)));
        {
            let iter = shape.indices();
            assert_eq!(iter.count(), 6);
            let mut iter = shape.indices();
            iter.next();
            assert_eq!(iter.count(), 5);
        }
        assert_eq!(shape.indices().last(), Some((A(1), B(2))));
        assert_eq!(shape.indices().nth(3), Some((A(1), B(0))));
    }
}
