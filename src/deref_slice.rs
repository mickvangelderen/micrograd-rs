/// This module provides `DerefSlice` and `DerefSliceMut`. These traits allow us
/// to obtain the item type `T` for a container `A` which implements
/// `Deref<[T]>`. This is useful because it allows us to abstract over borrowed
/// (&[T], &mut [T]) and owned (Vec<T>, Box<[T]>) providers of contiguous
/// arrays.
use std::ops::Deref;
use std::ops::DerefMut;

mod sealed {
    pub trait Sealed {}
}

pub trait DerefSlice: Deref<Target = [Self::Item]> + sealed::Sealed {
    type Item;
}

pub trait DerefSliceMut: DerefSlice + DerefMut<Target = [Self::Item]> {}

impl<T, A> sealed::Sealed for A where A: Deref<Target = [T]> {}

impl<T, A> DerefSlice for A
where
    A: Deref<Target = [T]>,
{
    type Item = T;
}

impl<T, A> DerefSliceMut for A where A: DerefMut<Target = [T]> {}
