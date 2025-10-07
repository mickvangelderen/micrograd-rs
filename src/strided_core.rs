use std::{
    mem::{ManuallyDrop, MaybeUninit},
    ptr::addr_of,
};

pub struct S<T> {
    pub count: T,
    pub stride: T,
}

pub(crate) const fn array_uninit<T, const N: usize>() -> [MaybeUninit<T>; N] {
    [const { MaybeUninit::uninit() }; N]
}

// TODO: Replace with stable version when stabilized.
pub(crate) const unsafe fn array_assume_init<T, const N: usize>(array: [MaybeUninit<T>; N]) -> [T; N] {
    ::core::mem::transmute_copy(&array)
}

pub(crate) const fn manually_drop_inner_ref<T>(slot: &ManuallyDrop<T>) -> &T {
    // SAFETY:
    // - same layout
    unsafe { std::mem::transmute(slot) }
}

#[macro_export]
macro_rules! const_for {
    ($var:pat_param in $range:expr => $body:stmt) => {
        let ::core::ops::Range { start: mut index, end } = $range;
        while index < end {
            let $var = index;
            $body
            index += 1;
        }
    };
}

#[inline]
const fn stride<const N: usize>(count: [usize; N]) -> [usize; N] {
    let mut stride = array_uninit();
    let mut mul = 1;
    const_for!(n in 0..N => {
        stride[n].write(mul);
        mul *= count[n];
    });
    unsafe { array_assume_init(stride) }
}

#[inline]
const fn new<const N: usize>(count: [usize; N]) -> S<[usize; N]> {
    S {
        count,
        stride: stride(count),
    }
}

#[inline]
const fn ravel<const N: usize>(S { count, stride }: S<[usize; N]>, index: [usize; N]) -> usize {
    let mut out = 0;

    const_for!(n in 0..N => {
        debug_assert!(index[n] < count[n]);
        out += index[n] * stride[n];
    });

    out
}

#[inline]
const fn unravel<const N: usize>(S { count, stride }: S<[usize; N]>, index: usize) -> [usize; N] {
    let mut out = array_uninit();

    let mut rem = index;
    const_for!(n in 0..N => {
        let s = stride[n];
        let i = (rem / s) % count[n];
        out[n].write(i);
        rem -= n * s;
    });

    unsafe { array_assume_init(out) }
}

#[inline]
const fn count<const N: usize>(shape: S<[usize; N]>) -> usize {
    let mut out = 1;

    const_for!(n in 0..N => {
        out *= shape.count[n];
    });

    out
}

#[inline]
const fn into_aos<const N: usize>(S { count, stride }: S<[usize; N]>) -> [S<usize>; N] {
    let mut shape = array_uninit();

    const_for!(n in 0..N => {
        shape[n].write(S { count: count[n], stride: stride[n] });
    });

    unsafe { array_assume_init(shape) }
}

#[inline]
const fn into_soa<T, const N: usize>(shape: [S<T>; N]) -> S<[T; N]> {
    let mut stride = array_uninit();
    let mut count = array_uninit();

    let shape = ManuallyDrop::new(shape);

    // NOTE: Work around our inability to obtain a reference to the inner value in a const context.
    let shape: &[S<T>; N] = manually_drop_inner_ref(&shape);

    const_for!(n in 0..N => {
        let shape = &shape[n];
        // SAFETY: The original will not be dropped and we only create one copy of each stride.
        stride[n].write(unsafe { addr_of!(shape.stride).read() });
        // SAFETY: The original will not be dropped and we only create one copy of each count.
        count[n].write(unsafe { addr_of!(shape.count).read() });
    });

    // SAFETY: All elements have been written to.
    let stride = unsafe { array_assume_init(stride) };
    // SAFETY: All elements have been written to.
    let count = unsafe { array_assume_init(count) };

    S { stride, count }
}
