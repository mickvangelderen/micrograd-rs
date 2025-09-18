// TODO: make this work I guess, I started with Tensor and ended up writing the View abstraction (multi-dimensional contiguous array view). 

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
