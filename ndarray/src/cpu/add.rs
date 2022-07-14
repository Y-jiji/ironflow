use super::super::proto::NDArray;
use super::device::*;
use std::ops::Add;

use async_std::task::*;

macro_rules! auto_impl {($($tok:ident), *) => {$(

impl Add for NDArray<$tok, SingleThreadNaive> {
    // consume self and another piece of data, get a new added one
    type Output = JoinHandle<NDArray<$tok, SingleThreadNaive>>;
    fn add(self, rhs: Self) -> Self::Output {
        spawn(async move {
            assert!(self.size == rhs.size, "have different size!");
            {let locked_self_data = &mut *self.data.lock().unwrap();
            let locked_rhs_data  = &*rhs.data.lock().unwrap();
            for i in 0..locked_self_data.len() {
                locked_self_data[i] += locked_rhs_data[i];
            }} /* the mutable ref to lhs.data dropped after this scope */
            return self;
        })
    }
}

impl Add for &NDArray<$tok, SingleThreadNaive> {
    type Output = JoinHandle<NDArray<$tok, SingleThreadNaive>>;
    fn add(self, rhs: Self) -> Self::Output {
        let lhs = self.deep_clone().unwrap();
        let rhs = rhs.clone();
        spawn(async move {
            assert!(lhs.size == rhs.size, "have different size!");
            {let locked_lhs_data = &mut *lhs.data.lock().unwrap();
            let locked_rhs_data  = &*rhs.data.lock().unwrap();
            for i in 0..locked_lhs_data.len() {
                locked_lhs_data[i] += locked_rhs_data[i];
            }} /* the mutable ref to lhs.data dropped after this scope */
            return lhs;
        })
    }
}

)*}}

auto_impl!{ i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, f32, f64 }

#[cfg(test)]
mod tests {
    use crate::{NDArray, cpu::device::SingleThreadNaive};
    use std::sync::{Mutex, Arc};
    use std::time::Instant;
    use async_std::task::*;
    use futures::future::join4;

    #[allow(unused_macros)]
    macro_rules! auto_impl_tests {($($tok:ident, $name:ident), *) => {$(
        #[test]
        fn $name() {
            let start = Instant::now();
            let a = NDArray::<$tok, SingleThreadNaive> {
                size: vec![4, 5, 6, 7, 8, 9, 10, 11, 12],
                data: Arc::new(Mutex::new(vec![1 as $tok; 4*5*6*7*8*9*10*11*12])),
                device: Arc::new(Mutex::new(SingleThreadNaive)),
                devptr: Arc::new(Mutex::new(())),
                            };
            let b = NDArray::<$tok, SingleThreadNaive> {
                size: vec![4, 5, 6, 7, 8, 9, 10, 11, 12],
                data: Arc::new(Mutex::new(vec![2 as $tok; 4*5*6*7*8*9*10*11*12])),
                device: Arc::new(Mutex::new(SingleThreadNaive)),
                devptr: Arc::new(Mutex::new(())),
                            };
            let c = NDArray::<$tok, SingleThreadNaive> {
                size: vec![4, 5, 6, 7, 8, 9, 10, 11, 12],
                data: Arc::new(Mutex::new(vec![3 as $tok; 4*5*6*7*8*9*10*11*12])),
                device: Arc::new(Mutex::new(SingleThreadNaive)),
                devptr: Arc::new(Mutex::new(())),
                            };
            let result = block_on(async { join4(&a+&b, &a+&b, &a+&b, &a+&b).await } );
            assert!(*result.0.data.lock().unwrap() == *c.data.lock().unwrap(), "1+2!=3");
            let end = Instant::now();
            println!("{:?}", end - start);
        }
    )*}}

    auto_impl_tests! {
        i8, test_add_i8,
        i16, test_add_i16,
        i32, test_add_i32,
        i64, test_add_i64,
        i128, test_add_i128,
        u8, test_add_u8,
        u16, test_add_u16,
        u32, test_add_u32,
        u64, test_add_u64,
        u128, test_add_u128,
        f32, test_add_f32,
        f64, test_add_f64
    }
}