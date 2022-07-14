use std::ops::Add;
use super::*;

use super::helpers::aspawn;

use crate::async_rt::task::JoinHandle;

type Err = 
<CPUUniStream as Device>::Err;

impl Add for NDArray<i32, CPUUniStream> {
    type Output = JoinHandle<Result<Self, Err>>;
    fn add(
        self, 
        rhs: Self,
    ) -> Self::Output {aspawn! {{
        if self.size != rhs.size { Err("unmatched size")? }
        {let lbuff = &mut *self.buff.lock().unwrap();
        let rbuff = &*rhs.buff.lock().unwrap();
        let step = std::mem::size_of::<i32>();
        let size = unsafe { lbuff.upper.offset_from(lbuff.lower) / (step as isize) };
        for x in 0..size {unsafe {
            *(lbuff.lower.offset(x) as *mut i32) += *(rbuff.lower.offset(x) as *mut i32); 
        }}} // drop lbuff here
        Ok(self)
    }}}
}