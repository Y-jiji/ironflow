use std::ops::Div;
use super::*;


use crate::async_rt::task::JoinHandle;

type Err = 
<CpuUniStream as Device>::Err;

macro_rules! impl_div_for_types {($($t: ty), *) => {$(

impl Div for NDArray<$t, CpuUniStream> {
    type Output = JoinHandle<Result<Self, Err>>;
    fn div(
        self, 
        rhs: Self,
    ) -> Self::Output {aspawn! {{
        if self.size != rhs.size { Err("unmatched size")? }
        // get a deep copy of self
        let out = self.deep_clone()?;
        // get locked lhs buff
        {let lbuff = &mut *out.buff.lock().unwrap();
        // get locked rhs buff
        let rbuff = &*rhs.buff.lock().unwrap();
        // get step size
        let step = std::mem::size_of::<$t>() as isize;
        let size = unsafe { lbuff.upper.offset_from(lbuff.lower) / step };
        for x in 0..size {unsafe {
            *(lbuff.lower.offset(x*step) as *mut $t) /= 
            *(rbuff.lower.offset(x*step) as *mut $t); 
        }}} // drop lbuff here
        Ok(out)
    }}}
}

)*};}

impl_div_for_types!{ i32, i64, f32, f64 }