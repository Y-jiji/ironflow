use crate::*;
use super::*;
use std::ffi::c_void;

impl From<NDArray<i32, CpuUniStream>>
for Vec<i32> {
    fn from(x: NDArray<i32, CpuUniStream>) -> Self {
        let len = x.size.into_iter().reduce(|x,y| x*y).unwrap_or(0);
        let ret = vec![0; len];
        x.device.lock().unwrap().memcpy(
            ret.as_ptr() as *mut c_void, 0, 
            x.buff.lock().unwrap().lower, 0, 
            len * std::mem::size_of::<i32>()).unwrap();
        ret
    }
}