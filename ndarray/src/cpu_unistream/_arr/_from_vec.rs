use crate::*;
use super::*;
use std::ffi::c_void;

use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

impl From<(Vec<i32>, Arc<Mutex<CpuUniStream>>)>
for NDArray<i32, CpuUniStream> {
    fn from(x: (Vec<i32>, Arc<Mutex<CpuUniStream>>)) -> Self {
        let ptr = x.0.as_ptr() as *mut c_void;
        let len = x.0.len() * std::mem::size_of::<i32>();
        let buff = Buffer::new(x.1.clone(), len).unwrap();
        x.1.lock().unwrap()
        .memcpy(buff.lower, 0, ptr, 0, len)
        .unwrap();
        Self {
            device: x.1.clone(),
            buff: Arc::new(Mutex::new(buff)),
            size: vec![x.0.len(); 1],
            __type: PhantomData
        }
    }
}