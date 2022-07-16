use std::marker::PhantomData;
use std::sync::{Arc, RwLock};
use crate::Buffer;

pub
struct NDArray<ValT, DevT> {
    /// marker of value type
    pub(crate)
    _val : PhantomData<ValT>,
    /// buffer might be actively modified by device
    pub(crate) 
    buff : Arc<RwLock<Buffer<DevT>>>, 
    /// meta data about shape
    pub(crate) 
    size : Vec<usize>,
}

