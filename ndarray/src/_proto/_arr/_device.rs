use std::sync::{RwLock, Arc};

pub 
type Ptr = *mut std::ffi::c_void;

pub
type DevBuf<DevT> = Arc<RwLock<DevT>>;

pub
struct Buffer<DevT> {
    range: (Ptr, Ptr),
    device: DevT,
}

