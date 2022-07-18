use std::sync::Arc;
use parking_lot::Mutex;
use std::marker::PhantomData;
use std::ffi::c_void;

// a global guard for memory management
static MALLOC_GUARD: Arc<Mutex<PhantomData<()>>> = Arc::new(Mutex::new(PhantomData));

pub(crate) fn malloc(memlen: usize) -> *mut c_void {
    MALLOC_GUARD.lock();
    return unsafe {libc::malloc(memlen)};
}

pub(crate) fn free(ptr: *mut c_void) {
    MALLOC_GUARD.lock();
    unsafe {libc::free(ptr)};
}

