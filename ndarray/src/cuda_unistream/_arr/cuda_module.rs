use super::cuda_binding::*;
use super::cuda_error::*;
use std::ffi::c_void;

#[derive(Debug)]
pub(crate) 
struct CudaModule {inner: *mut CUmod_st} // wrapper for CUmod

#[derive(Debug)]
pub(crate) 
struct CudaFn {pub(crate) inner: *const CUfunc_st}  // wrapper for CUfunc

impl CudaModule {
    pub(crate)
    fn new(data: &str) -> CudaResult<Self> {
        let mut inner = std::ptr::null_mut();
        unsafe { cuModuleLoadData(
            &mut inner as *mut *mut CUmod_st, 
            data.as_ptr() as *const c_void )
        }.wrap_val(())?;
        Ok(Self { inner })
    }
    pub(crate)
    fn get_fn(&self, name: &str) -> CudaResult<CudaFn> {
        let mut inner = std::ptr::null_mut();
        unsafe { cuModuleGetFunction(
            &mut inner, self.inner, name.as_ptr() as *const i8) }
        .wrap_val(CudaFn { inner })
    }
}

impl Drop for CudaModule {
    fn drop(self: &mut CudaModule) {
        #[cfg(test)]
        println!("drop {self:?}");
        unsafe { cuModuleUnload(self.inner) }
        .wrap_val(()).unwrap();
    }
}
