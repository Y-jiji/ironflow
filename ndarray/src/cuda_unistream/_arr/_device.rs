use std::ffi::c_void;
use super::cuda_module::*;
use super::cuda_binding::*;
use super::cuda_error::*;
use crate::*;

pub(crate) 
struct CudaLaunchLayout {
    grid  : (usize, usize, usize),
    block : (usize, usize, usize),
}

#[derive(Debug)]
pub(crate)
struct CudaMemPool;

/*******************************************************************************
 * This struct manages a cuda stream and the operator bindings and also a piece
 * of binded memory. 
 *******************************************************************************/
#[derive(Debug)]
pub 
struct CudaUniStream {
    inner   : *mut CUstream_st ,
    module  : CudaModule       ,
    membase : *mut c_void      ,
    memsize : usize            ,
    mempool : CudaMemPool      ,
}

impl Clone 
for CudaUniStream 
{ fn clone(&self) -> Self { Self::new(self.memsize).unwrap() } }

unsafe impl Send
for CudaUniStream {}

/// hard coded kernel data (a monolithic ops.ptx file compiled from cusrc)
const KERNEL_FN: &str = include_str!(concat!(env!("OUT_DIR"), "/ops.ptx"));

impl CudaUniStream {
    /****************************************************************************
     * create a stream with memory and module bindings
     ****************************************************************************/
    pub
    fn new(memsize: usize)
    -> CudaResult<Self> {
        let inner 
            = CudaUniStream::init_stream()?;
        let membase
            = CudaUniStream::init_memory(inner, memsize)?;
        let module 
            = CudaModule::new(KERNEL_FN)?;
        Ok(Self { inner, membase, memsize, module, mempool: CudaMemPool })
    }
    /****************************************************************************
     * create a stream
     ****************************************************************************/
    fn init_stream()
    -> CudaResult<*mut CUstream_st> {
        let mut strmptr : *mut CUstream_st = std::ptr::null_mut();
        let error_n = 
            unsafe { cudaStreamCreate(&mut strmptr) };
        CudaError::from(error_n).wrap_val(strmptr)
    }
    /****************************************************************************
     * initialize memory for the current stream
     ****************************************************************************/
    fn init_memory(strmptr : *mut CUstream_st, memsize: usize) 
    -> CudaResult<*mut c_void> {
        let mut devptr = std::ptr::null_mut();
        let error_n = 
            unsafe { cudaMallocAsync(&mut devptr, memsize as u64, strmptr) };
        CudaError::from(error_n).wrap_val(devptr)
    }
    /****************************************************************************
     * launch a procedure call on the current CudaStream
     * in this implementation this function is usually called with `cujob` macro
     ****************************************************************************/
    pub(crate)
    fn launch(&self, name: &str, layout: CudaLaunchLayout, param: &[*mut c_void])
    -> CudaResult<()> {
        /* launch a kernel on current stream */
        let f: CudaFn = self.module.get_fn(name)?;
        let CudaLaunchLayout {
            grid: (gridx, gridy, gridz), 
            block: (blockx, blocky, blockz) 
        } = layout;
        let error = 
            unsafe {cuLaunchKernel(
                f.inner as *mut CUfunc_st,
                gridx as u32, gridy as u32, gridz as u32,
                blockx as u32, blocky as u32, blockz as u32,
                0, self.inner,
                param.as_ptr() as *mut *mut c_void, 
                std::ptr::null_mut()
            )};
        error.wrap_val(())
    }
    /****************************************************************************
     * synchronize current stream with current cpu thread
     ****************************************************************************/
    pub(crate)
    fn sync(&self) -> CudaResult<()> {
        CudaError::
        from( unsafe { cudaStreamSynchronize(self.inner) } ).wrap_val(())
    }
}

impl Drop for CudaUniStream {
    /****************************************************************************
     * drop CudaUniStream, if error occurs, the only choice is to panic. 
     ****************************************************************************/
    fn drop(self: &mut CudaUniStream) {
        #[cfg(test)] println!("drop {self:?}");
        // Launch a memfree job
        CudaError
        ::from(unsafe { 
            cudaFreeAsync(self.membase as *mut std::ffi::c_void, self.inner) 
        }).wrap_val(())
        .expect(&format!(
            "cudaFreeAsync {{lower:{:?}, upper:{:?}}} failed, or something bad happened before", 
            self.membase, unsafe {self.membase.add(self.memsize)}
        ));
        // Then sync this stream to make sure all procedures are done. 
        self.sync()
            .expect(&format!("cudaStreamSynchronize ({:?}) failed", self.inner));
        // Try to destroy the current stream
        CudaError::
        from(unsafe { cudaStreamDestroy(self.inner) }).wrap_val(())
        .expect(&format!("cudaStreamDestroy ({:?}) failed", self.inner));
    }
}

const H2D: i32 = cudaMemcpyKind_cudaMemcpyHostToDevice;
const D2H: i32 = cudaMemcpyKind_cudaMemcpyDeviceToHost;
const D2D: i32 = cudaMemcpyKind_cudaMemcpyDeviceToDevice;
// const H2H: i32 = cudaMemcpyKind_cudaMemcpyHostToHost; <==== not used in this crate

#[allow(unused)]
impl Device for CudaUniStream {
    type Err = CudaError;
    fn memcpy(
        &self,
        dst: Ptr, dstnum: usize,
        src: Ptr, srcnum: usize,
        len: usize,
    ) -> Result<(), Self::Err> {
        let error_n = match (dstnum, srcnum) {
            (0, 0) => 
                unsafe {libc::memcpy(dst, src, len); 0},
            (0, 1) => 
                unsafe {cudaMemcpyAsync(dst, src, len as u64, D2H, self.inner)},
            (1, 0) =>
                unsafe {cudaMemcpyAsync(dst, src, len as u64, H2D, self.inner)},
            (1, 1) =>
                unsafe {cudaMemcpyAsync(dst, src, len as u64, D2D, self.inner)},
            (_, _) =>
                panic!("Neither on device stream, nor on host memory!")
        };
        CudaError::from(error_n).wrap_val(())
    }
    fn del_buff(
        &mut self,
        lower: Ptr,
        upper: Ptr,
        devnum: usize,
    ) -> Result<(), Self::Err> {
        panic!("#TODO(Y-jiji)");
        Ok(())
    }
    fn memset(
        &self,
        dst: Ptr, dstnum: usize,
        len: usize, _0: bool,
    ) -> Result<(), Self::Err> {
        panic!("#TODO(Y-jiji)");
        Ok(())
    }
    fn new_buff(
        &mut self,
        len: usize,
    ) -> Result<(Ptr, Ptr, usize), Self::Err> {
        panic!("#TODO(Y-jiji)");
        Ok((
            self.membase, 
            unsafe{self.membase.add(self.memsize)}, 
            1 as usize
        ))    
    }
    fn new_err(
        msg: &str
    ) -> Result<(), Self::Err> {
        panic!("#TODO(Y-jiji)");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        let x = CudaUniStream::new(100).unwrap();
    }
}