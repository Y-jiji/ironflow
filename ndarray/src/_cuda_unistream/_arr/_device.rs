use crate::*;
use std::{
    fmt::*,
    error::Error,
    result::Result,
    sync::Arc, pin::Pin
};

use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

include!(concat!(env!("OUT_DIR"), "/nvidia_toolkit/cuda.rs"));

impl Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl Error for CudaError {}

type CudaResult<T> = Result<T, CudaError>;

impl CudaError {
    pub(crate) fn wrap_val<T>(self, val: T) -> CudaResult<T> {
        match self {
            Self::CUDA_SUCCESS => Ok(val),
            _else => Err(_else),
        }
    }
}

struct CudaLaunchLayout {
    grid: (usize, usize, usize),
    block: (usize, usize, usize)
}

struct CudaFn {
    inner: *const CUfunc_st
}

#[derive(Debug)]
struct CudaModule {
    inner: *mut CUmod_st,
}

impl CudaModule {
    fn new(data: &[&str]) -> CudaResult<Self> {
        let mut inner = std::ptr::null_mut();
        for ptx in data {
            unsafe { cuModuleLoadData(
                &mut inner as *mut *mut CUmod_st,
                (*ptx).as_ptr() as Ptr )
            }.wrap_val(())?
        }
        Ok(Self { inner })
    }
    fn get_fn(&self, name: &str) -> CudaResult<CudaFn> {
        let mut inner = std::ptr::null_mut();
        println!("{name:?}");
        unsafe { cuModuleGetFunction(
            &mut inner, self.inner, name.as_ptr() as *const i8) }
        .wrap_val(CudaFn { inner })
    }
}

impl Drop for CudaModule {
    fn drop(self: &mut CudaModule) {
        println!("drop {self:?}");
        unsafe { cuModuleUnload(self.inner) }
        .wrap_val(()).unwrap();
    }
}

unsafe extern "C" fn callback_wrapper<T>(callback: Ptr)
where
    T: FnOnce() + Send,
{
    // Stop panics from unwinding across the FFI
    let _ = std::panic::catch_unwind(|| {
        let callback: Box<T> = Box::from_raw(callback as *mut T);
        callback();
    });
}

impl CudaStream {
    const H2D: i32 = cudaMemcpyKind_cudaMemcpyHostToDevice;
    const H2H: i32 = cudaMemcpyKind_cudaMemcpyHostToHost;
    const D2H: i32 = cudaMemcpyKind_cudaMemcpyDeviceToHost;
    const D2D: i32 = cudaMemcpyKind_cudaMemcpyDeviceToDevice;
    pub fn new(memsize: usize, kernelfn: &[&str])
    -> CudaResult<Self> {
        let inner = CudaStream::init_stream()?;
        let membase = CudaStream::init_memory(inner, memsize)?;
        let module = CudaModule::new(kernelfn)?;
        Ok(Self { inner: Arc::new(Mutex::new(inner)), membase, memsize, module })
    }
    fn init_stream()
    -> CudaResult<*mut CUstream_st> {
        let mut strmptr : *mut CUstream_st = std::ptr::null_mut();
        let error_n = unsafe { cudaStreamCreate(&mut strmptr) };
        CudaError::from(error_n).wrap_val(strmptr)
    }
    fn init_memory(strmptr : *mut CUstream_st, memsize: usize)
    -> CudaResult<Ptr> {
        let mut devptr = std::ptr::null_mut();
        let error_n = unsafe { cudaMallocAsync(
            &mut devptr, memsize as u64, strmptr) };
        unsafe { cuStreamSynchronize(strmptr) }.wrap_val(())?;
        CudaError::from(error_n).wrap_val(devptr)
    }
    fn host_malloc(memsize: usize) {

    }
    fn add_callback<T> (
        &self, 
        callback: Box<T>
    ) -> CudaResult<()>
    where T: FnOnce() + Send {
        unsafe {cuLaunchHostFunc(
            *self.inner.lock(),
            Some(callback_wrapper::<T>),
            Box::into_raw(callback) as Ptr,
        )}.wrap_val(())
    }
    pub fn launch(&self, name: &str, layout: CudaLaunchLayout, param: &[Ptr])
    -> CudaResult<()> {
        let f: CudaFn = self.module.get_fn(name)?;
        println!("function name: {name}");
        let CudaLaunchLayout {
            grid: (gridx, gridy, gridz),
            block: (blockx, blocky, blockz)
        } = layout;
        let error_n = unsafe {cuLaunchKernel(
            f.inner as *mut CUfunc_st,
            gridx as u32, gridy as u32, gridz as u32,
            blockx as u32, blocky as u32, blockz as u32,
            0, *self.inner.lock(),
            param.as_ptr() as *mut Ptr,
            std::ptr::null_mut() as *mut Ptr
        )};
        CudaError::from(error_n).wrap_val(())
    }
    pub fn sync(&self) -> CudaResult<()> {
        unsafe { cuStreamSynchronize(*self.inner.lock()) }.wrap_val(())
    }
}


#[derive(Debug)]
struct CudaStream {
    inner: Arc<Mutex<*mut CUstream_st>>,
    membase: Ptr,
    memsize: usize,
    module: CudaModule
}

impl Device for CudaStream
{
/* -------------------------------------------------------------------------------------------------------- */

    // Device specific error type
    type Err = CudaError;

/* -------------------------------------------------------------------------------------------------------- */

    // Copy a piece of memory (dst and src segements should not overlap)
    fn memcpy(
        &self, 
        dst: Ptr, dstdev: usize, 
        src: Ptr, srcdev: usize, 
        memlen: usize
    ) -> CudaResult<()> {
        let count = memlen as u64;
        // determine the type of this copy action
        let cpyty = match (dstdev, srcdev) {
            (0, 0) => Self::H2H ,
            (0, 1) => Self::H2D ,
            (1, 0) => Self::D2H ,
            (1, 1) => Self::D2D ,
            (_, _) => panic!("No such memcpy type for unistream")
        };
        // launch the memcpy on current stream
        let error_n = unsafe { cudaMemcpyAsync(
            dst, src, count, cpyty, *self.inner.lock()) };
        // wrap the error number to result about whether its launch succeeded 
        CudaError::from(error_n).wrap_val(())
    }

    // Allocate a piece of memory of device, other buffers might be swapped out
    fn memnew(
        &self,
        memlen: usize,
        dstdev: usize,
    ) -> Result<MemSeg, Self::Err> {
        match dstdev {
            0 => {
                let lower = seal_libc::malloc(memlen);
                let upper = unsafe{lower.add(memlen)};
                return Ok(MemSeg {lower, upper, devnr: 0})
            },
            1 => {
            },
            _ => panic!("No such physical device on unistream")
        }
        panic!("#TODO(@Y-jiji)");
    }

    // Delete a piece of memory, lock must be kept to ensure unique access
    fn memdel(
        &self,
        memseg: MemSeg
    ) -> Result<(), Self::Err> {
        panic!("#TODO(@Y-jiji)");
    }

/* -------------------------------------------------------------------------------------------------------- */

    // Aquire a write lock on buffer, put it to the device indicated by devnr
    // Then, downgrade the lock to a reader lock and return it
    fn bufswp<'a>(
        &self,
        memseg : &'a Arc<RwLock<MemSeg>>,
        dstdev : usize
    ) -> Result<RwLockReadGuard<'a, MemSeg>, Self::Err> {
        // See if the memseg is already on the destination device
        let memseg_read = memseg.read_recursive();
        if (*memseg_read).devnr == dstdev { return Ok(memseg_read); }
        drop(memseg_read);
        // Use write lock to modify the memory segements
        let mut memseg_write = memseg.write();
        // Check twice, because there might be multiple references to one buffer instance here. 
        if (*memseg_write).devnr == dstdev { return Ok(RwLockWriteGuard::downgrade(memseg_write)); }
        // Get a piece of new memory on destination device
        let memlen = memseg_write.len();
        let newseg = self.memnew(memlen, dstdev)?;
        // Now that memory is logically assigned to this buffer, we copy memory and wait for completion.
        self.memcpy(
            newseg.lower, newseg.devnr,
            memseg_write.lower, memseg_write.devnr, 
            memlen
        )?;
        // Add a check point before completion.
        // Then spin and wait for it. (this is not the best idea, but the safest)
        let mut ckpt = Pin::from(Box::new(false));
        self.add_callback(Box::new(|| {*ckpt.as_mut() = true;}))?;
        while !*ckpt.as_ref() {};
        // Change the state of memory segment
        *memseg_write = newseg;
        // Return read guard downgraded from write guard
        Ok(RwLockWriteGuard::downgrade(memseg_write))
    }

    // Get a new piece of buffer
    fn bufnew(
        &self,
        len : usize
    ) -> Result<DevBuf<Self>, Self::Err> {
        panic!("TODO(@Y-jiji)");
    }

    // Drop an old piece of buffer
    fn bufdel(
        &self,
        lower : Ptr,
    ) -> Result<(), Self::Err> {
        panic!("TODO(@Y-jiji)");
    }

/* -------------------------------------------------------------------------------------------------------- */
}

impl Drop for CudaStream {
    fn drop(self: &mut CudaStream) {
        println!("drop {self:?}");
        unsafe {cuMemFreeAsync(
            self.membase as CUdeviceptr,
            *self.inner.lock())}
            .wrap_val(()).unwrap();
        self.sync().unwrap();
        let error_n = unsafe {cudaStreamDestroy(*self.inner.lock())};
        CudaError::from(error_n).wrap_val(()).unwrap();
    }
}
