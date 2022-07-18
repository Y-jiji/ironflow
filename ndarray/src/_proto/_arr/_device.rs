use parking_lot::{RwLock, RwLockReadGuard};
use std::sync::Arc;
use std::fmt::Debug;

pub 
type Ptr = *mut std::ffi::c_void;

pub(crate)
trait Device 
where Self      : Sized+Debug,
      Self::Err : Debug 
{
/* -------------------------------------------------------------------------------------------------------- */
/*                                                  types                                                   */
/* -------------------------------------------------------------------------------------------------------- */
    /// Device specific error type
    type Err;

/* -------------------------------------------------------------------------------------------------------- */
/*                                      deal with raw memory on device                                      */
/* -------------------------------------------------------------------------------------------------------- */
    /// Copy a piece of memory (Mind that dst and src segements should not *overlap*). 
    fn memcpy(
        &self,
        dstptr: Ptr, dstdev: usize,
        srcptr: Ptr, srcdev: usize,
        memlen: usize
    ) -> Result<(), Self::Err>;

    /// Allocate a piece of memory
    /// 
    /// *Development Tips:* 
    ///  
    /// When swapping out a buffer, first acquire the buffer's write lock, before calling memdel. 
    /// This trick ensures bufswp's intergrity. 
    fn memnew(
        &self,
        memlen: usize,
        dstdev: usize,
    ) -> Result<MemSeg, Self::Err>;

    /// Deallocate a piece of memory. Unregister it from memory pool. 
    fn memdel(
        &self,
        memseg: MemSeg
    ) -> Result<(), Self::Err>;

/* -------------------------------------------------------------------------------------------------------- */
/*                                          sealed buffer actions                                           */
/* -------------------------------------------------------------------------------------------------------- */
    /// Acquire a write lock on buffer, put it to the device indicated by devnr. 
    /// Then, downgrade the lock to a reader lock and return it. 
    fn bufswp<'a>(
        &self,
        memseg : &'a Arc<RwLock<MemSeg>>,
        dstdev : usize
    ) -> Result<RwLockReadGuard<'a, MemSeg>, Self::Err>;

    /// Get a new piece of buffer. 
    fn bufnew(
        &self,
        len : usize
    ) -> Result<DevBuf<Self>, Self::Err>;

    /// Drop an old piece of buffer. 
    fn bufdel(
        &self,
        lower : Ptr,
    ) -> Result<(), Self::Err>;

/* -------------------------------------------------------------------------------------------------------- */
}

#[derive(Debug)]
pub(crate)
struct MemSeg {
    pub(crate) lower: Ptr  ,         // lower bound
    pub(crate) upper: Ptr  ,         // upper bound
    pub(crate) devnr: usize,         // which physical device
}

impl MemSeg {
    pub(crate)
    fn len(&self) -> usize {
        self.upper as usize -
        self.lower as usize
    }
}

#[derive(Debug)]
pub(crate) 
struct DevBuf<DevT>
where DevT: Device
{
    memseg : Arc<RwLock<MemSeg>>,        // (lower, upper, devnr)
    device : DevT,                       // |dev|ice |m|emory |m|anager, contains some inner mutable state
}

unsafe impl<DevT> Send for DevBuf<DevT>
where DevT: Device {}