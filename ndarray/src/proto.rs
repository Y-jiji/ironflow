/* ---------------------------------------------------------------------------------------------------- *
 * public behaviours of different implementations of heterogenous devices
 * ---------------------------------------------------------------------------------------------------- */
use std::sync::{Arc, Mutex, PoisonError, MutexGuard};
/* ---------------------------------------------------------------------------------------------------- *
 * DevPtr : public behaviours of device pointers
 * ---------------------------------------------------------------------------------------------------- */
 pub trait DevPtr {
    /* ---------------------------------------------------------------------------------------------------- *
     * nullptr() -> Self
     * return : a null pointer
     * ---------------------------------------------------------------------------------------------------- */
    fn nullptr() -> Self;
}

/* ---------------------------------------------------------------------------------------------------- *
 * MemPtr : public behaviours of host memory pointers
 * ---------------------------------------------------------------------------------------------------- */
 pub trait MemPtr where 
     Self : Sized {
    /* ---------------------------------------------------------------------------------------------------- *
     * nullptr() -> Self
     * return : a 
     * ---------------------------------------------------------------------------------------------------- */
    fn from_vec<ValT>(v: &Vec<ValT>) -> Self;
}

/* ---------------------------------------------------------------------------------------------------- *
 * Device : public behaviours of devices (or streams with allocated space)
 * However, note that specific computations are implemented in ndarray, remember trait Device is an int
 * -erface for memory management, and does nothing about unifying computational things. 
 * ---------------------------------------------------------------------------------------------------- */
pub trait Device where
    Self         : Sized, 
    Self::DevPtr : DevPtr,
    Self::MemPtr : MemPtr {

    type DevPtr; /* pointer to device memory */
    type MemPtr; /* pointer to (host's) main memory */
    type MemErr; /* memory error */
    /* ---------------------------------------------------------------------------------------------------- *
     * new() -> Result<Self, Self::MemErr>
     * input  : initialize total memory on the physical Device
     * return : Ok(DevPtr) if allocation succeed, Err(MemErr) if allocation failed
     * ---------------------------------------------------------------------------------------------------- */
    fn new(capacity: usize) -> Result<Self, Self::MemErr>;
    /* ---------------------------------------------------------------------------------------------------- *
     * malloc(usize) -> Result<Ptr<T>, MemErr>
     * input  : usize, the size of targeted memory
     * return : Ok(DevPtr) if allocation succeed, Err(MemErr) if failed
     * ---------------------------------------------------------------------------------------------------- */
    fn malloc(self, 
        size_in_bytes: usize
    ) -> Result<Self::DevPtr, Self::MemErr>;
    /* ---------------------------------------------------------------------------------------------------- *
     * memcpy_as_dst(Self, MemPtr, DevPtr) -> Result<(), MemErr>
     * behaviour : copy bytes from host to device
     * return : Ok(()) if memcpy succeed, Err(MemErr) if failed
     * ---------------------------------------------------------------------------------------------------- */
    fn memcpy_as_dst(self, 
        src: Self::MemPtr, dst: Self::DevPtr, 
        size_in_bytes: usize
    ) -> Result<(), Self::MemErr>;
    /* ---------------------------------------------------------------------------------------------------- *
     * memcpy_as_src(Self, DevPtr, MemPtr) -> Result<(), MemErr>
     * behaviour : copy bytes from device to host
     * return : Ok(()) if memcpy succeed, Err(MemErr) if failed
     * ---------------------------------------------------------------------------------------------------- */
    fn memcpy_as_src(self, 
        src: Self::DevPtr, dst: Self::MemPtr, 
        size_in_bytes: usize
    ) -> Result<(), Self::MemErr>;
    /* ---------------------------------------------------------------------------------------------------- *
     * memcpy_inside(Self, DevPtr, DevPtr) -> Result<(), MemErr>
     * behaviour : copy bytes from device to device
     * return : Ok(()) if memcpy succeed, Err(MemErr) if failed
     * ---------------------------------------------------------------------------------------------------- */
    fn memcpy_inside(self, 
        src: Self::DevPtr, dst: Self::DevPtr,
        size_in_bytes: usize,
    ) -> Result<(), Self::MemErr>;
}

#[derive(Clone, Debug)]
pub struct NDArray<ValT, DevT> where
    ValT: Clone,
    DevT: Device {
    /*-------------------------------- things about host memory --------------------------------*/
    pub size : Vec<usize>,             // size, namely shape
    pub data : Arc<Mutex<Vec<ValT>>>,  // flattened data of this tensor (may outlive this struct)
    pub memptr : DevT::MemPtr,         // binded memory pointer to self.data's inner bytes
    /*------------------------------------------------------------------------------------------*/

    /*------------------------------- things about device memory -------------------------------*/
    pub device : Arc<Mutex<DevT>>,     // binded device/computation stream, shared
    pub devptr : DevT::DevPtr,         // binded device pointer, uniquely owned by one ndarray
    /*------------------------------------------------------------------------------------------*/
}

impl<ValT, DevT> NDArray<ValT, DevT> where
    ValT: Clone,
    DevT: Device {
    pub fn deep_clone(&self) -> Result<Self, PoisonError<MutexGuard<Vec<ValT>>>> {match self.data.lock() {
        Ok(lock) => Ok(Self { 
            size: self.size.clone(), 
            data: Arc::new(Mutex::new((*lock).clone())), /* the most important thing happens here */
            device: self.device.clone(),
            devptr: DevT::DevPtr::nullptr(),
            memptr: DevT::MemPtr::from_vec(&*lock),
        }),
        Err(e) => Err(e)
    }}
}