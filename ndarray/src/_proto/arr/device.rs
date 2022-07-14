/** 
 * Device: public behaviours of a device
 * This is supposed to be implemented 
 */
pub(crate) trait Device {
    type Error;
    type Ptr;
    fn memcpy_as_dst(dst: Self::Ptr, src: Self::Ptr, len: isize) -> Result<(), Self::Error>;
    fn memcpy_as_src(dst: Self::Ptr, src: Self::Ptr, len: isize) -> Result<(), Self::Error>;
    fn memcpy_in_dev(dst: Self::Ptr, src: Self::Ptr, len: isize) -> Result<(), Self::Error>;
}

/** 
 * DevMem: device memory management
 * This is supposed to be implemented without calling malloc. 
 */
pub(crate) trait DevMem: Device {
    fn malloc(len: isize) -> Result<Self::Ptr, Self::Error>;
    fn free(ptr: Self::Ptr) -> Result<(), Self::Error>;
}

/** 
 * DevBuf: device buffer that works like an array
 * Unlike Vec type, it will keep its length, before it is consumed by something. 
 */
pub(crate) struct DevBuf<ValT, DevT: Device> {
    pub(crate) ptr: *const std::ffi::c_void,  /* base pointer */
    pub(crate) len: isize,                    /* memory length */
    pub(crate) dev: DevT,                     /* device type */
    pub(crate) num: isize,                    /* index of physical device */
    pub(crate) val_t: std::marker::PhantomData<ValT>     /* keep track of val_t */
}

/** 
 * Data: 
 * manage data on host as Vec, 
 * manage data on device as DevBuf
 */
pub(crate) enum Data<ValT, DevT: Device> {
    Host(Vec<ValT>),
    Device(DevBuf<ValT, DevT>)
}

