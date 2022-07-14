/** Device: public behaviours of a device */
pub(crate) trait Device {
    type Error;
    type Ptr;
    fn memcpy_as_dst(dst: Self::Ptr, src: Self::Ptr, len: isize) -> Result<(), Self::Error>;
    fn memcpy_as_src(dst: Self::Ptr, src: Self::Ptr, len: isize) -> Result<(), Self::Error>;
    fn memcpy_in_dev(dst: Self::Ptr, src: Self::Ptr, len: isize) -> Result<(), Self::Error>;
}

/** DevMem: device memory management */
pub(crate) trait DevMem: Device {
    fn malloc(len: isize) -> Result<Self::Ptr, Self::Error>;
    fn free(ptr: Self::Ptr) -> Result<(), Self::Error>;
}

/** DevNum: Marker enum for DevBuf */
pub(crate) enum DevNum {
    Host,
    Device(usize),
}

/** DevBuf: device buffer that works like an array */
pub(crate) struct DevBuf<DevT: Device> {
    ptr: *const std::ffi::c_void,
    len: isize,
    dev: DevT,
    num: DevNum
}