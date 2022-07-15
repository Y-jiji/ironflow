use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use std::marker::{Send, Sync};

pub(crate) 
type Ptr = *mut std::ffi::c_void;

/// ----------------------------------------------------------------------------------- ///
///     Tips: 
///     Device is always wrapped in Arc Mutex, thus playing with evil things will not
///     do any harm in async code, since the coroutine aquired the lock will not be
///     suspended, making the device operations always atomic
/// ----------------------------------------------------------------------------------- ///
pub trait Device 
where Self: Send + Clone + Debug, 
      Self::Err: Debug + Clone, 
{
/// ------------------------------------------------------------------------------------ ///
///     Device specific error
/// ------------------------------------------------------------------------------------ ///
    type Err;
/// ------------------------------------------------------------------------------------ ///
///     unified call of memcpy
/// ------------------------------------------------------------------------------------ ///
    fn memcpy(
        &self,
        dst: Ptr, dstnum: usize,
        src: Ptr, srcnum: usize,
        len: usize,
    ) -> Result<(), Self::Err>;
/// ------------------------------------------------------------------------------------ ///
///     unified call of memset
///     set all 1 bits or all 1bits
/// ------------------------------------------------------------------------------------ ///
    fn memset(
        &self,
        dst: Ptr, dstnum: usize,
        len: usize, _0: bool,
    ) -> Result<(), Self::Err>;
/// ------------------------------------------------------------------------------------ ///
///     change internal states of this device to occupy this piece of memory
///     get a piece of memory as (lower, upper, devnum)
/// ------------------------------------------------------------------------------------ ///
    fn new_buff(
        &mut self,
        len: usize,
    ) -> Result<(Ptr, Ptr, usize), Self::Err>;
/// ------------------------------------------------------------------------------------ ///
///     create a new Err(Self::Err)
/// ------------------------------------------------------------------------------------ ///
    fn new_err(
        msg: &str
    ) -> Result<(), Self::Err>;
/// ------------------------------------------------------------------------------------ ///
///     change internal states of this device to free this piece of memory
/// ------------------------------------------------------------------------------------ ///
    fn del_buff(
        &mut self,
        lower: Ptr,
        upper: Ptr,
        devnum: usize,
    ) -> Result<(), Self::Err>;
}

#[derive(Debug, Clone)]
pub struct Buffer<DevT>
where DevT: Device + Clone 
{
    pub(crate) lower  : Ptr,
    pub(crate) upper  : Ptr,
    pub(crate) device : Arc<Mutex<DevT>>,
    pub(crate) devnum : usize,
}

unsafe impl<DevT> Send
for Buffer<DevT>
where DevT: Device + Clone {}

unsafe impl<DevT> Sync
for Buffer<DevT>
where DevT: Device + Clone {}

impl<DevT: Device + Clone> Buffer<DevT> {
    pub(crate) 
    fn new(
        device: Arc<Mutex<DevT>>, 
        len: usize
    ) -> Result<Self, DevT::Err> {
        let locked_device = &mut *device.lock().unwrap();
        let (lower, upper, devnum) = locked_device.new_buff(len)?;
        Ok(Self { lower, upper, device: device.clone(), devnum })
    }
    pub(crate) 
    fn deep_clone(
        &self
    ) -> Result<Self, DevT::Err> {
        let locked_device = &mut *self.device.lock().unwrap();
        let len = self.upper as usize - self.lower as usize;
        let (lower, upper, devnum) = locked_device.new_buff(len)?;
        locked_device.memcpy(
            lower, devnum,
            self.lower, self.devnum, len
        )?;
        Ok(Self { lower, upper, device: self.device.clone(), devnum })
    }
}

impl<DevT: Device + Clone> Drop
for Buffer<DevT> {
    fn drop(&mut self) {
        self.device
            .lock().unwrap()
            .del_buff(self.lower, self.upper, self.devnum)
            .unwrap();
    }
}