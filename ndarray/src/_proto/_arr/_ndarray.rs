use super::*;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

macro_rules! ArcMutexNew {($x: expr) => { Arc::new(Mutex::new($x)) };}

/**
 * Tips: 
 * Async runtime guarntees that if you don't call .await or spawn when you lock a Mutex(or a Rwlock).
 * The deadlock will not happen. 
 */

#[derive(Debug, Clone)]
pub struct NDArray<ValT, DevT>
where ValT: Clone + Sized, 
      DevT: Device + Clone,
{
    pub(crate) buff   : Arc<Mutex<Buffer<DevT>>>,
    pub(crate) size   : Vec<usize>,
    pub(crate) device : Arc<Mutex<DevT>>,
    pub(crate) __type : PhantomData<ValT>,
}

impl<ValT, DevT> NDArray<ValT, DevT>
where ValT: Clone + Sized,
      DevT: Device + Clone, 
{
    pub
    fn deep_clone(
        &self,
    ) -> Result<Self, DevT::Err> {
        Ok(Self {
            buff: ArcMutexNew!(self.buff.lock().unwrap().deep_clone()?),
            size: self.size.clone(),
            device: Arc::clone(&self.device),
            __type: self.__type,
        })
    }
    pub(crate)
    fn new_uninit(
        device: Arc<Mutex<DevT>>,
        size: Vec<usize>
    ) -> Result<Self, DevT::Err> {
        let len = size.clone()
            .into_iter()
            .reduce(|a, b| a*b)
            .unwrap_or(0)
            * std::mem::size_of::<ValT>();
        Ok(Self {
            buff: ArcMutexNew!(Buffer::new(device.clone(), len)?),
            size, device,
            __type: PhantomData,
        })
    }
    pub
    fn zero(
        device: Arc<Mutex<DevT>>,
        size: Vec<usize>
    ) -> Result<Self, DevT::Err> {
        let new = Self::new_uninit(device.clone(), size)?;
        {let buff = new.buff.lock().unwrap();
        (&mut *device.lock().unwrap()).memset(
            buff.lower, buff.devnum, 
            buff.upper as usize - buff.lower as usize, true)?;} // buff droped here
        Ok(new)
    }
}