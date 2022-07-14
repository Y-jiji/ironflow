use std::sync::{Arc, Mutex};

pub(crate) struct NDArray<ValT, DevT: super::device::Device> {
    pub(crate) data   : Arc<Mutex<super::device::Data<ValT, DevT>>>,    // flattened data
    pub(crate) size   : std::vec::Vec<usize>,                       // shape of this ndarray
    pub(crate) valt   : std::marker::PhantomData<ValT>, // marker for value type
    pub(crate) device : Arc<Mutex<DevT>>                // device information
}

unsafe impl<ValT, DevT> std::marker::Send 
for NDArray<ValT, DevT>
where DevT : super::device::Device {}

