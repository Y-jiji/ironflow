pub(crate) struct NDArray<ValT, DevT: super::device::Device> {
    buff   : super::device::DevBuf<DevT>,    /* flattened data        */
    size   : std::vec::Vec<usize>,           /* shape of this ndarray */
    valt   : std::marker::PhantomData<ValT>, /* marker for value type */
    device : DevT                            /* device information    */
}

