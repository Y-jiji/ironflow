/* ---------------------------------------------------------------------------------------------------- *
 * Device : public behaviours of devices (or streams with allocated space)
 *      Ptr    : ptr type on this device
 *      malloc : get some space on this device
 *      memcpy : copy from another device, usually host
 *      memset : fill all-1 or all-0 bits into pointed memory
 *      run    : start this device, or start a manager/stream of this device
 *      submit : submit a job to this device
 * ---------------------------------------------------------------------------------------------------- */
pub trait Device {
    type Ptr<T>;
    type MemErr;
    /* ---------------------------------------------------------------------------------------------------- *
     * malloc(usize) -> Result<Ptr<T>, MemErr>
     * input  : 
     * return : 
     * ---------------------------------------------------------------------------------------------------- */
    fn malloc<MType>(self, size_in_bytes: usize) -> Result<Ptr<MType>, MemErr>;
    /* ---------------------------------------------------------------------------------------------------- *
     * memcpy(Self, &MType, Ptr) -> Result<(), MemErr>
     * input  : 
     * return : 
     * ---------------------------------------------------------------------------------------------------- */
    fn memcpy<MType>(self, src: &MType, dst: Ptr) -> Result<(), MemErr>;
    fn submit();
    /* ---------------------------------------------------------------------------------------------------- *
     * run(Self) -> JoinHandle
     * input  : Self: a struct implements 
     * return : return the join handle of spawned device
     * ---------------------------------------------------------------------------------------------------- */
    fn run(self);
}