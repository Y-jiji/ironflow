/* ---------------------------------------------------------------------------------------------------- *
 * single_thread_naive: a really simple implementation that does anything useful.
 * However, it is good enough to debug the overlying tensor and nn crate, and also have a standard fold
 * -er structure, which will be useful when implement something on more complicated heterogeneous devic
 * -es. A not-suprising fact is that if you lauch a lot of operate and task::block_on them later, execu
 * -tion time will be less than simply compute with a single thread. 
 * ---------------------------------------------------------------------------------------------------- */

pub mod device;
pub mod add;
pub mod sub;
pub mod from_val;
pub mod from_dev;
pub mod ord;