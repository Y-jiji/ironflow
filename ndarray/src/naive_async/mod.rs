/* ---------------------------------------------------------------------------------------------------- *
 * single_thread_naive: a really simple implementation that does anything useful.
 * However, it is good enough to debug the overlying tensor and nn crate, and also have a standard fold
 * -er structure, which will be useful when implement something on more complicated heterogeneous devic
 * -es. A not-suprising fact is that if you lauch a lot of operate and task::block_on them later, execu
 * -tion time will be less than simply compute with a single thread. 
 * ---------------------------------------------------------------------------------------------------- */

mod device;
mod add;
mod sub;
mod div;
mod mul;
mod from_val;
mod from_dev;
mod ord;

pub use device::*;
pub use add::*;
pub use sub::*;
pub use div::*;
pub use mul::*;
pub use from_val::*;
pub use from_dev::*;
pub use ord::*;
