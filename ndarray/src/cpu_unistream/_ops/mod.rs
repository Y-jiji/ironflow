mod _add;
pub use _add::*;
mod _sub;
pub use _sub::*;
mod _div;
pub use _div::*;
mod _mul;
pub use _mul::*;

use super::_arr::*;
use crate::*;

#[macro_use]
mod helpers {
    #[macro_export]
    macro_rules! aspawn {
        ($x: expr) => {
            crate::async_rt::task::spawn(async move {
                $x
            })
        };
    }
}