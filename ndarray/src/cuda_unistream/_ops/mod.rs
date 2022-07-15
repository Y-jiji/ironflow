

#[macro_use]
mod helpers {
    #[macro_export]
    macro_rules! cujob {
        // full layout syntax
        ($stream: ident{$gx:expr, $gy:expr, $gz:expr; $bx:expr, $by:expr, $bz:expr}$fn_name: ident ($($x: expr),*)) => {
            $stream.launch(
                stringify!($fn_name),
                CudaLaunchLayout { grid:($gx, $gy, $gz), block: ($bx, $by, $bz) },
                &[$((&$x as *const _)  as *mut std::ffi::c_void,)*]
            )
        };
        // short handed layout syntax
        ($stream: ident{$bx:expr, $by:expr, $bz:expr}$fn_name: ident ($($x: expr),*)) => {
            $stream.launch(
                stringify!($fn_name),
                CudaLaunchLayout { grid:(1, 1, 1), block: ($bx, $by, $bz) },
                &[$((&$x as *const _)  as *mut std::ffi::c_void,)*]
            )
        };
    }
}