pub use ink_core::*;
pub mod compute;
pub mod gpu;

pub const SCALAR_DEVICE_RS: &str = include_str!("../devices/scalar.rs");
pub const COMPUTE_DEVICE_RS: &str = include_str!("../devices/compute.rs");
pub const SCALAR_DEVICE_JS: &str = include_str!("../devices/scalar.mjs");
pub const COMPUTE_DEVICE_JS: &str = include_str!("../devices/compute.mjs");
