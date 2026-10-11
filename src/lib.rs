pub use ink_core::*;
pub mod compute;
pub mod gpu;

/// General recursive state definitions execute in the literal host lane.
/// Device eligibility is separate from language/representation coverage.
pub fn lower_definition(checked: &ink_core::definition::CheckedDefinition) -> LangResult<ink_lowering_c::definition::Emission> {
    ink_lowering_c::definition::lower(checked)
}

pub const SCALAR_DEVICE_RS: &str = include_str!("../devices/scalar.rs");
pub const COMPUTE_DEVICE_RS: &str = include_str!("../devices/compute.rs");
pub const SCALAR_DEVICE_JS: &str = include_str!("../devices/scalar.mjs");
pub const COMPUTE_DEVICE_JS: &str = include_str!("../devices/compute.mjs");

/// Standalone recursive representations execute in the literal C/native host lane.
pub fn lower_representation(checked: &ink_core::representation::CheckedRepresentation) -> LangResult<ink_lowering_c::representation_program::Emission> {
    ink_lowering_c::representation_program::lower(checked)
}

/// Browser representations use the literal JavaScript host for recursive state.
pub fn lower_representation_web(checked: &ink_core::representation::CheckedRepresentation) -> LangResult<String> {
    ink_lowering_js::representation_program::lower(checked)
}
