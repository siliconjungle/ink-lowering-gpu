use ink_core::{core::Type, LangResult};
pub fn expression(n: &str, a: &[String], first: Option<&Type>) -> LangResult<String> {
    let x = &a[0];
    Ok(match n {
        "sqrt" => format!("ink_f32_sqrt({x})"),
        "f32_div" => format!("ink_f32_div({x},{})", a[1]),
        "floor" | "ceil" | "trunc" | "round_ties_even" => format!(
            "ink_f32_integral({x},{}u)",
            match n {
                "trunc" => 0,
                "floor" => 1,
                "ceil" => 2,
                _ => 3,
            }
        ),
        "is_nan" => format!("ink_f32_nan({x})"),
        "is_finite" => format!("(({x})&0x7fffffffu)<0x7f800000u"),
        "f32_bits" | "f32_from_bits" => x.clone(),
        "f32_from_u64" => format!("ink_f32_from_u64({x})"),
        "f32_from_u32" => format!("ink_f32_from_u64(InkU64({x},0u))"),
        "f32_from_i32" => format!("ink_f32_from_i32({x})"),
        "popcount" | "leading_zeros" | "trailing_zeros" => {
            let name = match n {
                "popcount" => "countOneBits",
                "leading_zeros" => "countLeadingZeros",
                _ => "countTrailingZeros",
            };
            if first == Some(&Type::U64) {
                match n{"popcount"=>format!("(countOneBits(({x}).lo)+countOneBits(({x}).hi))"),"leading_zeros"=>format!("select(countLeadingZeros(({x}).lo)+32u,countLeadingZeros(({x}).hi),({x}).hi!=0u)"),_=>format!("select(countTrailingZeros(({x}).hi)+32u,countTrailingZeros(({x}).lo),({x}).lo!=0u)")}
            } else {
                format!("{name}(u32({x}))")
            }
        }
        _ => return Err(format!("primitive {n} requires host value execution")),
    })
}
