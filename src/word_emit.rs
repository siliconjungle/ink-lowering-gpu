//! Literal fixed-width word operations; no rewrite or profitability decisions.
use ink_core::{core::Type, LangResult};
pub fn expression(name: &str, args: &[String], ty: &Type) -> LangResult<String> {
    if !matches!(ty, Type::U32 | Type::U64 | Type::I32) || args.len() != if name == "word_not" { 1 } else { 2 } {
        return Err("GPU word operation type or arity".into());
    }
    let a=&args[0];
    if name=="word_not" {
        return Ok(if *ty==Type::U64 {format!("InkU64(~({a}).lo,~({a}).hi)")}else{format!("(~({a}))")});
    }
    let b=&args[1];
    if let Some(op)=match name {"word_and"=>Some("&"),"word_or"=>Some("|"),"word_xor"=>Some("^"),_=>None} {
        return Ok(if *ty==Type::U64 {format!("InkU64(({a}).lo{op}({b}).lo,({a}).hi{op}({b}).hi)")}
            else {format!("(({a}){op}({b}))")});
    }
    if *ty==Type::U64 {
        let suffix=match name {"wrapping_shl"=>"shl","wrapping_shr"=>"shr","rotate_left"=>"rol","rotate_right"=>"ror",_=>return Err("unsupported GPU word primitive".into())};
        return Ok(format!("ink_u64_{suffix}({a},{b})"));
    }
    let count=format!("(({b})&31u)");
    match name {
        "wrapping_shl"=>Ok(format!("(({a})<<{count})")),
        "wrapping_shr"=>Ok(format!("(({a})>>{count})")),
        "rotate_left"|"rotate_right"=>{
            let bits=if *ty==Type::I32 {format!("bitcast<u32>({a})")}else{a.clone()};
            let (first,second)=if name=="rotate_left"{("<<",">>")}else{(">>","<<")};
            let out=format!("(({bits}){first}{count})|(({bits}){second}((0u-{count})&31u))");
            Ok(if *ty==Type::I32 {format!("bitcast<i32>({out})")}else{format!("({out})")})
        }
        _=>Err("unsupported GPU word primitive".into()),
    }
}
