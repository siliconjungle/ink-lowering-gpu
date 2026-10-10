module LiteralGraph;
pub fn mapped_filter(xs:List<u32>,k:u32)->List<u32>{return xs.map(fn(x)=>x+k).filter(fn(x)=>x>k).map_indexed(fn(i)=>fn(x)=>x+i);}
pub fn ordered(xs:List<u32>,seed:u32,k:u32)->u32{return foldr(xs.filter(fn(x)=>x<seed).map(fn(x)=>x+k),seed,fn(x)=>fn(a)=>choose(a<seed,x+a*k,x-a));}
pub fn paired(xs:List<u32>,ys:List<u32>,k:u32)->List<u32>{return xs.map(fn(x)=>x+k).zip(ys.filter(fn(y)=>y>k),fn(x)=>fn(y)=>x-y).map(fn(x)=>x*k);}
pub fn joined(xs:List<i32>,ys:List<i32>)->List<i32>{return xs.scan().concat(ys.map(fn(y)=>y*3));}
pub fn ordered_float(xs:List<f32>,seed:f32)->f32{return foldr(xs.map(fn(x)=>x+seed),seed,fn(x)=>fn(a)=>x-a);}
pub fn sum_float(xs:List<f32>)->f32{return sum(xs.map(fn(x)=>x));}
pub fn count_filtered(xs:List<u32>,k:u32)->u64{return count(xs.filter(fn(x)=>x>k));}
pub fn helper_pipeline(xs:List<u32>,k:u32)->List<u32>{return selected(xs,k).map(fn(x)=>x+k);}
fn selected(xs:List<u32>,k:u32)->List<u32>{return xs.filter(fn(x)=>x<k);}
pub fn safe_read(xs:List<u32>,k:u32)->u32{return xs.map(fn(x)=>x+k).at_or(k,k+1);}
pub fn scalar_scope(x:u32)->u32{return choose(true,quot_or(x+7,3,99),x)+x;}
