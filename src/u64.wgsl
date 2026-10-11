// Exact wrapping u64 over two storage lanes. No floating point intermediates.
struct InkU64 {lo:u32,hi:u32,}
struct InkDiv64 {quotient:InkU64,remainder:InkU64,}
fn ink_u64_eq(a:InkU64,b:InkU64)->bool{return a.lo==b.lo&&a.hi==b.hi;}
fn ink_u64_lt(a:InkU64,b:InkU64)->bool{return a.hi<b.hi||(a.hi==b.hi&&a.lo<b.lo);}
fn ink_u64_add(a:InkU64,b:InkU64)->InkU64{let lo=a.lo+b.lo;return InkU64(lo,a.hi+b.hi+select(0u,1u,lo<a.lo));}
fn ink_u64_sub(a:InkU64,b:InkU64)->InkU64{return InkU64(a.lo-b.lo,a.hi-b.hi-select(0u,1u,a.lo<b.lo));}
fn ink_u64_mul(a:InkU64,b:InkU64)->InkU64{let al=a.lo&65535u;let ah=a.lo>>16u;let bl=b.lo&65535u;let bh=b.lo>>16u;let p=al*bl;let x=ah*bl;let y=al*bh;let mid=(p>>16u)+(x&65535u)+(y&65535u);let high=ah*bh+(x>>16u)+(y>>16u)+(mid>>16u);return InkU64((mid<<16u)|(p&65535u),high+a.lo*b.hi+a.hi*b.lo);}
fn ink_u64_div(a:InkU64,b:InkU64)->InkDiv64{var q=InkU64(0u,0u);var r=InkU64(0u,0u);var cursor=64u;loop{if(cursor==0u){break;}cursor--;var bit=0u;if(cursor>=32u){bit=(a.hi>>(cursor-32u))&1u;}else{bit=(a.lo>>cursor)&1u;}let overflow=r.hi>>31u;r=InkU64((r.lo<<1u)|bit,(r.hi<<1u)|(r.lo>>31u));if(overflow!=0u||!ink_u64_lt(r,b)){r=ink_u64_sub(r,b);if(cursor>=32u){q.hi|=1u<<(cursor-32u);}else{q.lo|=1u<<cursor;}}}return InkDiv64(q,r);}
