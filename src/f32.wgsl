// Binary32 evaluated with integer lanes: round-to-nearest, ties-to-even,
// gradual underflow, signed zero and canonical arithmetic NaN.
fn ink_f32_nan(x:u32)->bool{return (x&0x7fffffffu)>0x7f800000u;}
fn ink_f32_eq(a:u32,b:u32)->bool{return !ink_f32_nan(a)&&!ink_f32_nan(b)&&(a==b||((a|b)&0x7fffffffu)==0u);}
fn ink_f32_lt(a:u32,b:u32)->bool{
 if(ink_f32_nan(a)||ink_f32_nan(b)||ink_f32_eq(a,b)){return false;}
 if((a^b)&0x80000000u)!=0u{return (a&0x80000000u)!=0u;}
 if((a&0x80000000u)!=0u){return a>b;}return a<b;
}
fn ink_f32_jam(x:u32,n:u32)->u32{if(n==0u){return x;}if(n>=32u){return select(0u,1u,x!=0u);}return (x>>n)|select(0u,1u,(x&((1u<<n)-1u))!=0u);}
fn ink_f32_round(sign:u32,exponent:i32,extended:u32)->u32{
 var e=exponent;var s=extended;
 if(e<1i){s=ink_f32_jam(s,u32(1i-e));e=1i;}
 let tail=s&7u;var sig=s>>3u;
 if(tail>4u||(tail==4u&&(sig&1u)!=0u)){sig++;}
 if(sig>=0x1000000u){sig>>=1u;e++;}
 if(e>=255i){return sign|0x7f800000u;}
 if(sig==0u){return sign;}
 let raw=select(u32(e),0u,e==1i&&sig<0x800000u);
 return sign|(raw<<23u)|(sig&0x7fffffu);
}
fn ink_f32_add(a:u32,b:u32)->u32{
 if(ink_f32_nan(a)||ink_f32_nan(b)){return 0x7fc00000u;}
 var x=a;var y=b;if((x&0x7fffffffu)<(y&0x7fffffffu)){x=b;y=a;}
 let ax=x&0x7fffffffu;let ay=y&0x7fffffffu;let sign=x&0x80000000u;
 if(ax==0x7f800000u){if(ay==ax&&((x^y)&0x80000000u)!=0u){return 0x7fc00000u;}return x;}
 let rx=(x>>23u)&255u;let ry=(y>>23u)&255u;var e=i32(max(rx,1u));
 var sx=((x&0x7fffffu)|select(0u,0x800000u,rx!=0u))<<3u;
 let sy=ink_f32_jam(((y&0x7fffffu)|select(0u,0x800000u,ry!=0u))<<3u,u32(e)-max(ry,1u));
 if(((x^y)&0x80000000u)==0u){sx+=sy;if(sx>=0x8000000u){sx=ink_f32_jam(sx,1u);e++;}}
 else{sx-=sy;if(sx==0u){return 0u;}loop{if(sx>=0x4000000u||e<=1i){break;}sx<<=1u;e--;}}
 return ink_f32_round(sign,e,sx);
}
fn ink_f32_sub(a:u32,b:u32)->u32{return ink_f32_add(a,b^0x80000000u);}
fn ink_f32_mul(a:u32,b:u32)->u32{
 if(ink_f32_nan(a)||ink_f32_nan(b)){return 0x7fc00000u;}
 let aa=a&0x7fffffffu;let bb=b&0x7fffffffu;let sign=(a^b)&0x80000000u;
 if(aa==0x7f800000u||bb==0x7f800000u){if(aa==0u||bb==0u){return 0x7fc00000u;}return sign|0x7f800000u;}
 if(aa==0u||bb==0u){return sign;}
 let ra=(a>>23u)&255u;let rb=(b>>23u)&255u;var ea=i32(max(ra,1u));var eb=i32(max(rb,1u));
 var sa=(a&0x7fffffu)|select(0u,0x800000u,ra!=0u);var sb=(b&0x7fffffu)|select(0u,0x800000u,rb!=0u);
 loop{if(sa>=0x800000u){break;}sa<<=1u;ea--;}
 loop{if(sb>=0x800000u){break;}sb<<=1u;eb--;}
 let product=ink_u64_mul(InkU64(sa,0u),InkU64(sb,0u));let high=(product.hi&0x8000u)!=0u;
 let shift=select(20u,21u,high);let extended=(product.lo>>shift)|(product.hi<<(32u-shift))|select(0u,1u,(product.lo&((1u<<shift)-1u))!=0u);
 return ink_f32_round(sign,ea+eb-127i+select(0i,1i,high),extended);
}
