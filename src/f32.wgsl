// Binary32 carried in integer lanes: round-to-nearest, ties-to-even,
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
 if(ay==0u){if(ax==0u){return x&y;}return x;}
 if(ax==0x7f800000u){if(ay==ax&&((x^y)&0x80000000u)!=0u){return 0x7fc00000u;}return x;}
 let rx=(x>>23u)&255u;let ry=(y>>23u)&255u;var e=i32(max(rx,1u));
 var sx=((x&0x7fffffu)|select(0u,0x800000u,rx!=0u))<<3u;
 let sy=ink_f32_jam(((y&0x7fffffu)|select(0u,0x800000u,ry!=0u))<<3u,u32(e)-max(ry,1u));
 if(((x^y)&0x80000000u)==0u){sx+=sy;if(sx>=0x8000000u){sx=ink_f32_jam(sx,1u);e++;}}
 else{sx-=sy;if(sx==0u){return 0u;}let shift=min(countLeadingZeros(sx)-5u,u32(e-1i));sx<<=shift;e-=i32(shift);}
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
 let ashift=countLeadingZeros(sa)-8u;sa<<=ashift;ea-=i32(ashift);
 let bshift=countLeadingZeros(sb)-8u;sb<<=bshift;eb-=i32(bshift);
 // A power-of-two factor changes only the exponent of a normal result.
 let power=(sa==0x800000u)||(sb==0x800000u);
 let exponent=ea+eb-127i;
 if(power&&exponent>0i&&exponent<255i){return sign|(u32(exponent)<<23u)|((sa|sb)&0x7fffffu);}
 let product=ink_u64_mul(InkU64(sa,0u),InkU64(sb,0u));let high=(product.hi&0x8000u)!=0u;
 let shift=select(20u,21u,high);let extended=(product.lo>>shift)|(product.hi<<(32u-shift))|select(0u,1u,(product.lo&((1u<<shift)-1u))!=0u);
 return ink_f32_round(sign,ea+eb-127i+select(0i,1i,high),extended);
}
// A native operation can propose a normal result. Accept it only when the
// exact integer-scaled input lies between the candidate's rounding midpoints.
// Zero accepts the candidate; -1 or +1 points to its adjacent lane.
// The candidate checks return 2 when their midpoint scaling is inapplicable.
fn ink_f32_rounding_direction(value:InkU64,low:InkU64,high:InkU64,sig:u32)->i32{
 let even=(sig&1u)==0u;
 if(ink_u64_lt(value,low)||(!even&&ink_u64_eq(value,low))){return -1i;}
 if(ink_u64_lt(high,value)||(!even&&ink_u64_eq(value,high))){return 1i;}
 return 0i;
}
fn ink_f32_sqrt_candidate_direction(x:u32,raw:u32,candidate:u32)->i32{
 // At an exact power of two, the predecessor has a different ULP.
 let fraction=candidate&0x7fffffu;
 if(candidate<0x800000u||candidate>=0x7f800000u||fraction==0u){return 2i;}
 let shift=i32(raw)-2i*i32(candidate>>23u)+152i;
 if(shift<25i||shift>26i){return 2i;}
 let s=u32(shift);let input=(x&0x7fffffu)|0x800000u;
 let scaled=InkU64(input<<s,input>>(32u-s));
 let sig=fraction|0x800000u;
 // sqrt(input) is between the midpoints iff input is between their squares.
 let lo=InkU64(sig*2u-1u,0u);let hi=InkU64(sig*2u+1u,0u);
 return ink_f32_rounding_direction(scaled,ink_u64_mul(lo,lo),ink_u64_mul(hi,hi),sig);
}
fn ink_f32_div_candidate_direction(a:u32,b:u32,exponent_diff:i32,candidate:u32)->i32{
 let fraction=candidate&0x7fffffu;
 if(candidate<0x800000u||candidate>=0x7f800000u||fraction==0u){return 2i;}
 let shift=exponent_diff-i32(candidate>>23u)+151i;
 if(shift<24i||shift>25i){return 2i;}
 let s=u32(shift);let input=(a&0x7fffffu)|0x800000u;
 let scaled=InkU64(input<<s,input>>(32u-s));
 let sig=fraction|0x800000u;
 let divisor=InkU64((b&0x7fffffu)|0x800000u,0u);
 // Compare input against each midpoint times the positive divisor.
 return ink_f32_rounding_direction(scaled,ink_u64_mul(InkU64(sig*2u-1u,0u),divisor),
     ink_u64_mul(InkU64(sig*2u+1u,0u),divisor),sig);
}
fn ink_f32_sqrt(x:u32)->u32{
 let a=x&0x7fffffffu;if(a==0u){return x;}
 if(a>0x7f800000u||(x&0x80000000u)!=0u){return 0x7fc00000u;}
 if(a==0x7f800000u){return x;}
 let raw=x>>23u;
 if(raw!=0u){
  let candidate=bitcast<u32>(sqrt(bitcast<f32>(x)));
  let direction=ink_f32_sqrt_candidate_direction(x,raw,candidate);
  if(direction==0i){return candidate;}
  // A neighboring estimate still has to pass the exact midpoint check.
  if(abs(direction)==1i){
   let adjusted=u32(i32(candidate)+direction);
   if(ink_f32_sqrt_candidate_direction(x,raw,adjusted)==0i){return adjusted;}
  }
 }
 var m=(x&0x7fffffu)|select(0u,0x800000u,raw!=0u);var e=i32(max(raw,1u))-127i;
 let normalize=countLeadingZeros(m)-8u;m<<=normalize;e-=i32(normalize);
 let odd=e&1i;var half=(e-odd)/2i;let shift=23u+u32(odd);
 let n=InkU64(m<<shift,m>>(32u-shift));var lo=0u;var hi=0x1000000u;
 loop{if(lo+1u>=hi){break;}let mid=(lo+hi)>>1u;let sq=ink_u64_mul(InkU64(mid,0u),InkU64(mid,0u));if(!ink_u64_lt(n,sq)){lo=mid;}else{hi=mid;}}
 let sq=ink_u64_mul(InkU64(lo,0u),InkU64(lo,0u));let rem=ink_u64_sub(n,sq);
 if(ink_u64_lt(InkU64(lo,0u),rem)){lo++;}
 if(lo==0x1000000u){lo>>=1u;half++;}
 return (u32(half+127i)<<23u)|(lo&0x7fffffu);
}
fn ink_f32_div(a:u32,b:u32)->u32{
 let aa=a&0x7fffffffu;let bb=b&0x7fffffffu;let sign=(a^b)&0x80000000u;
 if(ink_f32_nan(a)||ink_f32_nan(b)||(aa==0u&&bb==0u)||(aa==0x7f800000u&&bb==aa)){return 0x7fc00000u;}
 if(aa==0x7f800000u||bb==0u){return sign|0x7f800000u;}
 if(aa==0u||bb==0x7f800000u){return sign;}
 let ra=aa>>23u;let rb=bb>>23u;let exponent_diff=i32(ra)-i32(rb);
 if(ra!=0u&&rb!=0u&&bb<=0x7e800000u&&abs(exponent_diff)<=124i){
  let candidate=bitcast<u32>(bitcast<f32>(aa)/bitcast<f32>(bb));
  let direction=ink_f32_div_candidate_direction(aa,bb,exponent_diff,candidate);
  if(direction==0i){return sign|candidate;}
  if(abs(direction)==1i){
   let adjusted=u32(i32(candidate)+direction);
   if(ink_f32_div_candidate_direction(aa,bb,exponent_diff,adjusted)==0i){return sign|adjusted;}
  }
 }
 var ea=i32(max(ra,1u))-127i;var eb=i32(max(rb,1u))-127i;
 var sa=(a&0x7fffffu)|select(0u,0x800000u,ra!=0u);var sb=(b&0x7fffffu)|select(0u,0x800000u,rb!=0u);
 let ashift=countLeadingZeros(sa)-8u;sa<<=ashift;ea-=i32(ashift);
 let bshift=countLeadingZeros(sb)-8u;sb<<=bshift;eb-=i32(bshift);
 var e=ea-eb;if(sa<sb){sa<<=1u;e--;}
 // The normalized ratio is in [1,2). Generate its fractional bits with
 // a remainder smaller than the 24-bit divisor, so one u32 lane suffices.
 var remainder=sa-sb;var quotient=1u;
 for(var i=0u;i<26u;i++){
  remainder<<=1u;quotient<<=1u;
  if(remainder>=sb){remainder-=sb;quotient|=1u;}
 }
 let extended=quotient|select(0u,1u,remainder!=0u);
 return ink_f32_round(sign,e+127i,extended);
}
// mode: 0 trunc, 1 floor, 2 ceil, 3 ties-to-even.
fn ink_f32_integral(x:u32,mode:u32)->u32{
 let a=x&0x7fffffffu;let sign=x&0x80000000u;
 if(a>0x7f800000u){return 0x7fc00000u;}if(a>=0x4b000000u||a==0u){return x;}
 let e=i32((x>>23u)&255u)-127i;
 if(e<0i){var up=false;if(mode==1u){up=sign!=0u;}else if(mode==2u){up=sign==0u;}else if(mode==3u){up=a>0x3f000000u;}return sign|select(0u,0x3f800000u,up);}
 let shift=u32(23i-e);let mask=(1u<<shift)-1u;let fraction=a&mask;var rounded=a&~mask;var up=false;
 if(fraction!=0u){if(mode==1u){up=sign!=0u;}else if(mode==2u){up=sign==0u;}else if(mode==3u){let half=1u<<(shift-1u);up=fraction>half||(fraction==half&&((rounded>>shift)&1u)!=0u);}}
 if(up){rounded+=1u<<shift;}return sign|rounded;
}
fn ink_f32_from_u64(x:InkU64)->u32{
 if(ink_u64_eq(x,InkU64(0u,0u))){return 0u;}
 let leading=select(countLeadingZeros(x.lo)+32u,countLeadingZeros(x.hi),x.hi!=0u);let e=63u-leading;
 if(e<=23u){return ((e+127u)<<23u)|((x.lo<<(23u-e))&0x7fffffu);}
 let shift=e-23u;var sig=0u;var rem=InkU64(0u,0u);var half=InkU64(0u,0u);
 if(shift<32u){sig=(x.lo>>shift)|(x.hi<<(32u-shift));rem=InkU64(x.lo&((1u<<shift)-1u),0u);half=InkU64(1u<<(shift-1u),0u);}
 else if(shift==32u){sig=x.hi;rem=InkU64(x.lo,0u);half=InkU64(0x80000000u,0u);}
 else{sig=x.hi>>(shift-32u);rem=InkU64(x.lo,x.hi&((1u<<(shift-32u))-1u));half=InkU64(0u,1u<<(shift-33u));}
 if(ink_u64_lt(half,rem)||(ink_u64_eq(rem,half)&&(sig&1u)!=0u)){sig++;}
 var exponent=e+127u;if(sig==0x1000000u){sig>>=1u;exponent++;}
 return (exponent<<23u)|(sig&0x7fffffu);
}
fn ink_f32_from_i32(x:i32)->u32{let neg=x<0i;let magnitude=select(u32(x),0u-u32(x),neg);return ink_f32_from_u64(InkU64(magnitude,0u))|select(0u,0x80000000u,neg);}
