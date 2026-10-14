import assert from 'node:assert/strict';
import {createDevice,graphCapacity,dispatchCount} from '../devices/compute.mjs';
globalThis.GPUBufferUsage={MAP_READ:1,COPY_SRC:4,COPY_DST:8,STORAGE:128};globalThis.GPUShaderStage={COMPUTE:4};globalThis.GPUMapMode={READ:1};
let live=0,peak=0,allocations=0,dispatches=0;
// Each dispatch records the pipeline and bindings current at that point, as
// WebGPU does; one pass may carry many dispatches.
// Mapped bytes are poisoned on readback unmap. Only the first owning copy may
// escape; stripping a header must not copy the already-owned readback again.
const mappedReadback=new WeakSet(),ownedReadback=new WeakSet();let headerCopies=0;
const originalSlice=Uint32Array.prototype.slice;
Uint32Array.prototype.slice=function(...args){if(ownedReadback.has(this.buffer))headerCopies++;const result=originalSlice.apply(this,args);if(mappedReadback.has(this.buffer))ownedReadback.add(result.buffer);return result;};
const device={limits:{maxStorageBuffersPerShaderStage:8,maxComputeWorkgroupsPerDimension:65535,maxBufferSize:1<<25,maxStorageBufferBindingSize:1<<25},lost:new Promise(()=>{}),pushErrorScope(){},async popErrorScope(){return null},
 createBuffer({size,usage}){live+=size;peak=Math.max(peak,live);allocations++;const data=new ArrayBuffer(size);if(usage&GPUBufferUsage.MAP_READ)mappedReadback.add(data);let dead=false;return {size,data,getMappedRange(){return data},unmap(){if(usage&GPUBufferUsage.MAP_READ)new Uint32Array(data).fill(0xdeadbeef)},async mapAsync(){},destroy(){if(!dead){live-=size;dead=true}}};},
 createBindGroupLayout(x){return x},createPipelineLayout(x){return x},createShaderModule(x){return x},async createComputePipelineAsync({compute}){return {kind:compute.module.code,getBindGroupLayout(){return {}}}},createBindGroup(x){return x},
 createCommandEncoder(){const commands=[];return {beginComputePass(){let pipe,group;return {setPipeline(p){pipe=p},setBindGroup(_,g){group=g},dispatchWorkgroups(){const pipe0=pipe,group0=group;commands.push(()=>{const pipe=pipe0,group=group0;dispatches++;const b=group.entries.map(e=>new Uint32Array(e.resource.buffer.data));const out=b.at(-1);if(pipe.kind==='filter'){let n=0;for(let i=0;i<b[0][0];i++)if(b[0][i+1]>b[1][0])out[++n]=b[0][i+1];out[0]=n;}else if(pipe.kind==='direct-map'){const dst=b[1];dst[0]=b[0][0];for(let i=0;i<b[0][0];i++)dst[i+1]=(b[0][i+1]+1)>>>0;}else if(pipe.kind==='map'){out[0]=b[0][0];for(let i=0;i<b[0][0];i++)out[i+1]=(b[0][i+1]+1)>>>0;}else if(pipe.kind==='sum'){let n=0;for(let i=0;i<b[0][0];i++)n=(n+b[0][i+1])>>>0;out[0]=n;}else throw Error('mock shader');})},end(){}}},copyBufferToBuffer(a,offset,b,target,bytes){commands.push(()=>new Uint8Array(b.data,target,bytes).set(new Uint8Array(a.data,offset,bytes)))},finish(){return commands}}},queue:{submit(lists){for(const list of lists)for(const f of list)f()}},destroy(){}};
Object.defineProperty(globalThis,'navigator',{value:{gpu:{async requestAdapter(){return {async requestDevice(){return device}}}}}});
globalThis.fetch=async url=>({async text(){return String(url).split('/').at(-1)}});
const list={kind:'list',element:{kind:'u32'}},word={kind:'u32'};
const graph=(op,result,args,kind,indices)=>({nodes:[{id:0,op,args:args.map(param=>({param})),result,capacity:{kind,args:indices},shader:op,ordered:op!=='map',workgroup_size:op==='map'?64:1}],output:{node:0}});
const functions=[{graph_required:true,graph:graph('filter',list,[0,1],'same',[0]),result:list},{shader:'direct-map',graph_required:false,graph:graph('map',list,[0],'same',[0]),result:list},{graph_required:true,graph:graph('sum',word,[0],'one',[]),result:word}];
const engine=createDevice({root:new URL('http://mock/'),enabled:true,onLost(){},fail(s){throw Error(s)},width(t){return t.kind==='list'?this?.width?.(t.element)??1:1},unpack(w,t){return t.kind==='list'?[...w]:w[0]},MAX_BYTES:1024*1024});
function plan(input,iterations=1){return {inputs:new Map([['xs',{index:0}]]),feedback:new Map(),constants:[{words:Uint32Array.from(input),type:list,count:input.length},{words:Uint32Array.of(2),type:word,count:1}],steps:[{f:functions[0],args:[{kind:'constant',index:0,count:input.length},{kind:'constant',index:1,count:1}],count:input.length},{f:functions[1],args:[{kind:'step',index:0,count:input.length}],count:input.length},{f:functions[2],args:[{kind:'step',index:1,count:input.length}],count:1}],outputs:['result'],ids:new Map([['result',2]]),iterations,reason:null};}
const p=plan([1,3,2,4],3);
assert.deepEqual(engine.preparation(p),{device:false,shaders:[['filter',false],['map',false],['sum',false]]});
await engine.prepare(p);
assert.deepEqual(engine.preparation(p),{device:true,shaders:[['filter',true],['map',true],['sum',true]]});
assert.deepEqual(engine.preparation({steps:[{f:{shader:'unprepared'}}]}),{device:true,shaders:[['unprepared',false]]});
const preparation=engine.preparation(p);preparation.shaders[0][1]=false;
assert.equal(engine.preparation(p).shaders[0][1],true);
const result=await engine.run(p);
assert.equal(result.values.result,9);assert.equal(result.counters.dispatches,9);assert.equal(dispatches,9);assert.equal(result.counters.upload_bytes,24);assert.equal(result.counters.readback_bytes,4);assert.equal(result.counters.live_buffer_bytes,peak);assert.equal(allocations,6);assert.equal(live,0);assert.deepEqual([...p.constants[0].words],[1,3,2,4]);
const empty=plan([]);await engine.prepare(empty);assert.equal((await engine.run(empty)).values.result,0);assert.equal(live,0);
for(const input of [[1,3,2,4],[0,1],[]]) {
  const feedback=plan(input,3);feedback.feedback.set('xs',1);
  await engine.prepare(feedback);const actual=await engine.run(feedback);
  const final=input.filter(x=>x>2).map(x=>x+3);
  assert.equal(actual.values.result,final.reduce((a,b)=>a+b,0));
  assert.equal(actual.counters.upload_bytes,input.length*4+8);
  assert.equal(actual.counters.readback_bytes,4);
  assert.equal(actual.counters.dispatches,9);
  assert.equal(actual.counters.resident_copy_bytes,4*(input.length+1)*4);
  assert.equal(live,0);
}
// Empty graphs expose identity inputs, so feedback destinations alias outputs.
// An unused filter forces graph execution. Copying destinations immediately
// would corrupt swaps; copying after the final iteration would alter outputs.
for(const iterations of [1,2,3]) {
  const identity={graph_required:true,result:list,graph:{nodes:[],output:{param:0}}};
  const p={iterations,reason:null,inputs:new Map([['a',{index:0}],['b',{index:1}]]),
    constants:[{words:Uint32Array.of(1,2),type:list,count:2},{words:Uint32Array.of(7,9),type:list,count:2},{words:Uint32Array.of(2),type:word,count:1}],
    steps:[{f:identity,args:[{kind:'constant',index:0,count:2}],count:2},{f:identity,args:[{kind:'constant',index:1,count:2}],count:2},{f:functions[0],args:[{kind:'constant',index:0,count:2},{kind:'constant',index:2,count:1}],count:2}],
    feedback:new Map([['a',1],['b',0]]),outputs:['a','b'],ids:new Map([['a',0],['b',1]])};
  await engine.prepare(p);const result=await engine.run(p);
  assert.deepEqual(result.values,iterations%2?{a:[1,2],b:[7,9]}:{a:[7,9],b:[1,2]});
  assert.equal(result.counters.resident_copy_bytes,(iterations-1)*48);
  assert.equal(result.counters.upload_bytes,28);assert.equal(result.counters.readback_bytes,24);
  assert.equal(result.counters.dispatches,iterations);assert.equal(live,0);
}
const mismatch=plan([1]);mismatch.feedback.set('xs',2);await engine.prepare(mismatch);
await assert.rejects(engine.run(mismatch),/capacity\/type mismatch/);assert.equal(live,0);
const bounded=createDevice({root:new URL('http://mock/'),enabled:true,onLost(){},fail(s){throw Error(s)},width:()=>1,unpack:()=>0,MAX_BYTES:70});
const over=plan([1,2,3,4]);over.feedback.set('xs',1);await bounded.prepare(over);
await assert.rejects(bounded.run(over),/buffer limits/);assert.equal(live,0);bounded.close();
assert.deepEqual(graphCapacity({nodes:[{args:[{param:0},{param:1}],capacity:{kind:'min',args:[0,1]}}],output:{node:0}},[7,3]),{capacities:[3],outputCount:3});
assert.throws(()=>graphCapacity({nodes:[{args:[{param:0},{param:1}],capacity:{kind:'sum',args:[0,1]}}],output:{node:0}},[0xffffffff,1]),/capacity/);
// Captured graph sessions retain buffers/inputs, preserve dynamic extents,
// return owned results and allocate no new device buffers on repeated runs.
const residentPlan=plan([1,3,2,4]);const resident=await engine.capture(residentPlan),allocated=allocations;
assert.equal(resident.setup.upload_bytes,24);assert.ok(live>0);
residentPlan.constants[0].words.fill(99);
const first=await resident.run();assert.equal(first.values.result,9);assert.equal(first.counters.upload_bytes,0);assert.equal(first.counters.readback_bytes,4);assert.equal(first.counters.dispatches,3);assert.equal(first.counters.live_buffer_bytes,live);
assert.equal((await resident.run()).values.result,9);assert.equal(allocations,allocated);assert.equal(first.values.result,9);resident.close();resident.close();assert.equal(live,0);await assert.rejects(resident.run(),/closed/);
const legacy={feedback:new Map(),constants:[{words:Uint32Array.from([0,4294967295]),type:list,count:2}],steps:[{f:functions[1],args:[{kind:'constant',index:0,count:2}],count:2}],outputs:['result'],ids:new Map([['result',0]]),iterations:1,reason:null};
await engine.prepare(legacy);const complete=await engine.run(legacy);assert.deepEqual(complete.values.result,[1,0]);legacy.constants[0].words.fill(42);await engine.run(legacy);assert.deepEqual(complete.values.result,[1,0],'complete outputs survive subsequent call and caller mutation');legacy.constants[0].words.set([0,4294967295]);
const legacyCall=await engine.capture(legacy);legacy.constants[0].words.fill(99);const a=await legacyCall.run();assert.deepEqual(a.values.result,[1,0]);assert.equal(a.counters.upload_bytes,0);assert.equal(a.counters.readback_bytes,12);assert.equal(a.counters.dispatches,1);const aCopy=[...a.values.result];await legacyCall.run();assert.deepEqual(a.values.result,aCopy);legacyCall.close();assert.equal(live,0);assert.deepEqual(a.values.result,[1,0],'retained output survives close');assert.deepEqual(complete.values.result,[1,0]);assert.equal(headerCopies,0,'owned readback header stripping uses a borrowed subview consumed before return');Uint32Array.prototype.slice=originalSlice;
assert.equal(dispatchCount(p),3);
const overDispatch={...p,iterations:1_000_000,steps:[{f:{graph_required:true,graph:{nodes:[{schedule:{passes:Array(100).fill({})}}]}}}]};
assert.equal(dispatchCount(overDispatch),100);
await assert.rejects(engine.prepare(overDispatch),/dispatch budget/);
const slicePlan=plan([1,3,2,4]);slicePlan.ids.set('filtered',0);
await engine.prepare(slicePlan);const sliceState=await engine.open(slicePlan);await engine.advance(sliceState,slicePlan,1);
const shortSlice=await engine.read(sliceState,slicePlan,{step:'filtered',slice:[99,3]});
assert.deepEqual({value:shortSlice.value,length:shortSlice.length,start:shortSlice.start},{value:[],length:2,start:2});
await assert.rejects(engine.advance(sliceState,overDispatch,1_000_000),/dispatch budget/);
engine.closeSession(sliceState);
engine.close();console.log('PASS prepared graph/legacy residency, input capture, no repeated allocations, logical extents, owned output, cleanup');console.log('PASS graph capacity, mixed dynamic lengths, iteration buffer reuse, exact counters, empty stages, input nonmutation, resident feedback, alias swaps, capacity and allocation guards');

for(const value of [0,1,16,0xffffffff]) assert.deepEqual(graphCapacity({nodes:[{args:[],capacity:{kind:'fixed',value,args:[]}}],output:{node:0}},[]),{capacities:[value],outputCount:value});
for(const value of [-1,0x100000000,NaN,Infinity,0.5,undefined]) assert.throws(()=>graphCapacity({nodes:[{args:[],capacity:{kind:'fixed',value,args:[]}}],output:{node:0}},[]),/capacity/);
