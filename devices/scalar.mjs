const B={MAP_READ:1,COPY_SRC:4,COPY_DST:8,STORAGE:128};
const SCOPES={sum:[false,true,false,true],count:[false,true],offsets:[false,true,true],scatter:[false,true,false,false,false]};
export function createDevice({asset,onLost}){
  let device,initialising,gpuError;
  const pipelines=new Map();
  async function ensureGPU(){
    if(gpuError)throw Error(gpuError);
    if(!initialising)initialising=(async()=>{
      if(!globalThis.navigator?.gpu)throw Error('WebGPU unavailable');
      const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error('no WebGPU adapter');
      device=await adapter.requestDevice();
      device.lost.then(info=>{gpuError=`GPU device lost: ${info.message}`;onLost();});
      return device;
    })().catch(e=>{gpuError=String(e);throw e;});
    return initialising;
  }
  async function pipeline(name,access){
    if(pipelines.has(name))return pipelines.get(name);
    const code=await (await asset(name)).text();
    const layout=device.createBindGroupLayout({entries:access.map((write,binding)=>({binding,visibility:4,buffer:{type:write?'storage':'read-only-storage'}}))});
    const compiled=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module:device.createShaderModule({code}),entryPoint:'main'}});
    pipelines.set(name,compiled);return compiled;
  }
  async function prepare(f,args){
    await ensureGPU();const cap=Math.max(1,args[f.gpu.list_param].length),groups=Math.ceil(cap/256);
    if(cap*4>device.limits.maxStorageBufferBindingSize||cap*4>device.limits.maxBufferSize||groups>device.limits.maxComputeWorkgroupsPerDimension)throw Error('input exceeds GPU buffer/dispatch limits');
    const filters=f.gpu.stages.filter(s=>s.kind==='filter').length;
    if(cap*4*(1+f.gpu.stages.length+filters)+groups*8*filters>268435456)throw Error('GPU working buffers exceed runtime allocation budget');
    if(filters&&cap>1048576)throw Error('input exceeds this backend’s stable-filter prefix budget');
    for(const stage of f.gpu.stages)await pipeline(stage.shader,stage.kind==='filter'?[false,true,false,true,false,true,true]:[false,true,false,true,false]);
    if(f.gpu.stages.some(s=>s.kind==='filter')){await pipeline('filter-offsets.wgsl',SCOPES.offsets);await pipeline('filter-scatter.wgsl',SCOPES.scatter);}
    await pipeline(`${f.gpu.aggregate}.wgsl`,SCOPES[f.gpu.aggregate]);
  }
  async function gpu(f,args){
    await prepare(f,args);
    const resources=[];device.pushErrorScope('out-of-memory');device.pushErrorScope('validation');
    let scopesOpen=2;
    try{
      const allocate=(size,usage=B.STORAGE|B.COPY_DST|B.COPY_SRC)=>{const buffer=device.createBuffer({size:Math.max(4,size),usage});resources.push(buffer);return buffer;};
      const upload=v=>{const b=allocate(v.byteLength);device.queue.writeBuffer(b,0,v);return b;};
      const xs=args[f.gpu.list_param];let cap=Math.max(1,xs.length),data=upload(xs),length=upload(new Uint32Array([xs.length]));
      const parameters=upload(new Uint32Array(args.map((v,i)=>i===f.gpu.list_param?0:typeof v==='boolean'?(v?1:0):v)));
      const encoder=device.createCommandEncoder();
      function dispatch(name,buffers,groups){
        const compiled=pipelines.get(name),pass=encoder.beginComputePass();pass.setPipeline(compiled);
        pass.setBindGroup(0,device.createBindGroup({layout:compiled.getBindGroupLayout(0),entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))}));
        pass.dispatchWorkgroups(Math.max(1,groups));pass.end();
      }
      for(const stage of f.gpu.stages){
        const output=allocate(cap*4),outlen=allocate(4),groups=Math.ceil(cap/256);
        if(stage.kind==='map')dispatch(stage.shader,[data,output,length,outlen,parameters],groups);
        else{
          const prefixes=allocate(cap*4),counts=allocate(groups*4),offsets=allocate(groups*4);
          dispatch(stage.shader,[data,output,length,outlen,parameters,prefixes,counts],groups);
          dispatch('filter-offsets.wgsl',[counts,offsets,outlen],Math.ceil(groups/64));
          dispatch('filter-scatter.wgsl',[data,output,length,prefixes,offsets],groups);
        }
        data=output;length=outlen;
      }
      if(f.gpu.aggregate==='count'){const output=allocate(4);dispatch('count.wgsl',[length,output],1);data=output;}
      else{
        do{const groups=Math.ceil(cap/256),output=allocate(groups*4),outlen=allocate(4);dispatch('sum.wgsl',[data,output,length,outlen],groups);data=output;length=outlen;cap=groups;}while(cap>1);
      }
      const readback=allocate(4,B.COPY_DST|B.MAP_READ);encoder.copyBufferToBuffer(data,0,readback,0,4);device.queue.submit([encoder.finish()]);
      await readback.mapAsync(1);const value=new Uint32Array(readback.getMappedRange())[0];readback.unmap();
      const validation=await device.popErrorScope();scopesOpen--;
      const allocation=await device.popErrorScope();scopesOpen--;
      const error=validation??allocation;if(error)throw Error(error.message);
      return f.result==='u64'?BigInt(value):value;
    }finally{
      try{while(scopesOpen>0){scopesOpen--;await device.popErrorScope();}}
      finally{for(const buffer of resources)buffer.destroy();}
    }
  }

  return {prepare,run:gpu,close(){device?.destroy();pipelines.clear();gpuError="engine disposed";}};
}
