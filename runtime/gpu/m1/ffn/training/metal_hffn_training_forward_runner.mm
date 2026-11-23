#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "hffn_route_codec.h"

// ReLU compatibility entry: inference-owned arithmetic, caller-owned z/h.
struct Args { uint32_t input=128, hidden=512, output=128, n=64, degree=13, warmup=30, repeats=200; bool untimed=false; const char* path=nullptr; };
static uint32_t number(const char* x) { return uint32_t(std::strtoul(x, nullptr, 10)); }
static bool option(int& i, int argc, const char* argv[], const char* name, uint32_t& value) { if (std::strcmp(argv[i], name) || ++i >= argc) return false; value = number(argv[i]); return true; }
static bool parse(int argc, const char* argv[], Args& a) { for (int i=1; i<argc; ++i) { if (!std::strcmp(argv[i],"--untimed")) { a.untimed=true; continue; } if (option(i,argc,argv,"--input",a.input) || option(i,argc,argv,"--hidden",a.hidden) || option(i,argc,argv,"--output-size",a.output) || option(i,argc,argv,"--n",a.n) || option(i,argc,argv,"--degree",a.degree) || option(i,argc,argv,"--warmup",a.warmup) || option(i,argc,argv,"--repeats",a.repeats)) continue; if (!std::strcmp(argv[i],"--output") && ++i<argc) { a.path=argv[i]; continue; } return false; } return a.path && a.input && a.hidden && a.output && a.n && a.degree<=a.input && a.degree<=a.output && a.input<=65535u && uint64_t(a.hidden)*a.degree<INT32_MAX && uint64_t(a.n)*std::max({a.input,a.hidden,a.output})<(uint64_t(1)<<32); }

static id<MTLBuffer> buffer(id<MTLDevice> d, const void* data, size_t bytes) { id<MTLBuffer> b=[d newBufferWithLength:bytes options:MTLResourceStorageModeShared]; std::memcpy(b.contents, data, bytes); return b; }
static void dispatch(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p, NSArray<id<MTLBuffer>>* buffers, MTLSize groups) { [e setComputePipelineState:p]; for(NSUInteger i=0;i<buffers.count;++i) [e setBuffer:buffers[i] offset:0 atIndex:i]; [e dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(512,1,1)]; }
static bool untimed_run=false;
static double run(id<MTLCommandQueue> queue, id<MTLComputePipelineState> up, id<MTLComputePipelineState> down, NSArray<id<MTLBuffer>>* ub, NSArray<id<MTLBuffer>>* db, MTLSize ug, MTLSize dg, uint32_t repeats) { id<MTLCommandBuffer> cmd=[queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder]; for(uint32_t k=0;k<repeats;++k) { dispatch(e,up,ub,ug); dispatch(e,down,db,dg); } [e endEncoding]; [cmd commit]; [cmd waitUntilCompleted]; if(cmd.status==MTLCommandBufferStatusError) { std::fprintf(stderr,"Metal: %s\n",[[cmd.error.localizedDescription description] UTF8String]); std::exit(1); } if(untimed_run)return 0; return (cmd.GPUEndTime-cmd.GPUStartTime)*1000.0/double(repeats); }
static double max_error(const std::vector<float>& ref, const float* got) { double error=0; for(size_t i=0;i<ref.size();++i) {if(!std::isfinite(ref[i]) || !std::isfinite(got[i]))std::exit(1);error=std::max(error,std::abs(double(ref[i]-got[i])));} return error; }

int main(int argc, const char* argv[]) { @autoreleasepool {
  Args a; if(!parse(argc,argv,a)) { std::fprintf(stderr,"usage: --input I --hidden F --output-size O --n N --degree D --output file\n"); return 2; }
  untimed_run=a.untimed;
  const uint32_t edges=a.hidden*a.degree; const HffnRouteCodec codec(a.hidden,a.degree); const bool compact16=codec.compact16;
  std::vector<float> x(a.input*a.n), up(edges), down(edges), bup(a.hidden), bdown(a.output), z_ref(a.hidden*a.n), h_ref(a.hidden*a.n), y_ref(a.output*a.n);
  std::vector<uint16_t> up_index(edges), route16(edges); std::vector<uint32_t> route32(edges), offset(a.output+1,0);
  for(uint32_t i=0;i<a.input;++i) for(uint32_t n=0;n<a.n;++n) x[i*a.n+n]=float(int((i*17u+n*13u)%29u)-14)*.02f;
  for(uint32_t r=0;r<a.hidden;++r) { bup[r]=float(int(r%9u)-4)*.01f; for(uint32_t e=0;e<a.degree;++e) { const uint32_t p=r*a.degree+e; up_index[p]=uint16_t((r*19u+e*7u)%a.input); up[p]=float(int((p*3u)%17u)-8)*.015f; down[p]=float(int((p*5u)%19u)-9)*.012f; ++offset[(r*17u+e*5u)%a.output+1u]; } }
  for(uint32_t o=0;o<a.output;++o) { offset[o+1]+=offset[o]; bdown[o]=float(int(o%11u)-5)*.008f; }
  std::vector<uint32_t> cursor=offset;
  for(uint32_t p=0;p<edges;++p) { const uint32_t r=p/a.degree,e=p%a.degree,o=(r*17u+e*5u)%a.output,q=cursor[o]++; if(compact16) route16[q]=codec.encode16(r,e); else route32[q]=codec.encode32(r,e); }
  for(uint32_t r=0;r<a.hidden;++r) for(uint32_t n=0;n<a.n;++n) { float v=bup[r]; for(uint32_t e=0;e<a.degree;++e) { const uint32_t p=r*a.degree+e; v+=up[p]*x[uint32_t(up_index[p])*a.n+n]; } z_ref[r*a.n+n]=v; h_ref[r*a.n+n]=std::max(v,0.0f); }
  for(uint32_t o=0;o<a.output;++o) for(uint32_t n=0;n<a.n;++n) { float v=bdown[o]; for(uint32_t q=offset[o];q<offset[o+1];++q) { const uint32_t p=compact16?codec.decode16(route16[q]):codec.decode32(route32[q]); v+=down[p]*h_ref[(p/a.degree)*a.n+n]; } y_ref[o*a.n+n]=v; }
  id<MTLDevice> device=MTLCreateSystemDefaultDevice(); if(!device) return 1; NSError* error=nil;
  const char *directory=std::getenv("HFFN_INFERENCE_FORWARD_DIR");if(!directory)return 2;
  auto library=[&](NSString *name){NSString *path=[[NSString stringWithUTF8String:directory] stringByAppendingPathComponent:name];NSString *source=[NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];id<MTLLibrary> result=source?[device newLibraryWithSource:source options:nil error:&error]:nil;if(!result){std::fprintf(stderr,"inference source missing/invalid\n");std::exit(1);}return result;};
  id<MTLLibrary> up_library=library(@"up.metal"),down_library=library(@"down.metal");
  auto pipeline=^(NSString* name) { NSError* e=nil; id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:[([name isEqualToString:@"hffn_inference_up"] ? up_library : down_library) newFunctionWithName:name] error:&e]; if(!p) { std::fprintf(stderr,"pipeline: %s\n",[[e.localizedDescription description] UTF8String]); std::exit(1); } return p; };
  id<MTLComputePipelineState> up_pipe=pipeline(@"hffn_inference_up"), down_pipe=pipeline(@"hffn_inference_down"); std::vector<float> z(a.hidden*a.n), h(a.hidden*a.n), y(a.output*a.n);
  const void* routes=compact16?static_cast<const void*>(route16.data()):static_cast<const void*>(route32.data()); const size_t route_bytes=size_t(edges)*(compact16?sizeof(uint16_t):sizeof(uint32_t));
  id<MTLBuffer> xb=buffer(device,x.data(),x.size()*4),uib=buffer(device,up_index.data(),up_index.size()*2),upb=buffer(device,up.data(),up.size()*4),bupb=buffer(device,bup.data(),bup.size()*4),zb=buffer(device,z.data(),z.size()*4),hb=buffer(device,h.data(),h.size()*4),ob=buffer(device,offset.data(),offset.size()*4),rb=buffer(device,routes,route_bytes),db=buffer(device,down.data(),down.size()*4),bdb=buffer(device,bdown.data(),bdown.size()*4),yb=buffer(device,y.data(),y.size()*4);
  std::vector<uint32_t> uoff(a.hidden+1),index(up_index.begin(),up_index.end()),hidden(edges),slots(edges);
  for(uint32_t r=0;r<=a.hidden;++r)uoff[r]=r*a.degree;
  for(uint32_t q=0;q<edges;++q){uint32_t p=compact16?codec.decode16(route16[q]):codec.decode32(route32[q]);slots[q]=p;hidden[q]=p/a.degree;}
  uint32_t one=1;float gate=1;
  id nsb=buffer(device,&a.n,4),uob=buffer(device,uoff.data(),uoff.size()*4),indexb=buffer(device,index.data(),index.size()*4),hiddenb=buffer(device,hidden.data(),hidden.size()*4),slotb=buffer(device,slots.data(),slots.size()*4),oneb=buffer(device,&one,4),gateb=buffer(device,&gate,4);
  NSArray<id<MTLBuffer>>* ub=@[nsb,uob,indexb,upb,bupb,xb,hb,zb];NSArray<id<MTLBuffer>>* dbs=@[nsb,ob,hiddenb,db,slotb,hb,gateb,bdb,yb,oneb];
  NSString *planpath=[[NSString stringWithUTF8String:directory] stringByAppendingPathComponent:@"dispatch.json"];
  NSDictionary *plan=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:planpath] options:0 error:&error];
  if(!plan || [plan[@"input"] unsignedIntValue]!=a.input || [plan[@"hidden"] unsignedIntValue]!=a.hidden || [plan[@"output"] unsignedIntValue]!=a.output || [plan[@"n"] unsignedIntValue]!=a.n)return 2;
  MTLSize ug=MTLSizeMake([plan[@"up_groups"] unsignedIntValue],1,1),dg=MTLSizeMake([plan[@"down_groups"] unsignedIntValue],1,1);
  id<MTLCommandQueue> queue=[device newCommandQueue];std::vector<double> samples;
  if(a.untimed){run(queue,up_pipe,down_pipe,ub,dbs,ug,dg,1);samples.assign(7,0.0);}
  else{run(queue,up_pipe,down_pipe,ub,dbs,ug,dg,a.warmup);for(int i=0;i<7;++i)samples.push_back(run(queue,up_pipe,down_pipe,ub,dbs,ug,dg,a.repeats));std::sort(samples.begin(),samples.end());}
  NSDictionary *result=@{@"kind":@"hffn_metal_training_forward_relu",@"route":compact16?@"compact16":@"compact32",@"dimensions":@{@"input":@(a.input),@"hidden":@(a.hidden),@"output":@(a.output),@"N":@(a.n),@"degree":@(a.degree)},@"buffers":@{@"z_dense":@YES,@"h_dense":@YES,@"b_up":@YES,@"b_down":@YES},@"median_forward_gpu_ms":a.untimed?(id)[NSNull null]:@(samples[3]),@"max_abs_error":@{@"z":@(max_error(z_ref,(float*)zb.contents)),@"h":@(max_error(h_ref,(float*)hb.contents)),@"y":@(max_error(y_ref,(float*)yb.contents))}};
  NSString *json=[[NSString alloc] initWithData:[NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:&error] encoding:NSUTF8StringEncoding];
  if(![json writeToFile:[NSString stringWithUTF8String:a.path] atomically:YES encoding:NSUTF8StringEncoding error:&error]) { std::fprintf(stderr,"write: %s\n",[[error.localizedDescription description] UTF8String]); return 1; } std::printf("%s",json.UTF8String);
}}
