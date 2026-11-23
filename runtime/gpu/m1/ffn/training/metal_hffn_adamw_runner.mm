#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Isolated compact AdamW microbenchmark.  Slots are ordered as
// [up active values | down active values | b_up | b_down].  No slot outside
// this pool exists on the HFFN side.
struct Args { uint32_t input=128, hidden=512, output=128, degree=13, warmup=30, repeats=500, step=1, verify_steps=10, dispatches=1, threads=256; const char* path=nullptr; const char* dump_prefix=nullptr; };
struct Params { float lr,beta1,beta2,eps,wd,step_size,beta2_scale; };
static uint32_t active_threads=256;

static uint32_t number(const char* x) { return uint32_t(std::strtoul(x,nullptr,10)); }
static bool option(int& i,int argc,const char* argv[],const char* name,uint32_t& x) { if(std::strcmp(argv[i],name)||++i>=argc)return false;x=number(argv[i]);return true; }
static bool parse(int argc,const char* argv[],Args& a) {
  for(int i=1;i<argc;++i) {
    if(option(i,argc,argv,"--input",a.input)||option(i,argc,argv,"--hidden",a.hidden)||option(i,argc,argv,"--output-size",a.output)||option(i,argc,argv,"--degree",a.degree)||option(i,argc,argv,"--warmup",a.warmup)||option(i,argc,argv,"--repeats",a.repeats)||option(i,argc,argv,"--step",a.step)||option(i,argc,argv,"--verify-steps",a.verify_steps)||option(i,argc,argv,"--dispatches",a.dispatches)||option(i,argc,argv,"--threads",a.threads))continue;
    if(!std::strcmp(argv[i],"--output")&&++i<argc){a.path=argv[i];continue;}
    if(!std::strcmp(argv[i],"--dump-prefix")&&++i<argc){a.dump_prefix=argv[i];continue;}
    return false;
  }
  active_threads=a.threads;
  return a.path && a.input>0 && a.hidden>0 && a.output>0 && a.degree>0 && a.degree<=a.input && a.degree<=a.output && a.step>0 && (a.dispatches==1 || a.dispatches==4) && a.threads>0 && a.threads<=1024;
}
static std::string source() { return R"MSL(
#include <metal_stdlib>
using namespace metal;
struct Params { float lr,beta1,beta2,eps,wd,step_size,beta2_scale; };
kernel void adamw_compact(device float* p[[buffer(0)]],device const float* g[[buffer(1)]],device float* m[[buffer(2)]],device float* v[[buffer(3)]],constant Params& q[[buffer(4)]],uint gid[[thread_position_in_grid]]){
  float mi=q.beta1*m[gid]+(1.0f-q.beta1)*g[gid];
  float vi=q.beta2*v[gid]+(1.0f-q.beta2)*g[gid]*g[gid];
  float pi=p[gid]*(1.0f-q.lr*q.wd);
  m[gid]=mi;v[gid]=vi;p[gid]=pi-q.step_size*(mi/(sqrt(vi)/q.beta2_scale+q.eps));
}
)MSL"; }
static id<MTLBuffer> buffer(id<MTLDevice> d,const void* data,NSUInteger n){id<MTLBuffer>b=[d newBufferWithLength:n options:MTLResourceStorageModeShared];std::memcpy(b.contents,data,n);return b;}
static double timed(id<MTLCommandQueue> q,id<MTLComputePipelineState> p,NSArray<id<MTLBuffer>>* bs,uint32_t n,uint32_t repeats,uint32_t threads){id<MTLCommandBuffer>c=[q commandBuffer];id<MTLComputeCommandEncoder>e=[c computeCommandEncoder];[e setComputePipelineState:p];for(NSUInteger i=0;i<bs.count;++i)[e setBuffer:bs[i] offset:0 atIndex:i];[e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];for(uint32_t r=1;r<repeats;++r)[e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];[e endEncoding];[c commit];[c waitUntilCompleted];if(c.status==MTLCommandBufferStatusError){std::fprintf(stderr,"Metal: %s\n",[[c.error.localizedDescription description] UTF8String]);std::exit(1);}return(c.GPUEndTime-c.GPUStartTime)*1000.0/double(repeats);}
static double timed(id<MTLCommandQueue> q,id<MTLComputePipelineState> p,NSArray<id<MTLBuffer>>* bs,uint32_t n,uint32_t repeats){return timed(q,p,bs,n,repeats,active_threads);}
static double timed_segments(id<MTLCommandQueue> q,id<MTLComputePipelineState> p,NSArray<id<MTLBuffer>>* bs,const std::vector<uint32_t>& cuts,uint32_t repeats,uint32_t threads){id<MTLCommandBuffer>c=[q commandBuffer];id<MTLComputeCommandEncoder>e=[c computeCommandEncoder];[e setComputePipelineState:p];for(uint32_t r=0;r<repeats;++r)for(size_t s=0;s+1<cuts.size();++s){const NSUInteger offset=NSUInteger(cuts[s])*sizeof(float);for(NSUInteger i=0;i<4;++i)[e setBuffer:bs[i] offset:offset atIndex:i];[e setBuffer:bs[4] offset:0 atIndex:4];[e dispatchThreads:MTLSizeMake(cuts[s+1]-cuts[s],1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];}[e endEncoding];[c commit];[c waitUntilCompleted];if(c.status==MTLCommandBufferStatusError){std::fprintf(stderr,"Metal: %s\n",[[c.error.localizedDescription description] UTF8String]);std::exit(1);}return(c.GPUEndTime-c.GPUStartTime)*1000.0/double(repeats);}
static double timed_segments(id<MTLCommandQueue> q,id<MTLComputePipelineState> p,NSArray<id<MTLBuffer>>* bs,const std::vector<uint32_t>& cuts,uint32_t repeats){return timed_segments(q,p,bs,cuts,repeats,active_threads);}
static void cpu_one(std::vector<float>& p,const std::vector<float>& g,std::vector<float>& m,std::vector<float>& v,const Params& q){for(size_t i=0;i<p.size();++i){m[i]=q.beta1*m[i]+(1-q.beta1)*g[i];v[i]=q.beta2*v[i]+(1-q.beta2)*g[i]*g[i];p[i]*=1-q.lr*q.wd;p[i]-=q.step_size*(m[i]/(std::sqrt(v[i])/q.beta2_scale+q.eps));}}
static Params adamw_params(uint32_t step){const float lr=1e-3f,b1=.9f,b2=.999f;return Params{lr,b1,b2,1e-8f,.01f,float(double(lr)/(1.0-std::pow(double(b1),double(step)))),float(std::sqrt(1.0-std::pow(double(b2),double(step))))};}
static bool dump(const char* prefix,const char* suffix,const float* data,size_t n){std::string path=std::string(prefix)+suffix;FILE*f=std::fopen(path.c_str(),"wb");return f&&std::fwrite(data,sizeof(float),n,f)==n&&!std::fclose(f);}
int main(int argc,const char* argv[]){@autoreleasepool{Args a;if(!parse(argc,argv,a)){std::fprintf(stderr,"usage: --input I --hidden H --output-size O --degree D --output file [--warmup W --repeats R --step S --verify-steps K --dispatches 1|4 --dump-prefix path]\n");return 2;}const uint32_t up=a.hidden*a.degree,down=a.hidden*a.degree,total=up+down+a.hidden+a.output;std::vector<float> p(total),g(total),m(total,0),v(total,0);for(uint32_t i=0;i<total;++i){p[i]=float(int(i%31)-15)*0.007f;g[i]=float(int((i*11u)%37u)-18)*0.003f;}Params q=adamw_params(a.step);id<MTLDevice>d=MTLCreateSystemDefaultDevice();if(!d)return 1;NSError*err=nil;id<MTLLibrary>lib=[d newLibraryWithSource:[NSString stringWithUTF8String:source().c_str()] options:nil error:&err];if(!lib){std::fprintf(stderr,"MSL: %s\n",[[err.localizedDescription description] UTF8String]);return 1;}id<MTLComputePipelineState>pipe=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"adamw_compact"] error:&err];if(!pipe){std::fprintf(stderr,"pipeline: %s\n",[[err.localizedDescription description] UTF8String]);return 1;}id<MTLBuffer>pb=buffer(d,p.data(),total*4u),gb=buffer(d,g.data(),total*4u),mb=buffer(d,m.data(),total*4u),vb=buffer(d,v.data(),total*4u),qb=buffer(d,&q,sizeof(q));NSArray<id<MTLBuffer>>*bs=@[pb,gb,mb,vb,qb];id<MTLCommandQueue>queue=[d newCommandQueue];const std::vector<uint32_t>cuts{0,up,up+down,up+down+a.hidden,total};auto measure=[&](uint32_t repeats){return a.dispatches==1?timed(queue,pipe,bs,total,repeats):timed_segments(queue,pipe,bs,cuts,repeats);};std::vector<float> refp=p,refm=m,refv=v;for(uint32_t k=0;k<a.verify_steps;++k){q=adamw_params(a.step+k);std::memcpy(qb.contents,&q,sizeof(q));cpu_one(refp,g,refm,refv,q);measure(1);}float*op=(float*)pb.contents;float*om=(float*)mb.contents;float*ov=(float*)vb.contents;double error=0;for(uint32_t i=0;i<total;++i)error=std::max(error,std::max(std::abs(double(op[i]-refp[i])),std::max(std::abs(double(om[i]-refm[i])),std::abs(double(ov[i]-refv[i])))));q=adamw_params(a.step);std::memcpy(qb.contents,&q,sizeof(q));std::memcpy(pb.contents,p.data(),total*4u);std::memcpy(mb.contents,m.data(),total*4u);std::memcpy(vb.contents,v.data(),total*4u);measure(a.warmup);std::vector<double>samples;for(int i=0;i<7;++i)samples.push_back(measure(a.repeats));std::sort(samples.begin(),samples.end());op=(float*)pb.contents;om=(float*)mb.contents;ov=(float*)vb.contents;if(a.dump_prefix&&(!dump(a.dump_prefix,"_p.f32",op,total)||!dump(a.dump_prefix,"_m.f32",om,total)||!dump(a.dump_prefix,"_v.f32",ov,total))){std::fprintf(stderr,"dump failed\n");return 1;}const uint64_t compact=uint64_t(total)*16u,dense_slots=uint64_t(a.hidden)*(a.input+a.output)+a.hidden+a.output,dense=dense_slots*16u;NSString*json=[NSString stringWithFormat:@"{\n  \"kind\": \"hffn_metal_compact_adamw\",\n  \"dimensions\": {\"input\": %u, \"hidden\": %u, \"output\": %u, \"degree\": %u},\n  \"active_slots\": {\"up\": %u, \"down\": %u, \"b_up\": %u, \"b_down\": %u, \"total\": %u},\n  \"adamw\": {\"lr\": %.9g, \"beta1\": %.9g, \"beta2\": %.9g, \"eps\": %.9g, \"weight_decay\": %.9g, \"step\": %u},\n  \"memory_bytes\": {\"compact_p_g_m_v\": %llu, \"dense_iso_support_p_g_m_v\": %llu},\n  \"warmup\": %u, \"repeats\": %u, \"verify_steps\": %u, \"dispatches\": %u,\n  \"median_gpu_ms\": %.9f,\n  \"max_abs_error_vs_cpu\": %.9g\n}\n",a.input,a.hidden,a.output,a.degree,up,down,a.hidden,a.output,total,q.lr,q.beta1,q.beta2,q.eps,q.wd,a.step,compact,dense,a.warmup,a.repeats,a.verify_steps,a.dispatches,samples[3],error];if(![json writeToFile:[NSString stringWithUTF8String:a.path] atomically:YES encoding:NSUTF8StringEncoding error:&err]){std::fprintf(stderr,"write: %s\n",[[err.localizedDescription description] UTF8String]);return 1;}std::printf("%s",json.UTF8String);}}
