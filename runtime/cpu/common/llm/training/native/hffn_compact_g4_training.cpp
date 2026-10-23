// Compact hidden-major HFFN training kernel with fixed persistent G4 workers.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "g4_contract.h"

namespace {
inline float gelu_grad(float x) { const float c=0.5f*(1.0f+std::erf(x*0.7071067811865475f)); return c+x*std::exp(-0.5f*x*x)*0.3989422804014327f; }
inline int64_t now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
inline void nt_to_t(const float* src,float* dst,int n,int h) { for(int b=0;b<n;++b) for(int i=0;i<h;++i) dst[static_cast<size_t>(i)*n+b]=src[static_cast<size_t>(b)*h+i]; }
inline void t_to_nt(const float* src,float* dst,int n,int h) { for(int b=0;b<n;++b) for(int i=0;i<h;++i) dst[static_cast<size_t>(b)*h+i]=src[static_cast<size_t>(i)*n+b]; }
inline int tile_for(int n) { return std::min(n,128); }
inline void validate(const int32_t* groups,int n,int h,int f) { if(n<=0||h<=0||f<=0||groups[0]!=0||groups[kG4Groups]!=f) throw std::runtime_error("invalid compact G4 dimensions"); for(int g=0;g<kG4Groups;++g) if(groups[g]>groups[g+1]) throw std::runtime_error("non-monotonic compact G4 groups"); }

// Inert worker-record layouts preserve the offsets and generated code of the
// retained backward/input path; no forward/output method or dispatch remains.
struct FJob { const float* x; const int32_t* groups; const float* bup; const int32_t* upoff; const int16_t* upidx; const float* upval; const int32_t* downoff; const int16_t* downidx; const float* downval; int n,h,begin,end; float* z; float* act; float* ylocal; bool write_down; };
struct OJob { const float* act; const int32_t* offsets; const int32_t* slots; const int16_t* hidden; const float* down; const float* bdown; const int32_t* groups; int n,h; float* yt; };
struct BJob { const float* x; const float* gy; const float* z; const float* act; const int32_t* groups; const int32_t* upoff; const int16_t* upidx; const float* upval; const int32_t* downoff; const int16_t* downidx; const float* downval; int n,h,begin,end; float* gxlocal; float* gz; float* gup; float* gdown; float* gbup; bool write_gx; bool compute_up; };
// Reverse schedules reference the same hidden-major slots as the canonical
// HFFN storage.  They only change output/input ownership for the two global
// reductions; they do not duplicate weights, gradients or optimizer state.
struct XJob { const float* x; const float* gz; const int32_t* offsets; const int32_t* slots; const int16_t* hidden; const float* up; const int32_t* groups; int n,h; float* gxt; float* gup; };
template <int kBatch>
inline void xgroup_fixed(const XJob& j,int g) { for(int in=j.groups[g];in<j.groups[g+1];++in){float* dst=j.gxt+static_cast<size_t>(in)*kBatch;const float* x=j.x+static_cast<size_t>(in)*kBatch;for(int b=0;b<kBatch;++b)dst[b]=0.0f;for(int q=j.offsets[in];q<j.offsets[in+1];++q){const float* z=j.gz+static_cast<size_t>(j.hidden[q])*kBatch;const float w=j.up[j.slots[q]];float dot=0.0f;for(int b=0;b<kBatch;++b){dst[b]+=z[b]*w;dot+=z[b]*x[b];}j.gup[j.slots[q]]+=dot;}} }
inline void xgroup(const XJob& j,int g) { if(j.n==8){xgroup_fixed<8>(j,g);return;}if(j.n==32){xgroup_fixed<32>(j,g);return;}if(j.n==64){xgroup_fixed<64>(j,g);return;}for(int in=j.groups[g];in<j.groups[g+1];++in){float* dst=j.gxt+static_cast<size_t>(in)*j.n;const float* x=j.x+static_cast<size_t>(in)*j.n;for(int b=0;b<j.n;++b)dst[b]=0.0f;for(int q=j.offsets[in];q<j.offsets[in+1];++q){const float* z=j.gz+static_cast<size_t>(j.hidden[q])*j.n;const float w=j.up[j.slots[q]];float dot=0.0f;for(int b=0;b<j.n;++b){dst[b]+=z[b]*w;dot+=z[b]*x[b];}j.gup[j.slots[q]]+=dot;}} }
inline void bgroup8(const BJob& j,int g) {
    for(int hidden=j.groups[g];hidden<j.groups[g+1];++hidden){const int us=j.upoff[hidden],ue=j.upoff[hidden+1],ds=j.downoff[hidden],de=j.downoff[hidden+1];float gh0=0,gh1=0,gh2=0,gh3=0,gh4=0,gh5=0,gh6=0,gh7=0;
        for(int p=ds;p<de;++p){const float* gy=j.gy+static_cast<size_t>(j.downidx[p])*8;const float* a=j.act+static_cast<size_t>(hidden)*8;const float w=j.downval[p];j.gdown[p]+=a[0]*gy[0]+a[1]*gy[1]+a[2]*gy[2]+a[3]*gy[3]+a[4]*gy[4]+a[5]*gy[5]+a[6]*gy[6]+a[7]*gy[7];gh0+=w*gy[0];gh1+=w*gy[1];gh2+=w*gy[2];gh3+=w*gy[3];gh4+=w*gy[4];gh5+=w*gy[5];gh6+=w*gy[6];gh7+=w*gy[7];}
        const float* z=j.z+static_cast<size_t>(hidden)*8;float* gz=j.gz+static_cast<size_t>(hidden)*8;gz[0]=gh0*gelu_grad(z[0]);gz[1]=gh1*gelu_grad(z[1]);gz[2]=gh2*gelu_grad(z[2]);gz[3]=gh3*gelu_grad(z[3]);gz[4]=gh4*gelu_grad(z[4]);gz[5]=gh5*gelu_grad(z[5]);gz[6]=gh6*gelu_grad(z[6]);gz[7]=gh7*gelu_grad(z[7]);j.gbup[hidden]+=gz[0]+gz[1]+gz[2]+gz[3]+gz[4]+gz[5]+gz[6]+gz[7];
        if(j.compute_up) for(int p=us;p<ue;++p){const int in=j.upidx[p];const float* x=j.x+static_cast<size_t>(in)*8;const float w=j.upval[p];j.gup[p]+=gz[0]*x[0]+gz[1]*x[1]+gz[2]*x[2]+gz[3]*x[3]+gz[4]*x[4]+gz[5]*x[5]+gz[6]*x[6]+gz[7]*x[7];if(j.write_gx){float* out=j.gxlocal+(static_cast<size_t>(g)*j.h+in)*8;out[0]+=gz[0]*w;out[1]+=gz[1]*w;out[2]+=gz[2]*w;out[3]+=gz[3]*w;out[4]+=gz[4]*w;out[5]+=gz[5]*w;out[6]+=gz[6]*w;out[7]+=gz[7]*w;}}
    }
}
inline void bgroup(const BJob& j,int g) {
    if(j.n==8&&j.begin==0&&j.end==8){bgroup8(j,g);return;}
    for(int hidden=j.groups[g];hidden<j.groups[g+1];++hidden) {
        const int us=j.upoff[hidden],ue=j.upoff[hidden+1],ds=j.downoff[hidden],de=j.downoff[hidden+1]; float bgrad=0.0f;
        for(int b=j.begin;b<j.end;++b) { float gh=0.0f; for(int p=ds;p<de;++p){const float v=j.gy[static_cast<size_t>(j.downidx[p])*j.n+b];j.gdown[p]+=j.act[static_cast<size_t>(hidden)*j.n+b]*v;gh+=j.downval[p]*v;} const float gz=gh*gelu_grad(j.z[static_cast<size_t>(hidden)*j.n+b]);j.gz[static_cast<size_t>(hidden)*j.n+b]=gz;bgrad+=gz;if(j.compute_up)for(int p=us;p<ue;++p){const int in=j.upidx[p];j.gup[p]+=gz*j.x[static_cast<size_t>(in)*j.n+b];if(j.write_gx)j.gxlocal[(static_cast<size_t>(g)*j.h+in)*j.n+b]+=gz*j.upval[p];}}
        j.gbup[hidden]+=bgrad;
    }
}
// One pool owns the fixed G4 group assignment for backward and input-gradient work.
// Only three native worker threads exist; group 0 runs on the calling thread.
class G4Pool {
public:
    G4Pool(){for(int g=1;g<kG4Groups;++g) workers[g-1]=std::thread(&G4Pool::loop,this,g);}
    ~G4Pool(){{std::lock_guard<std::mutex>l(mu);stop=true;}start.notify_all();for(auto&t:workers)if(t.joinable())t.join();}
    void backward(const BJob& job){run_backward(job);}
    void input(const XJob& job){run_input(job);}
private:
    void run_backward(const BJob& job){{std::lock_guard<std::mutex>l(mu);bcurrent=job;kind=Kind::Backward;done=0;++generation;}start.notify_all();bgroup(job,0);wait();}
    void run_input(const XJob& job){{std::lock_guard<std::mutex>l(mu);xcurrent=job;kind=Kind::Input;done=0;++generation;}start.notify_all();xgroup(job,0);wait();}
    void wait(){std::unique_lock<std::mutex>l(mu);finish.wait(l,[&]{return done==kG4Groups-1;});}
    enum class Kind { Forward, Backward, Output, Input };
    void loop(int g){uint64_t seen=0;for(;;){Kind local_kind;BJob bj{};XJob xj{};{std::unique_lock<std::mutex>l(mu);start.wait(l,[&]{return stop||generation!=seen;});if(stop)return;seen=generation;local_kind=kind;if(local_kind==Kind::Backward)bj=bcurrent;else xj=xcurrent;}if(local_kind==Kind::Backward)bgroup(bj,g);else xgroup(xj,g);{std::lock_guard<std::mutex>l(mu);++done;}finish.notify_one();}}
    std::mutex mu;std::condition_variable start,finish;std::thread workers[kG4Groups-1];FJob fcurrent{};BJob bcurrent{};OJob ocurrent{};XJob xcurrent{};Kind kind=Kind::Forward;uint64_t generation=0;int done=0;bool stop=false;
};
inline G4Pool& g4pool(){static G4Pool p;return p;}
}

extern "C" void hffn_compact_g4_backward_nt_f32(const float* gynt,const float* xt,const float* z,const float* act,const int32_t* groups,const int32_t* upoff,const int16_t* upidx,const float* upval,const int32_t* downoff,const int16_t* downidx,const float* downval,int n,int h,int f,float* gyt,float* gxnt,float* gxt,float* gxlocal,float* gz,float* gup,float* gdown,float* gbup,float* gbdown,int64_t* timing){
    validate(groups,n,h,f);const int64_t t0=now_ns();nt_to_t(gynt,gyt,n,h);const int64_t t1=now_ns();const int upnnz=upoff[f],downnnz=downoff[f];std::fill(gxt,gxt+static_cast<size_t>(h)*n,0.0f);std::fill(gxlocal,gxlocal+static_cast<size_t>(kG4Groups)*h*n,0.0f);std::fill(gup,gup+upnnz,0.0f);std::fill(gdown,gdown+downnnz,0.0f);std::fill(gbup,gbup+f,0.0f);for(int out=0;out<h;++out){float sum=0.0f;for(int b=0;b<n;++b)sum+=gyt[static_cast<size_t>(out)*n+b];gbdown[out]=sum;}BJob base{xt,gyt,z,act,groups,upoff,upidx,upval,downoff,downidx,downval,n,h,0,n,gxlocal,gz,gup,gdown,gbup,true,true};const int tile=tile_for(n);if(n<=2){for(int g=0;g<kG4Groups;++g)bgroup(base,g);}else if(tile>=n)g4pool().backward(base);else for(int begin=0;begin<n;begin+=tile){BJob c=base;c.begin=begin;c.end=std::min(begin+tile,n);g4pool().backward(c);}for(int in=0;in<h;++in)for(int b=0;b<n;++b){float v=0.0f;for(int g=0;g<kG4Groups;++g)v+=gxlocal[(static_cast<size_t>(g)*h+in)*n+b];gxt[static_cast<size_t>(in)*n+b]=v;}const int64_t t2=now_ns();t_to_nt(gxt,gxnt,n,h);const int64_t t3=now_ns();timing[0]=t1-t0;timing[1]=t2-t1;timing[2]=t3-t2;
}

extern "C" void hffn_compact_g4_backward_dual_nt_f32(const float* gynt,const float* xt,const float* z,const float* act,const int32_t* hidden_groups,const int32_t* upoff,const int16_t* upidx,const float* upval,const int32_t* downoff,const int16_t* downidx,const float* downval,const int32_t* up_in_offsets,const int32_t* up_slots,const int16_t* up_hidden,const int32_t* input_groups,int n,int h,int f,float* gyt,float* gxnt,float* gxt,float* gz,float* gup,float* gdown,float* gbup,float* gbdown,int64_t* timing){
    validate(hidden_groups,n,h,f);const int64_t t0=now_ns();nt_to_t(gynt,gyt,n,h);const int64_t t1=now_ns();const int upnnz=upoff[f],downnnz=downoff[f];std::fill(gup,gup+upnnz,0.0f);std::fill(gdown,gdown+downnnz,0.0f);std::fill(gbup,gbup+f,0.0f);for(int out=0;out<h;++out){float sum=0.0f;for(int b=0;b<n;++b)sum+=gyt[static_cast<size_t>(out)*n+b];gbdown[out]=sum;}BJob hidden{xt,gyt,z,act,hidden_groups,upoff,upidx,upval,downoff,downidx,downval,n,h,0,n,nullptr,gz,gup,gdown,gbup,false,false};const int tile=tile_for(n);if(n<=2){for(int g=0;g<kG4Groups;++g)bgroup(hidden,g);}else if(tile>=n)g4pool().backward(hidden);else for(int begin=0;begin<n;begin+=tile){BJob c=hidden;c.begin=begin;c.end=std::min(begin+tile,n);g4pool().backward(c);}const int64_t t2=now_ns();XJob input{xt,gz,up_in_offsets,up_slots,up_hidden,upval,input_groups,n,h,gxt,gup};g4pool().input(input);const int64_t t3=now_ns();t_to_nt(gxt,gxnt,n,h);const int64_t t4=now_ns();timing[0]=t1-t0;timing[1]=t2-t1;timing[2]=t3-t2;timing[3]=t4-t3;
}

// The compact post-attention transaction owns LN2 -> [H,N] directly, then
// executes exactly one persistent-G4 FFN dispatch.  It never materializes an
// [N,H] LN2 temporary or a transposed FFN output merely to transpose it back.
inline void ln2_backward_nt(const float* gy,const float* x,const float* gamma,const float* mean,const float* invstd,int n,int h,float* gx,float* ggamma,float* gbeta){
    std::fill(ggamma,ggamma+h,0.0f);std::fill(gbeta,gbeta+h,0.0f);
    for(int b=0;b<n;++b){const float* xb=x+static_cast<size_t>(b)*h;const float* yb=gy+static_cast<size_t>(b)*h;float* out=gx+static_cast<size_t>(b)*h;const float mu=mean[b],inv=invstd[b];float s1=0.0f,s2=0.0f;for(int i=0;i<h;++i){const float dx=yb[i]*gamma[i];s1+=dx;s2+=dx*(xb[i]-mu);ggamma[i]+=yb[i]*(xb[i]-mu)*inv;gbeta[i]+=yb[i];}const float invsq=inv*inv;for(int i=0;i<h;++i){const float dx=yb[i]*gamma[i];out[i]=inv*(dx-s1/h-(xb[i]-mu)*invsq*s2/h);}}
}

extern "C" void hffn_compact_g4_backward_post_attention_layernorm_nt_f32(const float* gout,const float* residual,const float* gamma,const float* mean,const float* invstd,const float* xt,const float* z,const float* act,const int32_t* groups,const int32_t* upoff,const int16_t* upidx,const float* upval,const int32_t* downoff,const int16_t* downidx,const float* downval,int n,int h,int f,float* gyt,float* gln2,float* gxlocal,float* gz,float* gup,float* gdown,float* gbup,float* gbdown,float* gresidual,float* gattn,float* ggamma,float* gbeta,int64_t* timing){
    validate(groups,n,h,f);const int64_t t0=now_ns();nt_to_t(gout,gyt,n,h);const int64_t t1=now_ns();const int upnnz=upoff[f],downnnz=downoff[f];std::fill(gxlocal,gxlocal+static_cast<size_t>(kG4Groups)*h*n,0.0f);std::fill(gup,gup+upnnz,0.0f);std::fill(gdown,gdown+downnnz,0.0f);std::fill(gbup,gbup+f,0.0f);for(int outi=0;outi<h;++outi){float sum=0.0f;for(int b=0;b<n;++b)sum+=gyt[static_cast<size_t>(outi)*n+b];gbdown[outi]=sum;}BJob base{xt,gyt,z,act,groups,upoff,upidx,upval,downoff,downidx,downval,n,h,0,n,gxlocal,gz,gup,gdown,gbup,true,true};const int tile=tile_for(n);if(n<=2){for(int g=0;g<kG4Groups;++g)bgroup(base,g);}else if(tile>=n)g4pool().backward(base);else for(int begin=0;begin<n;begin+=tile){BJob c=base;c.begin=begin;c.end=std::min(begin+tile,n);g4pool().backward(c);}const int64_t t2=now_ns();
    for(int b=0;b<n;++b)for(int in=0;in<h;++in){float v=0.0f;for(int g=0;g<kG4Groups;++g)v+=gxlocal[(static_cast<size_t>(g)*h+in)*n+b];gln2[static_cast<size_t>(b)*h+in]=v;}const int64_t t3=now_ns();ln2_backward_nt(gln2,residual,gamma,mean,invstd,n,h,gresidual,ggamma,gbeta);for(int i=0;i<n*h;++i){gresidual[i]+=gout[i];gattn[i]=gout[i];}const int64_t t4=now_ns();timing[0]=t1-t0;timing[1]=t2-t1;timing[2]=t3-t2;timing[3]=t4-t3;timing[4]=t4-t0;
}

extern "C" void hffn_compact_g4_adamw_f32(float* up,const float* gup,float* um,float* uv,int upnnz,float* down,const float* gdown,float* dm,float* dv,int downnnz,float* bup,const float* gbup,float* bum,float* buv,int f,float* bdown,const float* gbdown,float* bdm,float* bdv,int h,float lr,float beta1,float beta2,float eps,float wd,int64_t step){
    if(step<=0)throw std::runtime_error("invalid AdamW step");const float ss=static_cast<float>(static_cast<double>(lr)/(1.0-std::pow(static_cast<double>(beta1),static_cast<double>(step))));const float b2s=static_cast<float>(std::sqrt(1.0-std::pow(static_cast<double>(beta2),static_cast<double>(step))));auto one=[&](float&p,float g,float&m,float&v){m=beta1*m+(1-beta1)*g;v=beta2*v+(1-beta2)*g*g;if(wd)p*=1-lr*wd;p-=ss*(m/(std::sqrt(v)/b2s+eps));};for(int i=0;i<upnnz;++i)one(up[i],gup[i],um[i],uv[i]);for(int i=0;i<downnnz;++i)one(down[i],gdown[i],dm[i],dv[i]);for(int i=0;i<f;++i)one(bup[i],gbup[i],bum[i],buv[i]);for(int i=0;i<h;++i)one(bdown[i],gbdown[i],bdm[i],bdv[i]);
}

// Parameter gradients only: the existing hidden-owned computation, without
// clearing, writing, reducing or transposing any input-gradient buffer.
extern "C" void hffn_compact_g4_backward_weights_nt_f32(const float* gynt,const float* xt,const float* z,const float* act,const int32_t* groups,const int32_t* upoff,const int16_t* upidx,const float* upval,const int32_t* downoff,const int16_t* downidx,const float* downval,int n,int h,int f,float* gyt,float* gz,float* gup,float* gdown,float* gbup,float* gbdown,int64_t* timing){
    validate(groups,n,h,f);const int64_t t0=now_ns();nt_to_t(gynt,gyt,n,h);const int64_t t1=now_ns();const int upnnz=upoff[f],downnnz=downoff[f];std::fill(gup,gup+upnnz,0.0f);std::fill(gdown,gdown+downnnz,0.0f);std::fill(gbup,gbup+f,0.0f);for(int out=0;out<h;++out){float sum=0.0f;for(int b=0;b<n;++b)sum+=gyt[static_cast<size_t>(out)*n+b];gbdown[out]=sum;}BJob base{xt,gyt,z,act,groups,upoff,upidx,upval,downoff,downidx,downval,n,h,0,n,nullptr,gz,gup,gdown,gbup,false,true};const int tile=tile_for(n);if(n<=2){for(int g=0;g<kG4Groups;++g)bgroup(base,g);}else if(tile>=n)g4pool().backward(base);else for(int begin=0;begin<n;begin+=tile){BJob c=base;c.begin=begin;c.end=std::min(begin+tile,n);g4pool().backward(c);}const int64_t t2=now_ns();timing[0]=t1-t0;timing[1]=t2-t1;timing[2]=0;
}
