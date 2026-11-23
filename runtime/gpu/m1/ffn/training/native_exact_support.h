#pragma once
#include <set>
#include <stdexcept>
#include <cstdint>

// Cold validated HFFN descriptor/stream, copied into the one native owner.
// No converted export is written and no array is consulted in train_step.
struct NativeExactSupport {
  uint32_t input=0,hidden=0,output=0,degree=1;
  std::vector<uint32_t> up_offsets,down_offsets,up_idx,down_idx;
  std::vector<float> up_val,down_val,b_up,b_down;
  void validate() {
    if(!input || !hidden || !output || input>=INT32_MAX || hidden>=INT32_MAX || output>=INT32_MAX ||
       up_idx.empty() || down_idx.empty() || uint64_t(up_idx.size())+down_idx.size()+hidden+output>=uint64_t(INT32_MAX) ||
       up_offsets.size()!=size_t(hidden)+1 || down_offsets.size()!=size_t(hidden)+1 ||
       up_val.size()!=up_idx.size() || down_val.size()!=down_idx.size() || b_up.size()!=hidden || b_down.size()!=output)
      throw std::invalid_argument("invalid exact support extent");
    degree=1;
    auto check=[&](const auto &offsets,const auto &index,const auto &values,uint32_t bound){
      if(offsets.front()!=0 || offsets.back()!=index.size())throw std::invalid_argument("support endpoint");
      for(uint32_t r=0;r<hidden;++r){
        if(offsets[r]>offsets[r+1] || offsets[r+1]>index.size())throw std::invalid_argument("support offsets");
        degree=std::max(degree,offsets[r+1]-offsets[r]);
        std::set<uint32_t> seen;
        for(uint32_t p=offsets[r];p<offsets[r+1];++p)
          if(index[p]>=bound || !std::isfinite(values[p]) || !seen.insert(index[p]).second)
            throw std::invalid_argument("support index/duplicate/nonfinite");
      }
    };
    check(up_offsets,up_idx,up_val,input);check(down_offsets,down_idx,down_val,output);
    for(float value:b_up)if(!std::isfinite(value))throw std::invalid_argument("nonfinite hidden bias");
    for(float value:b_down)if(!std::isfinite(value))throw std::invalid_argument("nonfinite output bias");
    unsigned bits=1;while((uint64_t(1)<<bits)<degree)++bits;
    if((uint64_t(hidden-1)<<bits)+degree-1>UINT32_MAX)throw std::invalid_argument("route codec overflow");
  }
  bool uniform_shared_degree() const {
    for(uint32_t r=0;r<=hidden;++r)
      if(uint64_t(up_offsets[r])!=uint64_t(r)*degree || uint64_t(down_offsets[r])!=uint64_t(r)*degree)return false;
    return true;
  }
  template<class T> static void copy(std::vector<T> &out,HffnTrainSpan span,uint64_t count) {
    if(span.elements!=count || !span.data || reinterpret_cast<uintptr_t>(span.data)%alignof(T))
      throw std::invalid_argument("invalid/alignment exact support span");
    const T *begin=static_cast<const T*>(span.data);out.assign(begin,begin+count);
  }
  void load(const HffnTrainSupport &s) {
    if(s.version!=1 || !s.input || !s.hidden || !s.output || s.input>=INT32_MAX || s.hidden>=INT32_MAX || s.output>=INT32_MAX ||
       !s.up_idx.elements || !s.down_idx.elements || s.up_idx.elements>=INT32_MAX || s.down_idx.elements>=INT32_MAX ||
       s.up_idx.elements+s.down_idx.elements+s.hidden+s.output>=uint64_t(INT32_MAX))
      throw std::invalid_argument("invalid exact support descriptor");
    input=s.input;hidden=s.hidden;output=s.output;
    copy(up_offsets,s.up_offsets,uint64_t(hidden)+1);copy(down_offsets,s.down_offsets,uint64_t(hidden)+1);
    copy(up_idx,s.up_idx,s.up_idx.elements);copy(down_idx,s.down_idx,s.down_idx.elements);
    copy(up_val,s.up_val,s.up_idx.elements);copy(down_val,s.down_val,s.down_idx.elements);
    copy(b_up,s.b_up,hidden);copy(b_down,s.b_down,output);validate();
  }
  template<class T> static void read(std::vector<T> &out,size_t count){
    out.resize(count);
    if(std::fread(out.data(),sizeof(T),count,stdin)!=count)throw std::invalid_argument("truncated support transport");
  }
  void load(uint32_t i,uint32_t h,uint32_t o){
    uint32_t header[8];
    if(std::fread(header,4,8,stdin)!=8 || header[0]!=0x4846464eu || header[1]!=1 ||
       header[2]!=i || header[3]!=h || header[4]!=o || !header[5] || !header[6] || header[7]!=0 ||
       uint64_t(header[5])+header[6]+h+o>=uint64_t(INT32_MAX))throw std::invalid_argument("invalid support transport header");
    input=i;hidden=h;output=o;
    read(up_offsets,size_t(h)+1);read(down_offsets,size_t(h)+1);
    read(up_idx,header[5]);read(down_idx,header[6]);read(up_val,header[5]);read(down_val,header[6]);read(b_up,h);read(b_down,o);
    if(std::fgetc(stdin)!=EOF)throw std::invalid_argument("trailing support transport bytes");
    validate();
  }
};

static void exact_replace(std::string& s,const std::string& old,const std::string& value) {
  size_t pos=0;
  while((pos=s.find(old,pos))!=std::string::npos){s.replace(pos,old.size(),value);pos+=value.size();}
}

// Reuse the kernel bodies and reductions. Only slot/hidden-row
// address expressions, compact segment boundaries and cold bindings change.
static std::string exact_support_source(std::string s,uint32_t ue,uint32_t de) {
  const std::string extra="constant uint UCOUNT="+std::to_string(ue)+"u,DCOUNT="+std::to_string(de)+
    "u;\ninline uint slot(Route x,device const uint*offsets){return offsets[uint(x)>>EB]+(uint(x)&EM);}\n";
  const size_t insertion=s.find("inline uint slot(");
  if(insertion==std::string::npos)throw std::runtime_error("native slot generator changed");
  s.insert(insertion,extra);
  const std::map<std::string,unsigned> bindings={
    {"bdown",7},{"bhidden",6},{"binput",8},
    {"bdown_owner_tile",6},{"bhidden_owner_tile",6},{"binput_owner_tile",8},
    {"bdown_bias_owner_tile",6},{"bdown_bhidden_fused_owner_tile",7},
    {"binput_owner_tile_fused1",8}};
  size_t pos=0;
  while((pos=s.find("kernel void ",pos))!=std::string::npos){
    const size_t name_start=pos+12, paren=s.find('(',name_start), brace=s.find('{',paren);
    const std::string name=s.substr(name_start,paren-name_start);
    int depth=0;size_t end=brace;
    for(;end<s.size();++end){if(s[end]=='{')++depth;else if(s[end]=='}' && --depth==0){++end;break;}}
    std::string kernel=s.substr(pos,end-pos);
    auto entry=bindings.find(name);
    if(entry!=bindings.end()){
      const size_t builtin=kernel.find(",uint",kernel.find("[[buffer("));
      if(builtin==std::string::npos)throw std::runtime_error("native kernel arguments changed");
      kernel.insert(builtin,",device const uint*support_offsets[[buffer("+std::to_string(entry->second)+")]]");
      exact_replace(kernel,"slot(route[q])","slot(route[q],support_offsets)");
      exact_replace(kernel,"for(uint e=0;e<D;++e){uint p=r*D+e,o=target[p];",
                           "for(uint p=support_offsets[r];p<support_offsets[r+1u];++p){uint o=target[p];");
      exact_replace(kernel,"for(uint e=0;e<D;++e){uint p=r*D+e;",
                           "for(uint p=support_offsets[r];p<support_offsets[r+1u];++p){");
      exact_replace(kernel,"2u*F*D","(UCOUNT+DCOUNT)");
      exact_replace(kernel,"F*D+p","UCOUNT+p");
    } else if(name=="reduce_flat_partials") {
      exact_replace(kernel,"2u*F*D","(UCOUNT+DCOUNT)");
    }
    s.replace(pos,end-pos,kernel);pos+=kernel.size();
  }
  return s;
}
