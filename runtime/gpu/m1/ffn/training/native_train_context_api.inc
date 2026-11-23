static thread_local std::string api_error;
template<class Function, class Result>
static Result api_call(Function &&function, Result failure) {
  @autoreleasepool {
    @try {
      try { Result result=function();api_error.clear();return result; }
      catch(const std::exception &error) { api_error=error.what(); }
    } @catch(NSException *error) {
      api_error=error.reason ? error.reason.UTF8String : "Objective-C exception";
    }
    return failure;
  }
}
extern "C" const char *hffn_train_error(void) { return api_error.c_str(); }
extern "C" HffnTrainContext *hffn_train_create(int argc, const char *const *argv,
                                               const char *bundle) {
  return api_call([&]() -> HffnTrainContext * {
    if (!bundle || argc < 0 || (argc && !argv)) throw std::invalid_argument("invalid construction arguments");
    std::vector<const char *> args={"hffn_native_context"};
    for (int i=0;i<argc;++i) {
      if (!argv[i]) throw std::invalid_argument("null argument");
      if (!std::strcmp(argv[i],"--support-stdin") || !std::strcmp(argv[i],"--output") || !std::strcmp(argv[i],"--dump-prefix") ||
          !std::strcmp(argv[i],"--profile-phases") || !std::strcmp(argv[i],"--verify-steps") ||
          !std::strcmp(argv[i],"--untimed") || !std::strcmp(argv[i],"--warmup") || !std::strcmp(argv[i],"--repeats")) throw std::invalid_argument("CLI-only option in context constructor");
      args.push_back(argv[i]);
    }
    args.push_back("--output");args.push_back("unused-context-output");
    HffnTrainContext *context=nullptr;
    const int code=prepare_and_run(int(args.size()),args.data(),&context,bundle);
    if(code || !context)throw std::runtime_error("context construction rejected (exit "+std::to_string(code)+")");
    return context;
  },static_cast<HffnTrainContext *>(nullptr));
}
extern "C" int hffn_train_step(HffnTrainContext *context) {
  return api_call([&] {
    if(!context)throw std::invalid_argument("null context");
    // Unified exception handling in execute() poisons a GPU-failed context.
    context->step();return 0;
  },1);
}
extern "C" int hffn_train_bind_flat_parameters(HffnTrainContext *context, void *metal_buffer,
                                               uint64_t byte_offset, uint64_t count) {
  return api_call([&] {
    if(!context || !metal_buffer)throw std::invalid_argument("null context/buffer");
    context->bind_flat_parameters((__bridge id<MTLBuffer>)metal_buffer,byte_offset,count);
    return 0;
  },1);
}
extern "C" int hffn_train_bind_activations(HffnTrainContext *context, uint32_t slots,
    void *const *x_buffers, const uint64_t *x_offsets, void *const *gy_buffers, const uint64_t *gy_offsets,
    void *gx_buffer, uint64_t gx_offset) {
  return api_call([&] {
    if(!context)throw std::invalid_argument("null context");
    std::vector<id<MTLBuffer>> x, gy;std::vector<uint64_t> xo, gyo;
    if(x_buffers){
      if(!x_offsets)throw std::invalid_argument("null x offsets");
      for(uint32_t k=0;k<slots;++k){
        if(!x_buffers[k])throw std::invalid_argument("null x buffer");
        x.push_back((__bridge id<MTLBuffer>)x_buffers[k]);xo.push_back(x_offsets[k]);
      }
    }
    if(gy_buffers){
      if(!gy_offsets)throw std::invalid_argument("null gy offsets");
      for(uint32_t k=0;k<slots;++k){
        if(!gy_buffers[k])throw std::invalid_argument("null gy buffer");
        gy.push_back((__bridge id<MTLBuffer>)gy_buffers[k]);gyo.push_back(gy_offsets[k]);
      }
    }
    context->bind_activations(x,xo,gy,gyo,(__bridge id<MTLBuffer>)gx_buffer,gx_offset);
    return 0;
  },1);
}
extern "C" int hffn_train_stage_begin(HffnTrainContext *context) {
  return api_call([&] { if(!context)throw std::invalid_argument("null context"); context->staged_begin(); return 0; },1);
}
extern "C" int hffn_train_set_update_scalars(HffnTrainContext *context,const float *values) {
  return api_call([&] { if(!context)throw std::invalid_argument("null context"); context->set_update_scalars(values); return 0; },1);
}
extern "C" int hffn_train_enable_caller_managed_step(HffnTrainContext *context) {
  return api_call([&] { if(!context)throw std::invalid_argument("null context"); context->enable_caller_managed_step(); return 0; },1);
}
extern "C" int hffn_train_restore_optimizer(HffnTrainContext *context,const float *first,const float *second,uint64_t count,uint32_t completed) {
  return api_call([&] { if(!context)throw std::invalid_argument("null context"); context->restore_optimizer(first,second,count,completed); return 0; },1);
}
extern "C" int hffn_train_stage_optimizer_begin(HffnTrainContext *context) {
  return api_call([&] { if(!context)throw std::invalid_argument("null context"); context->staged_optimizer_begin(); return 0; },1);
}
extern "C" int hffn_train_stage_forward(HffnTrainContext *context,uint32_t slot) {
  return api_call([&] { if(!context)throw std::invalid_argument("null context"); context->staged_forward(slot); return 0; },1);
}
extern "C" int hffn_train_stage_backward(HffnTrainContext *context,uint32_t slot) {
  return api_call([&] { if(!context)throw std::invalid_argument("null context"); context->staged_backward(slot); return 0; },1);
}
extern "C" int hffn_train_stage_update(HffnTrainContext *context) {
  return api_call([&] { if(!context)throw std::invalid_argument("null context"); context->update(); return 0; },1);
}
extern "C" int hffn_train_stage_clear_gradients(HffnTrainContext *context) {
  return api_call([&] { if(!context)throw std::invalid_argument("null context"); context->clear_gradients(); return 0; },1);
}
extern "C" int hffn_train_stage_view(HffnTrainContext *context,const char *name,uint32_t slot,HffnTrainView *view) {
  return api_call([&] {
    if(!context || !name || !view)throw std::invalid_argument("null context/name/view");
    if(context->state==HffnTrainContext::State::failed)throw std::invalid_argument("context failed");
    const auto state=context->state;
    if(!std::strcmp(name,"g_up") || !std::strcmp(name,"g_down") || !std::strcmp(name,"g_bup") || !std::strcmp(name,"g_bdown")) {
      if(state!=HffnTrainContext::State::accumulating || context->next_micro!=context->config.accum_steps)
        throw std::invalid_argument("gradient view requires completed accumulation");
      *view=context->view(name,slot);return 0;
    }
    if(std::strcmp(name,"x") && std::strcmp(name,"gy") && std::strcmp(name,"y") && std::strcmp(name,"gx"))
      throw std::invalid_argument("not a staged activation view");
    if(!std::strcmp(name,"x") && (state!=HffnTrainContext::State::accumulating || slot!=context->next_micro))
      throw std::invalid_argument("x view requires next forward");
    if((!std::strcmp(name,"gy") || !std::strcmp(name,"y")) &&
       (state!=HffnTrainContext::State::forward_ready || slot!=context->next_micro))
      throw std::invalid_argument("view requires completed forward");
    if(!std::strcmp(name,"gx") && (state!=HffnTrainContext::State::accumulating ||
       !context->next_micro || slot+uint64_t(1)!=context->next_micro))
      throw std::invalid_argument("gx view requires completed backward");
    *view=context->view(name,slot);return 0;
  },1);
}
static HffnTrainContext *create_exact_impl(int argc,const char *const *argv,
    const char *bundle,const HffnTrainSupport *support,void *external,
    uint64_t offset,uint64_t count) {
  return api_call([&]() -> HffnTrainContext * {
    if(!support || !bundle || argc<0 || (argc && !argv))
      throw std::invalid_argument("invalid exact construction arguments");
    // Validate descriptor before parsing configuration or allocating on GPU.
    NativeExactSupport checked;checked.load(*support);
    std::vector<std::string> owned={"hffn_native_exact_context"};
    for(int i=0;i<argc;++i){
      if(!argv[i])throw std::invalid_argument("null configuration argument");
      const std::string value=argv[i];
      for(const char *option:{"--input","--hidden","--output-size","--degree","--support-stdin",
          "--output","--dump-prefix","--profile-phases","--verify-steps","--untimed","--warmup","--repeats"})
        if(value==option)throw std::invalid_argument("descriptor dimension or CLI-only exact constructor option");
      owned.push_back(value);
    }
    owned.insert(owned.end(),{"--input",std::to_string(checked.input),"--hidden",std::to_string(checked.hidden),
        "--output-size",std::to_string(checked.output),"--degree","1","--output","unused-context-output"});
    std::vector<const char*> args;for(const auto &value:owned)args.push_back(value.c_str());
    HffnTrainContext *context=nullptr;
    const int code=prepare_and_run(int(args.size()),args.data(),&context,bundle,support,external,offset,count);
    if(code || !context)throw std::runtime_error("exact context construction rejected (exit "+std::to_string(code)+")");
    return context;
  },static_cast<HffnTrainContext *>(nullptr));
}
extern "C" HffnTrainContext *hffn_train_create_exact(int argc,const char *const *argv,
    const char *bundle,const HffnTrainSupport *support) {
  return create_exact_impl(argc,argv,bundle,support,nullptr,0,0);
}
extern "C" HffnTrainContext *hffn_train_create_exact_with_parameters(int argc,const char *const *argv,
    const char *bundle,const HffnTrainSupport *support,void *external,uint64_t offset,uint64_t count) {
  if (!external) { api_error="null external parameter buffer"; return nullptr; }
  return create_exact_impl(argc,argv,bundle,support,external,offset,count);
}
extern "C" int hffn_train_view(HffnTrainContext *context,const char *name,uint32_t slot,HffnTrainView *view) {
  return api_call([&] {
    if(!context || !view)throw std::invalid_argument("null context/view");
    if(context->state!=HffnTrainContext::State::ready)throw std::invalid_argument("view requires ready context");
    *view=context->view(name,slot);return 0;
  },1);
}
extern "C" int hffn_train_reset(HffnTrainContext *context,const float *values,uint64_t count) {
  return api_call([&] {
    if(!context)throw std::invalid_argument("null context");
    context->reset(values,count);return 0;
  },1);
}
extern "C" uint32_t hffn_train_completed_updates(HffnTrainContext *context) { return context ? context->completed : 0; }
extern "C" void hffn_train_destroy(HffnTrainContext *context) { delete context; }
extern "C" int hffn_training_cli(int argc,const char *argv[]) {
  const int result=api_call([&] {return prepare_and_run(argc,argv,nullptr,nullptr);},1);
  if(result && !api_error.empty())std::fprintf(stderr,"native training: %s\n",api_error.c_str());
  return result;
}
