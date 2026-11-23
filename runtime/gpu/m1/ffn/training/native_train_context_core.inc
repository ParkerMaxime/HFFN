// Included once in the native implementation, after the unchanged helpers.
// This context is the execution owner for both the public API and CLI.
struct HffnTrainContext {
  Args config;
  id<MTLCommandQueue> queue;
  NSArray *pipelines, *bindings, *grids;
  NSArray *verify_ps, *verify_bs, *verify_gs, *update_ps, *update_bs, *update_gs;
  NSArray *forward_ps, *forward_bs, *forward_gs;
  NSArray *backward_ps, *backward_bs, *backward_gs;
  NSArray *micro_bindings;
  NSDictionary *buffers;
  std::unique_ptr<NativeDiagnostics> diagnostics;
  id<MTLBuffer> parameters[4], gradients[4], first[4], second[4], scalars[4];
  size_t offsets[4], counts[4];
  uint32_t total, completed = 0, next_micro = 0;
  enum class State { ready, accumulating, forward_ready, failed } state = State::ready;
  // Declared after every member of the control's layout, so that no field of the hot path changes its offset.
  id<MTLBuffer> external_parameters = nil;
  size_t external_parameter_offset = 0;
  // Staged-path bindings, declared last so that no member of the control's layout moves (audit V2).
  NSArray *micro_forward_bindings, *micro_backward_bindings;
  // Caller-owned activation buffers, declared after every earlier member.
  NSArray *external_x = nil, *external_gy = nil;
  id<MTLBuffer> external_gx = nil;
  std::vector<size_t> external_x_offsets, external_gy_offsets;
  size_t external_gx_offset = 0;
  // Set only by bind_flat_parameters: that entry does not remap the staged tables, whereas a pool
  // supplied at creation is in every table already. The staged refusals test this member, not external_parameters.
  bool pool_bound_late = false;
  // Caller-managed correction mode, declared last: set once by enable_caller_managed_step, never
  // cleared. scalars_ready is cleared by begin() and set by set_update_scalars(); it is read only in that mode.
  bool caller_managed_step = false, scalars_ready = false;

  id<MTLBuffer> buffer(NSString *name) const { return buffers[name]; }
  float *part(id<MTLBuffer> const *group, unsigned i) const {
    return static_cast<float *>(group[i].contents) + offsets[i] +
        (group[i] == external_parameters ? external_parameter_offset : 0);
  }
  HffnTrainContext(const Args &a, id<MTLCommandQueue> q, NSArray *ps,
      NSArray *bs, NSArray *gs, NSDictionary *owners,
      std::unique_ptr<NativeDiagnostics> trace)
      : config(a), queue(q), pipelines(ps), bindings(bs), grids(gs),
        buffers(owners), diagnostics(std::move(trace)) {
    config.path = config.dump_prefix = nullptr;
    const uint32_t E = a.up_count, ED = a.down_count;
    total = E + ED + a.hidden + a.output;
    const char *pnames[] = {"upb", "db", "bub", "bdb"};
    const char *gnames[] = {"gub", "gdb", "gbub", "gbdb"};
    const char *mnames[] = {"umb", "dmb", "bumb", "bdmb"};
    const char *vnames[] = {"uvb", "dvb", "buvb", "bdvb"};
    const char *qnames[] = {"pub", "pdb", "pbub", "pbdb"};
    const size_t flat_offsets[] = {0, E, E + ED, E + ED + a.hidden};
    for (unsigned i = 0; i < 4; ++i) {
      counts[i] = i == 0 ? E : (i == 1 ? ED : (i == 2 ? a.hidden : a.output));
      offsets[i] = a.flat_adamw_pool ? flat_offsets[i] : 0;
      parameters[i] = buffer(a.flat_adamw_pool ? @"flat_pb" : @(pnames[i]));
      gradients[i] = buffer(a.flat_adamw_pool ? @"flat_gb" : @(gnames[i]));
      first[i] = buffer(a.flat_adamw_pool ? @"flat_mb" : @(mnames[i]));
      second[i] = buffer(a.flat_adamw_pool ? @"flat_vb" : @(vnames[i]));
      scalars[i] = buffer(@(qnames[i]));
    }
    const NSUInteger forward_count = a.fdown_group_private ? 3u : 2u;
    const NSUInteger verify_count = a.tiled_n ? 9 :
        (a.owner_tile ? (a.binput_separate ? 7 : 6) +
                            (a.fdown_group_private ? 1u : 0u) : 5);
    if(ps.count>bs.count || ps.count>gs.count ||
       ps.count!=verify_count+(a.flat_adamw_pool?1:4))
      throw std::invalid_argument("inconsistent legacy stage selection");
    auto prefix = [](NSArray *array, NSUInteger start, NSUInteger size) {
      return [array subarrayWithRange:NSMakeRange(start, size)];
    };
    verify_ps = prefix(ps, 0, verify_count);
    verify_bs = prefix(bs, 0, verify_count);
    verify_gs = prefix(gs, 0, verify_count);
    update_ps = prefix(ps, verify_count, a.flat_adamw_pool ? 1 : 4);
    update_bs = prefix(bs, verify_count, a.flat_adamw_pool ? 1 : 4);
    update_gs = prefix(gs, verify_count, a.flat_adamw_pool ? 1 : 4);
    forward_ps = prefix(ps, 0, forward_count);
    forward_bs = prefix(bs, 0, forward_count);
    forward_gs = prefix(gs, 0, forward_count);
    backward_ps = prefix(ps, forward_count, verify_count - forward_count);
    backward_bs = prefix(bs, forward_count, verify_count - forward_count);
    backward_gs = prefix(gs, forward_count, verify_count - forward_count);
    NSMutableArray *all_micro = [NSMutableArray array];
    NSMutableArray *all_forward = [NSMutableArray array], *all_backward = [NSMutableArray array];
    NSArray *xs = buffers[@"x_micro"], *gys = buffers[@"gy_micro"];
    auto replace = [](NSMutableArray *all, NSUInteger stage, NSUInteger index, id value) {
      NSMutableArray *row = [all[stage] mutableCopy];
      row[index] = value;
      all[stage] = [row copy];
    };
    for (uint32_t k = 0; k < a.accum_steps; ++k) {
      NSMutableArray *all = [verify_bs mutableCopy];
      replace(all, 0, 5, xs[k]);
      if (a.tiled_n) {
        replace(all, 2, 1, gys[k]); replace(all, 5, 1, gys[k]);
        replace(all, 7, 0, xs[k]);
      } else {
        const NSUInteger shift = forward_count - 2u;
        const NSUInteger gy_index = a.owner_tile && a.bdown_fused &&
            a.owner_tile_n == 128u ? 2u : 1u;
        replace(all, 2u + shift, 1, gys[k]);
        replace(all, 3u + shift, gy_index, gys[k]);
        replace(all, 4u + shift, 0, xs[k]);
      }
      [all_micro addObject:[all copy]];
      [all_forward addObject:prefix(all, 0, forward_count)];
      [all_backward addObject:prefix(all, forward_count, verify_count - forward_count)];
    }
    micro_bindings = [all_micro copy];
    micro_forward_bindings = [all_forward copy];
    micro_backward_bindings = [all_backward copy];
  }
  double execute(NSArray *ps, NSArray *bs, NSArray *gs, uint32_t repeats = 1) {
    if (state == State::failed) throw std::runtime_error("context failed; reconstruct it");
    try { return run(queue, ps, bs, gs, repeats, *diagnostics); }
    catch (...) { state = State::failed; throw; }
  }
  void begin() {
    if (state != State::ready) throw std::runtime_error("context is not ready");
    // Caller-managed mode: no native scalar is written and the counter is not read; the setter must arm this update.
    if (caller_managed_step) { scalars_ready = false; next_micro = 0; state = State::accumulating; return; }
    if (completed == UINT32_MAX) throw std::overflow_error("AdamW update counter exhausted");
    for (unsigned i = 0; i < 4; ++i) {
      AdamWParams value = adamw_params(uint32_t(counts[i]), completed + 1);
      std::memcpy(scalars[i].contents, &value, sizeof(value));
    }
    if (config.flat_adamw_pool) {
      AdamWParams value = adamw_params(total, completed + 1);
      std::memcpy(buffer(@"flat_qb").contents, &value, sizeof(value));
    }
    next_micro = 0;
    state = State::accumulating;
  }
  void set_update_scalars(const float *values) {
    if (state != State::accumulating) throw std::invalid_argument("invalid scalar setter order");
    if (!values) throw std::invalid_argument("null scalar values");
    float q[7];
    std::memcpy(q, values, sizeof(q));
    void *dest[5] = {};
    for (unsigned i = 0; i < 4; ++i) {
      dest[i] = scalars[i].contents;
      if (!dest[i]) throw std::runtime_error("scalar buffer is not host accessible");
    }
    if (config.flat_adamw_pool) {
      dest[4] = buffer(@"flat_qb").contents;
      if (!dest[4]) throw std::runtime_error("flat scalar buffer is not host accessible");
    }
    static_assert(sizeof(AdamWParams) == sizeof(q), "unexpected scalar layout");
    for (void *dst : dest) if (dst) std::memcpy(dst, q, sizeof(q));
    scalars_ready = true;
  }
  // Cold, irreversible: the caller supplies the seven scalars of every update; begin() no longer rejects on the native counter.
  void enable_caller_managed_step() {
    if (state != State::ready) throw std::invalid_argument("caller-managed step mode requires a ready context");
    caller_managed_step = true;
  }
  double accumulate(uint32_t k) {
    if (state != State::accumulating || k != next_micro || k >= config.accum_steps)
      throw std::invalid_argument("invalid accumulation order");
    double result = execute(verify_ps, micro_bindings[k], verify_gs);
    ++next_micro;
    return result;
  }
  // The staged bindings are not remapped by bind_flat_parameters: refuse a pool bound that way (fail closed).
  void staged_begin() {
    if (pool_bound_late) throw std::runtime_error("staged stages are not qualified with an external parameter pool");
    begin();
  }
  void restore_optimizer(const float *m, const float *v, uint64_t count, uint32_t step) {
    if (state != State::ready) throw std::invalid_argument("restore requires ready context");
    if (!m || !v || count != total) throw std::invalid_argument("invalid optimizer state spans");
    const uint64_t bytes = count * sizeof(float);
    const uintptr_t ms = reinterpret_cast<uintptr_t>(m), vs = reinterpret_cast<uintptr_t>(v);
    if (ms > UINTPTR_MAX - bytes || vs > UINTPTR_MAX - bytes)
      throw std::invalid_argument("optimizer state address overflow");
    for (unsigned i = 0; i < 4; ++i) {
      if (first[i].storageMode == MTLStorageModePrivate || second[i].storageMode == MTLStorageModePrivate ||
          !first[i].contents || !second[i].contents)
        throw std::invalid_argument("optimizer moments are not host accessible");
      id<MTLBuffer> groups[] = {parameters[i], gradients[i], first[i], second[i]};
      for (id<MTLBuffer> b : groups) {
        if (b.storageMode == MTLStorageModePrivate) continue;
        const uintptr_t start = reinterpret_cast<uintptr_t>(b.contents);
        if ((ms < start + b.length && start < ms + bytes) ||
            (vs < start + b.length && start < vs + bytes))
          throw std::invalid_argument("optimizer source aliases context state");
      }
    }
    uint64_t at = 0;
    for (unsigned i = 0; i < 4; ++i) {
      std::memcpy(part(first, i), m + at, counts[i] * sizeof(float));
      std::memcpy(part(second, i), v + at, counts[i] * sizeof(float));
      at += counts[i];
    }
    completed = step;
  }
  // Caller supplies every compact gradient after begin; use the existing update pipeline.
  void staged_optimizer_begin() {
    staged_begin();
    next_micro = config.accum_steps;
  }
  double staged_forward(uint32_t k) {
    if (state != State::accumulating || k != next_micro || k >= config.accum_steps)
      throw std::invalid_argument("invalid staged forward order");
    double result = execute(forward_ps, micro_forward_bindings[k], forward_gs);
    state = State::forward_ready;
    return result;
  }
  double staged_backward(uint32_t k) {
    if (state != State::forward_ready || k != next_micro || k >= config.accum_steps)
      throw std::invalid_argument("invalid staged backward order");
    double result = execute(backward_ps, micro_backward_bindings[k], backward_gs);
    ++next_micro;
    state = State::accumulating;
    return result;
  }
  double update() {
    if (state != State::accumulating || next_micro != config.accum_steps)
      throw std::invalid_argument("incomplete gradient accumulation");
    if (caller_managed_step && !scalars_ready)
      throw std::invalid_argument("caller-managed step: update scalars not supplied for this update");
    double result = execute(update_ps, update_bs, update_gs);
    // Caller-managed mode: the native counter is diagnostic only and saturates at UINT32_MAX.
    if (caller_managed_step) { completed += completed != UINT32_MAX; state = State::ready; return result; }
    ++completed;
    state = State::ready;
    return result;
  }
  double step() {
    if (caller_managed_step) throw std::invalid_argument("caller-managed step mode requires the staged entries");
    begin();
    if (config.accum_steps == 1) {
      double result = execute(pipelines, bindings, grids);
      ++completed;
      state = State::ready;
      return result;
    }
    double result = 0;
    for (uint32_t k = 0; k < config.accum_steps; ++k) result += accumulate(k);
    return result + update();
  }
  // Caller-owned optimiser path: the caller has read the finished gradient pool through the stage view and consumed it; this
  // clears exactly the pool counts (host memset on the Shared pool, valid because the stages are synchronous) and returns to
  // `ready` without counting a native update.
  void clear_gradients() {
    if (state == State::ready) throw std::invalid_argument("nothing to clear");
    if (state != State::accumulating || next_micro != config.accum_steps)
      throw std::invalid_argument("incomplete gradient accumulation");
    for (unsigned i = 0; i < 4; ++i)
      if (gradients[i].storageMode == MTLStorageModePrivate || !gradients[i].contents)
        throw std::invalid_argument("private gradient pool is not covered");
    for (unsigned i = 0; i < 4; ++i) std::memset(part(gradients, i), 0, counts[i] * sizeof(float));
    state = State::ready;
  }
  void bind_flat_parameters(id<MTLBuffer> external, uint64_t byte_offset, uint64_t count) {
    if (state != State::ready || completed || external_parameters)
      throw std::invalid_argument("binding requires a fresh ready context");
    if (!config.flat_adamw_pool || !external || external.device != queue.device)
      throw std::invalid_argument("incompatible parameter buffer or device");
    if (count != total || byte_offset % sizeof(float) || byte_offset > external.length ||
        uint64_t(count) * sizeof(float) > external.length - byte_offset)
      throw std::invalid_argument("invalid parameter buffer extent");
    id<MTLBuffer> old = parameters[0];
    auto remap = [&](NSArray *rows) -> NSArray * {
      NSMutableArray *out = [NSMutableArray arrayWithCapacity:rows.count];
      for (NSArray *row in rows) {
        NSMutableArray *entries = [NSMutableArray arrayWithCapacity:row.count];
        for (id entry in row) {
          if ([entry isKindOfClass:[NSValue class]]) {
            BufferBinding original;
            [(NSValue *)entry getValue:&original];
            if ((__bridge id<MTLBuffer>)original.buffer == old) {
              [entries addObject:(::view(external, NSUInteger(byte_offset) + original.offset))];
              continue;
            }
          } else if (entry == old) {
            [entries addObject:(::view(external, NSUInteger(byte_offset)))];
            continue;
          }
          [entries addObject:entry];
        }
        [out addObject:[entries copy]];
      }
      return [out copy];
    };
    NSArray *new_bindings=remap(bindings), *new_verify=remap(verify_bs),
            *new_update=remap(update_bs), *new_forward=remap(forward_bs),
            *new_backward=remap(backward_bs);
    NSMutableArray *new_micro=[NSMutableArray arrayWithCapacity:micro_bindings.count];
    for (NSArray *micro in micro_bindings) [new_micro addObject:remap(micro)];
    bindings=new_bindings; verify_bs=new_verify; update_bs=new_update;
    forward_bs=new_forward; backward_bs=new_backward; micro_bindings=[new_micro copy];
    external_parameters=external;
    pool_bound_late=true;
    external_parameter_offset=byte_offset/sizeof(float);
    for (unsigned i=0;i<4;++i) parameters[i]=external;
  }
  // Cold, once per kind: the staged tables read caller-owned buffers (at byte offsets) instead of the native x / gy / gx.
  // An empty vector keeps that kind native. Every table that holds a native buffer of a bound kind is rebuilt by buffer
  // identity (as bind_flat_parameters does); staged_*, begin, update, run and enc are untouched. Ordering with the
  // caller's own queue is the caller's contract (synchronise before the first stage and after the last).
  void bind_activations(const std::vector<id<MTLBuffer>> &x, const std::vector<uint64_t> &x_off,
                        const std::vector<id<MTLBuffer>> &gy, const std::vector<uint64_t> &gy_off,
                        id<MTLBuffer> gx, uint64_t gx_off) {
    if (state != State::ready) throw std::invalid_argument("activation binding requires a ready context");
    if (pool_bound_late) throw std::invalid_argument("activation binding is not qualified with an external parameter pool");
    if (x.empty() && gy.empty() && !gx) throw std::invalid_argument("nothing to bind");
    if ((!x.empty() && external_x != nil) || (!gy.empty() && external_gy != nil) || (gx && external_gx != nil))
      throw std::invalid_argument("activation buffers are already bound");
    if (!x.empty() && (x.size() != config.accum_steps || x_off.size() != x.size()))
      throw std::invalid_argument("x binding needs one buffer per accumulation slot");
    if (!gy.empty() && (gy.size() != config.accum_steps || gy_off.size() != gy.size()))
      throw std::invalid_argument("gy binding needs one buffer per accumulation slot");
    NSArray *native_x = buffers[@"x_micro"], *native_gy = buffers[@"gy_micro"];
    // Only the regimes the parity covers (audit R2): a bound buffer replaces a native one by identity, so a variant whose
    // kernels do not write every gx element (tiled_n) or read x at another table index is refused, not trusted.
    if (config.tiled_n || config.persistent_n || config.owner_tile_n != 128u ||
        config.binput_separate || config.binput_gup_hidden || config.bdown_gup_fused || config.fdown_group_private ||
        config.fdown_segmented_route || config.fdown_output_tile4 || config.packet_binput_staged ||
        config.packet_binput_staged64 ||
        config.n >= 8192u)
      throw std::invalid_argument("activation binding is not qualified with this configuration");
    struct Range { id<MTLBuffer> buffer; uint64_t offset, length; };
    std::vector<Range> ranges;
    // Ranges bound by an earlier call count too (audit R1): the overlap rule holds across calls.
    for (size_t k = 0; k < external_x.count; ++k) ranges.push_back({external_x[k], external_x_offsets[k], [native_x[k] length]});
    for (size_t k = 0; k < external_gy.count; ++k) ranges.push_back({external_gy[k], external_gy_offsets[k], [native_gy[k] length]});
    if (external_gx) ranges.push_back({external_gx, external_gx_offset, buffer(@"gxb").length});
    struct Swap { id<MTLBuffer> old, now; uint64_t offset; size_t hits; };
    std::vector<Swap> swaps;
    auto accept = [&](id<MTLBuffer> external, uint64_t offset, id<MTLBuffer> native, const char *what) {
      if (!external || !native) throw std::invalid_argument(std::string("null ")+what+" buffer");
      if (external.device != queue.device) throw std::invalid_argument(std::string(what)+" buffer is on another device");
      const uint64_t need = native.length;
      if (offset % sizeof(float)) throw std::invalid_argument(std::string(what)+" offset is not 4-byte aligned");
      if (offset > external.length || need > external.length - offset)
        throw std::invalid_argument(std::string(what)+" buffer is too small for its extent");
      for (id key in buffers) {
        id value = buffers[key];
        NSArray *list = [value isKindOfClass:[NSArray class]] ? value : @[ value ];
        for (id item in list) if (item == external) throw std::invalid_argument(std::string(what)+" buffer is a native context buffer");
      }
      // Overlap by byte range: one buffer object (a heap-backed tensor view), or two Shared objects over one memory
      // span. Two distinct Private objects over one allocation cannot be compared (stated limit).
      for (const Range &r : ranges) {
        bool overlap = false;
        if (r.buffer == external) overlap = offset < r.offset + r.length && r.offset < offset + need;
        else if (external.storageMode != MTLStorageModePrivate && r.buffer.storageMode != MTLStorageModePrivate) {
          const uintptr_t lo = uintptr_t(external.contents) + offset, rlo = uintptr_t(r.buffer.contents) + r.offset;
          overlap = lo < rlo + r.length && rlo < lo + need;
        }
        if (overlap) throw std::invalid_argument(std::string(what)+" range overlaps another bound range");
      }
      ranges.push_back({external, offset, need});
    };
    NSMutableArray *kept_x = [NSMutableArray array], *kept_gy = [NSMutableArray array];
    for (size_t k = 0; k < x.size(); ++k) {
      accept(x[k], x_off[k], native_x[k], "x");
      swaps.push_back({native_x[k], x[k], x_off[k], 0}); [kept_x addObject:x[k]];
    }
    for (size_t k = 0; k < gy.size(); ++k) {
      accept(gy[k], gy_off[k], native_gy[k], "gy");
      swaps.push_back({native_gy[k], gy[k], gy_off[k], 0}); [kept_gy addObject:gy[k]];
    }
    if (gx) { accept(gx, gx_off, buffer(@"gxb"), "gx"); swaps.push_back({buffer(@"gxb"), gx, gx_off, 0}); }
    auto remap = [&](NSArray *rows) -> NSArray * {
      NSMutableArray *out = [NSMutableArray arrayWithCapacity:rows.count];
      for (NSArray *row in rows) {
        NSMutableArray *entries = [NSMutableArray arrayWithCapacity:row.count];
        for (id entry in row) {
          bool replaced = false;
          for (Swap &swap : swaps) {
            if ([entry isKindOfClass:[NSValue class]]) {
              BufferBinding original;
              [(NSValue *)entry getValue:&original];
              if ((__bridge id<MTLBuffer>)original.buffer == swap.old) {
                [entries addObject:(::view(swap.now, NSUInteger(swap.offset) + original.offset))];
                ++swap.hits; replaced = true; break;
              }
            } else if (entry == swap.old) {
              [entries addObject:(::view(swap.now, NSUInteger(swap.offset)))];
              ++swap.hits; replaced = true; break;
            }
          }
          if (!replaced) [entries addObject:entry];
        }
        [out addObject:[entries copy]];
      }
      return [out copy];
    };
    auto remap_all = [&](NSArray *tables) {
      NSMutableArray *out = [NSMutableArray arrayWithCapacity:tables.count];
      for (NSArray *table in tables) [out addObject:remap(table)];
      return [out copy];
    };
    NSArray *new_bindings = remap(bindings), *new_verify = remap(verify_bs), *new_update = remap(update_bs),
            *new_forward = remap(forward_bs), *new_backward = remap(backward_bs),
            *new_micro = remap_all(micro_bindings), *new_micro_forward = remap_all(micro_forward_bindings),
            *new_micro_backward = remap_all(micro_backward_bindings);
    // Fail closed: a bound buffer that no table reads would be a missed substitution.
    for (const Swap &swap : swaps)
      if (!swap.hits) throw std::runtime_error("bound activation buffer is not read by any staged table");
    bindings = new_bindings; verify_bs = new_verify; update_bs = new_update;
    forward_bs = new_forward; backward_bs = new_backward; micro_bindings = new_micro;
    micro_forward_bindings = new_micro_forward; micro_backward_bindings = new_micro_backward;
    if (!x.empty()) { external_x = [kept_x copy]; external_x_offsets.assign(x_off.begin(), x_off.end()); }
    if (!gy.empty()) { external_gy = [kept_gy copy]; external_gy_offsets.assign(gy_off.begin(), gy_off.end()); }
    if (gx) { external_gx = gx; external_gx_offset = gx_off; }
  }
  void reset(const float *packed, uint64_t count) {
    if (state != State::ready) throw std::invalid_argument("reset requires ready context");
    if (!packed || count != total) throw std::invalid_argument("invalid reset parameter count");
    // Metal Private storage is decided by storageMode, never by `contents`: on this machine most Private
    // MPS buffers report a non-null `contents` that must not be dereferenced.
    const bool private_parameters = parameters[0].storageMode == MTLStorageModePrivate;
    if (private_parameters && !external_parameters)
      throw std::invalid_argument("private reset requires a bound external parameter pool");
    if (!private_parameters && !parameters[0].contents)
      throw std::invalid_argument("parameter buffer has no CPU view");
    const uintptr_t source = reinterpret_cast<uintptr_t>(packed);
    if (source > UINTPTR_MAX - count * sizeof(float)) throw std::invalid_argument("reset address overflow");
    const uintptr_t end = source + count * sizeof(float);
    // No source alias with live native state: reject before clearing anything.
    for (unsigned i = 0; i < 4; ++i) {
      id<MTLBuffer> groups[] = {parameters[i], gradients[i], first[i], second[i]};
      for (id<MTLBuffer> b : groups) {
        if (b.storageMode == MTLStorageModePrivate) continue;
        uintptr_t start = reinterpret_cast<uintptr_t>(b.contents);
        if (source < start + b.length && start < end)
          throw std::invalid_argument("reset source aliases context state");
      }
    }
    for (uint64_t i = 0; i < count; ++i)
      if (!std::isfinite(packed[i])) throw std::invalid_argument("nonfinite reset parameter");
    if (private_parameters) {
      // Stage exactly the compact vector, blit it into the caller's buffer at its view offset and wait.
      // The caller has synchronized its own queue before this call and does not use the tensor meanwhile.
      id<MTLBuffer> staging = [queue.device newBufferWithBytes:packed length:count*sizeof(float)
                                                       options:MTLResourceStorageModeShared];
      if (!staging) throw std::runtime_error("reset staging allocation failed");
      id<MTLCommandBuffer> command = [queue commandBuffer];
      if (!command) throw std::runtime_error("reset command buffer creation failed");
      id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
      if (!encoder) throw std::runtime_error("reset blit encoder creation failed");
      // A submitted transfer may partially write; poison the context on any failure.
      state = State::failed;
      [encoder copyFromBuffer:staging sourceOffset:0 toBuffer:parameters[0]
            destinationOffset:external_parameter_offset*sizeof(float) size:count*sizeof(float)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error("private parameter reset transfer failed");
      state = State::ready;
    }
    size_t at = 0;
    for (unsigned i = 0; i < 4; ++i) {
      if (!private_parameters) std::memcpy(part(parameters, i), packed + at, counts[i] * sizeof(float));
      std::memset(part(gradients, i), 0, counts[i] * sizeof(float));
      std::memset(part(first, i), 0, counts[i] * sizeof(float));
      std::memset(part(second, i), 0, counts[i] * sizeof(float));
      at += counts[i];
    }
    completed = next_micro = 0;
  }
  HffnTrainView view(const char *name, uint32_t slot) const {
    if (!name) throw std::invalid_argument("missing view name");
    id<MTLBuffer> b = nil;
    size_t offset = 0, count = 0;
    const char *names[] = {"up", "down", "bup", "bdown"};
    for (unsigned i = 0; i < 4; ++i) {
      if (!std::strcmp(name, names[i])) b = parameters[i];
      else if (std::string(name) == std::string("g_") + names[i]) b = gradients[i];
      else if (std::string(name) == std::string("m_") + names[i]) b = first[i];
      else if (std::string(name) == std::string("v_") + names[i]) b = second[i];
      if (b) {
        offset = (offsets[i] + (b == external_parameters ? external_parameter_offset : 0)) * sizeof(float);
        count = counts[i]; break;
      }
    }
    if (!b && (!std::strcmp(name, "x") || !std::strcmp(name, "gy"))) {
      if (slot >= config.accum_steps) throw std::invalid_argument("input slot out of range");
      const bool is_x = !std::strcmp(name, "x");
      NSArray *all = buffers[is_x ? @"x_micro" : @"gy_micro"];
      b = all[slot]; count = b.length / sizeof(float);
      NSArray *bound = is_x ? external_x : external_gy;
      if (bound != nil) { b = bound[slot]; offset = (is_x ? external_x_offsets : external_gy_offsets)[slot]; }
    }
    if (!b) {
      const char *views[] = {"z", "h", "y", "gz", "gx"};
      const char *keys[] = {"zb", "hb", "yb", "gzb", "gxb"};
      for (unsigned i = 0; i < 5; ++i) if (!std::strcmp(name, views[i])) {
        b = buffer(@(keys[i])); count = b.length / sizeof(float);
        if (i == 4 && external_gx != nil) { b = external_gx; offset = external_gx_offset; }
        break;
      }
    }
    if (!b) throw std::invalid_argument("unknown state view");
    void *host = b.storageMode == MTLStorageModePrivate ? nullptr :
        static_cast<char *>(b.contents) + offset;
    return {(__bridge void *)b, host, count, offset, uint32_t(b.storageMode)};
  }
};
