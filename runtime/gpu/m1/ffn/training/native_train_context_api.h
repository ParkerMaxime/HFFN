#ifndef HFFN_NATIVE_TRAIN_CONTEXT_API_H
#define HFFN_NATIVE_TRAIN_CONTEXT_API_H
#include <stdint.h>
#define HFFN_TRAIN_API __attribute__((visibility("default")))
#ifdef __cplusplus
extern "C" {
#endif
typedef struct HffnTrainContext HffnTrainContext;
typedef struct {
  void *metal_buffer;  /* Borrowed; valid until context destruction. */
  void *data;          /* NULL for Private storage. */
  uint64_t elements;
  uint64_t byte_offset;
  uint32_t storage_mode;
} HffnTrainView;
/* Cold descriptor; all arrays are contiguous, aligned, native-endian.
 * Offsets/indices are uint32_t, values/biases float32. The caller owns valid
 * spans until create_exact returns; the context retains its own GPU storage.
 */
typedef struct { const void *data; uint64_t elements; } HffnTrainSpan;
typedef struct {
  uint32_t version, input, hidden, output;
  HffnTrainSpan up_offsets, down_offsets, up_idx, down_idx;
  HffnTrainSpan up_val, down_val, b_up, b_down;
} HffnTrainSupport;
/* Legacy homogeneous fixture/configuration constructor. */
HFFN_TRAIN_API HffnTrainContext *hffn_train_create(int argc, const char *const *argv,
                                   const char *inference_bundle);
/* Descriptor dimensions are authoritative; configuration argv contains only
 * execution/storage selectors (N,K,...), never dimensions/degree/CLI options.
 */
HFFN_TRAIN_API HffnTrainContext *hffn_train_create_exact(int argc, const char *const *argv,
                                  const char *inference_bundle,
                                  const HffnTrainSupport *support);
HFFN_TRAIN_API HffnTrainContext *hffn_train_create_exact_with_parameters(
    int argc, const char *const *argv, const char *inference_bundle,
    const HffnTrainSupport *support, void *metal_buffer, uint64_t byte_offset, uint64_t count);
HFFN_TRAIN_API int hffn_train_step(HffnTrainContext *context);
/* Cold binding (opt-in): caller retains the tensor, context retains the
 * Metal buffer, and producer/consumer queues must be synchronized externally. */
HFFN_TRAIN_API int hffn_train_bind_flat_parameters(HffnTrainContext *context, void *metal_buffer,
                                                  uint64_t byte_offset, uint64_t count);
/* Cold, once per kind, in state ready: the staged tables read caller-owned Metal buffers (borrowed handles, byte offsets)
 * instead of the native x / gy / gx. x_buffers / gy_buffers are arrays of `slots` handles (slots == accumulation steps) or
 * NULL to keep that kind native; gx_buffer may be NULL. The caller synchronises its own queue before the first stage and
 * after the last and keeps the buffers alive and unmoved until destruction. No unbind: a new context returns to native. */
HFFN_TRAIN_API int hffn_train_bind_activations(HffnTrainContext *context, uint32_t slots,
    void *const *x_buffers, const uint64_t *x_offsets, void *const *gy_buffers, const uint64_t *gy_offsets,
    void *gx_buffer, uint64_t gx_offset);
/* Staged synchronous interface; call begin, then forward/backward for each
 * microbatch in order, then update. Each call handles the full N-token block.
 * Views are borrowed and remain valid only until context destruction. */
HFFN_TRAIN_API int hffn_train_stage_begin(HffnTrainContext *context);
/* Set the seven float32 AdamW scalars of the coming update, legal only during accumulation (after stage_begin, before
 * stage_update); begin() rewrites them from the native constants at the next step, so a call acts for one update only.
 * values[0..6] = lr, beta1, beta2, eps, weight decay, step_size, beta2_scale, with step_size = lr / (1 - beta1^t) and
 * beta2_scale = sqrt(1 - beta2^t) computed by the caller for its own step t. The call does not validate the values: a
 * non-finite or out-of-range scalar yields a non-finite or meaningless update, the caller's responsibility. Returns nonzero
 * with hffn_train_error() set on a null context, a null pointer or a call outside accumulation; nothing is written then. */
HFFN_TRAIN_API int hffn_train_set_update_scalars(HffnTrainContext *context, const float *values);
/* Cold, irreversible opt-in of the caller-managed correction mode (no disable: a new context returns to the default mode). Legal only
 * in the ready state; a second call in that state is accepted and changes nothing. In this mode:
 *  - the caller's own step counter is the authority for the bias corrections: before EVERY hffn_train_stage_update it supplies the
 *    seven scalars of that update through hffn_train_set_update_scalars (after the begin of that update; the last valid call wins);
 *  - stage_begin / stage_optimizer_begin neither read the native counter (no exhaustion refusal) nor write native scalars; they
 *    clear the per-update marker, which only a valid setter call of the same update sets (a call before begin, with null values or
 *    on a null context is rejected and sets nothing);
 *  - stage_update without that marker is rejected with hffn_train_error() set: nothing is written, the state stays accumulating, the
 *    gradients stay in the pool and the caller may still supply the scalars and call stage_update again;
 *  - hffn_train_completed_updates is diagnostic only: it counts updates and saturates at 0xFFFFFFFF, it never feeds a correction and
 *    is not the consumer's step; restore_optimizer still writes it (any uint32 value);
 *  - hffn_train_step (the non-staged step) is rejected, because it gives the caller no window to supply the scalars.
 * Synchronisation and aliases are those of the setter and of the staged entries (synchronous stages, host writes of Shared scalar
 * buffers, values copied before any write). Returns 0, or non-zero with hffn_train_error() set on a null context or a context that is
 * not ready; nothing is changed then. The default mode (no call) is unchanged. */
HFFN_TRAIN_API int hffn_train_enable_caller_managed_step(HffnTrainContext *context);
/* Cold restore: two caller-owned readable host spans of count floats, packed in up/down/bup/bdown order.
 * Caller synchronizes producers; no source span may overlap parameter, gradient or moment buffers. The context must be ready; all
 * validation precedes the first write and the step counter is written last, so a rejection changes nothing. Non-finite moment bits are
 * copied unchanged. Safety against the GPU rests on every stage of the context being synchronous: no command of this context is in
 * flight when the call runs. A consumer that reads the moment views from its own asynchronous queue must synchronize that queue
 * itself (caller's responsibility). Parameters are not restored by this call (use hffn_train_reset first). */
HFFN_TRAIN_API int hffn_train_restore_optimizer(HffnTrainContext *context, const float *first,
                                               const float *second, uint64_t count, uint32_t completed);
/* Begin an optimizer-only step: staged_begin() (same refusals: externally bound pool via bind_flat_parameters, context not ready,
 * counter exhausted) then next_micro = accum_steps, so that no micro-batch runs. Contract of the caller:
 *  - overwrite EVERY compact gradient (g_up, g_down, g_bup, g_bdown, through hffn_train_stage_view) after this call and before
 *    hffn_train_stage_update; the update is then the unchanged one (bitwise equal to the update of a context that ran its own
 *    backward and had the same gradient written over its pool);
 *  - a group the caller does not write, or writes partly, is updated with whatever the pool holds (zeros after a previous update);
 *    the entry detects no missing or stale gradient;
 *  - the gradient views are valid only after this call (the stage view refuses before the accumulation is complete) and are cleared by the update;
 *  - with accum_steps > 1 the entry skips the micro-batches by construction: the gradient written by the caller is the whole
 *    accumulated gradient of the step;
 *  - qualified for the flat and the segmented AdamW pool and for a pool supplied at creation; not qualified together with
 *    set_update_scalars, stage_clear_gradients, a forward after the entry, or --private-readonly.
 * Returns 0, or non-zero with hffn_train_error() set on a null context or a call outside the ready state; nothing is changed then. */
HFFN_TRAIN_API int hffn_train_stage_optimizer_begin(HffnTrainContext *context);
HFFN_TRAIN_API int hffn_train_stage_forward(HffnTrainContext *context, uint32_t slot);
HFFN_TRAIN_API int hffn_train_stage_backward(HffnTrainContext *context, uint32_t slot);
HFFN_TRAIN_API int hffn_train_stage_update(HffnTrainContext *context);
/* After the last backward stage the compact gradient pool is readable through hffn_train_stage_view("g_up", "g_down",
 * "g_bup", "g_bdown"; borrowed pointers, no copy). A caller that consumes it with its own optimiser then calls
 * hffn_train_stage_clear_gradients (pool zeroed, state ready, no native update counted); the caller has synchronised. */
HFFN_TRAIN_API int hffn_train_stage_clear_gradients(HffnTrainContext *context);
HFFN_TRAIN_API int hffn_train_stage_view(HffnTrainContext *context, const char *name,
                                        uint32_t slot, HffnTrainView *view);
HFFN_TRAIN_API int hffn_train_view(HffnTrainContext *context, const char *name, uint32_t slot,
                    HffnTrainView *view);
HFFN_TRAIN_API int hffn_train_reset(HffnTrainContext *context, const float *parameters,
                     uint64_t count);
HFFN_TRAIN_API uint32_t hffn_train_completed_updates(HffnTrainContext *context);
HFFN_TRAIN_API const char *hffn_train_error(void);
HFFN_TRAIN_API void hffn_train_destroy(HffnTrainContext *context);
HFFN_TRAIN_API int hffn_training_cli(int argc, const char *argv[]);
#ifdef __cplusplus
}
#endif
#endif
