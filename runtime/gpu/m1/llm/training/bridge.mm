// native_ffn_adapter bridge (README.md of this leaf). Loaded only by
// native_ffn_adapter.py when the adapter is created. Three host services, nothing else:
//   buffer_info       the MTLBuffer handle, byte offset, storage mode and length behind a torch MPS tensor;
//   BlitQueue.copy    one persistent adapter queue; copies the four compact gradient views of the native
//                     context (borrowed handles, nonzero offsets) into grad_flat with one blit and a blocking
//                     wait (step 6). The F2 hold gates that blit on a persistent shared event;
//   increment_version the autograd version bump the native write bypasses (step 6).
// No kernel, no FFN arithmetic. Built without ARC: the installed PyTorch OperationUtils.h uses manual
// retain/release.
#include <torch/extension.h>
#include <torch/csrc/autograd/variable.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/native/mps/OperationUtils.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace {

id<MTLDevice> mps_device() {
  id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
  TORCH_CHECK(device != nil, "MPS device unavailable");
  return device;
}

// The Metal buffer of a contiguous float32 MPS tensor, with its extent checked against the buffer.
id<MTLBuffer> tensor_buffer(const torch::Tensor& value, const char* name) {
  TORCH_CHECK(value.device().is_mps(), name, ": expected an MPS tensor");
  TORCH_CHECK(value.scalar_type() == at::kFloat, name, ": expected float32");
  TORCH_CHECK(value.is_contiguous(), name, ": expected a contiguous tensor");
  TORCH_CHECK(value.numel() > 0, name, ": expected a non-empty tensor");
  id<MTLBuffer> buffer = at::native::mps::getMTLBufferStorage(value);
  TORCH_CHECK(buffer != nil, name, ": MPS tensor has no Metal buffer");
  TORCH_CHECK(buffer.device == mps_device(), name, ": buffer is not on the MPS device");
  const uint64_t offset = uint64_t(value.storage_offset()) * sizeof(float);
  const uint64_t length = uint64_t(value.numel()) * sizeof(float);
  TORCH_CHECK(offset <= buffer.length && length <= buffer.length - offset, name, ": view exceeds its buffer");
  return buffer;
}

std::string command_error(id<MTLCommandBuffer> command) {
  NSError* error = command.error;
  return error != nil ? std::string(error.localizedDescription.UTF8String) : std::string("no error description");
}

}  // namespace

// (handle, byte_offset, storage_mode, buffer_length). The handle is the borrowed MTLBuffer pointer; the caller
// keeps the tensor alive for as long as the handle is used.
std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> buffer_info(torch::Tensor value) {
  id<MTLBuffer> buffer = tensor_buffer(value, "tensor");
  return {uint64_t(reinterpret_cast<uintptr_t>(buffer)), uint64_t(value.storage_offset()) * sizeof(float),
          uint64_t(buffer.storageMode), uint64_t(buffer.length)};
}

// Bumps the autograd version counter of a tensor written outside torch (step 6 of the native backward).
void increment_version(torch::Tensor value) { torch::autograd::impl::bump_version(value); }

// One persistent command queue per native context, created at prepare(N), never inside a call. The shared event
// is created with it and used only by the F2 hold.
class BlitQueue {
 public:
  BlitQueue() {
    @autoreleasepool {
      id<MTLDevice> device = mps_device();
      queue_ = [device newCommandQueue];
      TORCH_CHECK(queue_ != nil, "Metal queue creation failed");
      event_ = [device newSharedEvent];
      if (event_ == nil) {
        [queue_ release];
        queue_ = nil;
        TORCH_CHECK(false, "MTLSharedEvent creation failed");
      }
      event_.signaledValue = 0;
    }
  }
  BlitQueue(const BlitQueue&) = delete;
  BlitQueue& operator=(const BlitQueue&) = delete;
  ~BlitQueue() { close(); }

  // Copies, in order, `count` float32 from each borrowed (handle, byte_offset, count) source into `destination`
  // at cumulative offsets: one command buffer, one blit encoder. Every source is checked before any encoding.
  // Without hold: commit and wait (blocking). With hold (F2 only): the blit first waits on the shared event, is
  // committed, and completes only after release().
  void copy(torch::Tensor destination, const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>& sources,
            bool hold) {
    TORCH_CHECK(queue_ != nil, "blit queue is closed");
    TORCH_CHECK(pending_ == nil, "a held copy is still pending");
    id<MTLBuffer> target = tensor_buffer(destination, "destination");
    uint64_t total = 0;
    for (const auto& source : sources) {
      const uint64_t handle = std::get<0>(source), offset = std::get<1>(source), count = std::get<2>(source);
      TORCH_CHECK(handle != 0, "null source buffer");
      TORCH_CHECK(offset % sizeof(float) == 0, "unaligned source offset");
      id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)reinterpret_cast<void*>(uintptr_t(handle));
      TORCH_CHECK(buffer.device == target.device, "source and destination are on different devices");
      TORCH_CHECK(offset <= buffer.length && count * sizeof(float) <= buffer.length - offset,
                  "source view exceeds its buffer");
      total += count;
    }
    TORCH_CHECK(total == uint64_t(destination.numel()), "source counts do not cover the destination");
    @autoreleasepool {
      id<MTLCommandBuffer> command = [queue_ commandBuffer];
      TORCH_CHECK(command != nil, "Metal command buffer creation failed");
      if (hold) [command encodeWaitForEvent:event_ value:event_value_ + 1];
      id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
      TORCH_CHECK(encoder != nil, "Metal blit encoder creation failed");
      uint64_t cursor = uint64_t(destination.storage_offset()) * sizeof(float);
      for (const auto& source : sources) {
        const uint64_t bytes = std::get<2>(source) * sizeof(float);
        if (bytes) {
          id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)reinterpret_cast<void*>(uintptr_t(std::get<0>(source)));
          [encoder copyFromBuffer:buffer
                     sourceOffset:NSUInteger(std::get<1>(source))
                         toBuffer:target
                destinationOffset:NSUInteger(cursor)
                             size:NSUInteger(bytes)];
        }
        cursor += bytes;
      }
      [encoder endEncoding];
      [command commit];
      if (hold) {
        ++event_value_;
        pending_ = [command retain];
        return;
      }
      [command waitUntilCompleted];
      TORCH_CHECK(command.status == MTLCommandBufferStatusCompleted, "gradient blit failed: ", command_error(command));
    }
  }

  // The reverse of `copy`: copies `source` (a contiguous float32 MPS tensor, e.g. flat.grad) into
  // the borrowed (handle, byte_offset, count) destinations at cumulative element offsets: one command buffer, one blit
  // encoder, commit and blocking wait. Every destination is checked before any encoding.
  void copy_in(torch::Tensor source, const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>& destinations) {
    TORCH_CHECK(queue_ != nil, "blit queue is closed");
    TORCH_CHECK(pending_ == nil, "a held copy is still pending");
    id<MTLBuffer> origin = tensor_buffer(source, "source");
    uint64_t total = 0;
    for (const auto& destination : destinations) {
      const uint64_t handle = std::get<0>(destination), offset = std::get<1>(destination), count = std::get<2>(destination);
      TORCH_CHECK(handle != 0, "null destination buffer");
      TORCH_CHECK(offset % sizeof(float) == 0, "unaligned destination offset");
      id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)reinterpret_cast<void*>(uintptr_t(handle));
      TORCH_CHECK(buffer.device == origin.device, "source and destination are on different devices");
      TORCH_CHECK(offset <= buffer.length && count * sizeof(float) <= buffer.length - offset,
                  "destination view exceeds its buffer");
      total += count;
    }
    TORCH_CHECK(total == uint64_t(source.numel()), "destination counts do not cover the source");
    @autoreleasepool {
      id<MTLCommandBuffer> command = [queue_ commandBuffer];
      TORCH_CHECK(command != nil, "Metal command buffer creation failed");
      id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
      TORCH_CHECK(encoder != nil, "Metal blit encoder creation failed");
      uint64_t cursor = uint64_t(source.storage_offset()) * sizeof(float);
      for (const auto& destination : destinations) {
        const uint64_t bytes = std::get<2>(destination) * sizeof(float);
        if (bytes) {
          id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)reinterpret_cast<void*>(uintptr_t(std::get<0>(destination)));
          [encoder copyFromBuffer:origin
                     sourceOffset:NSUInteger(cursor)
                         toBuffer:buffer
                destinationOffset:NSUInteger(std::get<1>(destination))
                             size:NSUInteger(bytes)];
        }
        cursor += bytes;
      }
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      TORCH_CHECK(command.status == MTLCommandBufferStatusCompleted, "gradient entry blit failed: ", command_error(command));
    }
  }

  // F2 only: signals the event of the held copy and waits for its completion.
  void release() {
    TORCH_CHECK(pending_ != nil, "no held copy");
    const bool completed = finish_pending();
    TORCH_CHECK(completed, "held gradient blit failed");
  }

  // Finishes a held copy if any, then releases the queue and the event. Idempotent.
  void close() {
    finish_pending();
    [event_ release];
    event_ = nil;
    [queue_ release];
    queue_ = nil;
  }

 private:
  bool finish_pending() {
    if (pending_ == nil) return true;
    event_.signaledValue = event_value_;
    [pending_ waitUntilCompleted];
    const bool completed = pending_.status == MTLCommandBufferStatusCompleted;
    [pending_ release];
    pending_ = nil;
    return completed;
  }

  id<MTLCommandQueue> queue_ = nil;
  id<MTLSharedEvent> event_ = nil;
  id<MTLCommandBuffer> pending_ = nil;
  uint64_t event_value_ = 0;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("STORAGE_PRIVATE") = uint64_t(MTLStorageModePrivate);
  m.def("buffer_info", &buffer_info);
  m.def("increment_version", &increment_version);
  pybind11::class_<BlitQueue>(m, "BlitQueue")
      .def(pybind11::init<>())
      .def("copy", &BlitQueue::copy, pybind11::arg("destination"), pybind11::arg("sources"),
           pybind11::arg("hold") = false, pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("copy_in", &BlitQueue::copy_in, pybind11::arg("source"), pybind11::arg("destinations"),
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("release", &BlitQueue::release, pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("close", &BlitQueue::close, pybind11::call_guard<pybind11::gil_scoped_release>());
}
