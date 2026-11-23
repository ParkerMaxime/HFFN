"""Cold backward plan lowering for native and Torch entry adapters.

One stage vocabulary, explicit ABI ordering/bindings, no hot plan interpreter.
GPU arithmetic remains owned by fused_edges, backward_sums and activation.
"""
from pathlib import Path
import argparse

_TORCH_ORDER = {'token_major': ('hidden', 'down', 'up', 'up_bias', 'down_bias', 'input'), 'feature_persistent': ('hidden', 'down', 'input'), 'feature_owner_tile': ('hidden', 'down', 'input', 'reduce'), 'feature_owner_tile_fused': ('hidden_down', 'down_bias', 'input', 'reduce'), 'feature_owner_tile_fused_direct': ('hidden_down', 'down_bias', 'input', 'reduce'), 'feature_owner_tile_fused_balanced': ('hidden_down', 'down_bias', 'input', 'reduce'), 'feature_owner_tile_fused_gup': ('hidden_down_up', 'down_bias', 'input', 'reduce'), 'feature_owner_tile_fused_gup_split4': ('hidden_down_up', 'down_bias', 'input', 'reduce')}
_TORCH_BINDINGS = {'token_major': {'hidden': ('_dispatch', 'hidden_backward', ['gy', 'context.z', 'self.down_offsets', 'self.down_idx', 'self.down_val', 'gz'], [('threads', 'gz.numel()')]), 'down': ('_dispatch', 'grad_down', ['gy', 'context.act', 'self.down_slot_hidden', 'self.down_idx', 'gdown'], [('threads', 'gdown.numel()')]), 'up': ('_dispatch', 'grad_up', ['gz', 'context.x', 'self.up_slot_hidden', 'self.up_idx', 'gup'], [('threads', 'gup.numel()')]), 'up_bias': ('_dispatch', 'grad_b_up', ['gz', 'gbup'], [('threads', 'gbup.numel()')]), 'down_bias': ('_dispatch', 'grad_b_down', ['gy', 'gbdown'], [('threads', 'gbdown.numel()')]), 'input': ('_dispatch', 'grad_x_token_major', ['gz', 'self.up_owner_offsets', 'self.up_route', 'self.up_offsets', 'self.up_val', 'gx'], [('threads', 'gx.numel()')])}, 'feature_persistent': {'hidden': ('_dispatch_persistent', 'hidden_backward_feature_persistent', ['gy_feature', 'context.z', 'self.down_offsets', 'self.down_idx', 'self.down_val', 'gz', 'gbup'], [('owners', 'self.spec.d_ff')]), 'down': ('_dispatch_persistent', 'down_grad_feature_persistent', ['context.act', 'gy_feature', 'self.down_owner_offsets', 'self.down_route', 'self.down_offsets', 'gdown', 'gbdown'], [('owners', 'self.spec.d_model')]), 'input': ('_dispatch_persistent', 'input_grad_feature_persistent', ['context.kernel_x', 'gz', 'self.up_owner_offsets', 'self.up_route', 'self.up_offsets', 'self.up_val', 'gup', 'gx_feature'], [('owners', 'self.spec.d_model')])}, 'feature_owner_tile': {'hidden': ('_dispatch_owner_tile', 'hidden_backward_owner_tile', ['gy_feature', 'context.z', 'self.down_offsets', 'self.down_idx', 'self.down_val', 'gz', 'partial'], [('owners', 'self.spec.d_ff')]), 'down': ('_dispatch_owner_tile', 'down_grad_owner_tile', ['context.act', 'gy_feature', 'self.down_owner_offsets', 'self.down_route', 'self.down_offsets', 'partial'], [('owners', 'self.spec.d_model')]), 'input': ('_dispatch_owner_tile', 'input_grad_owner_tile', ['context.kernel_x', 'gz', 'self.up_owner_offsets', 'self.up_route', 'self.up_offsets', 'self.up_val', 'partial', 'gx_feature'], [('owners', 'self.spec.d_model')]), 'reduce': ('_dispatch', 'reduce_owner_tile_grad', ['partial', 'gflat'], [('threads', 'gflat.numel()')])}, 'feature_owner_tile_fused': {'hidden_down': ('_dispatch_owner_tile', 'hidden_down_grad_owner_tile_fused', ['gy_feature', 'context.z', 'context.act', 'self.down_offsets', 'self.down_idx', 'self.down_val', 'gz', 'partial'], [('owners', 'self.spec.d_ff')]), 'down_bias': ('_dispatch_owner_tile', 'down_bias_owner_tile', ['gy_feature', 'partial'], [('owners', 'self.spec.d_model')]), 'input': ('_dispatch_owner_tile', 'input_grad_owner_tile_fused', ['context.kernel_x', 'gz', 'self.up_owner_offsets', 'self.up_route', 'self.up_offsets', 'self.up_val', 'partial', 'gx_feature'], [('owners', 'self.spec.d_model')]), 'reduce': ('_dispatch', 'reduce_owner_tile_grad', ['partial', 'gflat'], [('threads', 'gflat.numel()')])}, 'feature_owner_tile_fused_direct': {'hidden_down': ('_dispatch_owner_tile', 'hidden_down_grad_owner_tile_fused', ['gy_feature', 'context.z', 'context.act', 'self.down_offsets', 'self.down_idx', 'self.down_val', 'gz', 'partial'], [('owners', 'self.spec.d_ff')]), 'down_bias': ('_dispatch_owner_tile', 'down_bias_owner_tile', ['gy_feature', 'partial'], [('owners', 'self.spec.d_model')]), 'input': ('_dispatch_owner_tile', 'input_grad_owner_tile_fused_direct', ['context.kernel_x', 'gz', 'self.up_owner_offsets', 'self.up_route_slot', 'self.up_route_hidden', 'self.up_val', 'partial', 'gx_feature'], [('owners', 'self.spec.d_model')]), 'reduce': ('_dispatch', 'reduce_owner_tile_grad', ['partial', 'gflat'], [('threads', 'gflat.numel()')])}, 'feature_owner_tile_fused_balanced': {'hidden_down': ('_dispatch_owner_tile', 'hidden_down_grad_owner_tile_fused_balanced', ['gy_feature', 'context.z', 'context.act', 'self.down_offsets', 'self.down_idx', 'self.down_val', 'self.down_hidden_schedule', 'gz', 'partial'], [('owners', 'self.spec.d_ff')]), 'down_bias': ('_dispatch_owner_tile', 'down_bias_owner_tile', ['gy_feature', 'partial'], [('owners', 'self.spec.d_model')]), 'input': ('_dispatch_owner_tile', 'input_grad_owner_tile_fused_balanced', ['context.kernel_x', 'gz', 'self.up_owner_offsets', 'self.up_route', 'self.up_offsets', 'self.up_val', 'self.up_input_schedule', 'partial', 'gx_feature'], [('owners', 'self.spec.d_model')]), 'reduce': ('_dispatch', 'reduce_owner_tile_grad', ['partial', 'gflat'], [('threads', 'gflat.numel()')])}, 'feature_owner_tile_fused_gup': {'hidden_down_up': ('_dispatch_owner_tile', 'hidden_down_up_grad_owner_tile_fused', ['gy_feature', 'context.z', 'context.act', 'self.down_offsets', 'self.down_idx', 'self.down_val', 'context.kernel_x', 'self.up_offsets', 'self.up_idx', 'gz', 'partial'], [('owners', 'self.spec.d_ff')]), 'down_bias': ('_dispatch_owner_tile', 'down_bias_owner_tile', ['gy_feature', 'partial'], [('owners', 'self.spec.d_model')]), 'input': ('_dispatch_owner_tile', 'input_grad_x_owner_tile', ['gz', 'self.up_owner_offsets', 'self.up_route', 'self.up_offsets', 'self.up_val', 'gx_feature'], [('owners', 'self.spec.d_model')]), 'reduce': ('_dispatch', 'reduce_owner_tile_grad', ['partial', 'gflat'], [('threads', 'gflat.numel()')])}, 'feature_owner_tile_fused_gup_split4': {'hidden_down_up': ('_dispatch_owner_tile_split4', 'hidden_down_up_grad_owner_tile_fused_split4', ['gy_feature', 'context.z', 'context.act', 'self.down_offsets', 'self.down_idx', 'self.down_val', 'context.kernel_x', 'self.up_offsets', 'self.up_idx', 'gz', 'partial'], [('owners', 'self.spec.d_ff')]), 'down_bias': ('_dispatch_owner_tile', 'down_bias_owner_tile', ['gy_feature', 'partial'], [('owners', 'self.spec.d_model')]), 'input': ('_dispatch_owner_tile', 'input_grad_x_owner_tile', ['gz', 'self.up_owner_offsets', 'self.up_route', 'self.up_offsets', 'self.up_val', 'gx_feature'], [('owners', 'self.spec.d_model')]), 'reduce': ('_dispatch', 'reduce_owner_tile_grad', ['partial', 'gflat'], [('threads', 'gflat.numel()')])}}
_TORCH_SETUP = {'token_major': 'gx = torch.empty_like(context.x)', 'feature_persistent': 'gy_feature = gy.transpose(0, 1).contiguous()\ngx_feature = torch.empty_like(context.kernel_x)\nkeepalive.extend((gy_feature, gx_feature))', 'feature_owner_tile': "gy_feature = gy.transpose(0, 1).contiguous()\ngx_feature = torch.empty_like(context.kernel_x)\ntiles = (self.n + 127) // 128\npartial = torch.empty(self.flat.numel() * tiles, device='mps', dtype=torch.float32)\nkeepalive.extend((gy_feature, gx_feature, partial))", 'feature_owner_tile_fused': "gy_feature = gy.transpose(0, 1).contiguous()\ngx_feature = torch.empty_like(context.kernel_x)\ntiles = (self.n + 127) // 128\npartial = torch.empty(self.flat.numel() * tiles, device='mps', dtype=torch.float32)\nkeepalive.extend((gy_feature, gx_feature, partial))", 'feature_owner_tile_fused_direct': "gy_feature = gy.transpose(0, 1).contiguous()\ngx_feature = torch.empty_like(context.kernel_x)\ntiles = (self.n + 127) // 128\npartial = torch.empty(self.flat.numel() * tiles, device='mps', dtype=torch.float32)\nkeepalive.extend((gy_feature, gx_feature, partial))", 'feature_owner_tile_fused_balanced': "gy_feature = gy.transpose(0, 1).contiguous()\ngx_feature = torch.empty_like(context.kernel_x)\ntiles = (self.n + 127) // 128\npartial = torch.empty(self.flat.numel() * tiles, device='mps', dtype=torch.float32)\nkeepalive.extend((gy_feature, gx_feature, partial))", 'feature_owner_tile_fused_gup': "gy_feature = gy.transpose(0, 1).contiguous()\ngx_feature = torch.empty_like(context.kernel_x)\ntiles = (self.n + 127) // 128\npartial = torch.empty(self.flat.numel() * tiles, device='mps', dtype=torch.float32)\nkeepalive.extend((gy_feature, gx_feature, partial))", 'feature_owner_tile_fused_gup_split4': "gy_feature = gy.transpose(0, 1).contiguous()\ngx_feature = torch.empty_like(context.kernel_x)\ntiles = (self.n + 127) // 128\npartial = torch.empty(self.flat.numel() * tiles, device='mps', dtype=torch.float32)\nkeepalive.extend((gy_feature, gx_feature, partial))"}
_TORCH_FINISH = {'token_major': '', 'feature_persistent': 'gx = gx_feature.transpose(0, 1).contiguous()', 'feature_owner_tile': 'gx = gx_feature.transpose(0, 1).contiguous()', 'feature_owner_tile_fused': 'gx = gx_feature.transpose(0, 1).contiguous()', 'feature_owner_tile_fused_direct': 'gx = gx_feature.transpose(0, 1).contiguous()', 'feature_owner_tile_fused_balanced': 'gx = gx_feature.transpose(0, 1).contiguous()', 'feature_owner_tile_fused_gup': 'gx = gx_feature.transpose(0, 1).contiguous()', 'feature_owner_tile_fused_gup_split4': 'gx = gx_feature.transpose(0, 1).contiguous()'}
_TORCH_PREFIX = "if gy.device.type != 'mps' or gy.dtype != torch.float32 or tuple(gy.shape) != (self.n, self.spec.d_model):\n    raise ValueError(f'expected contiguous MPS float32 grad [{self.n},{self.spec.d_model}]')\ngy = gy.contiguous()\ngz = torch.empty(self.spec.d_ff, self.n, device='mps', dtype=torch.float32)\ngflat = torch.empty_like(self.flat)\nu = self.up_val.numel()\nd = self.down_val.numel()\nbu = self.b_up.numel()\ngup = gflat[:u]\ngdown = gflat[u:u + d]\ngbup = gflat[u + d:u + d + bu]\ngbdown = gflat[u + d + bu:]\nkeepalive: list[torch.Tensor] = [context.x, context.kernel_x, context.z, context.act, gy, gz]"
_TORCH_RETURN = "return {'grad_x': gx, 'grad_up': gup, 'grad_down': gdown, 'grad_b_up': gbup, 'grad_b_down': gbdown, 'grad_z': gz, 'grad_flat': gflat, '_mps_keepalive': tuple(keepalive)}"
_NATIVE_ORDER = {'persistent': ('down', 'hidden', 'input'), 'persistent_flat': ('down', 'hidden', 'input'), 'owner_fused': ('down_bias', 'hidden_down', 'input', 'reduce'), 'owner_separate': ('down', 'hidden', 'input', 'up', 'reduce'), 'owner': ('down', 'hidden', 'input', 'reduce'), 'tiled': ('down', 'reduce_down', 'down_bias', 'hidden', 'up_bias', 'input', 'reduce_up')}
_NATIVE_BINDINGS = {'tiled': {'down': 'pp(@"bdown_tile")', 'reduce_down': 'pp(@"reduce_edge")', 'down_bias': 'pp(@"reduce_output")', 'hidden': 'pp(@"bhidden_tile")', 'up_bias': 'pp(@"reduce_hidden")', 'input': 'pp(@"binput_tile")', 'reduce_up': 'pp(@"reduce_edge")'}, 'owner': {'down': 'pp(@"bdown_owner_tile")', 'hidden': 'pp(@"bhidden_owner_tile")', 'input': '(a.binput_fused1 && a.owner_tile_n == 128u ? pp(@"binput_owner_tile_fused1")\n             : a.binput_fused2 ? pp(@"binput_owner_tile_fused2")\n             : a.binput_split4 ? pp(@"binput_owner_tile_split4")\n             : a.binput_split2 ? pp(@"binput_owner_tile_split2")\n             : a.binput_edge_block4 ? pp(@"binput_owner_tile_block4")\n                                  : pp(@"binput_owner_tile"))', 'reduce': 'pp(@"reduce_flat_partials")'}, 'owner_separate': {'down': 'pp(@"bdown_owner_tile")', 'hidden': 'pp(@"bhidden_owner_tile")', 'input': 'pp(@"binput_gx_owner_tile")', 'up': '(a.binput_gup_hidden ? pp(@"binput_gup_hidden_tile")\n                                  : pp(@"binput_gup_owner_tile"))', 'reduce': 'pp(@"reduce_flat_partials")'}, 'owner_fused': {'down_bias': 'pp(@"bdown_bias_owner_tile")', 'hidden_down': '(a.bdown_gup_fused ? pp(@"bdown_bhidden_gup_fused_owner_tile")\n                                : pp(@"bdown_bhidden_fused_owner_tile"))', 'input': '(a.bdown_gup_fused ? pp(@"binput_gx_only_owner_tile")\n             : a.packet_binput_staged ? pp(@"binput_packet_staged_owner_tile")\n             : a.packet_binput_staged64 ? pp(@"binput_packet_staged64_owner_tile")\n             : a.binput_fused1 && a.owner_tile_n == 128u ? pp(@"binput_owner_tile_fused1")\n             : pp(@"binput_owner_tile"))', 'reduce': 'pp(@"reduce_flat_partials")'}, 'persistent_flat': {'down': 'pp(@"bdown")', 'hidden': 'pp(@"bhidden")', 'input': 'pp(@"binput")'}, 'persistent': {'down': 'pp(@"bdown")', 'hidden': 'pp(@"bhidden")', 'input': 'pp(@"binput")'}}
_NATIVE_PREFIX = {'tiled': ['pp(@"fup")', 'pp(@"fdown")'], 'owner': ['pp(@"fup")', 'pp(@"fdown")'], 'owner_separate': ['pp(@"fup")', 'pp(@"fdown")'], 'owner_fused': ['pp(@"fup")', 'pp(@"fdown")'], 'persistent_flat': ['pp(@"fup")', 'pp(@"fdown")'], 'persistent': ['pp(@"fup")', 'pp(@"fdown")']}
_NATIVE_SUFFIX = {'tiled': ['a.flat_adamw_pool ? pp(@"adamw") : pp(@"adamw")', 'pp(@"adamw")', 'pp(@"adamw")', 'pp(@"adamw")'], 'owner': ['pp(@"adamw")'], 'owner_separate': ['pp(@"adamw")'], 'owner_fused': ['pp(@"adamw")'], 'persistent_flat': ['pp(@"adamw")'], 'persistent': ['pp(@"adamw")', 'pp(@"adamw")', 'pp(@"adamw")', 'pp(@"adamw")']}

# Roles denote semantic work; ABI bindings retain layout-specific addressing,
# ownership, fusion and independent-stage ordering. The compiler is shared.
_STAGE_ROLES = frozenset({
    'hidden', 'down', 'up', 'up_bias', 'down_bias', 'input',
    'hidden_down', 'hidden_down_up', 'reduce', 'reduce_down', 'reduce_up',
})


def stage_plan(profile, abi):
    """Return the immutable stage sequence for an explicit ABI policy."""
    orders = {'torch': _TORCH_ORDER, 'native': _NATIVE_ORDER}
    stages = orders[abi][profile]
    validate_plan(stages)
    return stages


def validate_plan(stages):
    if not stages or len(set(stages)) != len(stages):
        raise ValueError('empty or duplicate backward stage')
    if not set(stages) <= _STAGE_ROLES:
        raise ValueError('unknown backward stage')
    producers = [s for s in stages if s in ('hidden', 'hidden_down', 'hidden_down_up')]
    if len(producers) != 1 or 'input' not in stages:
        raise ValueError('backward needs one hidden adjoint producer and input adjoint')
    hidden = stages.index(producers[0])
    if stages.index('input') <= hidden or ('up' in stages and stages.index('up') <= hidden):
        raise ValueError('input/up stage must follow its hidden adjoint')
    if 'reduce' in stages and stages[-1] != 'reduce':
        raise ValueError('partial reduction must follow all partial writers')
    return stages


def torch_backward_source(profile):
    """Lower a plan once to explicit existing dispatch calls, never a hot loop."""
    stages = stage_plan(profile, 'torch')
    bindings = _TORCH_BINDINGS[profile]
    if set(stages) != set(bindings):
        raise ValueError('missing or extra Torch stage binding')
    chunks = [_TORCH_PREFIX, _TORCH_SETUP[profile]]
    for stage in stages:
        dispatch, kernel, arguments, keywords = bindings[stage]
        args = ['self.lib.' + kernel, *arguments]
        args.extend(name + '=' + value for name, value in keywords)
        chunks.append('self.' + dispatch + '(' + ', '.join(args) + ')')
    chunks.extend((_TORCH_FINISH[profile], _TORCH_RETURN))
    body = '\n'.join(c for c in chunks if c)
    return 'def backward(self, context, gy):\n' + '\n'.join('    ' + line for line in body.splitlines()) + '\n'


def compile_torch_backward(torch):
    """Compile immutable ABI code at import; Torch owns tensors and lifetime."""
    implementations = {}
    for profile in _TORCH_ORDER:
        source = torch_backward_source(profile)
        namespace = {'torch': torch}
        exec(compile(source, '<hffn_backward_plan:' + profile + '>', 'exec'), namespace)
        implementation = namespace['backward']
        implementation.__hffn_source__ = source
        implementations[profile] = implementation
    return implementations


def native_pipeline_source(profile):
    stages = stage_plan(profile, 'native')
    bindings = _NATIVE_BINDINGS[profile]
    if set(stages) != set(bindings):
        raise ValueError('missing or extra native stage binding')
    expressions = [*_NATIVE_PREFIX[profile], *(bindings[s] for s in stages), *_NATIVE_SUFFIX[profile]]
    return '@[' + ','.join(expressions) + ']'


def emit_header(path):
    lines = ['// Generated by hffn_m1.backward_plan; do not edit.', '#pragma once']
    for profile in _NATIVE_ORDER:
        # Physical newlines are immaterial to C++ and are removed from macros.
        value = ' '.join(native_pipeline_source(profile).splitlines())
        lines.append('#define HFFN_M1_PLAN_' + profile.upper() + ' ' + value)
    Path(path).write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emit-header', type=Path, required=True)
    emit_header(parser.parse_args().emit_header)
