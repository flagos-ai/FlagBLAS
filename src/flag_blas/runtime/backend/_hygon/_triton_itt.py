"""FlagBLAS 侧对 Triton HCU 后端 per-kernel 编译选项的运行时注入。

背景
----
Triton (flagtree 0.6.1+hcu3.6) 的下列编译开关由全局 knob 控制（env 动态读取），
仅支持进程级，但 Hygon gfx936 上它们对 kernel 的影响按 shape 分化，需要 per-kernel 控制：

1. in-thread-transpose (ITT): env ``TRITON_HIP_USE_IN_THREAD_TRANSPOSE``。
   大/方 shape 受益 +5~15%，瘦/小 shape 受损（v1.2 已按 shape 分发）。
2. buffer ops: env ``AMDGCN_USE_BUFFER_OPS``（默认 True）。
   大/中输出 shape 受益 -1~-4%，但 2048x2048x16384 +43% 严重受损、
   128x4096x1024 +3.8% 等小 shape 轻微受损（v1.3 按 shape 分发）。

方案
----
不改 site-packages 源码，在 FlagBLAS 导入时做两处运行时注入：

1. 动态给 ``HIPOptions``（frozen dataclass）增加字段 ``use_in_thread_transpose``
   与 ``use_buffer_ops``（默认 None）。Triton 的 ``parse_options`` 按
   ``HIPOptions.__dataclass_fields__`` 从 launch kwargs 过滤编译选项，且编译缓存
   key（``compute_cache_key`` 用 ``str(options)``）基于 dataclass 字段序列化——
   新字段被自动吸收并区分不同变体的编译产物。
2. 包装 ``HIPBackend.make_ttgir``：编译时按 ``options`` 中的字段值临时设置对应
   env（knob 每次访问动态读 env），编译后恢复 env。字段为 None 时不动 env
   （保持全局默认，产物与未注入时完全一致）。

幂等：重复调用安全；若 site-packages 已手工打了同款 patch（字段已存在），跳过注入。
"""

import dataclasses
import os
import threading

# (HIPOptions 字段名, env 名) 列表——新增 per-kernel 编译选项在此登记
_OPTION_FIELDS = (
    ("use_in_thread_transpose", "TRITON_HIP_USE_IN_THREAD_TRANSPOSE"),
    ("use_buffer_ops", "AMDGCN_USE_BUFFER_OPS"),
)
# frozen dataclass 重新生成 __init__/__repr__/__eq__/__hash__ 前需要删除的旧方法
# （dataclass 的 _set_new_attribute 不覆盖已有属性）
_DATACLASS_METHODS = (
    "__init__", "__repr__", "__eq__", "__hash__", "__setattr__", "__delattr__",
    "__match_args__",
)

_installed = False
_install_lock = threading.Lock()
# make_ttgir 编译期间 env 切换的互斥（编译为同步流程，锁仅防多线程编译交错）
_env_lock = threading.Lock()


def install_triton_itt_patch():
    """在 Triton HCU 后端注入 per-kernel 编译选项支持。

    Returns:
        bool: True 表示注入完成（或已支持），False 表示当前环境无 Triton HCU 后端。
    """
    global _installed
    if _installed:
        return True
    with _install_lock:
        if _installed:
            return True
        try:
            from triton.backends.hcu import compiler_hcu as _chcu
        except ImportError:
            return False

        hip_options = _chcu.HIPOptions
        hip_backend = _chcu.HIPBackend

        # 1) HIPOptions 动态加字段（site-packages 已手工 patch 则跳过）
        for field_name, _env_name in _OPTION_FIELDS:
            if field_name not in hip_options.__dataclass_fields__:
                annotations = dict(getattr(hip_options, "__annotations__", {}))
                annotations[field_name] = bool
                hip_options.__annotations__ = annotations
                # 类属性默认值：重新生成 __init__ 时避免 non-default follows default
                setattr(hip_options, field_name, None)
                hip_options.__dataclass_fields__[field_name] = dataclasses.field(default=None)
                for _name in _DATACLASS_METHODS:
                    if _name in hip_options.__dict__:
                        delattr(hip_options, _name)
                # 保持原始 @dataclass(frozen=True) 语义（frozen 会生成 __hash__ 与禁赋值 __setattr__）
                dataclasses.dataclass(hip_options, frozen=True)

        # 2) 包装 make_ttgir：编译时按 options 的字段值切换对应 env
        _make_ttgir = hip_backend.__dict__.get("make_ttgir")
        if _make_ttgir is not None and not getattr(_make_ttgir, "_flagblas_itt_patched", False):
            _orig_func = _make_ttgir.__func__

            def _make_ttgir_wrapper(mod, metadata, options):
                to_set = {}
                for field_name, env_name in _OPTION_FIELDS:
                    val = getattr(options, field_name, None)
                    if val is not None:
                        to_set[env_name] = "1" if val else "0"
                if not to_set:
                    return _orig_func(mod, metadata, options)
                with _env_lock:
                    saved = {env: os.environ.get(env) for env in to_set}
                    for env, val in to_set.items():
                        os.environ[env] = val
                    try:
                        return _orig_func(mod, metadata, options)
                    finally:
                        for env, val in saved.items():
                            if val is None:
                                os.environ.pop(env, None)
                            else:
                                os.environ[env] = val

            _make_ttgir_wrapper._flagblas_itt_patched = True
            hip_backend.make_ttgir = staticmethod(_make_ttgir_wrapper)

        _installed = True
        return True
