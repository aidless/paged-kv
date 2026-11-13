#!/usr/bin/env python3
"""把 cuda/ 下的源码打包成一个 Kaggle script kernel，然后由 kaggle CLI 推送。

为什么走这条路：作者当时的机器没有可用的 NVIDIA 驱动，而 Kaggle 的 T4 是可用的
CUDA 通道；Kaggle kernel 只接受单个代码文件，所以这里把 .cu/.h 源码 base64 内嵌进
kernel.py——**单一真相源仍然是 cuda/ 里的 .cu 文件**，kernel.py 是纯生成物，
改完源码重跑本脚本即可。

用法：
    python3 cuda/build_kaggle_kernel.py [输出目录]
然后：
    kaggle kernels push -p <输出目录>
    kaggle kernels status <username>/paged-kv-cuda-bench
    kaggle kernels output <username>/paged-kv-cuda-bench -p <输出目录>/out

注意：Kaggle 免费 GPU 需要账号完成手机验证；未验证时 push 会报
"Cannot use GPU: Account not verified"。
"""

import base64
import json
import os
import pathlib
import sys

KERNEL_SLUG = "paged-kv-cuda-bench"
KERNEL_TITLE = "paged-kv CUDA bench (fused softmax / silu_and_mul)"
# Kaggle account that owns the target kernel. Override with the
# PAGED_KV_KAGGLE_USER environment variable.
USERNAME = os.environ.get("PAGED_KV_KAGGLE_USER", "aidless")

HERE = pathlib.Path(__file__).resolve().parent

RUNNER = '''# 由 cuda/build_kaggle_kernel.py 生成——不要手改，改 cuda/ 下的源码后重跑生成脚本。
import base64, pathlib, subprocess, sys

SRC = {src_files}

work = pathlib.Path("/kaggle/working/cuda_src")
work.mkdir(parents=True, exist_ok=True)
for name, b64 in SRC.items():
    (work / name).write_bytes(base64.b64decode(b64))

print("=== nvcc ===")
r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
print(r.stdout or r.stderr)

print("=== nvidia-smi ===")
r = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version",
     "--format=csv,noheader"], capture_output=True, text=True)
print(r.stdout or r.stderr)
cc = r.stdout.strip().split(",")[1].strip() if r.returncode == 0 and "," in r.stdout else ""

# 优先用 -arch=native（nvcc 自己问驱动），拿不到再回退到检测到的 compute capability。
arch_args = ["-arch=native"]
if cc:
    fallback = "-arch=sm_" + cc.replace(".", "")
else:
    fallback = "-arch=sm_75"

# 注意 -Xptxas 与它的 -v 必须相邻，所以 arch 插在它们之后而不是中间。
base = ["nvcc", "-std=c++14", "-O3", "-lineinfo", "-Xptxas", "-v",
        "-o", "bench", "bench.cu", "fused_ops.cu"]

def with_arch(arch):
    return base[:6] + [arch] + base[6:]

r = subprocess.run(with_arch(arch_args[0]), cwd=work, capture_output=True, text=True)
if r.returncode != 0:
    print("native 编译失败，回退 " + fallback)
    print(r.stderr)
    r = subprocess.run(with_arch(fallback), cwd=work, capture_output=True, text=True)

print("=== nvcc ptxas 信息（寄存器 / shared memory 用量）===")
print(r.stderr)
if r.returncode != 0:
    print("BUILD_FAILED")
    sys.exit(1)

print("=== 运行基准 ===")
r = subprocess.run([str(work / "bench")], capture_output=True, text=True)
print(r.stdout)
if r.stderr:
    print(r.stderr, file=sys.stderr)
sys.exit(r.returncode)
'''


def collect_sources():
    """读 cuda/ 下参与编译的三个文件，base64 编码后塞进 runner。"""
    names = ["fused_ops.h", "fused_ops.cu", "bench.cu"]
    files = {}
    for n in names:
        data = (HERE / n).read_bytes()
        files[n] = base64.b64encode(data).decode("ascii")
    return files


def main():
    outdir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/paged-kv-cuda-kernel")
    outdir.mkdir(parents=True, exist_ok=True)

    files = collect_sources()
    body = "{\n" + ",\n".join(f'    "{k}": "{v}"' for k, v in files.items()) + ",\n}"
    (outdir / "kernel.py").write_text(RUNNER.replace("{src_files}", body), encoding="utf-8")

    meta = {
        "id": f"{USERNAME}/{KERNEL_SLUG}",
        "title": KERNEL_TITLE,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        # 私有：这是学习/作品集产物，没必要公开推给所有人看
        "is_private": True,
        # 基准要 GPU；不需要联网（源码已内嵌）
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (outdir / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"生成完成: {outdir}")
    for p in sorted(outdir.iterdir()):
        print(f"  {p.name}  {p.stat().st_size} bytes")
    print(f"\n推送: kaggle kernels push -p {outdir}")


if __name__ == "__main__":
    main()
