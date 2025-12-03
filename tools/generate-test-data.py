#!/usr/bin/env python3
"""
LZ4 GPU Test Data Generator (moved to tools)
生成具有重复模式的数据用于测试LZ4 GPU压缩解压的正确性
"""

import argparse
import os
import sys
import math
import random
from collections import defaultdict

def generate_repeated_pattern(size, pattern_size=64, pattern_density=0.7):
    """
    生成具有重复模式的数据

    Args:
        size: 总数据大小（字节）
        pattern_size: 重复模式大小（字节）
        pattern_density: 模式重复密度（0.0-1.0）
    """
    data = bytearray(size)

    # 生成基础重复模式
    pattern = bytearray()
    for i in range(pattern_size):
        if i % 4 == 0:
            pattern.append((i * 7 + 23) % 256)  # 伪随机模式
        elif i % 4 == 1:
            pattern.append(i % 256)
        elif i % 4 == 2:
            pattern.append((i * 13 + 47) % 256)
        else:
            pattern.append(255 - (i % 256))

    # 填充数据：以block为单位混合pattern和随机数据，增大复杂性
    pos = 0
    # choose chunk sizes randomly between 1x and 8x pattern_size to reduce predictability
    while pos < size:
        chunk_len = min(size - pos, random.randint(1, 8) * pattern_size)
        if random.random() < pattern_density:
            # 使用重复模式（但长度可变）
            for i in range(chunk_len):
                data[pos + i] = pattern[(i + pos) % pattern_size]
        else:
            # 使用随机数据
            for i in range(chunk_len):
                # add a little non-uniformity to distribution to avoid pure uniform randomness
                data[pos + i] = random.randint(0, 255)
        pos += chunk_len

    return bytes(data)


def _update_counts(counts, data):
    # counts is a list of 256 ints
    for b in data:
        counts[b] += 1


def shannon_entropy_from_counts(counts, total):
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c == 0:
            continue
        p = c / total
        ent -= p * math.log2(p)
    return ent


def gen_zero_chunks(size, chunk_size=128 * 1024):
    produced = 0
    while produced < size:
        chunk_len = min(chunk_size, size - produced)
        yield bytes([0]) * chunk_len
        produced += chunk_len


def gen_random_chunks(size, chunk_size=128 * 1024):
    produced = 0
    while produced < size:
        chunk_len = min(chunk_size, size - produced)
        yield os.urandom(chunk_len)
        produced += chunk_len


def gen_repeat_chunks(size, pattern_size=4, density=0.9995, symbols=1, noise=0.0001, chunk_size=64 * 1024):
    # Use a small symbols set to lower entropy compared to pure random
    sym = [random.randrange(0, 256) for _ in range(symbols)]
    produced = 0
    while produced < size:
        chunk_len = min(chunk_size, size - produced)
        arr = bytearray(chunk_len)
        pos = 0
        while pos < chunk_len:
            seg = min(chunk_len - pos, random.randint(1, 4) * pattern_size)
            if random.random() < density:
                # repeated region created using sym set
                for i in range(seg):
                    if random.random() < noise:
                        arr[pos + i] = random.randrange(0, 256)
                    else:
                        arr[pos + i] = sym[(i + produced + pos) % len(sym)]
            else:
                for i in range(seg):
                    arr[pos + i] = random.randrange(0, 256)
            pos += seg
        produced += chunk_len
        yield bytes(arr)


def gen_structured_chunks(size, structure_ratio=0.8, chunk_size=64 * 1024):
    produced = 0
    header_pattern = (b"{\"ts\":\"2025-11-27T00:00:00\", \"id\":\"user%03d\", \"action\": \"op\",\n}")
    while produced < size:
        chunk_len = min(chunk_size, size - produced)
        arr = bytearray()
        while len(arr) < chunk_len:
            if random.random() < 0.85:
                # structured line
                # uid, values with limited set to reduce entropy while preserving variety
                uid = random.randint(0, 31)
                val = random.randint(0, 255)
                ln = f"{{\"ts\":\"2025-11-27T00:00:{produced % 60:02d}\",\"uid\":\"u{uid:03d}\",\"val\":{val}}}\n".encode('utf-8')
            else:
                # small random blob
                ln = os.urandom(random.randint(8, 64))
            arr.extend(ln)
            if len(arr) > chunk_len:
                arr = arr[:chunk_len]
                break
        produced += chunk_len
        yield bytes(arr)


def gen_mixed_chunks(size, chunk_size=128 * 1024):
    produced = 0
    while produced < size:
        seg = min(chunk_size, size - produced)
        r = random.random()
        if r < 0.5:
            # repeat-ish
            for chunk in gen_repeat_chunks(seg, pattern_size=4, density=0.85, symbols=2, noise=0.002, chunk_size=seg):
                yield chunk
        elif r < 0.9:
            # structured-ish
            for chunk in gen_structured_chunks(seg, structure_ratio=0.85, chunk_size=seg):
                yield chunk
        else:
            # random-ish
            for chunk in gen_random_chunks(seg, chunk_size=seg):
                yield chunk
        produced += seg

def generate_structured_data(size, structure_ratio=0.8):
    """
    生成具有结构化重复模式的数据

    Args:
        size: 总数据大小
        structure_ratio: 结构化数据比例
    """
    data = bytearray(size)
    struct_size = int(size * structure_ratio)

    # 前半部分：结构化重复数据
    pattern = b"0123456789ABCDEF" * 4  # 64字节模式
    for i in range(struct_size // len(pattern) + 1):
        start_pos = i * len(pattern)
        end_pos = min(start_pos + len(pattern), struct_size)
        if start_pos < struct_size:
            data[start_pos:end_pos] = pattern[:end_pos - start_pos]

    # 后半部分：随机数据（但加入ASCII-like片段和垃圾二进制混合以提升复杂度）
    for i in range(struct_size, size):
        if random.random() < 0.15:
            # ASCII letters and punctuation
            data[i] = random.choice(b'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_@:.')
        else:
            data[i] = random.randint(0, 255)

    return bytes(data)


def _random_sizes_mb(count, min_mb=1, max_mb=512, seed=None):
    # Produce 'count' sizes (in bytes) between min_mb and max_mb by sampling
    # approximately log-uniformly to include small and very large files.
    if seed is not None:
        random.seed(seed)
    sizes = set()
    while len(sizes) < count:
        r = random.random()
        size_mb = int(math.exp(math.log(min_mb) + r * (math.log(max_mb) - math.log(min_mb))))
        if size_mb < 1:
            size_mb = 1
        if size_mb > max_mb:
            size_mb = max_mb
        sizes.add(size_mb)
    return sorted(sizes)


def generate_suite(out_dir, per_pattern=5, min_mb=1, max_mb=512, seed=None):
    """
    Generate a suite of diverse test files into out_dir.
    Files created include random sizes between min_mb and max_mb (MB) for a few
    patterns (zero, random, repeat, structured, mixed). Streaming generators are
    used to avoid high memory usage.
    """
    os.makedirs(out_dir, exist_ok=True)
    patterns = ["zero", "random", "repeat", "structured", "mixed"]
    print(f"Generating test suite in {out_dir} ...")

    def write_file_stream(outpath, gen):
        with open(outpath, 'wb') as f:
            for chunk in gen:
                f.write(chunk)

    # Determine sizes per pattern: a handful of random sizes (MB) between min_mb and max_mb
    for p in patterns:
        sizes_mb = _random_sizes_mb(per_pattern, min_mb=min_mb, max_mb=max_mb, seed=seed)
        for i, size_mb in enumerate(sizes_mb):
            fname = f"sample_{size_mb}mb_{p}_{i+1}.txt"
            outpath = os.path.join(out_dir, fname)
            if os.path.exists(outpath):
                print(f"Skipping existing {outpath}")
                continue
            size_val = size_mb * 1024 * 1024
            print(f"Generating {outpath} ({size_mb} MB, pattern={p})")
            if p == "zero":
                gen = gen_zero_chunks(size_val)
            elif p == "random":
                gen = gen_random_chunks(size_val)
            elif p == "repeat":
                # repeat: low-entropy defaults for suite
                gen = gen_repeat_chunks(size_val, pattern_size=1, density=0.9995, symbols=1, noise=0.0001)
            elif p == "structured":
                gen = gen_structured_chunks(size_val, structure_ratio=0.8)
            else:
                gen = gen_mixed_chunks(size_val)
            write_file_stream(outpath, gen)
    print("Test suite generation complete.")

def main():
    parser = argparse.ArgumentParser(description="生成LZ4 GPU测试数据")
    parser.add_argument("size", nargs='?', type=str, help="数据大小（如：64MB, 1GB）")
    parser.add_argument("--output", "-o", default="test_data.dat", help="输出文件名")
    parser.add_argument("--out-dir", dest="out_dir", default="samples", help="输出目录（用于 --suite）")
    parser.add_argument("--per-pattern", dest="per_pattern", type=int, default=3, help="每种模式生成的样本数量（用于 --suite）")
    parser.add_argument("--min-mb", dest="min_mb", type=int, default=1, help="最小文件大小（MB, 用于 --suite）")
    parser.add_argument("--max-mb", dest="max_mb", type=int, default=512, help="最大文件大小（MB, 用于 --suite）")
    parser.add_argument("--seed", dest="seed", type=int, default=None, help="随机种子（用于 --suite, 可复现）")
    parser.add_argument("--pattern", "-p", choices=["zero", "random", "repeat", "structured", "mixed"],
                       default="repeat", help="数据模式类型")
    parser.add_argument("--pattern-size", type=int, default=1, help="重复模式大小 (默认值1, 取值越小可获得更低熵)")
    parser.add_argument("--density", type=float, default=None, help="模式密度 (默认为基于模式的合理值)")
    parser.add_argument("--symbols", type=int, default=None, help="用于 repeat 尺度的符号数 (默认随模式) ")
    parser.add_argument("--noise", type=float, default=None, help="用于 repeat 模式的随机噪音比例 (默认随模式)")
    parser.add_argument("--suite", action="store_true", help="生成一套预定义的测试样例到 --out-dir 并退出")

    args = parser.parse_args()

    # If --suite requested, generate a suite and exit
    if args.suite:
        generate_suite(args.out_dir, per_pattern=args.per_pattern, min_mb=args.min_mb, max_mb=args.max_mb, seed=args.seed)
        return
        return

    if not args.size:
        parser.error("size is required unless --suite is specified")

    # 解析大小
    size_str = args.size.upper()
    if size_str.endswith('KB'):
        size = int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        size = int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        size = int(size_str[:-2]) * 1024 * 1024 * 1024
    elif size_str.endswith('B'):
        size = int(size_str[:-1])
    else:
        size = int(size_str)

    print(f"生成 {args.pattern} 模式测试数据，大小: {size:,} 字节")
    print(f"输出文件: {args.output}")

    # 根据模式生成数据
    # Use streaming writes for large files to limit memory usage
    def write_file_stream(outpath, generator_or_bytes):
        with open(outpath, 'wb') as f:
            if isinstance(generator_or_bytes, (bytes, bytearray)):
                f.write(generator_or_bytes)
            else:
                for chunk in generator_or_bytes:
                    f.write(chunk)

    if args.pattern == "zero":
        # streaming zeros
        def gen_zero_chunks():
            produced = 0
            block = 128 * 1024
            while produced < size:
                chunk_len = min(block, size - produced)
                yield bytes([0]) * chunk_len
                produced += chunk_len
        write_file_stream(args.output, gen_zero_chunks())
    elif args.pattern == "random":
        # streaming random generator
        def gen_random_chunks():
            produced = 0
            block = 128 * 1024
            while produced < size:
                chunk_len = min(block, size - produced)
                yield os.urandom(chunk_len)
                produced += chunk_len
        write_file_stream(args.output, gen_random_chunks())
    elif args.pattern == "repeat":
        # For repeat use top-level generator with low-entropy defaults
        pd = args.density if args.density is not None else 0.9995
        sy = args.symbols if args.symbols is not None else 1
        no = args.noise if args.noise is not None else 0.0001
        write_file_stream(args.output, gen_repeat_chunks(size, pattern_size=args.pattern_size, density=pd, symbols=sy, noise=no, chunk_size=64 * 1024))
    elif args.pattern == "structured":
        # streaming structured writer
        def gen_structured_chunks():
            pattern = b"0123456789ABCDEF" * 4
            produced = 0
            block = 64 * 1024
            while produced < size:
                chunk_len = min(block, size - produced)
                arr = bytearray(chunk_len)
                struct_density = args.density if args.density is not None else 0.8
                if produced < int(size * struct_density):
                    # structured area
                    for i in range(chunk_len):
                        arr[i] = pattern[(i + produced) % len(pattern)]
                else:
                    # semi-random with ASCII flavor
                    for i in range(chunk_len):
                        if random.random() < 0.15:
                            arr[i] = random.choice(b'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_@:.')
                        else:
                            arr[i] = random.getrandbits(8)
                produced += chunk_len
                yield bytes(arr)
        write_file_stream(args.output, gen_structured_chunks())
    else:  # mixed and other options fallback to 'mixed'
        # Use the top-level mixed generator which composes repeat/structured/random
        write_file_stream(args.output, gen_mixed_chunks(size, chunk_size=128 * 1024))

    # 写入文件 handled by the generators above; ensure output exists
    if not os.path.exists(args.output):
        print('Failed to create output, abort')
        sys.exit(1)

    fsize = os.path.getsize(args.output)
    print(f"成功生成 {fsize:,} 字节测试数据")

    # 显示数据特征
    # compute a sample-based Shannon entropy (bits per byte) to report 'data complexity'
    def shannon_entropy_sample(path, sample_size=256*1024):
        # read either the whole file if smaller or a sample slice
        sz = os.path.getsize(path)
        import math
        import random
        if sz == 0:
            return 0.0
        with open(path, 'rb') as fo:
            if sz <= sample_size:
                b = fo.read()
            else:
                # Pick deterministic samples evenly spaced across file to reduce
                # variance in entropy due to random sampling placement.
                # sample 4 evenly spaced slices (start, 1/3, 2/3, end) for more
                # representative coverage without randomness
                third = max(0, (sz - sample_size) // 3)
                offsets = [0, third, 2*third, max(0, sz - sample_size)]
                parts = []
                for off in offsets:
                    fo.seek(off)
                    parts.append(fo.read(sample_size))
                b = b"".join(parts)
        freq = [0] * 256
        for ch in b:
            freq[ch] += 1
        total = len(b)
        ent = 0.0
        for f in freq:
            if f:
                p = f / total
                ent -= p * math.log2(p)
        return ent

    entropy = shannon_entropy_sample(args.output)
    print(f"Shannon Entropy (bits/byte): {entropy:.3f} (max 8.0)")

if __name__ == "__main__":
    main()
