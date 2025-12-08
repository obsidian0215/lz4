#!/usr/bin/env python3
"""
Performance Analysis for LZ4 GPU Local vs Daemon Mode Tests

Analyzes results_agg.csv to:
1. Compare local vs daemon mode performance per sample
2. Identify performance bottlenecks (kernel, H2D, D2H transfers)
3. Find optimal configurations per sample
4. Generate summary statistics
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

def load_raw_results(csv_path):
    """Load raw results CSV and aggregate"""
    df = pd.read_csv(csv_path, na_values=['NA', 'N/A', ''])

    # Ensure numeric columns are properly typed
    numeric_cols = ['input_bytes', 'compressed_bytes',
                    'comp_total_ms', 'comp_kernel_ms', 'comp_h2d_ms', 'comp_d2h_ms',
                    'dec_total_ms', 'dec_kernel_ms', 'dec_h2d_ms', 'dec_d2h_ms',
                    'compression_ratio']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter to OK rows only
    df = df[df['ok'] == 'OK']

    # Aggregate by configuration
    group_cols = ['sample', 'accel', 'block_size', 'local', 'pinned', 'mode']
    agg_dict = {
        'input_bytes': 'first',
        'compressed_bytes': 'mean',
        'comp_total_ms': 'mean',
        'comp_kernel_ms': 'mean',
        'comp_h2d_ms': 'mean',
        'comp_d2h_ms': 'mean',
        'dec_total_ms': 'mean',
        'dec_kernel_ms': 'mean',
        'dec_h2d_ms': 'mean',
        'dec_d2h_ms': 'mean',
        'compression_ratio': 'mean',
        'run': 'count'
    }

    agg_df = df.groupby(group_cols).agg(agg_dict).reset_index()
    agg_df = agg_df.rename(columns={
        'run': 'runs',
        'compressed_bytes': 'avg_compressed_bytes',
        'comp_total_ms': 'avg_comp_total_ms',
        'comp_kernel_ms': 'avg_comp_kernel_ms',
        'comp_h2d_ms': 'avg_comp_h2d_ms',
        'comp_d2h_ms': 'avg_comp_d2h_ms',
        'dec_total_ms': 'avg_decomp_total_ms',
        'dec_kernel_ms': 'avg_decomp_kernel_ms',
        'dec_h2d_ms': 'avg_decomp_h2d_ms',
        'dec_d2h_ms': 'avg_decomp_d2h_ms',
        'compression_ratio': 'avg_compression_ratio'
    })

    return agg_df

def load_results(csv_path):
    """Load aggregated results CSV"""
    df = pd.read_csv(csv_path, na_values=['NA', 'N/A', ''])
    # Ensure numeric columns are properly typed
    numeric_cols = ['runs', 'successes', 'avg_compressed_bytes',
                    'avg_comp_total_ms', 'avg_comp_kernel_ms', 'avg_comp_h2d_ms', 'avg_comp_d2h_ms',
                    'avg_decomp_total_ms', 'avg_decomp_kernel_ms', 'avg_decomp_h2d_ms', 'avg_decomp_d2h_ms',
                    'avg_compression_ratio', 'success_rate']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def calculate_throughput(df):
    """Calculate throughput metrics"""
    # Get input size from sample name pattern or use compressed bytes * ratio
    df = df.copy()

    # Estimate input bytes from compressed_bytes * ratio
    df['est_input_mb'] = (df['avg_compressed_bytes'] * df['avg_compression_ratio']) / (1024 * 1024)

    # Compression throughput (MB/s)
    df['comp_throughput_total'] = df['est_input_mb'] / (df['avg_comp_total_ms'] / 1000)
    df['comp_throughput_kernel'] = df['est_input_mb'] / (df['avg_comp_kernel_ms'] / 1000)

    # Decompression throughput (MB/s)
    df['decomp_throughput_total'] = df['est_input_mb'] / (df['avg_decomp_total_ms'] / 1000)
    df['decomp_throughput_kernel'] = df['est_input_mb'] / (df['avg_decomp_kernel_ms'] / 1000)

    # Calculate overhead percentages
    df['comp_h2d_pct'] = (df['avg_comp_h2d_ms'] / df['avg_comp_total_ms']) * 100
    df['comp_d2h_pct'] = (df['avg_comp_d2h_ms'] / df['avg_comp_total_ms']) * 100
    df['comp_kernel_pct'] = (df['avg_comp_kernel_ms'] / df['avg_comp_total_ms']) * 100
    df['comp_overhead_pct'] = 100 - df['comp_kernel_pct'] - df['comp_h2d_pct'] - df['comp_d2h_pct']

    df['decomp_h2d_pct'] = (df['avg_decomp_h2d_ms'] / df['avg_decomp_total_ms']) * 100
    df['decomp_d2h_pct'] = (df['avg_decomp_d2h_ms'] / df['avg_decomp_total_ms']) * 100
    df['decomp_kernel_pct'] = (df['avg_decomp_kernel_ms'] / df['avg_decomp_total_ms']) * 100

    return df

def analyze_local_vs_daemon(df):
    """Compare local vs daemon mode performance"""
    print("\n" + "="*80)
    print("LOCAL vs DAEMON MODE COMPARISON")
    print("="*80)

    local_df = df[df['mode'] == 'local']
    daemon_df = df[df['mode'] == 'daemon']

    print(f"\nTotal configurations: Local={len(local_df)}, Daemon={len(daemon_df)}")

    # Overall averages
    print("\n--- Overall Averages ---")
    for mode, mode_df in [('Local', local_df), ('Daemon', daemon_df)]:
        if len(mode_df) == 0:
            print(f"  {mode}: No data")
            continue
        print(f"\n  {mode} Mode:")
        print(f"    Avg Compression Total:  {mode_df['avg_comp_total_ms'].mean():.2f} ms")
        print(f"    Avg Compression Kernel: {mode_df['avg_comp_kernel_ms'].mean():.2f} ms")
        print(f"    Avg Decompression Total:  {mode_df['avg_decomp_total_ms'].mean():.2f} ms")
        print(f"    Avg Decompression Kernel: {mode_df['avg_decomp_kernel_ms'].mean():.2f} ms")
        if 'comp_throughput_total' in mode_df.columns:
            print(f"    Avg Comp Throughput (total): {mode_df['comp_throughput_total'].mean():.2f} MB/s")
            print(f"    Avg Decomp Throughput (total): {mode_df['decomp_throughput_total'].mean():.2f} MB/s")

    # Speedup calculation (where both modes have data)
    if len(local_df) > 0 and len(daemon_df) > 0:
        print("\n--- Daemon vs Local Speedup ---")
        # Merge on common keys
        merge_keys = ['sample', 'accel', 'block_size', 'local', 'pinned']
        merged = pd.merge(local_df, daemon_df, on=merge_keys, suffixes=('_local', '_daemon'))

        if len(merged) > 0:
            merged['comp_speedup'] = merged['avg_comp_total_ms_local'] / merged['avg_comp_total_ms_daemon']
            merged['decomp_speedup'] = merged['avg_decomp_total_ms_local'] / merged['avg_decomp_total_ms_daemon']

            print(f"  Matched configurations: {len(merged)}")
            print(f"  Compression speedup (local/daemon): {merged['comp_speedup'].mean():.3f}x (median: {merged['comp_speedup'].median():.3f}x)")
            print(f"  Decompression speedup (local/daemon): {merged['decomp_speedup'].mean():.3f}x (median: {merged['decomp_speedup'].median():.3f}x)")

            # Cases where daemon is faster
            daemon_faster_comp = (merged['comp_speedup'] < 1).sum()
            daemon_faster_decomp = (merged['decomp_speedup'] < 1).sum()
            print(f"  Daemon faster for compression: {daemon_faster_comp}/{len(merged)} ({100*daemon_faster_comp/len(merged):.1f}%)")
            print(f"  Daemon faster for decompression: {daemon_faster_decomp}/{len(merged)} ({100*daemon_faster_decomp/len(merged):.1f}%)")

def analyze_by_sample(df):
    """Analyze performance per sample file"""
    print("\n" + "="*80)
    print("PER-SAMPLE ANALYSIS")
    print("="*80)

    samples = df['sample'].unique()
    print(f"\nTotal unique samples: {len(samples)}")

    sample_stats = []
    for sample in samples:
        sample_df = df[df['sample'] == sample]
        local_df = sample_df[sample_df['mode'] == 'local']
        daemon_df = sample_df[sample_df['mode'] == 'daemon']

        stats = {
            'sample': sample,
            'configs': len(sample_df),
            'local_configs': len(local_df),
            'daemon_configs': len(daemon_df),
        }

        # Best local config (by compression throughput)
        if len(local_df) > 0 and 'comp_throughput_total' in local_df.columns:
            best_local = local_df.loc[local_df['comp_throughput_total'].idxmax()]
            stats['best_local_comp_th'] = best_local['comp_throughput_total']
            stats['best_local_config'] = f"accel={best_local['accel']},bs={best_local['block_size']},local={best_local['local']},pinned={best_local['pinned']}"
            stats['local_avg_comp_ms'] = local_df['avg_comp_total_ms'].mean()
            stats['local_avg_kernel_ms'] = local_df['avg_comp_kernel_ms'].mean()

        # Best daemon config
        if len(daemon_df) > 0 and 'comp_throughput_total' in daemon_df.columns:
            best_daemon = daemon_df.loc[daemon_df['comp_throughput_total'].idxmax()]
            stats['best_daemon_comp_th'] = best_daemon['comp_throughput_total']
            stats['best_daemon_config'] = f"accel={best_daemon['accel']},bs={best_daemon['block_size']},local={best_daemon['local']},pinned={best_daemon['pinned']}"
            stats['daemon_avg_comp_ms'] = daemon_df['avg_comp_total_ms'].mean()
            stats['daemon_avg_kernel_ms'] = daemon_df['avg_comp_kernel_ms'].mean()

        sample_stats.append(stats)

    stats_df = pd.DataFrame(sample_stats)

    # Print top samples by throughput
    print("\n--- Top 10 Samples by Best Local Compression Throughput ---")
    if 'best_local_comp_th' in stats_df.columns:
        top_local = stats_df.nlargest(10, 'best_local_comp_th')[['sample', 'best_local_comp_th', 'best_local_config']]
        for _, row in top_local.iterrows():
            print(f"  {row['sample'][:50]:50s} {row['best_local_comp_th']:8.2f} MB/s  [{row['best_local_config']}]")

    print("\n--- Bottom 10 Samples by Best Local Compression Throughput (potential bottlenecks) ---")
    if 'best_local_comp_th' in stats_df.columns:
        bottom_local = stats_df.nsmallest(10, 'best_local_comp_th')[['sample', 'best_local_comp_th', 'best_local_config']]
        for _, row in bottom_local.iterrows():
            print(f"  {row['sample'][:50]:50s} {row['best_local_comp_th']:8.2f} MB/s  [{row['best_local_config']}]")

    return stats_df

def analyze_optimal_configs(df):
    """Find optimal configurations across all samples"""
    print("\n" + "="*80)
    print("OPTIMAL CONFIGURATION ANALYSIS")
    print("="*80)

    # Group by configuration parameters (excluding sample)
    config_cols = ['accel', 'block_size', 'local', 'pinned', 'mode']

    config_stats = df.groupby(config_cols).agg({
        'avg_comp_total_ms': 'mean',
        'avg_comp_kernel_ms': 'mean',
        'avg_decomp_total_ms': 'mean',
        'avg_decomp_kernel_ms': 'mean',
        'comp_throughput_total': 'mean',
        'decomp_throughput_total': 'mean',
        'sample': 'count'
    }).rename(columns={'sample': 'sample_count'}).reset_index()

    print("\n--- Top 10 Configurations by Average Compression Throughput (Local) ---")
    local_configs = config_stats[config_stats['mode'] == 'local'].nlargest(10, 'comp_throughput_total')
    for _, row in local_configs.iterrows():
        print(f"  accel={row['accel']:2}, bs={row['block_size']:5}, local={row['local']:3}, pinned={row['pinned']:9} | "
              f"Comp: {row['comp_throughput_total']:7.2f} MB/s | Decomp: {row['decomp_throughput_total']:7.2f} MB/s | "
              f"samples={row['sample_count']}")

    print("\n--- Top 10 Configurations by Average Compression Throughput (Daemon) ---")
    daemon_configs = config_stats[config_stats['mode'] == 'daemon'].nlargest(10, 'comp_throughput_total')
    for _, row in daemon_configs.iterrows():
        print(f"  accel={row['accel']:2}, bs={row['block_size']:5}, local={row['local']:3}, pinned={row['pinned']:9} | "
              f"Comp: {row['comp_throughput_total']:7.2f} MB/s | Decomp: {row['decomp_throughput_total']:7.2f} MB/s | "
              f"samples={row['sample_count']}")

    print("\n--- Top 10 Configurations by Average Decompression Throughput (Local) ---")
    local_decomp = config_stats[config_stats['mode'] == 'local'].nlargest(10, 'decomp_throughput_total')
    for _, row in local_decomp.iterrows():
        print(f"  accel={row['accel']:2}, bs={row['block_size']:5}, local={row['local']:3}, pinned={row['pinned']:9} | "
              f"Decomp: {row['decomp_throughput_total']:7.2f} MB/s | Comp: {row['comp_throughput_total']:7.2f} MB/s")

    return config_stats

def analyze_bottlenecks(df):
    """Analyze performance bottlenecks"""
    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)

    local_df = df[df['mode'] == 'local'].copy()

    if len(local_df) == 0:
        print("No local mode data available")
        return

    # Calculate time breakdown
    print("\n--- Compression Time Breakdown (Local Mode Averages) ---")
    avg_total = local_df['avg_comp_total_ms'].mean()
    avg_kernel = local_df['avg_comp_kernel_ms'].mean()
    avg_h2d = local_df['avg_comp_h2d_ms'].mean()
    avg_d2h = local_df['avg_comp_d2h_ms'].mean()
    avg_other = avg_total - avg_kernel - avg_h2d - avg_d2h

    print(f"  Total Time:     {avg_total:8.2f} ms (100%)")
    print(f"  Kernel Time:    {avg_kernel:8.2f} ms ({100*avg_kernel/avg_total:5.1f}%)")
    print(f"  H2D Transfer:   {avg_h2d:8.2f} ms ({100*avg_h2d/avg_total:5.1f}%)")
    print(f"  D2H Transfer:   {avg_d2h:8.2f} ms ({100*avg_d2h/avg_total:5.1f}%)")
    print(f"  Other Overhead: {avg_other:8.2f} ms ({100*avg_other/avg_total:5.1f}%)")

    print("\n--- Decompression Time Breakdown (Local Mode Averages) ---")
    avg_dec_total = local_df['avg_decomp_total_ms'].mean()
    avg_dec_kernel = local_df['avg_decomp_kernel_ms'].mean()
    avg_dec_h2d = local_df['avg_decomp_h2d_ms'].mean()
    avg_dec_d2h = local_df['avg_decomp_d2h_ms'].mean()
    avg_dec_other = avg_dec_total - avg_dec_kernel - avg_dec_h2d - avg_dec_d2h

    print(f"  Total Time:     {avg_dec_total:8.2f} ms (100%)")
    print(f"  Kernel Time:    {avg_dec_kernel:8.2f} ms ({100*avg_dec_kernel/avg_dec_total:5.1f}%)")
    print(f"  H2D Transfer:   {avg_dec_h2d:8.2f} ms ({100*avg_dec_h2d/avg_dec_total:5.1f}%)")
    print(f"  D2H Transfer:   {avg_dec_d2h:8.2f} ms ({100*avg_dec_d2h/avg_dec_total:5.1f}%)")
    print(f"  Other Overhead: {avg_dec_other:8.2f} ms ({100*avg_dec_other/avg_dec_total:5.1f}%)")

    # Identify bottleneck type per config
    print("\n--- Bottleneck Distribution (by dominant time component) ---")
    local_df['bottleneck'] = 'other'
    local_df.loc[local_df['avg_comp_kernel_ms'] >= local_df[['avg_comp_kernel_ms', 'avg_comp_h2d_ms', 'avg_comp_d2h_ms']].max(axis=1), 'bottleneck'] = 'kernel'
    local_df.loc[local_df['avg_comp_h2d_ms'] >= local_df[['avg_comp_kernel_ms', 'avg_comp_h2d_ms', 'avg_comp_d2h_ms']].max(axis=1), 'bottleneck'] = 'h2d_transfer'
    local_df.loc[local_df['avg_comp_d2h_ms'] >= local_df[['avg_comp_kernel_ms', 'avg_comp_h2d_ms', 'avg_comp_d2h_ms']].max(axis=1), 'bottleneck'] = 'd2h_transfer'

    bottleneck_counts = local_df['bottleneck'].value_counts()
    for bt, count in bottleneck_counts.items():
        print(f"  {bt}: {count} configs ({100*count/len(local_df):.1f}%)")

def analyze_by_parameter(df):
    """Analyze impact of each parameter"""
    print("\n" + "="*80)
    print("PARAMETER IMPACT ANALYSIS")
    print("="*80)

    local_df = df[df['mode'] == 'local']

    if len(local_df) == 0:
        print("No local mode data available")
        return

    # Impact of acceleration level
    print("\n--- Impact of Acceleration Level ---")
    accel_stats = local_df.groupby('accel').agg({
        'comp_throughput_total': 'mean',
        'decomp_throughput_total': 'mean',
        'avg_comp_total_ms': 'mean',
        'avg_compression_ratio': 'mean'
    })
    for accel, row in accel_stats.iterrows():
        print(f"  accel={accel}: Comp={row['comp_throughput_total']:.2f} MB/s, Decomp={row['decomp_throughput_total']:.2f} MB/s, "
              f"Ratio={row['avg_compression_ratio']:.3f}")

    # Impact of block size
    print("\n--- Impact of Block Size ---")
    block_stats = local_df.groupby('block_size').agg({
        'comp_throughput_total': 'mean',
        'decomp_throughput_total': 'mean',
        'avg_comp_total_ms': 'mean'
    })
    for bs, row in block_stats.iterrows():
        print(f"  block_size={bs}: Comp={row['comp_throughput_total']:.2f} MB/s, Decomp={row['decomp_throughput_total']:.2f} MB/s")

    # Impact of local workgroup size
    print("\n--- Impact of Local Workgroup Size ---")
    local_stats = local_df.groupby('local').agg({
        'comp_throughput_total': 'mean',
        'decomp_throughput_total': 'mean',
        'avg_comp_kernel_ms': 'mean'
    })
    for local, row in local_stats.iterrows():
        print(f"  local={local}: Comp={row['comp_throughput_total']:.2f} MB/s, Decomp={row['decomp_throughput_total']:.2f} MB/s, "
              f"Kernel={row['avg_comp_kernel_ms']:.2f} ms")

    # Impact of pinned memory
    print("\n--- Impact of Pinned Memory ---")
    pinned_stats = local_df.groupby('pinned').agg({
        'comp_throughput_total': 'mean',
        'decomp_throughput_total': 'mean',
        'avg_comp_h2d_ms': 'mean',
        'avg_comp_d2h_ms': 'mean'
    })
    for pinned, row in pinned_stats.iterrows():
        print(f"  pinned={pinned}: Comp={row['comp_throughput_total']:.2f} MB/s, Decomp={row['decomp_throughput_total']:.2f} MB/s, "
              f"H2D={row['avg_comp_h2d_ms']:.2f} ms, D2H={row['avg_comp_d2h_ms']:.2f} ms")

def main():
    if len(sys.argv) < 2:
        # Default path
        csv_path = "/root/lz4/lz4_gpu/ab_results/full_local_daemon_20251205_175201/results.csv"
    else:
        csv_path = sys.argv[1]

    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    print(f"Loading results from: {csv_path}")

    # Check if this is raw or aggregated data
    if 'results_agg' in csv_path:
        df = load_results(csv_path)
        print(f"Loaded {len(df)} aggregated configurations")
    else:
        df = load_raw_results(csv_path)
        print(f"Loaded and aggregated to {len(df)} configurations")

    # Calculate throughput metrics
    df = calculate_throughput(df)

    # Run analyses
    analyze_local_vs_daemon(df)
    sample_stats = analyze_by_sample(df)
    config_stats = analyze_optimal_configs(df)
    analyze_bottlenecks(df)
    analyze_by_parameter(df)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)

    local_df = df[df['mode'] == 'local']
    if len(local_df) > 0:
        best_config = local_df.loc[local_df['comp_throughput_total'].idxmax()]
        print(f"\n  Best Overall Configuration (by compression throughput):")
        print(f"    Mode: local")
        print(f"    Accel: {best_config['accel']}")
        print(f"    Block Size: {best_config['block_size']}")
        print(f"    Local Size: {best_config['local']}")
        print(f"    Pinned: {best_config['pinned']}")
        print(f"    Throughput: {best_config['comp_throughput_total']:.2f} MB/s (compression)")
        print(f"                {best_config['decomp_throughput_total']:.2f} MB/s (decompression)")

if __name__ == "__main__":
    main()
