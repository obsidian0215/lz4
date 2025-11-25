#!/bin/bash
# GPU性能控制统一接口
# 支持 NVIDIA, AMD, Intel GPU

set -e

GPU_VENDOR=""
GPU_DEVICE=""

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检测GPU类型
detect_gpu() {
    log_info "Detecting GPU..."

    if command -v nvidia-smi &>/dev/null; then
        GPU_VENDOR="nvidia"
        GPU_DEVICE=$(nvidia-smi -L 2>/dev/null | head -1 | cut -d: -f1 | cut -d' ' -f2 || echo "0")
        local gpu_name=$(nvidia-smi -L 2>/dev/null | head -1 | cut -d: -f2 | cut -d'(' -f1 || echo "Unknown")
        log_info "NVIDIA GPU detected: $gpu_name"
        return 0
    fi

    if command -v rocm-smi &>/dev/null; then
        GPU_VENDOR="amd"
        GPU_DEVICE="0"
        local gpu_name=$(rocm-smi --showproductname 2>/dev/null | grep "GPU" | head -1 || echo "Unknown AMD GPU")
        log_info "AMD GPU detected: $gpu_name"
        return 0
    fi

    if [ -e /sys/class/drm/card0/device/vendor ]; then
        local vendor=$(cat /sys/class/drm/card0/device/vendor 2>/dev/null)
        if [ "$vendor" = "0x8086" ]; then
            GPU_VENDOR="intel"
            GPU_DEVICE="0"
            log_info "Intel GPU detected"
            return 0
        fi
    fi

    # 尝试通过OpenCL检测
    if command -v clinfo &>/dev/null; then
        local cl_vendor=$(clinfo 2>/dev/null | grep "Device Vendor" | head -1 | awk '{print $3}')
        case "$cl_vendor" in
            NVIDIA)
                GPU_VENDOR="nvidia"
                GPU_DEVICE="0"
                log_warn "NVIDIA GPU detected via OpenCL (nvidia-smi not found)"
                return 0
                ;;
            Advanced|AMD)
                GPU_VENDOR="amd"
                GPU_DEVICE="0"
                log_warn "AMD GPU detected via OpenCL (rocm-smi not found)"
                return 0
                ;;
            Intel)
                GPU_VENDOR="intel"
                GPU_DEVICE="0"
                log_warn "Intel GPU detected via OpenCL"
                return 0
                ;;
        esac
    fi

    log_error "No GPU detected or unsupported vendor"
    return 1
}

# 查询GPU信息
query_gpu_info() {
    case $GPU_VENDOR in
        nvidia)
            echo "=== NVIDIA GPU Information ==="
            nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
            echo ""
            echo "Current Clock Frequencies:"
            nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory --format=csv,noheader
            echo ""
            echo "Max Clock Frequencies:"
            nvidia-smi --query-gpu=clocks.max.graphics,clocks.max.memory --format=csv,noheader
            echo ""
            echo "Power:"
            nvidia-smi --query-gpu=power.draw,power.limit --format=csv,noheader
            ;;
        amd)
            echo "=== AMD GPU Information ==="
            rocm-smi --showproductname --showid
            echo ""
            rocm-smi --showclocks
            echo ""
            rocm-smi --showpower
            ;;
        intel)
            echo "=== Intel GPU Information ==="
            if [ -e /sys/class/drm/card0/device/device ]; then
                echo "Device ID: $(cat /sys/class/drm/card0/device/device)"
            fi
            if [ -e /sys/class/drm/card0/gt_max_freq_mhz ]; then
                echo "Max Frequency: $(cat /sys/class/drm/card0/gt_max_freq_mhz) MHz"
                echo "Current Frequency: $(cat /sys/class/drm/card0/gt_cur_freq_mhz) MHz"
                echo "Min Frequency: $(cat /sys/class/drm/card0/gt_min_freq_mhz) MHz"
            fi
            ;;
        *)
            log_error "GPU vendor not detected"
            return 1
            ;;
    esac
}

# 设置GPU频率 (百分比: 0-100)
set_gpu_freq() {
    local freq_percent=$1

    if [ -z "$freq_percent" ] || [ "$freq_percent" -lt 0 ] || [ "$freq_percent" -gt 100 ]; then
        log_error "Invalid frequency percent: $freq_percent (must be 0-100)"
        return 1
    fi

    log_info "Setting GPU frequency to ${freq_percent}%..."

    case $GPU_VENDOR in
        nvidia)
            # NVIDIA需要先查询支持的频率
            local max_freq=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | head -1)
            if [ -z "$max_freq" ]; then
                log_error "Failed to query max GPU frequency"
                return 1
            fi
            local target_freq=$((max_freq * freq_percent / 100))
            log_info "Target frequency: ${target_freq} MHz (max: ${max_freq} MHz)"

            sudo nvidia-smi -pm 1 || log_warn "Failed to enable persistence mode"
            sudo nvidia-smi -lgc ${target_freq},${target_freq} || {
                log_error "Failed to set frequency. Try: sudo nvidia-smi -rac (reset)"
                return 1
            }
            ;;
        amd)
            # AMD使用性能级别 0-7
            local level=$((freq_percent * 7 / 100))
            log_info "Setting AMD performance level to $level"
            sudo rocm-smi --setperflevel manual || true
            sudo rocm-smi --setsclk $level || {
                log_error "Failed to set frequency"
                return 1
            }
            ;;
        intel)
            if [ ! -e /sys/class/drm/card0/gt_max_freq_mhz ]; then
                log_error "Intel GPU frequency control not available"
                return 1
            fi
            local max_freq=$(cat /sys/class/drm/card0/gt_RP0_freq_mhz 2>/dev/null || echo 1500)
            local rp1_freq=$(cat /sys/class/drm/card0/gt_RP1_freq_mhz 2>/dev/null || echo 300)
            local target_freq=$((max_freq * freq_percent / 100))

            # Intel GPU的实际最小频率通常是RP1而非RPn
            if [ $target_freq -lt $rp1_freq ]; then
                log_warn "Target ${target_freq} MHz < RP1 ${rp1_freq} MHz, using RP1"
                target_freq=$rp1_freq
            fi

            log_info "Target frequency: ${target_freq} MHz (range: ${rp1_freq}-${max_freq} MHz)"

            # Intel GPU约束：min必须<=max
            # 策略：先设置max到最大值，再设置min，最后设置实际的max
            # 这样可以避免违反约束
            echo $max_freq | sudo tee /sys/class/drm/card0/gt_max_freq_mhz >/dev/null || {
                log_error "Failed to set temporary max frequency"
                return 1
            }
            echo $target_freq | sudo tee /sys/class/drm/card0/gt_min_freq_mhz >/dev/null || {
                log_error "Failed to set min frequency"
                return 1
            }
            echo $target_freq | sudo tee /sys/class/drm/card0/gt_max_freq_mhz >/dev/null || {
                log_error "Failed to set max frequency"
                return 1
            }
            ;;
        *)
            log_error "Unknown GPU vendor"
            return 1
            ;;
    esac

    log_info "GPU frequency set successfully"
}

# 设置GPU功率限制 (瓦特)
set_gpu_power() {
    local power_watts=$1

    if [ -z "$power_watts" ] || [ "$power_watts" -le 0 ]; then
        log_error "Invalid power limit: $power_watts (must be > 0)"
        return 1
    fi

    log_info "Setting GPU power limit to ${power_watts}W..."

    case $GPU_VENDOR in
        nvidia)
            sudo nvidia-smi -pl $power_watts || {
                log_error "Failed to set power limit"
                return 1
            }
            ;;
        amd)
            # ROCm使用百分比调整 (相对默认功率)
            local default_power=$(rocm-smi --showpower | grep "Average" | awk '{print $5}' | head -1 | tr -d 'W')
            if [ -z "$default_power" ]; then
                default_power=200  # 假设默认200W
            fi
            local percent=$(( (power_watts - default_power) * 100 / default_power ))
            log_info "Power overdrive: ${percent}%"
            sudo rocm-smi --setpoweroverdrive $percent || {
                log_error "Failed to set power limit"
                return 1
            }
            ;;
        intel)
            local power_uw=$((power_watts * 1000000))
            local hwmon_path=$(find /sys/class/drm/card0/device/hwmon -name "power1_cap" 2>/dev/null | head -1)
            if [ -z "$hwmon_path" ]; then
                log_error "Intel GPU power control not available"
                return 1
            fi
            echo $power_uw | sudo tee $hwmon_path >/dev/null || {
                log_error "Failed to set power limit"
                return 1
            }
            ;;
        *)
            log_error "Unknown GPU vendor"
            return 1
            ;;
    esac

    log_info "GPU power limit set successfully"
}

# 重置GPU设置到默认
reset_gpu() {
    log_info "Resetting GPU to default settings..."

    case $GPU_VENDOR in
        nvidia)
            sudo nvidia-smi -pm 0 || true
            sudo nvidia-smi -rgc || log_warn "Failed to reset GPU clocks"
            sudo nvidia-smi -rac || log_warn "Failed to reset application clocks"
            ;;
        amd)
            sudo rocm-smi --setperflevel auto || true
            sudo rocm-smi --resetclocks || true
            sudo rocm-smi --setpoweroverdrive 0 || true
            ;;
        intel)
            if [ -e /sys/class/drm/card0/gt_max_freq_mhz ]; then
                local max_freq=$(cat /sys/class/drm/card0/gt_RP0_freq_mhz)
                echo $max_freq | sudo tee /sys/class/drm/card0/gt_max_freq_mhz >/dev/null || true
                local min_freq=$(cat /sys/class/drm/card0/gt_RPn_freq_mhz)
                echo $min_freq | sudo tee /sys/class/drm/card0/gt_min_freq_mhz >/dev/null || true
            fi
            ;;
    esac

    log_info "GPU reset complete"
}

# 设置GPU性能模式
set_gpu_mode() {
    local mode=$1

    case $mode in
        performance|max|high)
            log_info "Setting GPU to PERFORMANCE mode (100% frequency)..."
            set_gpu_freq 100
            ;;
        balanced|normal|medium)
            log_info "Setting GPU to BALANCED mode (70% frequency)..."
            set_gpu_freq 70
            ;;
        powersave|eco|low)
            log_info "Setting GPU to POWERSAVE mode (40% frequency)..."
            set_gpu_freq 40
            ;;
        *)
            log_error "Unknown mode: $mode"
            echo "Available modes: performance, balanced, powersave"
            return 1
            ;;
    esac
}

# 设置GPU温度限制 (仅NVIDIA)
set_gpu_temp_limit() {
    local temp_celsius=$1

    if [ -z "$temp_celsius" ] || [ "$temp_celsius" -le 0 ] || [ "$temp_celsius" -gt 100 ]; then
        log_error "Invalid temperature: $temp_celsius (must be 1-100°C)"
        return 1
    fi

    log_info "Setting GPU temperature limit to ${temp_celsius}°C..."

    case $GPU_VENDOR in
        nvidia)
            sudo nvidia-smi -gtt $temp_celsius || {
                log_error "Failed to set temperature limit"
                return 1
            }
            log_info "Temperature limit set successfully"
            ;;
        *)
            log_warn "Temperature limit not supported for $GPU_VENDOR GPU"
            return 1
            ;;
    esac
}

# 显示当前GPU状态 (简洁版)
show_gpu_status() {
    case $GPU_VENDOR in
        nvidia)
            echo "=== GPU Status ==="
            nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,clocks.current.graphics,power.draw --format=csv,noheader | while IFS=',' read name temp util_gpu util_mem freq power; do
                echo "  GPU: $name"
                echo "  Temperature: $temp"
                echo "  GPU Usage: $util_gpu"
                echo "  Memory Usage: $util_mem"
                echo "  Frequency: $freq"
                echo "  Power: $power"
            done
            ;;
        amd)
            echo "=== GPU Status ==="
            rocm-smi --showuse --showtemp --showclocks --showpower | grep -E "GPU|Temperature|Power|Level"
            ;;
        intel)
            echo "=== GPU Status ==="

            # 使用intel_gpu_top获取实时统计（如果可用）
            if command -v intel_gpu_top &>/dev/null; then
                # 获取一次采样数据（JSON格式）- 跳过第一个样本，使用第二个
                local gpu_data=$(timeout 3 intel_gpu_top -J -s 500 2>/dev/null | grep -A 100 "\"period\"" | tail -50 | tr '\n' ' ' | grep -o '{[^}]*"frequency"[^}]*}' | head -1 || echo "")

                if [ -n "$gpu_data" ]; then
                    # 解析频率
                    local freq_actual=$(echo "$gpu_data" | grep -o '"actual":[[:space:]]*[0-9.]*' | grep -o '[0-9.]*$')
                    local freq_req=$(echo "$gpu_data" | grep -o '"requested":[[:space:]]*[0-9.]*' | grep -o '[0-9.]*$')

                    # 重新获取完整数据以解析其他字段
                    gpu_data=$(timeout 3 intel_gpu_top -J -s 500 2>/dev/null | tail -100 | tr '\n' ' ')

                    local power_gpu=$(echo "$gpu_data" | grep -o '"GPU":[[:space:]]*[0-9.]*' | tail -1 | grep -o '[0-9.]*$')
                    local power_pkg=$(echo "$gpu_data" | grep -o '"Package":[[:space:]]*[0-9.]*' | tail -1 | grep -o '[0-9.]*$')
                    local rc6=$(echo "$gpu_data" | grep -o '"value":[[:space:]]*[0-9.]*' | grep -o '[0-9.]*$' | tail -1)
                    local render_busy=$(echo "$gpu_data" | grep -o '"Render/3D/0"[^}]*"busy":[[:space:]]*[0-9.]*' | grep -o '[0-9.]*$')

                    echo "  Frequency:"
                    [ -n "$freq_actual" ] && [ "$freq_actual" != "0.000000" ] && printf "    Actual:    %.0f MHz\n" "$freq_actual"
                    [ -n "$freq_req" ] && [ "$freq_req" != "0.000000" ] && printf "    Requested: %.0f MHz\n" "$freq_req"
                    echo ""
                    echo "  Usage:"
                    [ -n "$render_busy" ] && printf "    Render/3D: %.1f%%\n" "$render_busy"
                    [ -n "$rc6" ] && printf "    RC6 (idle): %.1f%%\n" "$rc6"
                    echo ""
                    echo "  Power:"
                    [ -n "$power_gpu" ] && printf "    GPU:       %.2f W\n" "$power_gpu"
                    [ -n "$power_pkg" ] && printf "    Package:   %.2f W\n" "$power_pkg"
                    echo ""
                fi
            fi

            # 后备方案：使用sysfs
            echo "  Frequency Settings:"
            echo "    Max:  $(cat /sys/class/drm/card0/gt_max_freq_mhz 2>/dev/null || echo 'N/A') MHz"
            echo "    Min:  $(cat /sys/class/drm/card0/gt_min_freq_mhz 2>/dev/null || echo 'N/A') MHz"

            # 显示Runtime
            if [ -r /sys/kernel/debug/dri/0/i915_engine_info ]; then
                local runtime=$(cat /sys/kernel/debug/dri/0/i915_engine_info 2>/dev/null | grep -A 10 "rcs0" | grep "Runtime:" | awk '{print $2}' || echo "")
                [ -n "$runtime" ] && echo "    Runtime:   $runtime (total active)"
            fi
            ;;
    esac
}

# 保存GPU配置到文件
save_gpu_profile() {
    local profile_name=${1:-default}
    local profile_file="$HOME/.gpu_profile_${profile_name}.conf"

    log_info "Saving GPU profile to $profile_file..."

    case $GPU_VENDOR in
        nvidia)
            {
                echo "GPU_VENDOR=nvidia"
                echo "GPU_FREQ=$(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits)"
                echo "GPU_POWER_LIMIT=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits)"
            } > "$profile_file"
            ;;
        amd)
            {
                echo "GPU_VENDOR=amd"
                echo "# AMD profile (manual implementation needed)"
            } > "$profile_file"
            ;;
        intel)
            {
                echo "GPU_VENDOR=intel"
                echo "GPU_MAX_FREQ=$(cat /sys/class/drm/card0/gt_max_freq_mhz)"
                echo "GPU_MIN_FREQ=$(cat /sys/class/drm/card0/gt_min_freq_mhz)"
            } > "$profile_file"
            ;;
    esac

    log_info "Profile saved: $profile_name"
}

# 加载GPU配置
load_gpu_profile() {
    local profile_name=${1:-default}
    local profile_file="$HOME/.gpu_profile_${profile_name}.conf"

    if [ ! -f "$profile_file" ]; then
        log_error "Profile not found: $profile_name"
        return 1
    fi

    log_info "Loading GPU profile: $profile_name..."

    source "$profile_file"

    case $GPU_VENDOR in
        nvidia)
            if [ -n "$GPU_FREQ" ]; then
                sudo nvidia-smi -lgc ${GPU_FREQ},${GPU_FREQ} || log_warn "Failed to restore frequency"
            fi
            if [ -n "$GPU_POWER_LIMIT" ]; then
                sudo nvidia-smi -pl $GPU_POWER_LIMIT || log_warn "Failed to restore power limit"
            fi
            ;;
        intel)
            if [ -n "$GPU_MAX_FREQ" ]; then
                echo $GPU_MAX_FREQ | sudo tee /sys/class/drm/card0/gt_max_freq_mhz >/dev/null
            fi
            if [ -n "$GPU_MIN_FREQ" ]; then
                echo $GPU_MIN_FREQ | sudo tee /sys/class/drm/card0/gt_min_freq_mhz >/dev/null
            fi
            ;;
    esac

    log_info "Profile loaded successfully"
}

# 实时监控GPU
monitor_gpu() {
    local interval=${1:-1}  # 默认1秒

    log_info "Monitoring GPU (Ctrl+C to stop)..."

    case $GPU_VENDOR in
        nvidia)
            nvidia-smi dmon -s pucvmet -d $interval || {
                log_warn "nvidia-smi dmon failed, falling back to nvidia-smi"
                watch -n $interval nvidia-smi
            }
            ;;
        amd)
            while true; do
                clear
                echo "=== AMD GPU Monitor (updating every ${interval}s) ==="
                rocm-smi --showuse --showpower --showtemp --showclocks
                sleep $interval
            done
            ;;
        intel)
            # 使用intel_gpu_top进行高级监控
            if command -v intel_gpu_top &>/dev/null; then
                log_info "Starting advanced GPU monitoring with intel_gpu_top..."

                # 初始化时读取硬件限制（不需要每次读取）
                local max_set=$(cat /sys/class/drm/card0/gt_max_freq_mhz 2>/dev/null || echo 0)
                local min_set=$(cat /sys/class/drm/card0/gt_min_freq_mhz 2>/dev/null || echo 0)

                # 使用文本模式（-l），设置环境变量COLUMNS避免自动换行
                COLUMNS=200 intel_gpu_top -l -s $((interval * 1000)) 2>/dev/null | while IFS= read -r line; do
                    # 跳过标题行
                    echo "$line" | grep -q "Freq MHz" && continue
                    echo "$line" | grep -q "req  act" && continue

                    # 解析数据行（格式：freq_req freq_act irq rc6 power_gpu power_pkg RCS% ...)
                    if echo "$line" | grep -qE '^[[:space:]]*[0-9]'; then
                        # 解析字段 - 使用数组避免变量个数限制
                        local fields=($line)

                        # 快速显示（先清屏再一次性输出）
                        clear
                        cat << EOF
========================================
  Intel GPU Monitor - $(date +%H:%M:%S)
========================================

Frequency:
  Requested:  ${fields[0]:-0} MHz
  Actual:     ${fields[1]:-0} MHz
  Range:      $min_set - $max_set MHz

Engine Usage:
  Render/3D:  ${fields[6]:-0.00}%
  Blitter:    ${fields[9]:-0.00}%
  Video0:     ${fields[12]:-0.00}%
  Video1:     ${fields[15]:-0.00}%
  VideoEnh:   ${fields[18]:-0.00}%

Power & State:
  GPU Power:  ${fields[4]:-0.00} W
  Package:    ${fields[5]:-0.00} W
  RC6 (idle): ${fields[3]:-0}%
  Interrupts: ${fields[2]:-0} /s

========================================
Press Ctrl+C to stop
EOF
                    fi
                done
            else
                # 降级到基础监控
                log_warn "intel_gpu_top not found, using basic monitoring"
                while true; do
                    clear
                    echo "========================================"
                    echo "  Intel GPU Monitor - $(date +%H:%M:%S)"
                    echo "========================================"
                    echo ""

                    echo "Frequency:"
                    echo "  Max Freq:  $(cat /sys/class/drm/card0/gt_max_freq_mhz) MHz"
                    echo "  Cur Freq:  $(cat /sys/class/drm/card0/gt_cur_freq_mhz) MHz"
                    echo "  Min Freq:  $(cat /sys/class/drm/card0/gt_min_freq_mhz) MHz"
                    echo ""

                    # 显示GPU活动信息
                    if [ -r /sys/kernel/debug/dri/0/i915_engine_info ]; then
                        echo "GPU Activity:"
                        local awake=$(cat /sys/kernel/debug/dri/0/i915_engine_info 2>/dev/null | grep -m1 "GT awake" | awk '{print $3}' || echo "unknown")
                        local runtime=$(cat /sys/kernel/debug/dri/0/i915_engine_info 2>/dev/null | grep -A 10 "rcs0" | grep "Runtime:" | awk '{print $2}' || echo "N/A")
                        echo "  GT Awake:  $awake"
                        echo "  Runtime:   $runtime (total active)"
                        echo ""
                    fi

                    echo "Hardware Limits:"
                    echo "  RP0 (Max): $(cat /sys/class/drm/card0/gt_RP0_freq_mhz) MHz"
                    echo "  RP1 (Eff): $(cat /sys/class/drm/card0/gt_RP1_freq_mhz) MHz"
                    echo "  RPn (Min): $(cat /sys/class/drm/card0/gt_RPn_freq_mhz) MHz"
                    echo ""

                    echo "========================================"
                    echo "Press Ctrl+C to stop (update: ${interval}s)"
                    sleep $interval
                done
            fi
            ;;
        *)
            log_error "Unknown GPU vendor"
            return 1
            ;;
    esac
}

# 显示帮助
show_help() {
    cat <<EOF
GPU Performance Control Tool

Usage: $0 <command> [arguments]

Commands:
    detect                  Detect GPU type
    info                    Show GPU detailed information
    status                  Show current GPU status (concise)
    freq <percent>          Set GPU frequency (0-100%)
    power <watts>           Set GPU power limit (watts)
    mode <mode>             Set performance mode (performance/balanced/powersave)
    temp <celsius>          Set temperature limit (NVIDIA only, 1-100°C)
    reset                   Reset GPU to default settings
    monitor [interval]      Monitor GPU in real-time (default 1s)
    save [profile]          Save current settings to profile
    load [profile]          Load settings from profile
    help                    Show this help message

Performance Modes:
    performance (max)       100% frequency, maximum performance
    balanced (normal)       70% frequency, balanced performance/power
    powersave (eco)         40% frequency, minimum power consumption

Examples:
    $0 detect                      # Detect GPU type
    $0 info                        # Show detailed GPU info
    $0 status                      # Show current status
    $0 freq 80                     # Set GPU to 80% frequency
    $0 mode performance            # Set to max performance
    $0 mode powersave              # Set to power saving mode
    $0 power 150                   # Set power limit to 150W
    $0 temp 80                     # Set temperature limit to 80°C (NVIDIA)
    $0 save gaming                 # Save current config as 'gaming' profile
    $0 load gaming                 # Load 'gaming' profile
    $0 monitor 2                   # Monitor GPU every 2 seconds
    $0 reset                       # Reset to defaults

Benchmark Workflow:
    # 1. Set max performance
    sudo $0 mode performance

    # 2. Run your benchmark
    ./param_scan.sh

    # 3. Reset to default
    sudo $0 reset

Frequency Sweep:
    for freq in 40 60 80 100; do
        sudo $0 freq \$freq
        ./benchmark.sh
    done

Note: Most commands require sudo privileges.
EOF
}

# 主函数
main() {
    local command=${1:-help}

    case $command in
        detect)
            detect_gpu
            ;;
        info)
            detect_gpu && query_gpu_info
            ;;
        status)
            detect_gpu && show_gpu_status
            ;;
        freq)
            if [ -z "$2" ]; then
                log_error "Missing frequency argument"
                echo "Usage: $0 freq <percent>"
                exit 1
            fi
            detect_gpu && set_gpu_freq "$2"
            ;;
        power)
            if [ -z "$2" ]; then
                log_error "Missing power argument"
                echo "Usage: $0 power <watts>"
                exit 1
            fi
            detect_gpu && set_gpu_power "$2"
            ;;
        mode)
            if [ -z "$2" ]; then
                log_error "Missing mode argument"
                echo "Usage: $0 mode <performance|balanced|powersave>"
                exit 1
            fi
            detect_gpu && set_gpu_mode "$2"
            ;;
        temp)
            if [ -z "$2" ]; then
                log_error "Missing temperature argument"
                echo "Usage: $0 temp <celsius>"
                exit 1
            fi
            detect_gpu && set_gpu_temp_limit "$2"
            ;;
        reset)
            detect_gpu && reset_gpu
            ;;
        monitor)
            detect_gpu && monitor_gpu "${2:-1}"
            ;;
        save)
            detect_gpu && save_gpu_profile "${2:-default}"
            ;;
        load)
            detect_gpu && load_gpu_profile "${2:-default}"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
