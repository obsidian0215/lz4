#!/bin/bash
# CPU性能控制和监控工具
# 支持频率调整、核心限制、性能模式切换、实时监控

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# 检测CPU信息
detect_cpu() {
    log_info "Detecting CPU..."

    if [ ! -f /proc/cpuinfo ]; then
        log_error "Cannot access /proc/cpuinfo"
        return 1
    fi

    local cpu_model=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
    local cpu_count=$(nproc)
    local cpu_cores=$(grep "cpu cores" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)

    log_info "CPU: $cpu_model"
    log_info "Logical CPUs: $cpu_count"
    log_info "Physical cores: ${cpu_cores:-N/A}"
}

# 显示CPU详细信息
show_cpu_info() {
    echo "=== CPU Information ==="

    # CPU型号
    local cpu_model=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
    echo "Model: $cpu_model"

    # CPU数量
    local cpu_count=$(nproc)
    echo "Logical CPUs: $cpu_count"

    # 物理核心
    local cpu_cores=$(grep "cpu cores" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
    echo "Physical cores: ${cpu_cores:-N/A}"

    # 当前频率
    echo ""
    echo "Current Frequencies:"
    if [ -d /sys/devices/system/cpu/cpu0/cpufreq ]; then
        for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
            local cpu_num=$(echo $cpu | grep -o 'cpu[0-9]*' | grep -o '[0-9]*')
            local freq=$(cat $cpu 2>/dev/null || echo 0)
            printf "  CPU%2d: %6d MHz\n" $cpu_num $((freq / 1000))
        done
    else
        echo "  Frequency scaling not available"
    fi

    # 频率范围
    if [ -f /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq ]; then
        echo ""
        echo "Frequency Range:"
        local min_freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq)
        local max_freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq)
        echo "  Min: $((min_freq / 1000)) MHz"
        echo "  Max: $((max_freq / 1000)) MHz"
    fi

    # Governor
    if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
        echo ""
        echo "Current Governor:"
        local governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
        echo "  $governor"

        echo ""
        echo "Available Governors:"
        cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors 2>/dev/null | tr ' ' '\n' | sed 's/^/  /'
    fi

    # 在线CPU
    echo ""
    echo "Online CPUs:"
    cat /sys/devices/system/cpu/online
}

# 显示简洁状态
show_cpu_status() {
    echo "=== CPU Status ==="

    # 平均频率
    if [ -d /sys/devices/system/cpu/cpu0/cpufreq ]; then
        local total_freq=0
        local count=0
        for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
            local freq=$(cat $cpu 2>/dev/null || echo 0)
            total_freq=$((total_freq + freq))
            count=$((count + 1))
        done
        local avg_freq=$((total_freq / count / 1000))
        echo "  Avg Frequency: $avg_freq MHz"
    fi

    # Governor
    if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
        echo "  Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
    fi

    # 在线CPU数量
    local online_cpus=$(cat /sys/devices/system/cpu/online | grep -o '-' | wc -l)
    if [ $online_cpus -eq 0 ]; then
        online_cpus=$(cat /sys/devices/system/cpu/online | tr ',' '\n' | wc -l)
    else
        online_cpus=$(nproc)
    fi
    echo "  Online CPUs: $online_cpus / $(nproc --all)"

    # Turbo状态
    if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
        local no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
        if [ "$no_turbo" = "0" ]; then
            echo "  Turbo Boost: Enabled"
        else
            echo "  Turbo Boost: Disabled"
        fi
    fi

    # CPU使用率（简单采样）
    if command -v mpstat &>/dev/null; then
        local idle=$(mpstat 1 1 | grep Average | awk '{print $NF}')
        local usage=$(awk "BEGIN {printf \"%.1f\", 100 - $idle}")
        echo "  CPU Usage: ${usage}%"
    fi
}

# 设置CPU频率 (百分比)
set_cpu_freq() {
    local freq_percent=$1

    if [ -z "$freq_percent" ] || [ "$freq_percent" -lt 0 ] || [ "$freq_percent" -gt 100 ]; then
        log_error "Invalid frequency percent: $freq_percent (must be 0-100)"
        return 1
    fi

    if [ ! -f /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq ]; then
        log_error "CPU frequency scaling not available"
        return 1
    fi

    log_info "Setting CPU frequency to ${freq_percent}%..."

    local max_freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq)
    local min_freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq)
    local target_freq=$(( (max_freq - min_freq) * freq_percent / 100 + min_freq ))

    log_info "Target frequency: $((target_freq / 1000)) MHz"

    # 设置所有CPU的频率
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq; do
        echo $target_freq | sudo tee $cpu > /dev/null || {
            log_error "Failed to set frequency"
            return 1
        }
    done

    log_info "CPU frequency set successfully"
}

# 设置Governor（性能策略）
set_governor() {
    local governor=$1

    if [ -z "$governor" ]; then
        log_error "Governor not specified"
        return 1
    fi

    if [ ! -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
        log_error "CPU governor control not available"
        return 1
    fi

    # 检查是否支持
    if ! grep -q "$governor" /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors 2>/dev/null; then
        log_error "Governor '$governor' not available"
        log_info "Available governors:"
        cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors
        return 1
    fi

    log_info "Setting CPU governor to '$governor'..."

    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo $governor | sudo tee $cpu > /dev/null || {
            log_error "Failed to set governor"
            return 1
        }
    done

    log_info "Governor set successfully"
}

# 设置Turbo Boost
set_turbo() {
    local action=$1

    if [ ! -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
        log_error "Turbo Boost control not available (Intel P-State not found)"
        return 1
    fi

    case $action in
        on|enable|1)
            log_info "Enabling Turbo Boost..."
            echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null || {
                log_error "Failed to enable Turbo Boost"
                return 1
            }
            log_info "Turbo Boost enabled"
            ;;
        off|disable|0)
            log_info "Disabling Turbo Boost..."
            echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null || {
                log_error "Failed to disable Turbo Boost"
                return 1
            }
            log_info "Turbo Boost disabled"
            ;;
        status)
            local no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
            if [ "$no_turbo" = "0" ]; then
                echo "Turbo Boost: Enabled"
            else
                echo "Turbo Boost: Disabled"
            fi
            ;;
        *)
            log_error "Invalid turbo action: $action"
            echo "Usage: $0 turbo <on|off|status>"
            return 1
            ;;
    esac
}

# 设置性能模式
set_cpu_mode() {
    local mode=$1

    case $mode in
        performance|max|high)
            log_info "Setting CPU to PERFORMANCE mode..."
            set_governor performance || return 1
            # 禁用turbo boost限制
            if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
                echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
            fi
            ;;
        balanced|normal|powersave)
            log_info "Setting CPU to BALANCED mode..."
            set_governor powersave || set_governor ondemand || return 1
            ;;
        eco|low)
            log_info "Setting CPU to ECO mode..."
            set_governor powersave || return 1
            # 启用turbo boost限制
            if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
                echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
            fi
            ;;
        *)
            log_error "Unknown mode: $mode"
            echo "Available modes: performance, balanced, eco"
            return 1
            ;;
    esac
}

# 限制CPU核心数
set_cpu_cores() {
    local num_cores=$1
    local total_cpus=$(nproc --all)

    if [ -z "$num_cores" ] || [ "$num_cores" -lt 1 ] || [ "$num_cores" -gt "$total_cpus" ]; then
        log_error "Invalid core count: $num_cores (must be 1-$total_cpus)"
        return 1
    fi

    log_info "Limiting to $num_cores CPU cores..."

    # 上线前num_cores个CPU
    for i in $(seq 0 $((num_cores - 1))); do
        if [ -f /sys/devices/system/cpu/cpu$i/online ]; then
            echo 1 | sudo tee /sys/devices/system/cpu/cpu$i/online > /dev/null 2>&1 || true
        fi
    done

    # 下线其余CPU
    for i in $(seq $num_cores $((total_cpus - 1))); do
        if [ -f /sys/devices/system/cpu/cpu$i/online ]; then
            echo 0 | sudo tee /sys/devices/system/cpu/cpu$i/online > /dev/null 2>&1 || {
                log_warn "Cannot offline CPU $i (may be protected)"
            }
        fi
    done

    log_info "CPU cores set to $num_cores"
}

# 重置CPU设置
reset_cpu() {
    log_info "Resetting CPU to default settings..."

    # 恢复所有CPU在线
    local total_cpus=$(nproc --all)
    for i in $(seq 1 $((total_cpus - 1))); do
        if [ -f /sys/devices/system/cpu/cpu$i/online ]; then
            echo 1 | sudo tee /sys/devices/system/cpu/cpu$i/online > /dev/null 2>&1 || true
        fi
    done

    # 恢复最大频率
    if [ -f /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq ]; then
        local max_freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq)
        for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq; do
            echo $max_freq | sudo tee $cpu > /dev/null 2>&1 || true
        done
    fi

    # 恢复governor
    if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
        set_governor powersave 2>/dev/null || set_governor ondemand 2>/dev/null || true
    fi

    # 启用turbo
    if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
        echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null 2>&1 || true
    fi

    log_info "CPU reset complete"
}

# 保存配置
save_cpu_profile() {
    local profile_name=${1:-default}
    local profile_file="$HOME/.cpu_profile_${profile_name}.conf"

    log_info "Saving CPU profile to $profile_file..."

    {
        echo "# CPU Profile: $profile_name"
        echo "GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo '')"
        echo "MAX_FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo '')"
        echo "ONLINE_CPUS=$(cat /sys/devices/system/cpu/online)"
        if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
            echo "NO_TURBO=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"
        fi
    } > "$profile_file"

    log_info "Profile saved: $profile_name"
}

# 加载配置
load_cpu_profile() {
    local profile_name=${1:-default}
    local profile_file="$HOME/.cpu_profile_${profile_name}.conf"

    if [ ! -f "$profile_file" ]; then
        log_error "Profile not found: $profile_name"
        return 1
    fi

    log_info "Loading CPU profile: $profile_name..."

    source "$profile_file"

    # 恢复governor
    if [ -n "$GOVERNOR" ]; then
        set_governor "$GOVERNOR" || log_warn "Failed to restore governor"
    fi

    # 恢复频率
    if [ -n "$MAX_FREQ" ]; then
        for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq; do
            echo $MAX_FREQ | sudo tee $cpu > /dev/null 2>&1 || true
        done
    fi

    # 恢复turbo设置
    if [ -n "$NO_TURBO" ] && [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
        echo $NO_TURBO | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null 2>&1 || true
    fi

    log_info "Profile loaded successfully"
}

# 实时监控CPU
monitor_cpu() {
    local interval=${1:-1}

    log_info "Monitoring CPU (Ctrl+C to stop)..."
    echo ""

    # 使用优化的实时监控（turbostat输出太冗长，默认使用清晰的内置监控）
    # 如果需要完整turbostat，运行: TURBOSTAT_MODE=1 ./cpu_control.sh monitor
    if [ "${TURBOSTAT_MODE:-0}" = "1" ] && command -v turbostat &>/dev/null; then
        log_info "Using turbostat (full detail mode)..."
        echo ""
        sudo turbostat --quiet --interval $interval \
            --show Core,CPU,Busy%,Bzy_MHz,IRQ,CoreTmp,PkgWatt 2>/dev/null || \
        sudo turbostat --interval $interval
    else
        # 使用优化的实时监控
        log_info "Using built-in real-time monitoring"
        if ! command -v mpstat &>/dev/null; then
            log_warn "Install 'sysstat' for better CPU usage stats: sudo apt install sysstat"
        fi
        echo ""

        # 获取CPU拓扑信息（只读一次）
        local total_cpus=$(nproc --all)
        local online_cpus=$(nproc)
        local min_freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq 2>/dev/null || echo 0)
        local max_freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 0)
        local governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'N/A')
        min_freq=$((min_freq / 1000))
        max_freq=$((max_freq / 1000))

        # 检查是否有温度传感器
        local has_temp=false
        [ -f /sys/class/thermal/thermal_zone0/temp ] && has_temp=true

        # 检查turbo状态
        local has_turbo=false
        [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ] && has_turbo=true

        # 检查RAPL功率接口
        local has_rapl=false
        local rapl_pkg_energy=""
        local rapl_cores_energy=""
        local rapl_scale=1

        if [ -d /sys/class/powercap/intel-rapl ]; then
            # 查找Package和Core能量文件
            for rapl_dir in /sys/class/powercap/intel-rapl/intel-rapl:*; do
                if [ -f "$rapl_dir/name" ]; then
                    local rapl_name=$(cat "$rapl_dir/name")
                    if [ "$rapl_name" = "package-0" ]; then
                        rapl_pkg_energy="$rapl_dir/energy_uj"
                        has_rapl=true
                    fi
                fi
                # 查找core子域
                for sub_dir in "$rapl_dir"/intel-rapl:*; do
                    if [ -f "$sub_dir/name" ]; then
                        local sub_name=$(cat "$sub_dir/name")
                        if [ "$sub_name" = "core" ]; then
                            rapl_cores_energy="$sub_dir/energy_uj"
                        fi
                    fi
                done
            done
        fi

        # 功率计算变量
        local prev_pkg_energy=0
        local prev_cores_energy=0
        local prev_time=0

        # 读取初始能量值
        if [ "$has_rapl" = true ]; then
            [ -f "$rapl_pkg_energy" ] && prev_pkg_energy=$(cat "$rapl_pkg_energy")
            [ -f "$rapl_cores_energy" ] && prev_cores_energy=$(cat "$rapl_cores_energy")
            prev_time=$(date +%s%N)
        fi

        # CPU使用率计算变量（整体）
        local prev_total=0
        local prev_idle=0

        # 各核使用率变量
        declare -A prev_cpu_total
        declare -A prev_cpu_idle

        while true; do
            # 一次性读取所有频率到关联数组（按CPU编号）
            declare -A freqs
            local total_freq=0
            local count=0

            # 获取所有CPU编号并排序
            local -a cpu_nums=()
            for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
                if [ -f "$cpu" ]; then
                    local num=$(echo $cpu | grep -o 'cpu[0-9]*' | grep -o '[0-9]*')
                    cpu_nums+=($num)
                fi
            done

            # 数值排序并去重
            cpu_nums=($(printf '%s\n' "${cpu_nums[@]}" | sort -n -u))

            # 按排序后的顺序读取频率
            for num in "${cpu_nums[@]}"; do
                local cpu="/sys/devices/system/cpu/cpu${num}/cpufreq/scaling_cur_freq"
                if [ -f "$cpu" ]; then
                    freqs[$num]=$(cat $cpu)
                    total_freq=$((total_freq + freqs[$num]))
                    count=$((count + 1))
                fi
            done

            # 读取其他状态（循环外）
            local temp=0
            [ "$has_temp" = true ] && temp=$(cat /sys/class/thermal/thermal_zone0/temp)

            local turbo_status=""
            if [ "$has_turbo" = true ]; then
                local no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
                [ "$no_turbo" = "0" ] && turbo_status="Enabled" || turbo_status="Disabled"
            fi

            local load=$(cat /proc/loadavg | awk '{print $1, $2, $3}')

            # 计算平均频率
            local avg_freq=$((total_freq / count / 1000))

            # 计算功率（如果支持RAPL）
            local pkg_power="N/A"
            local cores_power="N/A"

            if [ "$has_rapl" = true ]; then
                local curr_pkg_energy=0
                local curr_cores_energy=0
                local curr_time=$(date +%s%N)

                [ -f "$rapl_pkg_energy" ] && curr_pkg_energy=$(cat "$rapl_pkg_energy")
                [ -f "$rapl_cores_energy" ] && curr_cores_energy=$(cat "$rapl_cores_energy")

                if [ $prev_time -gt 0 ]; then
                    local time_diff=$(( (curr_time - prev_time) / 1000000 ))  # 转换为毫秒

                    if [ $time_diff -gt 0 ]; then
                        # 能量差值（微焦耳）
                        local pkg_energy_diff=$((curr_pkg_energy - prev_pkg_energy))
                        local cores_energy_diff=$((curr_cores_energy - prev_cores_energy))

                        # 处理溢出（RAPL计数器可能溢出）
                        if [ $pkg_energy_diff -lt 0 ]; then
                            pkg_energy_diff=$((pkg_energy_diff + 2147483647))
                        fi
                        if [ $cores_energy_diff -lt 0 ]; then
                            cores_energy_diff=$((cores_energy_diff + 2147483647))
                        fi

                        # 功率 = 能量差 / 时间差 (微焦耳/毫秒 = 毫瓦)
                        # 转换为瓦特: 毫瓦 / 1000
                        pkg_power=$(awk "BEGIN {printf \"%.2f\", ($pkg_energy_diff / $time_diff) / 1000}")
                        cores_power=$(awk "BEGIN {printf \"%.2f\", ($cores_energy_diff / $time_diff) / 1000}")
                    fi
                fi

                prev_pkg_energy=$curr_pkg_energy
                prev_cores_energy=$curr_cores_energy
                prev_time=$curr_time
            fi

            # 快速计算CPU使用率（从/proc/stat，无需等待）
            local cpu_usage="N/A"
            local cpu_idle="N/A"
            declare -A cpu_core_usage

            if [ -f /proc/stat ]; then
                # 读取所有CPU统计
                local stat_data=$(grep "^cpu" /proc/stat)

                # 总体CPU
                local stat_line=$(echo "$stat_data" | grep "^cpu ")
                local -a cpu_stats=($stat_line)
                local idle=${cpu_stats[4]}
                local iowait=${cpu_stats[5]:-0}
                local total=0
                for i in {1..7}; do
                    total=$((total + ${cpu_stats[$i]:-0}))
                done

                if [ $prev_total -gt 0 ]; then
                    local diff_total=$((total - prev_total))
                    local diff_idle=$((idle - prev_idle))
                    if [ $diff_total -gt 0 ]; then
                        cpu_usage=$(awk "BEGIN {printf \"%.1f\", 100 * ($diff_total - $diff_idle) / $diff_total}")
                        cpu_idle=$(awk "BEGIN {printf \"%.1f\", 100 * $diff_idle / $diff_total}")
                    fi
                fi

                prev_total=$total
                prev_idle=$idle

                # 各核CPU使用率
                while IFS= read -r line; do
                    if [[ $line =~ ^cpu([0-9]+) ]]; then
                        local cpu_num="${BASH_REMATCH[1]}"
                        local -a core_stats=($line)
                        local core_idle=${core_stats[4]}
                        local core_total=0
                        for i in {1..7}; do
                            core_total=$((core_total + ${core_stats[$i]:-0}))
                        done

                        if [ "${prev_cpu_total[$cpu_num]:-0}" -gt 0 ]; then
                            local core_diff_total=$((core_total - prev_cpu_total[$cpu_num]))
                            local core_diff_idle=$((core_idle - prev_cpu_idle[$cpu_num]))
                            if [ $core_diff_total -gt 0 ]; then
                                cpu_core_usage[$cpu_num]=$(awk "BEGIN {printf \"%.1f\", 100 * ($core_diff_total - $core_diff_idle) / $core_diff_total}")
                            else
                                cpu_core_usage[$cpu_num]="0.0"
                            fi
                        else
                            cpu_core_usage[$cpu_num]="N/A"
                        fi

                        prev_cpu_total[$cpu_num]=$core_total
                        prev_cpu_idle[$cpu_num]=$core_idle
                    fi
                done <<< "$stat_data"
            fi

            # 使用heredoc一次性输出所有内容
            clear
            cat << EOF
========================================
  CPU Monitor - $(date +%H:%M:%S)
========================================

CPU Frequencies (MHz):
EOF

            # 输出频率（5列布局，按CPU编号排序）
            local i=0
            for num in "${cpu_nums[@]}"; do
                printf "  CPU%2d:%4d" $num $((freqs[$num] / 1000))
                i=$((i + 1))

                # 每行5个CPU
                if [ $((i % 5)) -eq 0 ]; then
                    echo ""
                fi
            done

            # 换行（如果需要）
            [ $((i % 5)) -ne 0 ] && echo ""

            # 构建完整输出
            printf "\n  Average: %d MHz (Range: %d - %d MHz)\n" $avg_freq $min_freq $max_freq

            # CPU使用率
            echo ""
            echo "CPU Usage (%):"
            printf "  Overall: %5s%% (idle: %5s%%)\n" "$cpu_usage" "$cpu_idle"

            # 显示各核负载（5列布局）
            if [ ${#cpu_core_usage[@]} -gt 0 ]; then
                echo ""
                echo "  Per Core Usage:"
                local i=0
                for num in "${cpu_nums[@]}"; do
                    local usage="${cpu_core_usage[$num]:-N/A}"
                    printf "  CPU%2d:%5s%%" $num "$usage"
                    i=$((i + 1))

                    # 每行5个CPU
                    if [ $((i % 5)) -eq 0 ]; then
                        echo ""
                    fi
                done

                # 换行（如果需要）
                [ $((i % 5)) -ne 0 ] && echo ""
            fi

            # 状态信息
            cat << EOF

Status:
  Governor: $governor
  Online CPUs: $online_cpus / $total_cpus
EOF

            [ "$has_temp" = true ] && printf "  Temperature: %d°C\n" $((temp / 1000))
            [ "$has_turbo" = true ] && echo "  Turbo: $turbo_status"

            # 功率信息
            if [ "$has_rapl" = true ]; then
                cat << EOF

Power (RAPL):
  Package: $pkg_power W
EOF
                [ "$cores_power" != "N/A" ] && echo "  Cores:   $cores_power W"
            fi

            cat << EOF

Load Average (1m 5m 15m):
  $load

========================================
Press Ctrl+C to stop | Update: ${interval}s
EOF

            sleep $interval
        done
    fi
}

# 显示帮助
show_help() {
    cat <<EOF
CPU Performance Control & Monitoring Tool

Usage: $0 <command> [arguments]

Commands:
    detect                  Detect CPU information
    info                    Show detailed CPU information
    status                  Show current CPU status (concise)
    freq <percent>          Set CPU max frequency (0-100%)
    governor <name>         Set CPU governor (performance/powersave/ondemand/etc)
    mode <mode>             Set performance mode (performance/balanced/eco)
    turbo <on|off|status>   Enable/disable/check Turbo Boost
    cores <num>             Limit number of active CPU cores
    reset                   Reset CPU to default settings
    monitor [interval]      Monitor CPU in real-time (default 1s)
    save [profile]          Save current settings to profile
    load [profile]          Load settings from profile
    help                    Show this help message

Performance Modes:
    performance (max)       performance governor + turbo enabled
    balanced (normal)       powersave/ondemand governor
    eco (low)               powersave governor + turbo disabled

Turbo Boost Control:
    turbo on                Enable Turbo Boost (higher performance)
    turbo off               Disable Turbo Boost (lower power, stable freq)
    turbo status            Check current Turbo Boost state

Examples:
    $0 detect                      # Detect CPU
    $0 info                        # Show detailed info
    $0 status                      # Show current status
    $0 freq 80                     # Set max frequency to 80%
    $0 governor performance        # Set performance governor
    $0 mode performance            # Set to max performance
    $0 turbo off                   # Disable Turbo Boost
    $0 turbo on                    # Enable Turbo Boost
    $0 cores 4                     # Limit to 4 CPU cores
    $0 save benchmark              # Save current config as 'benchmark'
    $0 load benchmark              # Load 'benchmark' config
    $0 monitor 2                   # Monitor CPU every 2 seconds
    $0 reset                       # Reset to defaults

Benchmark Workflow:
    # 1. Set performance mode
    sudo $0 mode performance
    sudo $0 cores 8

    # 2. Run benchmark
    ./run_lzo_cpu.sh

    # 3. Reset
    sudo $0 reset

Frequency Sweep:
    for freq in 40 60 80 100; do
        sudo $0 freq \$freq
        ./benchmark.sh
    done

Note: Most commands require sudo privileges.
      Install 'sysstat' for better monitoring: sudo apt install sysstat
      Install 'linux-tools-common' for turbostat: sudo apt install linux-tools-common
EOF
}

# 主函数
main() {
    local command=${1:-help}

    case $command in
        detect)
            detect_cpu
            ;;
        info)
            show_cpu_info
            ;;
        status)
            show_cpu_status
            ;;
        freq)
            if [ -z "$2" ]; then
                log_error "Missing frequency argument"
                echo "Usage: $0 freq <percent>"
                exit 1
            fi
            set_cpu_freq "$2"
            ;;
        governor)
            if [ -z "$2" ]; then
                log_error "Missing governor argument"
                echo "Usage: $0 governor <name>"
                exit 1
            fi
            set_governor "$2"
            ;;
        mode)
            if [ -z "$2" ]; then
                log_error "Missing mode argument"
                echo "Usage: $0 mode <performance|balanced|eco>"
                exit 1
            fi
            set_cpu_mode "$2"
            ;;
        turbo)
            if [ -z "$2" ]; then
                log_error "Missing turbo argument"
                echo "Usage: $0 turbo <on|off|status>"
                exit 1
            fi
            set_turbo "$2"
            ;;
        cores)
            if [ -z "$2" ]; then
                log_error "Missing core count argument"
                echo "Usage: $0 cores <num>"
                exit 1
            fi
            set_cpu_cores "$2"
            ;;
        reset)
            reset_cpu
            ;;
        monitor)
            monitor_cpu "${2:-1}"
            ;;
        save)
            save_cpu_profile "${2:-default}"
            ;;
        load)
            load_cpu_profile "${2:-default}"
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
