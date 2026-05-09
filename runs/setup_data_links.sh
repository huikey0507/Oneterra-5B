#!/usr/bin/env bash
# 设置数据软链接脚本
# 用于创建 gen_seg_data/sota 和 ov_seg_data/sota 的软链接
# 
# 使用方法：
#   1. 在项目根目录运行（自动检测路径）：
#      bash runs/setup_data_links.sh
#   
#   2. 指定根目录和数据目录：
#      bash runs/setup_data_links.sh /path/to/X-SAM /path/to/X-SAM/datas

#######################################################################
#                          Logging                                    #
#######################################################################
log_time=$(date "+%Y-%m-%d %H:%M:%S")
log_format="[$log_time] [INFO] [setup_data_links.sh]"

#######################################################################
#                          Main Function                              #
#######################################################################
setup_data_links() {
    # 获取参数或使用默认值
    # 如果从项目根目录运行，自动检测路径
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local root_dir="${1:-$(realpath "$script_dir/../")}"
    local data_dir="${2:-$root_dir/datas}"
    local node_rank="${NODE_RANK:-0}"
    
    echo -e "$log_format Setting up data symbolic links..."
    echo -e "$log_format Current root directory: $root_dir"
    echo -e "$log_format Data directory: $data_dir"
    
    # 确保源数据目录存在
    sota_source_dir="$data_dir/sota"
    if [ ! -d "$sota_source_dir" ]; then
        echo -e "$log_format WARNING: Source data directory $sota_source_dir does not exist, skipping link setup."
        return 1
    fi
    
    # 获取源目录的绝对路径（处理挂载点变化）
    sota_abs_path=$(realpath "$sota_source_dir")
    echo -e "$log_format Source SOTA data absolute path: $sota_abs_path"
    
    # 设置 gen_seg_data/sota 的软链接
    if [ "$node_rank" = "0" ]; then
        gen_seg_sota_dir="$data_dir/gen_seg_data/sota"
        # 确保父目录存在
        mkdir -p "$(dirname $gen_seg_sota_dir)"
        
        # 如果目标已存在且不是软链接，先备份
        if [ -e "$gen_seg_sota_dir" ] && [ ! -L "$gen_seg_sota_dir" ]; then
            echo -e "$log_format Backing up existing $gen_seg_sota_dir"
            backup_name="${gen_seg_sota_dir}.backup.$(date +%Y%m%d_%H%M%S)"
            mv "$gen_seg_sota_dir" "$backup_name"
            echo -e "$log_format Backup saved to: $backup_name"
        fi
        
        # 删除现有软链接或目录，创建新的软链接
        rm -rf "$gen_seg_sota_dir"
        ln -sfn "$sota_abs_path" "$gen_seg_sota_dir"
        if [ -L "$gen_seg_sota_dir" ]; then
            echo -e "$log_format ✓ Created symlink: $gen_seg_sota_dir -> $sota_abs_path"
        else
            echo -e "$log_format ✗ Failed to create symlink: $gen_seg_sota_dir"
            return 1
        fi
        
        # 设置 ov_seg_data/sota 的软链接
        ov_seg_sota_dir="$data_dir/ov_seg_data/sota"
        # 确保父目录存在
        mkdir -p "$(dirname $ov_seg_sota_dir)"
        
        # 如果目标已存在且不是软链接，先备份
        if [ -e "$ov_seg_sota_dir" ] && [ ! -L "$ov_seg_sota_dir" ]; then
            echo -e "$log_format Backing up existing $ov_seg_sota_dir"
            backup_name="${ov_seg_sota_dir}.backup.$(date +%Y%m%d_%H%M%S)"
            mv "$ov_seg_sota_dir" "$backup_name"
            echo -e "$log_format Backup saved to: $backup_name"
        fi
        
        # 删除现有软链接或目录，创建新的软链接
        rm -rf "$ov_seg_sota_dir"
        ln -sfn "$sota_abs_path" "$ov_seg_sota_dir"
        if [ -L "$ov_seg_sota_dir" ]; then
            echo -e "$log_format ✓ Created symlink: $ov_seg_sota_dir -> $sota_abs_path"
        else
            echo -e "$log_format ✗ Failed to create symlink: $ov_seg_sota_dir"
            return 1
        fi
        
        # 验证软链接
        echo -e "$log_format Verifying symlinks..."
        if [ -L "$gen_seg_sota_dir" ] && [ -d "$gen_seg_sota_dir" ]; then
            echo -e "$log_format ✓ gen_seg_data/sota symlink is valid"
        else
            echo -e "$log_format ✗ gen_seg_data/sota symlink is invalid or broken"
            return 1
        fi
        
        if [ -L "$ov_seg_sota_dir" ] && [ -d "$ov_seg_sota_dir" ]; then
            echo -e "$log_format ✓ ov_seg_data/sota symlink is valid"
        else
            echo -e "$log_format ✗ ov_seg_data/sota symlink is invalid or broken"
            return 1
        fi
    fi
    
    # 等待 rank 0 完成软链接设置
    if [ "$node_rank" != "0" ]; then
        sleep 2
    fi
    
    echo -e "$log_format Data symbolic links setup completed."
    return 0
}

#######################################################################
#                          Main Entry                                #
#######################################################################
# 如果直接运行此脚本（而不是被 source），则执行设置
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    setup_data_links "$@"
    exit $?
fi

