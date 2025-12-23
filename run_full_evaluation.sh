#!/bin/bash
# 完整220条路线评估 - 后台CARLA + 实时输出
# 特点：
# 1. CARLA在后台无头模式运行（更快）
# 2. 终端实时显示评估进度和场景类型
# 3. 显示官方能力分类
# 4. 分批运行，更稳定

echo "=========================================="
echo "  完整220条路线评估（实时输出版本）"
echo "=========================================="
echo ""
echo "特点："
echo "  ✓ CARLA后台运行（无可视化，更快）"
echo "  ✓ 终端实时显示评估进度"
echo "  ✓ 显示每个路线的官方能力分类"
echo "  ✓ 分批运行（每批50条），更稳定"
echo ""
echo "预计时间：12-24小时"
echo ""
read -p "确定开始吗？(y/N): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "已取消"
    exit 0
fi

# 基本配置
BASE_PORT=30000
BASE_TM_PORT=50000
ROUTES_PER_BATCH=50
TOTAL_ROUTES=220
TEAM_AGENT=leaderboard/team_code/autopilot_agent.py
TEAM_CONFIG=autopilot_config.txt

cd /home/ajifang/b2drive

# 1. 清理环境
echo ""
echo "[1/5] 清理环境..."
pkill -9 -f CarlaUE4 2>/dev/null
sleep 3
rm -rf autopilot_full_results/
mkdir -p autopilot_full_results
echo "✓ 环境已清理"

# 2. 创建分批XML文件
echo ""
echo "[2/5] 创建分批路线文件..."
python3 << 'EOF'
import xml.etree.ElementTree as ET
import math

tree = ET.parse('leaderboard/data/bench2drive220.xml')
root = tree.getroot()
routes = list(root.findall('route'))

print(f"总路线数: {len(routes)}")

routes_per_batch = 50
batch_count = math.ceil(len(routes) / routes_per_batch)

print(f"分为 {batch_count} 批，每批 {routes_per_batch} 个路线")

for batch_idx in range(batch_count):
    start_idx = batch_idx * routes_per_batch
    end_idx = min((batch_idx + 1) * routes_per_batch, len(routes))
    batch_routes = routes[start_idx:end_idx]

    new_root = ET.Element('routes')
    for route in batch_routes:
        new_root.append(route)

    batch_filename = f'leaderboard/data/batch_{batch_idx}_full.xml'
    new_tree = ET.ElementTree(new_root)
    new_tree.write(batch_filename, encoding='utf-8', xml_declaration=True)

    print(f"  批次 {batch_idx}: 路线 {start_idx}-{end_idx-1} ({len(batch_routes)} 条)")

print("✓ 批次文件创建完成")
EOF

# 3. 运行分批评估
echo ""
echo "[3/5] 开始分批评估..."
BATCH_COUNT=$(ls leaderboard/data/batch_*_full.xml 2>/dev/null | wc -l)

START_TIME=$(date +%s)

for ((batch=0; batch<$BATCH_COUNT; batch++)); do
    BATCH_START_TIME=$(date +%s)

    echo ""
    echo "=========================================="
    echo "  批次 $((batch+1))/$BATCH_COUNT"
    echo "  路线范围: $((batch*50+1))-$((batch*50+50))"
    echo "=========================================="
    echo ""

    BATCH_ROUTES="leaderboard/data/batch_${batch}_full.xml"
    BATCH_CHECKPOINT="autopilot_full_results/batch_${batch}_result.json"

    # 使用run_evaluation.sh（会自动启动CARLA后台）
    # 实时输出会显示每个路线的场景类型和能力分类
    echo "$(date '+%Y-%m-%d %H:%M:%S') - 开始批次 $((batch+1))..."

    timeout 7200 bash leaderboard/scripts/run_evaluation.sh \
        $BASE_PORT $BASE_TM_PORT True \
        $BATCH_ROUTES $TEAM_AGENT $TEAM_CONFIG \
        $BATCH_CHECKPOINT "./autopilot_full_results/batch_${batch}" "only_traj" 0 \
        2>&1 | tee -a autopilot_full_results/batch_${batch}_log.txt

    EXIT_CODE=$?
    BATCH_END_TIME=$(date +%s)
    BATCH_DURATION=$((BATCH_END_TIME - BATCH_START_TIME))

    echo ""
    echo "批次 $((batch+1)) 耗时: $((BATCH_DURATION/60)) 分钟"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ 批次 $((batch+1)) 完成"
    elif [ $EXIT_CODE -eq 124 ]; then
        echo "⚠ 批次 $((batch+1)) 超时（2小时），可能部分完成"
    else
        echo "✗ 批次 $((batch+1)) 失败（退出码: $EXIT_CODE）"
    fi

    # 显示当前批次统计
    if [ -f "$BATCH_CHECKPOINT" ]; then
        echo ""
        echo "当前批次统计:"
        python3 << EOF
import json
try:
    with open('$BATCH_CHECKPOINT', 'r') as f:
        data = json.load(f)
    records = data.get('_checkpoint', {}).get('records', [])
    completed = sum(1 for r in records if r.get('status') in ['Completed', 'Perfect'])
    print(f"  完成路线: {completed}/{len(records)}")
except Exception as e:
    print(f"  无法读取统计: {e}")
EOF
    fi

    # 清理CARLA进程
    echo ""
    echo "清理CARLA进程..."
    pkill -9 -f CarlaUE4 2>/dev/null
    sleep 5

    # 显示总体进度
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    echo ""
    echo "总体进度: 批次 $((batch+1))/$BATCH_COUNT 完成"
    echo "已用时间: $((ELAPSED/3600)) 小时 $((ELAPSED%3600/60)) 分钟"

    # 估算剩余时间
    if [ $batch -gt 0 ]; then
        AVG_TIME=$((ELAPSED / (batch + 1)))
        REMAINING_BATCHES=$((BATCH_COUNT - batch - 1))
        ETA=$((AVG_TIME * REMAINING_BATCHES))
        echo "预计剩余: $((ETA/3600)) 小时 $((ETA%3600/60)) 分钟"
    fi

    echo ""
    echo "休息10秒后开始下一批..."
    sleep 10
done

# 4. 合并所有批次结果
echo ""
echo "[4/5] 合并所有批次结果..."
python3 << 'EOF'
import json
import glob

batch_files = sorted(glob.glob('autopilot_full_results/batch_*_result.json'))
print(f"找到 {len(batch_files)} 个批次结果文件")

all_records = []

for batch_file in batch_files:
    try:
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
            batch_records = batch_data.get('_checkpoint', {}).get('records', [])
            all_records.extend(batch_records)
            print(f"  {batch_file}: {len(batch_records)} 条记录")
    except Exception as e:
        print(f"  ✗ 读取 {batch_file} 失败: {e}")

print(f"\n总计: {len(all_records)}/220 条路线")

# 保存合并结果
final_result = {
    "_checkpoint": {
        "global_record": {},
        "progress": [len(all_records), 220],
        "records": all_records
    },
    "entry_status": "Completed",
    "eligible": True,
    "sensors": [],
    "values": [],
    "labels": []
}

with open('autopilot_full_results/full_evaluation.json', 'w') as f:
    json.dump(final_result, f, indent=2)

print("✓ 合并结果已保存: autopilot_full_results/full_evaluation.json")
EOF

# 5. 计算官方能力评分
echo ""
echo "[5/5] 计算官方能力评分..."
python tools/ability_benchmark.py -r autopilot_full_results/full_evaluation.json

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "  评估完成！"
echo "=========================================="
echo ""
echo "总耗时: $((TOTAL_DURATION/3600)) 小时 $((TOTAL_DURATION%3600/60)) 分钟"
echo ""
echo "结果文件:"
echo "  - autopilot_full_results/full_evaluation.json (详细结果)"
echo "  - autopilot_full_results/full_evaluation_ability.json (能力评分)"
echo ""
echo "查看能力评分:"
echo "  python tools/ability_benchmark.py -r autopilot_full_results/full_evaluation.json"
echo ""
