import random
import os

# 设置随机种子以确保结果可复现
random.seed(42)

# 输入文件路径 (相对路径)
data_file = 'fs_TKY_time_segment.txt'
# 输出文件路径 (相对路径)
train_file = 'fs_TKY_time_segment_train.txt'
test_file = 'fs_TKY_time_segment_test.txt'

# 检查输入文件是否存在
if not os.path.exists(data_file):
    print(f"错误: 输入文件 {data_file} 不存在")
    exit(1)

# 按用户分组存储轨迹
user_trajectories = {}

# 读取数据集并按用户分组
print("正在读取数据集...")
with open(data_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = list(map(int, line.split()))
        user_id = parts[0]
        trajectory = parts  # 完整的轨迹行，包括用户ID
        
        if user_id not in user_trajectories:
            user_trajectories[user_id] = []
        user_trajectories[user_id].append(trajectory)

print(f"共有 {len(user_trajectories)} 个用户的数据")

# 初始化训练集和测试集
train_trajectories = []
test_trajectories = []

# 对每个用户的数据进行8:2划分
print("正在进行8:2数据划分...")
for user_id, trajectories in user_trajectories.items():
    num_trajectories = len(trajectories)
    if num_trajectories < 5:  # 如果轨迹数量太少，全部放入训练集
        train_trajectories.extend(trajectories)
        print(f"用户 {user_id} 轨迹数量较少 ({num_trajectories}条)，全部放入训练集")
        continue
    
    # 打乱轨迹顺序
    random.shuffle(trajectories)
    
    # 计算训练集和测试集的分割点
    split_point = int(num_trajectories * 0.8)
    
    # 分配到训练集和测试集
    train_trajectories.extend(trajectories[:split_point])
    test_trajectories.extend(trajectories[split_point:])
    
    print(f"用户 {user_id}: 总轨迹数 {num_trajectories}, 训练集 {split_point}条, 测试集 {num_trajectories - split_point}条")

# 保存训练集
print("\n正在保存训练集...")
with open(train_file, 'w', encoding='utf-8') as f:
    for trajectory in train_trajectories:
        f.write(' '.join(map(str, trajectory)) + '\n')

# 保存测试集
print("正在保存测试集...")
with open(test_file, 'w', encoding='utf-8') as f:
    for trajectory in test_trajectories:
        f.write(' '.join(map(str, trajectory)) + '\n')

# 打印统计信息
total_train = len(train_trajectories)
total_test = len(test_trajectories)
total_all = total_train + total_test

train_percent = (total_train / total_all) * 100
test_percent = (total_test / total_all) * 100

print("\n数据划分完成!")
print(f"总轨迹数: {total_all}")
print(f"训练集轨迹数: {total_train} ({train_percent:.2f}%)")
print(f"测试集轨迹数: {total_test} ({test_percent:.2f}%)")
print(f"训练集文件已保存至: {train_file}")
print(f"测试集文件已保存至: {test_file}")