import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import timedelta
import os

# 简单直接的文件读取方式
def read_csv_with_encoding(file_path):
    try:
        print("使用 latin1 编码读取文件...")
        # 直接使用 latin1 编码，这个编码几乎可以读取任何字节
        with open(file_path, 'r', encoding='latin1') as f:
            lines = f.readlines()
        
        print(f"成功读取 {len(lines)} 行原始数据")
        
        # 手动解析数据
        data = []
        for i, line in enumerate(lines):
            try:
                parts = line.strip().split('\t')
                if len(parts) >= 8:  # 确保有足够的列
                    data.append(parts[:8])  # 只取前8列
            except Exception as e:
                print(f"跳过第 {i+1} 行: {e}")
                continue
        
        # 使用字典方式创建DataFrame，避免版本兼容性问题
        columns = ['user_id', 'venue_id', 'cat_id', 'cat_name',
                  'lat', 'lon', 'tz_offset', 'utc_time']
        
        # 转置数据并创建字典
        data_dict = {}
        for i, col in enumerate(columns):
            data_dict[col] = [row[i] if i < len(row) else '' for row in data]
        
        df = pd.DataFrame(data_dict)
        print(f"成功创建 DataFrame: {len(df)} 行数据")
        return df
        
    except Exception as e:
        print(f"读取文件失败: {e}")
        raise

# 核心函数: 基于时间间隔切分轨迹
def split_trajectories(user_data, time_threshold=timedelta(hours=8)):
    """
    按时间间隔切分用户的签到记录为多条轨迹
    :param user_data: 单个用户的签到记录（按时间排序）
    :param time_threshold: 时间间隔阈值（超过此值则分割轨迹）
    :return: 该用户的有效轨迹列表
    """
    # 按时间排序
    user_data = user_data.sort_values('utc_time')
    trajectories = []
    current_traj = []
    last_time = None

    for _, row in user_data.iterrows():
        if last_time is None:
            # 第一条记录
            current_traj.append(row)
            last_time = row['utc_time']
            continue

        # 计算时间差
        time_diff = row['utc_time'] - last_time

        # 如果时间差超过阈值，结束当前轨迹并开始新的
        if time_diff > time_threshold:
            if len(current_traj) > 0:
                trajectories.append(current_traj)
            current_traj = []

        # 添加当前点到轨迹
        current_traj.append(row)
        last_time = row['utc_time']

    # 添加最后的轨迹
    if len(current_traj) > 0:
        trajectories.append(current_traj)

    return trajectories

# 读取数据
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(script_dir, 'raw_data', 'dataset_TSMC2014_NYC.txt')
print(f"尝试读取文件: {data_file}")
df = read_csv_with_encoding(data_file)

# 安全处理时间列
def safe_parse_time(time_str):
    try:
        return pd.to_datetime(time_str, format='%a %b %d %H:%M:%S %z %Y')
    except:
        return pd.NaT

df['utc_time'] = df['utc_time'].apply(safe_parse_time)
df = df[df['utc_time'].notna()]
print(f"处理后数据集: {len(df)} 行, {df['user_id'].nunique()} 个用户")

# 过滤前数据统计
# print("过滤前数据统计:")
# print(f"  总用户数: {df['user_id'].nunique()}")
# print(f"  总POI数: {df['venue_id'].nunique()}")
# print(f"  总签到次数: {len(df)}")

# # 过滤低频用户：总签到次数少于10次的用户
# print("过滤低频用户...")
# user_checkin_counts = df['user_id'].value_counts()
# active_users = user_checkin_counts[user_checkin_counts >= 10].index
df = df  # 不过滤用户
# print(f"过滤后用户数: {df['user_id'].nunique()}")

# # 过滤POI前数据统计
# print(f"过滤POI前数据统计:")
# print(f"  总POI数: {df['venue_id'].nunique()}")
# print(f"  总签到次数: {len(df)}")

# # 过滤低频POI：被访问次数少于10次的POI
# print("过滤低频POI...")
# poi_visit_counts = df['venue_id'].value_counts()
# active_pois = poi_visit_counts[poi_visit_counts >= 10].index
df = df  # 不过滤POI
# print(f"过滤后POI数: {len(active_pois)}")
print(f"当前数据集: {len(df)} 行")

# 设置轨迹分割参数
TIME_THRESHOLD = timedelta(hours=8)  # 8小时无活动分割轨迹
MIN_POINTS_PER_TRAJ = 2  # 每条轨迹最少点数（包含头尾）
MAX_TRAJ_LENGTH = 100  # 每条轨迹最多点数
MIN_USER_TRAJ = 2  # 用户最少轨迹数
SAMPLE_USERS = 200  # 目标用户数

# 计算每个用户的轨迹数量
print("计算用户轨迹数量...")
user_traj_counts = {}
user_traj_data = {}

for user_id, group in df.groupby('user_id'):
    # 分割用户轨迹
    user_trajs = split_trajectories(group, TIME_THRESHOLD)

    # 过滤轨迹：长度在合理范围内
    valid_trajs = []
    for traj in user_trajs:
        point_count = len(traj)
        if MIN_POINTS_PER_TRAJ <= point_count <= MAX_TRAJ_LENGTH:
            valid_trajs.append(traj)

    # 不过滤用户轨迹数量
    if len(valid_trajs) >= 0:
        user_traj_counts[user_id] = len(valid_trajs)
        user_traj_data[user_id] = valid_trajs

# 按轨迹数量降序排序用户
sorted_users = sorted(user_traj_counts.items(), key=lambda x: x[1], reverse=True)
print(f"最多轨迹用户: {sorted_users[0][1]} 条轨迹")

# 选择前800个用户
selected_users = [user_id for user_id, _ in sorted_users[:SAMPLE_USERS]]
print(f"选择前 {len(selected_users)} 个最多轨迹用户")

# 生成轨迹序列
output_lines = []
user_counter = 1  # 用户id从1开始映射

# 创建venue映射，poi id从总用户数量+1开始映射
venue_mapping = {}
current_id = df['user_id'].nunique() + 1  # poi id从总用户数量+1开始映射

for user_id in selected_users:
    # 获取该用户的所有轨迹
    valid_trajs = user_traj_data[user_id]

    # 处理每条轨迹
    for traj in valid_trajs:
        poi_sequence = []
        for point in traj:
            venue_id = point['venue_id']

            # 更新venue映射
            if venue_id not in venue_mapping:
                venue_mapping[venue_id] = current_id
                current_id += 1

            # 添加POI点到序列
            poi_sequence.append(str(venue_mapping[venue_id]))

        # 创建轨迹行
        output_lines.append(f"{user_counter} {' '.join(poi_sequence)}")

    user_counter += 1

# 确保processed_data文件夹存在
processed_data_dir = os.path.join(script_dir, 'processed_data')
if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

# 保存结果
output_file = os.path.join(script_dir, 'processed_data', 'fs_NYC_200.txt')
with open(output_file, 'w') as f:
    f.write("\n".join(output_lines))

print(f"\n数据已保存至 {output_file}")

# # 保存用户映射
# user_mapping_df = pd.DataFrame({
#     'original_user_id': selected_users,
#     'mapped_user_id': range(1, len(selected_users) + 1)  # 从1开始
# })
# user_mapping_file = os.path.join(script_dir, 'processed_data', 'user_mapping_NYC.csv')
# user_mapping_df.to_csv(user_mapping_file, index=False)

# # 保存venue映射（新增）
# venue_mapping_df = pd.DataFrame({
#     'original_venue_id': list(venue_mapping.keys()),
#     'mapped_id': list(venue_mapping.values())
# })
# venue_mapping_file = os.path.join(script_dir, 'processed_data', 'venue_mapping_NYC.csv')
# venue_mapping_df.to_csv(venue_mapping_file, index=False)

print("\n处理完成!")
print(f"成功生成 {len(selected_users)} 个用户")
print(f"生成轨迹数: {len(output_lines)}")
print(f"最短轨迹: {min(len(line.split()) - 1 for line in output_lines)} 个POI点")
print(f"最长轨迹: {max(len(line.split()) - 1 for line in output_lines)} 个POI点")
print(f"平均轨迹长度: {np.mean([len(line.split()) - 1 for line in output_lines]):.1f} 个POI点")
print(f"映射场地数: {len(venue_mapping)}")
# print(f"venue映射文件已保存至 {venue_mapping_file}")