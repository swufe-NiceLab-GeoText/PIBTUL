
from collections import OrderedDict
import networkx as nx
from node2vec import Node2Vec
import numpy as np

# 读取数据并移除用户ID（保持原数据处理）
with open('data/processed_data/Gowalla_200.txt', 'r') as file:
    data = file.readlines()

# 解析数据并记录POI出现顺序（保持原有映射逻辑）
poi_sequences = []
poi_vocab = OrderedDict()
current_id = 1  # 从1开始编号

for line in data:
    numbers = line.strip().split()
    if len(numbers) < 2:
        continue
    raw_seq = list(map(int, numbers[1:]))

    # 转换原始POI序列为新ID（保持字符串格式）
    new_seq = []
    for poi in raw_seq:
        if poi not in poi_vocab:
            poi_vocab[poi] = current_id
            current_id += 1
        new_seq.append(str(poi_vocab[poi]))  # 保持字符串格式用于图构建

    poi_sequences.append(new_seq)

# 构建带权有向图（适配原有数据格式）
G = nx.DiGraph()

# 第一步：添加所有节点（确保孤立POI存在）
for raw_poi, mapped_id in poi_vocab.items():
    G.add_node(str(mapped_id))

# 第二步：添加带权边
for seq in poi_sequences:
    for i in range(len(seq) - 1):
        src = seq[i]
        dst = seq[i + 1]
        if G.has_edge(src, dst):
            G[src][dst]['weight'] += 1
        else:
            G.add_edge(src, dst, weight=1)

# 配置Node2Vec参数（优化轨迹特性）
node2vec = Node2Vec(
    G,
    dimensions=250,  # 保持与原维度一致
    walk_length=5,  # 适合长距离模式
    num_walks=100,  # 增加采样数量
    p=0.5,  # 降低返回概率（促进探索）
    q=2.0,  # 提高出入参数（强调序列顺序）
    weight_key='weight',  # 启用边权重
    workers=8
)

# 训练模型（适配加权图）
# 修改模型训练部分为：
model = node2vec.fit(
    window=3,
    min_count=1,
    negative=20,
    epochs=20,
    alpha=0.025,  # 正确参数名称（替代learning_rate）
    min_alpha=0.001,
    batch_words=10000,
    compute_loss=True,
    workers=8,
    seed=42
)
# 保存嵌入文件（严格保持原格式）
with open('data/processed_data/gowalla200_embedding_node2vec.dat', 'w') as f:
    # 写入文件头（总数=映射POI数 + 1）
    total_pois = len(poi_vocab) + 1
    f.write(f"{total_pois} {model.wv.vector_size}\n")

    # 新增0号POI向量（全零）
    f.write(f"0 {' '.join(['0'] * model.wv.vector_size)}\n")

    # 按原始映射顺序写入所有POI
    for raw_poi, mapped_id in poi_vocab.items():
        node_id = str(mapped_id)
        if node_id in model.wv:
            vector = model.wv[node_id]
        else:
            # 处理未训练到的POI（高斯随机初始化）
            vector = np.random.normal(scale=0.1, size=model.wv.vector_size)

        # 格式化为原始代码样式
        line = f"{mapped_id} {' '.join(map(str, vector))}\n"
        f.write(line)

# 验证嵌入完整性
print(f"Total POIs in vocab: {len(poi_vocab)}")
print(f"Embeddings saved: {total_pois}")
print(f"Vectors in model: {len(model.wv)}")