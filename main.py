import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import re
import sys
from collections import Counter
import os  # 新增：用于文件存在性判断
from torchcrf import CRF  # 新增：导入CRF类，解决报错问题

# ==================== 配置文件 ====================
CONFIG = {
    'model_path': 'ner_model_crf_final.pth',
    'force_retrain': False,  # 设置为True强制重新训练，False则优先加载已有模型
    'seq_len': None,  # 将从数据中获取
    'batch_size': 64,
    'num_epochs': 10,
    'learning_rate': 0.001,
    'hidden_dim': 128,
    'embedding_dim': 300
}

# ==================== 1. 加载预处理数据并验证 ====================
print("加载预处理数据...")
processed_inputs = np.load('processed_inputs.npy')
processed_outputs = np.load('processed_outputs.npy')

with open('word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)

with open('label2idx.pkl', 'rb') as f:
    label2idx = pickle.load(f)

with open('idx2label.pkl', 'rb') as f:
    idx2label = pickle.load(f)

# 验证标签映射
print("=" * 60)
print("标签映射验证:")
print("=" * 60)
for idx, label in idx2label.items():
    print(f"  索引 {idx} -> 标签 '{label}'")

# 检查输出数据的标签范围
unique_labels = np.unique(processed_outputs)
print(f"\n输出数据中的唯一标签索引: {unique_labels}")
print(f"最小标签索引: {unique_labels.min()}, 最大标签索引: {unique_labels.max()}")

# 如果标签索引不是从0开始，需要调整
if unique_labels.min() > 0:
    print("警告: 标签索引不是从0开始，这可能导致问题!")
    # 修正标签映射
    print("尝试修正标签映射...")
    # 创建新的标签映射，确保从0开始
    all_labels = ['<PAD>', 'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
    label2idx_fixed = {label: idx for idx, label in enumerate(all_labels)}
    idx2label_fixed = {idx: label for idx, label in enumerate(all_labels)}
    
    # 转换输出数据
    processed_outputs_fixed = np.zeros_like(processed_outputs)
    for old_idx, new_idx in zip(range(1, len(all_labels)), range(1, len(all_labels))):
        processed_outputs_fixed[processed_outputs == old_idx] = new_idx
    
    processed_outputs = processed_outputs_fixed
    label2idx = label2idx_fixed
    idx2label = idx2label_fixed
    
    print("标签映射已修正:")
    for idx, label in idx2label.items():
        print(f"  索引 {idx} -> 标签 '{label}'")

# 统计标签分布
print("\n标签分布统计:")
for idx in range(len(idx2label)):
    count = np.sum(processed_outputs == idx)
    percentage = count / processed_outputs.size * 100
    print(f"  {idx2label[idx]}({idx}): {count}次 ({percentage:.2f}%)")

seq_len = processed_inputs.shape[1]
vocab_size = len(word2idx) + 1
tag_size = len(label2idx)
CONFIG['seq_len'] = seq_len

print(f"\n序列长度: {seq_len}")
print(f"词汇表大小: {vocab_size}")
print(f"标签数量: {tag_size}")

# ==================== 2. 加载预训练词向量（优化后：保存/直接加载文件） ====================
def load_pretrained_embeddings(embedding_path, word2idx, embedding_dim=300, save_path='pretrained_embeddings.pt'):
    """
    加载预训练词向量（优化版：支持缓存保存/直接加载）
    :param embedding_path: 原始词向量文件路径
    :param word2idx: 词到索引的映射
    :param embedding_dim: 词向量维度
    :param save_path: 词向量缓存文件保存路径
    :return: torch.FloatTensor 预训练词向量
    """
    # 第一步：检查缓存文件是否存在，存在则直接加载返回
    if os.path.exists(save_path):
        print(f"\n发现词向量缓存文件: {save_path}，直接加载...")
        try:
            pretrained_embeds = torch.load(save_path)
            print("词向量缓存文件加载成功！")
            return pretrained_embeds
        except Exception as e:
            print(f"加载词向量缓存文件失败: {e}，将重新加载原始词向量...")
    
    # 第二步：缓存文件不存在/加载失败，按原逻辑加载原始词向量
    print(f"\n加载预训练词向量: {embedding_path}")
    
    vocab_size = len(word2idx) + 1
    embeddings = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
    embeddings[0] = np.zeros(embedding_dim)  # padding
    
    try:
        found_words = 0
        with open(embedding_path, 'r', encoding='utf-8') as f:
            # 读取第一行词数量和维度
            first_line = f.readline().strip()
            if len(first_line.split()) == 2:
                print(f"词向量文件头: {first_line}")
            else:
                f.seek(0)  # 重置文件指针
            
            for line_num, line in enumerate(tqdm(f, desc="加载词向量")):
                line = line.strip()
                if not line:
                    continue
                
                # 按空格分割
                parts = line.split()
                if len(parts) < embedding_dim + 1:
                    continue
                
                # 尝试从后往前找embedding_dim个数字
                vector_parts = []
                word_parts = []
                
                # 从后向前收集数字 避免词中有空格的问题
                for part in reversed(parts):
                    # 检查是否是数字（包含小数点、负号、科学计数法）
                    if re.match(r'^[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?$', part):
                        vector_parts.insert(0, part)
                        if len(vector_parts) == embedding_dim:
                            word_parts = parts[:-len(vector_parts)]
                            break
                
                if len(vector_parts) == embedding_dim:
                    try:
                        vector = np.array([float(x) for x in vector_parts])
                        word = ' '.join(word_parts) if word_parts else parts[0]
                        
                        if word in word2idx:
                            idx = word2idx[word]
                            embeddings[idx] = vector
                            found_words += 1
                    except:
                        continue
        
        print(f"成功加载 {found_words}/{len(word2idx)} 个词的预训练向量")
        
    except Exception as e:
        print(f"加载词向量时出错: {e}")
        print("使用随机初始化词向量")
    
    # 第三步：将生成的词向量保存到文件，方便后续直接加载
    pretrained_embeds = torch.FloatTensor(embeddings)
    try:
        torch.save(pretrained_embeds, save_path)
        print(f"词向量已保存到缓存文件: {save_path}")
    except Exception as e:
        print(f"保存词向量缓存文件失败: {e}")
    
    return pretrained_embeds

# 加载词向量
embedding_path = 'sgns.renmin.bigram-char'
pretrained_embeddings = load_pretrained_embeddings(embedding_path, word2idx)

# ==================== 3. BiLSTM+CRF模型（CRF报错已解决） ====================
class BiLSTM_CRF(nn.Module):
    """BiLSTM+CRF模型"""
    def __init__(self, vocab_size, tag_size, embedding_dim=300, hidden_dim=128):
        super(BiLSTM_CRF, self).__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 如果提供了预训练词向量
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True  # 允许微调
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.5
        )
        
        # 全连接层，将LSTM输出映射到标签空间
        self.fc = nn.Linear(hidden_dim * 2, tag_size)
        self.dropout = nn.Dropout(0.5)
        
        # CRF层（已导入torchcrf的CRF类，报错解决）
        self.crf = CRF(tag_size, batch_first=True)
        
    def forward(self, x, tags=None, mask=None):
        # 获取序列长度（非padding部分）
        if mask is None:
            mask = (x != 0).bool()
        
        # 嵌入层
        embeds = self.embedding(x)
        
        # LSTM层
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        
        # 全连接层，得到发射分数
        emissions = self.fc(lstm_out)
        
        # 如果有标签，计算CRF损失
        if tags is not None:
            # CRF层计算负对数似然损失
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            # 预测模式，使用Viterbi解码
            predictions = self.crf.decode(emissions, mask=mask)
            return predictions

# ==================== 4. 检查并加载已有模型 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}")

# 检查模型文件是否存在
model_file_exists = os.path.exists(CONFIG['model_path'])

if model_file_exists and not CONFIG['force_retrain']:
    print("\n" + "="*60)
    print("发现已训练模型，正在加载...")
    print("="*60)
    
    try:
        # 加载保存的模型
        checkpoint = torch.load(CONFIG['model_path'], map_location=device)
        
        # 从checkpoint中获取必要的配置信息
        loaded_word2idx = checkpoint.get('word2idx', word2idx)
        loaded_label2idx = checkpoint.get('label2idx', label2idx)
        loaded_idx2label = checkpoint.get('idx2label', idx2label)
        loaded_vocab_size = checkpoint.get('vocab_size', vocab_size)
        loaded_tag_size = checkpoint.get('tag_size', tag_size)
        
        # 使用加载的配置或默认配置
        model = BiLSTM_CRF(loaded_vocab_size, loaded_tag_size, 
                          CONFIG['embedding_dim'], CONFIG['hidden_dim'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        print("模型加载成功！")
        print(f"词汇表大小: {loaded_vocab_size}")
        print(f"标签数量: {loaded_tag_size}")
        
        # 更新全局变量（如果需要）
        word2idx = loaded_word2idx
        label2idx = loaded_label2idx
        idx2label = loaded_idx2label
        
        # 设置模型为评估模式
        model.eval()
        
        # 跳过训练，直接进入DEMO演示
        skip_training = True
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("将重新训练模型...")
        skip_training = False
else:
    if CONFIG['force_retrain']:
        print("\n强制重新训练模式，将忽略已有模型")
    else:
        print("\n未找到已训练模型，将开始训练")
    skip_training = False

# ==================== 5. 训练模型（仅在需要时执行） ====================
if not skip_training:
    print("\n" + "="*60)
    print("开始训练BiLSTM+CRF模型")
    print("="*60)
    
    # 准备数据加载器
    train_inputs = torch.LongTensor(processed_inputs)
    train_outputs = torch.LongTensor(processed_outputs)
    train_mask = (train_inputs != 0).bool()
    
    train_dataset = TensorDataset(train_inputs, train_outputs, train_mask)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # 初始化模型（如果需要）
    if 'model' not in locals():
        model = BiLSTM_CRF(vocab_size, tag_size, CONFIG['embedding_dim'], CONFIG['hidden_dim'])
        model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # ==================== 训练函数 ====================
    def train_epoch(model, data_loader, optimizer, device):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        for batch_idx, (inputs, labels, mask) in enumerate(tqdm(data_loader, desc="训练")):
            inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
            
            # 前向传播和计算损失（CRF损失）
            loss = model(inputs, tags=labels, mask=mask)
            
            # 计算准确率（用于调试）
            with torch.no_grad():
                predictions = model(inputs, mask=mask)
                
                # 由于CRF返回的是列表，需要处理
                for i in range(len(predictions)):
                    seq_len = mask[i].sum().item()
                    pred_seq = predictions[i][:seq_len]
                    label_seq = labels[i][:seq_len].cpu().tolist()
                    
                    # 计算准确率
                    correct = sum(1 for p, l in zip(pred_seq, label_seq) if p == l)
                    total_correct += correct
                    total_tokens += seq_len
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        return avg_loss, accuracy
    
    # ==================== 训练循环 ====================
    best_accuracy = 0
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        
        # 保存最佳模型
        if train_acc > best_accuracy:
            best_accuracy = train_acc
            torch.save(model.state_dict(), 'best_model_crf.pth')
            print(f"保存最佳模型，准确率: {train_acc:.4f}")
        
        # 更新学习率
        scheduler.step()
    
    print(f"\n训练完成，最佳准确率: {best_accuracy:.4f}")
    
    # ==================== 保存完整模型 ====================
    torch.save({
        'model_state_dict': model.state_dict(),
        'word2idx': word2idx,
        'label2idx': label2idx,
        'idx2label': idx2label,
        'vocab_size': vocab_size,
        'tag_size': tag_size,
        'seq_len': seq_len,
        'config': CONFIG
    }, CONFIG['model_path'])
    
    print(f"模型已保存为 '{CONFIG['model_path']}'")
    
else:
    print("\n" + "="*60)
    print("已跳过训练阶段，使用已训练模型")
    print("="*60)

# ==================== 6. 评估函数 ====================
def evaluate_model(model, data_loader, device):
    """评估模型性能"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels, mask in tqdm(data_loader, desc="评估"):
            inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
            
            # 获取预测（CRF解码）
            predictions = model(inputs, mask=mask)
            
            # 收集非padding的预测和标签
            for i in range(len(predictions)):
                seq_len = mask[i].sum().item()
                all_predictions.extend(predictions[i][:seq_len])
                all_labels.extend(labels[i][:seq_len].cpu().tolist())
    
    return all_predictions, all_labels

def calculate_token_accuracy(predictions, labels):
    """计算token级别的准确率"""
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    total = len(labels)
    return correct / total if total > 0 else 0

def calculate_entity_metrics(predictions, labels, idx2label):
    """计算实体级别的指标"""
    # 转换为标签名
    pred_labels = [idx2label.get(p, 'O') for p in predictions]
    true_labels = [idx2label.get(l, 'O') for l in labels]
    
    # 统计各标签数量
    pred_counter = Counter(pred_labels)
    true_counter = Counter(true_labels)
    
    # 简单的实体匹配
    tp, fp, fn = 0, 0, 0
    
    i = 0
    while i < len(pred_labels):
        if pred_labels[i].startswith('B-'):
            # 找到一个预测的实体
            entity_type = pred_labels[i][2:]
            end_idx = i + 1
            while end_idx < len(pred_labels) and pred_labels[end_idx] == f'I-{entity_type}':
                end_idx += 1
            
            # 检查是否匹配真实实体
            if i < len(true_labels) and true_labels[i].startswith('B-') and true_labels[i][2:] == entity_type:
                # 检查整个实体是否匹配
                match = True
                for j in range(i, end_idx):
                    if j >= len(true_labels) or true_labels[j] != pred_labels[j]:
                        match = False
                        break
                
                if match:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            else:
                fp += 1
            
            i = end_idx
        elif true_labels[i].startswith('B-') and pred_labels[i] == 'O':
            fn += 1
            i += 1
        else:
            i += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1, pred_counter, true_counter

# ==================== 7. DEMO演示函数 ====================
def predict_sentence(model, sentence, word2idx, idx2label, device, max_len=seq_len):
    """预测单个句子"""
    # 分词
    words = list(sentence.strip())
    
    # 转换为索引
    word_indices = [word2idx.get(word, 1) for word in words]  # 1是<UNK>
    
    # 填充
    if len(word_indices) > max_len:
        word_indices = word_indices[:max_len]
    else:
        padding_len = max_len - len(word_indices)
        word_indices = word_indices + [0] * padding_len
    
    # 预测
    input_tensor = torch.LongTensor([word_indices]).to(device)
    mask = (input_tensor != 0).bool().to(device)
    
    model.eval()
    with torch.no_grad():
        predictions = model(input_tensor, mask=mask)
    
    # 提取结果（CRF返回的是列表）
    pred_indices = predictions[0] if predictions else []
    pred_labels = []
    
    for i, idx in enumerate(pred_indices[:len(words)]):
        pred_labels.append(idx2label.get(idx, 'O'))
    
    return list(zip(words, pred_labels))

# ==================== 8. 模型性能评估 ====================
print("\n" + "="*60)
print("模型性能评估")
print("="*60)

# 加载测试数据
def load_test_data(file_path, word2idx, label2idx, max_len=seq_len):
    """加载测试数据"""
    inputs = []
    outputs = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        current_input = []
        current_output = []
        
        for line in f:
            line = line.strip()
            if not line:  # 空行表示句子结束
                if current_input:
                    # 填充或截断到固定长度
                    if len(current_input) > max_len:
                        current_input = current_input[:max_len]
                        current_output = current_output[:max_len]
                    else:
                        padding_len = max_len - len(current_input)
                        current_input.extend([0] * padding_len)
                        current_output.extend([0] * padding_len)
                    
                    inputs.append(current_input)
                    outputs.append(current_output)
                    
                    current_input = []
                    current_output = []
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                word = parts[0]
                label = parts[1]
                
                # 转换词和标签为索引
                word_idx = word2idx.get(word, 1)  # 1是<UNK>
                label_idx = label2idx.get(label, 0)  # 0是O标签
                
                current_input.append(word_idx)
                current_output.append(label_idx)
    
    # 处理最后一个句子
    if current_input:
        if len(current_input) > max_len:
            current_input = current_input[:max_len]
            current_output = current_output[:max_len]
        else:
            padding_len = max_len - len(current_input)
            current_input.extend([0] * padding_len)
            current_output.extend([0] * padding_len)
        
        inputs.append(current_input)
        outputs.append(current_output)
    
    return np.array(inputs), np.array(outputs)

# 如果有测试数据，加载并进行评估
try:
    test_file = './chinese/test_data'
    print(f"\n加载测试数据: {test_file}")
    test_inputs, test_outputs = load_test_data(test_file, word2idx, label2idx)
    
    # 转换为Tensor
    test_inputs_tensor = torch.LongTensor(test_inputs)
    test_outputs_tensor = torch.LongTensor(test_outputs)
    test_mask = (test_inputs_tensor != 0).bool()
    
    # 创建测试数据集
    test_dataset = TensorDataset(test_inputs_tensor, test_outputs_tensor, test_mask)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    print(f"测试数据: {len(test_inputs)}个样本")
    
    # 评估模型
    print("\n在测试集上评估模型...")
    test_predictions, test_labels = evaluate_model(model, test_loader, device)
    
    # 计算token准确率
    test_token_acc = calculate_token_accuracy(test_predictions, test_labels)
    print(f"测试集Token准确率: {test_token_acc:.4f}")
    
    # 计算实体级别的指标
    precision, recall, f1, pred_counter, true_counter = calculate_entity_metrics(test_predictions, test_labels, idx2label)
    
    print("\n预测标签分布:")
    for label, count in pred_counter.most_common():
        print(f"  {label}: {count}次 ({count/len(test_predictions)*100:.1f}%)")
    
    print("\n真实标签分布:")
    for label, count in true_counter.most_common():
        print(f"  {label}: {count}次 ({count/len(test_labels)*100:.1f}%)")
    
    print(f"\n实体识别指标 - 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")
    
except Exception as e:
    print(f"测试数据加载失败: {e}")
    print("将使用训练数据进行评估...")
    
    # 使用训练数据进行评估
    train_inputs = torch.LongTensor(processed_inputs)
    train_outputs = torch.LongTensor(processed_outputs)
    train_mask = (train_inputs != 0).bool()
    
    train_dataset = TensorDataset(train_inputs, train_outputs, train_mask)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    print("\n在训练集上评估模型...")
    train_predictions, train_labels = evaluate_model(model, train_loader, device)
    
    # 计算token准确率
    train_token_acc = calculate_token_accuracy(train_predictions, train_labels)
    print(f"训练集Token准确率: {train_token_acc:.4f}")
    
    # 计算实体级别的指标
    precision, recall, f1, pred_counter, true_counter = calculate_entity_metrics(train_predictions, train_labels, idx2label)
    
    print("\n预测标签分布:")
    for label, count in pred_counter.most_common():
        print(f"  {label}: {count}次 ({count/len(train_predictions)*100:.1f}%)")
    
    print("\n真实标签分布:")
    for label, count in true_counter.most_common():
        print(f"  {label}: {count}次 ({count/len(train_labels)*100:.1f}%)")
    
    print(f"\n实体识别指标 - 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")

# 示例句子
print("\n" + "="*60)
print("DEMO演示")
print("="*60)

# 加载最佳模型
model.load_state_dict(torch.load('best_model_crf.pth'))

# 示例句子
demo_sentences = [
    "中国共产党",
    "中国共产党是中国最大的政党",
    "江泽民",
    "坚决贯彻国家主义江泽民关于发展两岸关系",
    "海钓比赛地点在厦门和金门之间的海域",
    "中共中央站起来了",
    "俄罗斯",
    "新加坡",   
]

for sentence in demo_sentences:
    print(f"{sentence}")
    results = predict_sentence(model, sentence, word2idx, idx2label, device)
    lable_result = [x for _, x in results]
    print(" ".join(lable_result))

    



print("\n" + "="*60)
print("NER模型应用完成！")
print("="*60)
print(f"模型文件: {CONFIG['model_path']}")
print(f"下次运行将自动加载此模型，无需重新训练")
print(f"如需重新训练，请设置 CONFIG['force_retrain'] = True")