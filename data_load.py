import os
import numpy as np
from collections import Counter
from typing import List, Tuple

def load_and_preprocess_data(data_path: str, max_seq_len: int = 30):
    """
    加载并预处理数据
    
    参数:
    data_path: 数据文件路径
    max_seq_len: 最大序列长度，默认30
    
    返回:
    inputs: 预处理后的输入序列
    outputs: 预处理后的标签序列
    word2idx: 词汇到索引的映射
    label2idx: 标签到索引的映射
    idx2label: 索引到标签的映射
    """
    

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_sentence = []
    current_labels = []
    all_sentences = []
    all_labels = []

    
   
    
    for line in lines:
        line = line.strip()
        if not line:  # 空行表示句子结束
            if current_sentence:
                all_sentences.append(current_sentence)
                all_labels.append(current_labels)
                current_sentence = []
                current_labels = []
        else:
            parts = line.split()
            if len(parts) >= 2:  
                word = parts[0]
                label = parts[1]  
                current_sentence.append(word)
                current_labels.append(label)
    
    # 添加最后一个句子
    if current_sentence:
        all_sentences.append(current_sentence)
        all_labels.append(current_labels)
    
    print(f"总共加载了 {len(all_sentences)} 个句子")
    
    # 3. 构建词汇表和标签表
    # 构建词汇表（统计所有词语）
    word_counter = Counter()
    for sentence in all_sentences:
        word_counter.update(sentence)
    
    # 添加特殊标记
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in word_counter.most_common():
        word2idx[word] = len(word2idx)
    
    # 构建标签表（实验步骤中提到的7个标签）
    # 注意：这里按照文档中的7个标签定义
    labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
    label2idx = {label: idx + 1 for idx, label in enumerate(labels)}  # 从1开始编号
    label2idx['<PAD>'] = 0  # 填充标记为0
    idx2label = {idx: label for label, idx in label2idx.items()}
    
    # 4. 转换为索引并填充/截断
    processed_inputs = []
    processed_outputs = []
    
    for sentence, labels in zip(all_sentences, all_labels):
        # 转换为索引
        word_indices = [word2idx.get(word, word2idx['<UNK>']) for word in sentence]
        label_indices = [label2idx.get(label, 0) for label in labels]  # 未知标签设为0
        
        # 截断或填充
        if len(word_indices) > max_seq_len:
            word_indices = word_indices[:max_seq_len]
            label_indices = label_indices[:max_seq_len]
        else:
            padding_length = max_seq_len - len(word_indices)
            word_indices = word_indices + [word2idx['<PAD>']] * padding_length
            label_indices = label_indices + [label2idx['<PAD>']] * padding_length
        
        processed_inputs.append(word_indices)
        processed_outputs.append(label_indices)
    
    # 转换为numpy数组
    processed_inputs = np.array(processed_inputs, dtype=np.int32)
    processed_outputs = np.array(processed_outputs, dtype=np.int32)
    
    return processed_inputs, processed_outputs, word2idx, label2idx, idx2label

def validate_preprocessing(inputs, outputs, idx2label, word2idx, num_samples=3):
    """
    验证预处理步骤是否正确
    
    参数:
    inputs: 预处理后的输入
    outputs: 预处理后的输出
    idx2label: 索引到标签的映射
    word2idx: 词汇到索引的映射
    num_samples: 验证的样本数量
    """
    
    print("=" * 60)
    print("验证预处理结果:")
    print("=" * 60)
    
    # 1. 验证形状
    print(f"1. 输入数据形状: {inputs.shape}")
    print(f"   输出数据形状: {outputs.shape}")
    print(f"   所有序列长度应为30: {inputs.shape[1] == 30}")
    
    # 2. 验证填充是否正确
    print(f"\n2. 验证填充:")
    for i in range(min(num_samples, len(inputs))):
        original_len = np.sum(inputs[i] != word2idx['<PAD>'])
        print(f"   样本{i+1}: 原始长度={original_len}, 填充后长度={len(inputs[i])}")
    
    # 3. 显示样本
    print(f"\n3. 显示前{num_samples}个样本:")
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    for i in range(min(num_samples, len(inputs))):
        print(f"\n   样本{i+1}:")
        
        # 获取原始词和标签（去除填充）
        words = []
        labels = []
        for word_idx, label_idx in zip(inputs[i], outputs[i]):
            if word_idx != word2idx['<PAD>']:
                words.append(idx2word.get(word_idx, '<UNK>'))
                labels.append(idx2label.get(label_idx, '<PAD>'))
        
        print(f"   词: {' '.join(words)}")
        print(f"   标签: {' '.join(labels)}")
        
        # 验证对应关系
        if len(words) == len(labels):
            print(f"   词和标签数量匹配: ✓")
        else:
            print(f"   词和标签数量不匹配: ✗")
    
    # 4. 统计标签分布
    print(f"\n4. 标签分布统计:")
    unique_labels, counts = np.unique(outputs, return_counts=True)
    for label_idx, count in zip(unique_labels, counts):
        label_name = idx2label.get(label_idx, f"未知({label_idx})")
        print(f"   标签 {label_name}: {count} 次 ({count/len(outputs.flatten())*100:.2f}%)")
    
    # 5. 验证标签编号是否符合要求
    print(f"\n5. 验证标签编号:")
    expected_labels = {'O': 1, 'B-PER': 2, 'I-PER': 3, 'B-ORG': 4, 
                      'I-ORG': 5, 'B-LOC': 6, 'I-LOC': 7, '<PAD>': 0}
    
    all_correct = True
    for label_name, expected_idx in expected_labels.items():
        if label_name in idx2label.values():
            actual_idx = [k for k, v in idx2label.items() if v == label_name][0]
            if actual_idx == expected_idx:
                print(f"   {label_name}: 编号正确 ({actual_idx})")
            else:
                print(f"   {label_name}: 编号错误 (期望{expected_idx}, 实际{actual_idx})")
                all_correct = False
    
    return all_correct

