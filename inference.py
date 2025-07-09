import os
import numpy as np
import pandas as pd
from keras.models import load_model
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix

def load_raw_sequences(folder_path):
    """加载原始变长序列数据（与训练一致）"""
    sequences = []
    labels = []
    file_names = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            data = pd.read_csv(os.path.join(folder_path, file), header=None).values
            numeric_data = pd.to_numeric(data[:, 0], errors='coerce')  # 第一列输入特征
            label_data = pd.to_numeric(data[:, 1], errors='coerce')   # 第二列标签（如果有）
            sequences.append(numeric_data[~pd.isna(numeric_data)].astype('float32'))
            labels.append(label_data[~pd.isna(label_data)].astype('float32'))
            file_names.append(file)
    return sequences, labels, file_names

def normalize_sequences(sequences):
    """对变长序列进行归一化（与训练一致）"""
    scaler = joblib.load('scaler.pkl')
    return [scaler.transform(seq.reshape(-1, 1)).flatten() for seq in sequences]

def pad_sequences(sequences, pad_value=0.):
    """对变长序列进行padding，返回统一shape的ndarray和原始长度列表"""
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    padded = np.full((len(sequences), max_len, 1), pad_value, dtype='float32')
    for i, seq in enumerate(sequences):
        padded[i, :len(seq), 0] = seq
    return padded, lengths

def infer_on_folder(model, folder_path, threshold=0.5, save_result=True, evaluate=True):
    """对指定文件夹下的全部.csv文件进行推理和结果保存、可选评估"""
    sequences, labels, file_names = load_raw_sequences(folder_path)
    X = normalize_sequences(sequences)
    X_pad, lengths = pad_sequences(X)

    # 推理
    y_pred_prob = model.predict(X_pad, batch_size=16)
    y_pred_prob = [y_pred_prob[i, :lengths[i], 0] for i in range(len(lengths))]
    y_pred_label = [(probs >= threshold).astype(int) for probs in y_pred_prob]

    # 仅保存可视化图，不保存csv
    if save_result:
        os.makedirs('infer_results', exist_ok=True)
        for i, file in enumerate(file_names):
            plt.figure(figsize=(18, 4))
            plt.plot(sequences[i], label='Amplitude')
            event_idx = np.where(y_pred_label[i] == 1)[0]
            plt.scatter(event_idx, np.array(sequences[i])[event_idx], c='r', label='Event Detected')
            if len(labels[i]) == len(sequences[i]):
                true_idx = np.where(labels[i] == 1)[0]
                plt.scatter(true_idx, np.array(sequences[i])[true_idx], c='g', marker='x', label='True Event', alpha=0.6)
            plt.title(f'Inference on {file}')
            plt.xlabel('Time step')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join('infer_results', file.replace('.csv', '_plot.png')), dpi=200)
            plt.close()
    print(f"全部推理完成，图片保存在 infer_results 文件夹。")

    # 评估精度
    if evaluate:
        # 仅对有标签的样本进行评估
        y_true_flat, y_pred_flat = [], []
        for i in range(len(labels)):
            if len(labels[i]) == len(y_pred_label[i]) and len(labels[i]) > 0:
                y_true_flat.extend(labels[i])
                y_pred_flat.extend(y_pred_label[i])
        y_true_flat = np.array(y_true_flat).astype(int)
        y_pred_flat = np.array(y_pred_flat).astype(int)
        if len(y_true_flat) > 0:
            acc = accuracy_score(y_true_flat, y_pred_flat)
            prec = precision_score(y_true_flat, y_pred_flat, zero_division=0)
            rec = recall_score(y_true_flat, y_pred_flat, zero_division=0)
            f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
            print("\n推理精度评估汇总：")
            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {rec:.4f}")
            print(f"F1 score:  {f1:.4f}")
            print("\n详细分类报告：")
            print(classification_report(y_true_flat, y_pred_flat, digits=4))
            print("混淆矩阵：")
            print(confusion_matrix(y_true_flat, y_pred_flat))
        else:
            print("没有找到可用于评估的标签数据。")

if __name__ == "__main__":
    # 加载模型
    model = load_model('best_model.h5')
    # 指定用于推理的文件夹
    test_folder = 'test_data'  # 请将你的测试数据放在该目录下，每行为[amplitude, label]
    infer_on_folder(model, test_folder)