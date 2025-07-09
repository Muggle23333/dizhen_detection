import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import joblib
from collections import Counter
import tensorflow as tf
from matplotlib.gridspec import GridSpec
from keras.utils import Sequence

def plot_dataset_split(X_train, X_val, y_train, y_val):
    """可视化数据集划分情况"""
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # 样本长度分布
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist([len(x) for x in X_train], bins=30, alpha=0.7, label='训练集')
    ax1.hist([len(x) for x in X_val], bins=30, alpha=0.7, label='验证集')
    ax1.set_title('序列长度分布对比')
    ax1.set_xlabel('序列长度')
    ax1.set_ylabel('频数')
    ax1.legend()
    
    # 标签分布
    ax2 = fig.add_subplot(gs[0, 1:])
    train_counts = Counter(np.concatenate(y_train).flatten())
    val_counts = Counter(np.concatenate(y_val).flatten())
    ax2.bar(train_counts.keys(), train_counts.values(), alpha=0.7, label='训练集')
    ax2.bar(val_counts.keys(), val_counts.values(), alpha=0.7, label='验证集')
    ax2.set_title('标签类别分布对比')
    ax2.set_xlabel('类别')
    ax2.set_ylabel('数量')
    ax2.legend()
    
    # 样本示例可视化
    ax3 = fig.add_subplot(gs[1, :])
    #sample_idx = np.random.randint(0, len(X_train))
    sample_idx = 10
    ax3.plot(X_train[sample_idx], label='地震波形')
    event_points = np.where(y_train[sample_idx] == 1)[0]
    ax3.scatter(event_points, X_train[sample_idx][event_points], 
               color='red', label='地震事件')
    ax3.set_title(f'训练集样本示例 (ID: {sample_idx})')
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('振幅')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('dataset_split.png', dpi=300)
    plt.show()


def load_raw_sequences(folder_path):
    """加载原始变长序列数据"""
    sequences = []
    labels = []
    
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            data = pd.read_csv(os.path.join(folder_path, file), header=None).values
            numeric_data = pd.to_numeric(data[:, 0], errors='coerce')  # 非数值转为 NaN
            sequences.append(numeric_data[~pd.isna(numeric_data)].astype('float32'))  # 只保留有效数值
            numeric_data = pd.to_numeric(data[:, 1], errors='coerce')  # 非数值转为 NaN
            labels.append(numeric_data[~pd.isna(numeric_data)].astype('float32'))  # 只保留有效数值
    
    return sequences, labels

def normalize_sequences(sequences, mode='train'):
    """对变长序列进行归一化"""
    if mode == 'train':
        # 合并所有序列点计算统计量
        all_values = np.concatenate(sequences)
        scaler = StandardScaler().fit(all_values.reshape(-1, 1))
        joblib.dump(scaler, 'scaler.pkl')
    else:
        scaler = joblib.load('scaler.pkl')
    
    return [scaler.transform(seq.reshape(-1, 1)).flatten() for seq in sequences]

class EarthquakeDataGenerator(Sequence):
    """自定义数据生成器处理变长序列"""
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_X = [self.X[i] for i in batch_indices]
        batch_y = [self.y[i] for i in batch_indices]
        
        # 动态批处理：按当前批次最大长度处理
        max_len = max(len(x) for x in batch_X)
        X_batch = np.zeros((len(batch_X), max_len, 1))
        y_batch = np.zeros((len(batch_y), max_len, 1))
        
        for i, (x, y) in enumerate(zip(batch_X, batch_y)):
            X_batch[i, :len(x), 0] = x
            y_batch[i, :len(y), 0] = y
            
        return X_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def build_dynamic_lstm_model():
    """构建支持变长输入的LSTM模型"""
    model = Sequential([
        Masking(mask_value=0., input_shape=(None, 1)),  # 自动处理变长序列
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall')])
    return model    

# 配置
plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_training_curves(history):
    """绘制训练过程中的各项指标曲线"""
    metrics = ['loss', 'accuracy', 'precision', 'recall']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        ax.plot(history.history[metric], linewidth=2, label=f'训练{metric}')
        ax.plot(history.history[f'val_{metric}'], linewidth=2, label=f'验证{metric}')
        ax.set_title(f'{metric}变化曲线', fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=12)
        
        # 标记最佳值
        if 'val_' + metric in history.history:
            best_epoch = np.argmin(history.history['val_loss']) if metric == 'loss' else np.argmax(history.history[f'val_{metric}'])
            best_value = history.history[f'val_{metric}'][best_epoch]
            ax.axvline(best_epoch, color='r', linestyle='--', alpha=0.3)
            ax.annotate(f'最佳: {best_value:.4f}', 
                       xy=(best_epoch, best_value),
                       xytext=(10, 10), 
                       textcoords='offset points',
                       bbox=dict(boxstyle='round', fc='w'))
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    plt.show()
    

def evaluate_variable_length_model(model, generator):
    """评估变长序列模型"""
    print("\n验证集评估:")
    results = model.evaluate(generator, verbose=0)
    metrics = {
        'loss': results[0],
        'accuracy': results[1],
        'precision': results[2],
        'recall': results[3],
        'f1_score': 2 * (results[2] * results[3]) / (results[2] + results[3] + 1e-7)
    }
    
    for name, value in metrics.items():
        print(f"{name:>10}: {value:.4f}")


def train_model():
    # 1. 数据加载与预处理
    sequences, labels = load_raw_sequences(r'train_data')
    X = normalize_sequences(sequences)
    y = labels
    
    # 2. 数据集划分
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=[l[0] for l in labels]  # 按初始标签分层
    )
    
    # 3. 可视化数据集划分
    print("\n数据集划分统计:")
    print(f"训练集样本数: {len(X_train)} | 验证集样本数: {len(X_val)}")
    print(f"训练集平均长度: {np.mean([len(x) for x in X_train]):.1f}±{np.std([len(x) for x in X_train]):.1f}")
    print(f"验证集平均长度: {np.mean([len(x) for x in X_val]):.1f}±{np.std([len(x) for x in X_val]):.1f}")
    plot_dataset_split(X_train, X_val, y_train, y_val)
    
    # 4. 创建数据生成器
    train_gen = EarthquakeDataGenerator(X_train, y_train, batch_size=32)
    val_gen = EarthquakeDataGenerator(X_val, y_val, batch_size=32, shuffle=False)
    
    # 5. 模型构建
    model = build_dynamic_lstm_model()
    
    # 6. 训练配置
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, verbose=1),
        ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.CSVLogger('training_log.csv')
    ]
    
    # 7. 模型训练
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=2,
        callbacks=callbacks,
        verbose=1
    )
    
    # 8. 绘制训练曲线
    plot_training_curves(history)
    
    # 9. 最终评估
    print("\n最终模型评估:")
    model = tf.keras.models.load_model('best_model.h5')
    evaluate_variable_length_model(model, val_gen)

if __name__ == "__main__":
    # 检查GPU可用性
    print("="*50)
    print(f"TensorFlow 版本: {tf.__version__}")
    print(f"GPU 可用: {len(tf.config.list_physical_devices('GPU')) > 0}")
    if tf.config.list_physical_devices('GPU'):
        print(f"GPU 设备: {tf.test.gpu_device_name()}")
    print("="*50)
    
    train_model()