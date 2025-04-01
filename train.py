# D:\Action_Recognition_Project\scripts\train.py

import warnings
# 禁用 albumentations 版本检查的警告
warnings.filterwarnings("ignore", message="Error fetching version info")
warnings.filterwarnings("ignore", message="A new version of Albumentations is available")

import os
import json
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MultilabelAveragePrecision
from PIL import Image
import torchvision.models as models

# 自定义 Dataset 类
class ActionRecognitionDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, transform=None, minority_transform=None, minority_classes=None, label_mapping=None):
        """
        Args:
            json_file (str): 数据集的 JSON 文件路径。
            transform (albumentations.Compose, optional): 应用于样本的主转换。
            minority_transform (albumentations.Compose, optional): 应用于少数类样本的额外转换。
            minority_classes (list, optional): 需要额外增强的少数类名称列表。
            label_mapping (dict, optional): 类别名称到索引的映射。如果为 None，将自动生成。
        """
        self.data = self.load_data(json_file)
        self.transform = transform
        self.minority_transform = minority_transform
        self.minority_classes = minority_classes
        self.label_mapping = label_mapping if label_mapping else self.get_label_mapping()

    def load_data(self, json_file):
        """
        加载 JSON 文件中的数据。

        JSON 文件格式示例：
        [
            {
                "video_id": "v_Skiing_g01_c01.mp4",
                "timestamp": 0.066667,
                "image_path": "D:\\data\\UCF\\target\\vott-json-export\\v_Skiing_g01_c01.mp4_t=0.066667.jpg",
                "keypoints": { ... },
                "label": { ... }
            },
            ...
        ]
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data

    def get_label_mapping(self):
        """
        创建类别名称到索引的映射。

        返回：
            dict: 类别名称到索引的映射。
        """
        classes = set()
        for sample in self.data:
            regions = sample.get('label', {}).get('regions', [])
            for region in regions:
                tags = region.get('tags', [])
                for tag in tags:
                    classes.add(tag)
        classes = sorted(list(classes))
        return {cls: idx for idx, cls in enumerate(classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取样本数据
        sample = self.data[idx]
        img_path = sample.get('image_path')
        keypoints_data = sample.get('keypoints', {}).get('people', [])
        label_info = sample.get('label', {})
        regions = label_info.get('regions', [])

        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            logging.error(f"无法加载图像 {img_path}: {e}")
            # 返回一个全零的图像，避免训练中断
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # 提取关键点（假设只处理第一个人的关键点）
        if keypoints_data and len(keypoints_data) > 0:
            pose_keypoints = keypoints_data[0].get('pose_keypoints_2d', [])
            # 只取有效的关键点 (x, y)，忽略置信度
            keypoints = [pose_keypoints[i] for i in range(0, min(len(pose_keypoints), 75), 3) if pose_keypoints[i+2] > 0]
            keypoints = keypoints[:75]  # 确保长度一致，填充或截断
            keypoints += [0.0] * (75 - len(keypoints))  # 填充
        else:
            keypoints = [0.0] * 75  # 没有关键点时填充

        # 提取标签
        labels = [0] * len(self.label_mapping)
        for region in regions:
            tags = region.get('tags', [])
            for tag in tags:
                if tag in self.label_mapping:
                    labels[self.label_mapping[tag]] = 1

        # 检查是否属于少数类
        is_minority = False
        if self.minority_classes:
            for cls in self.minority_classes:
                cls_idx = self.label_mapping.get(cls, -1)
                if cls_idx != -1 and labels[cls_idx] == 1:
                    is_minority = True
                    break

        # 应用主转换
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']  # 这是一个 Tensor

        # 应用少数类的额外转换
        if is_minority and self.minority_transform:
            # 将 Tensor 转换为 numpy array (HWC)
            image_np = image.permute(1, 2, 0).cpu().numpy()
            augmented = self.minority_transform(image=image_np)
            image = augmented['image'].transpose(2, 0, 1)  # 转换回 CHW 格式的 numpy array

            # 将 numpy array 转换回 Tensor
            image = torch.from_numpy(image).float()

        # 转换为 Tensor
        keypoints = torch.tensor(keypoints, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.float)

        return image, keypoints, labels

# 自定义 CNN-LSTM 模型
class CNN_LSTM_Model(nn.Module):
    def __init__(self, cnn_output_size, keypoints_size, lstm_hidden_size, num_classes):
        super(CNN_LSTM_Model, self).__init__()
        # 使用预训练的 ResNet50 作为 CNN 特征提取器
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.cnn.fc = nn.Identity()  # 移除最后的全连接层

        # LSTM 部分
        self.lstm = nn.LSTM(input_size=cnn_output_size + keypoints_size, hidden_size=lstm_hidden_size, batch_first=True)

        # 最终分类层
        self.classifier = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, images, keypoints):
        # CNN 特征提取
        cnn_features = self.cnn(images)  # [batch_size, cnn_output_size]

        # 将关键点特征与 CNN 特征拼接
        combined_features = torch.cat((cnn_features, keypoints), dim=1)  # [batch_size, cnn_output_size + keypoints_size]

        # LSTM 期望输入为 [batch_size, seq_len, input_size]
        # 假设 seq_len=1，因为每个样本只有一个时间步
        lstm_input = combined_features.unsqueeze(1)  # [batch_size, 1, cnn_output_size + keypoints_size]

        lstm_out, _ = self.lstm(lstm_input)  # lstm_out: [batch_size, seq_len, lstm_hidden_size]
        lstm_out = lstm_out.squeeze(1)  # [batch_size, lstm_hidden_size]

        # 分类
        outputs = self.classifier(lstm_out)  # [batch_size, num_classes]

        return outputs

def extract_labels(data, label_mapping):
    """
    从 JSON 数据中提取标签，并转换为多标签二进制格式。

    Args:
        data (list): JSON 数据列表。
        label_mapping (dict): 类别名称到索引的映射。

    Returns:
        np.ndarray: 标签数组，形状为 (样本数, 类别数)。
    """
    labels_list = []
    for sample in data:
        labels = [0] * len(label_mapping)
        regions = sample.get('label', {}).get('regions', [])
        for region in regions:
            tags = region.get('tags', [])
            for tag in tags:
                if tag in label_mapping:
                    labels[label_mapping[tag]] = 1
        labels_list.append(labels)
    return np.array(labels_list)

def calculate_class_weights(labels):
    """
    计算每个类别的权重，使用逆频率法。

    参数：
        labels (np.ndarray): 标签数组，形状为 (样本数, 类别数)。

    返回：
        np.ndarray: 类别权重数组，形状为 (类别数,).
    """
    class_counts = labels.sum(axis=0)
    class_weights = 1.0 / (class_counts + 1e-6)  # 避免除零
    class_weights = class_weights / class_weights.sum() * len(class_weights)  # 归一化
    return class_weights

def print_class_distribution(labels, label_mapping, dataset_type="Train"):
    class_counts = labels.sum(axis=0)
    print(f"{dataset_type} 类别分布：")
    for cls, idx in label_mapping.items():
        print(f"  类别 '{cls}': {int(class_counts[idx])} 个样本")

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    preds_list = []
    labels_list = []
    for images, keypoints, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        keypoints = keypoints.to(device)
        labels = labels.to(device).float()

        # 前向传播
        outputs = model(images, keypoints)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # 记录预测和标签
        preds = torch.sigmoid(outputs) > 0.5
        preds_list.append(preds.cpu().numpy())
        labels_list.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    preds = np.vstack(preds_list)
    labels = np.vstack(labels_list)

    # 计算指标
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)

    return epoch_loss, accuracy, precision, recall, f1

def validate(model, dataloader, criterion, device, map_calculator):
    model.eval()
    running_loss = 0.0
    preds_list = []
    labels_list = []
    with torch.no_grad():
        for images, keypoints, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            keypoints = keypoints.to(device)
            labels = labels.to(device).float()

            outputs = model(images, keypoints)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            preds = torch.sigmoid(outputs) > 0.5
            preds_list.append(preds.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    preds = np.vstack(preds_list)
    labels = np.vstack(labels_list)

    # 计算指标
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)

    # 计算 mAP
    preds_tensor = torch.sigmoid(torch.from_numpy(preds).float()).to(device)
    labels_tensor = torch.from_numpy(labels).long().to(device)
    map_score = map_calculator(preds_tensor, labels_tensor).item()

    # 计算不同阈值下的 AP
    thresholds = [0.5, 0.75]
    ap_threshold_dict = {}
    for threshold in thresholds:
        binary_preds = (preds > threshold).astype(int)
        ap_per_class = []
        for i in range(labels.shape[1]):
            if np.sum(binary_preds[:, i]) == 0:
                # 如果没有预测为该类别的样本，则 AP 为 0
                ap = 0.0
            else:
                ap = average_precision_score(labels[:, i], preds[:, i])
            ap_per_class.append(ap)
        map_score_threshold = np.mean(ap_per_class)
        ap_threshold_dict[f'mAP@{threshold}'] = map_score_threshold

    return epoch_loss, accuracy, precision, recall, f1, map_score, ap_threshold_dict

def evaluate_test_set(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds_list = []
    labels_list = []
    with torch.no_grad():
        for images, keypoints, labels in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            keypoints = keypoints.to(device)
            labels = labels.to(device).float()

            outputs = model(images, keypoints)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            preds = torch.sigmoid(outputs) > 0.5
            preds_list.append(preds.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    test_loss = running_loss / len(dataloader.dataset)
    preds = np.vstack(preds_list)
    labels = np.vstack(labels_list)

    # 计算指标
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    map_score = average_precision_score(labels, preds, average='macro')

    return test_loss, accuracy, precision, recall, f1, map_score, labels, preds

def main():
    # 确保保存路径存在
    os.makedirs(r'D:\Action_Recognition_Project\data', exist_ok=True)
    os.makedirs(r'D:\Action_Recognition_Project\models', exist_ok=True)
    os.makedirs(r'D:\Action_Recognition_Project\reports', exist_ok=True)
    os.makedirs(r'D:\Action_Recognition_Project\logs', exist_ok=True)
    os.makedirs(r'D:\Action_Recognition_Project\metrics', exist_ok=True)  
    os.makedirs(r'D:\Action_Recognition_Project\plots', exist_ok=True)    

    # 配置日志
    logging.basicConfig(
        filename=r'D:\Action_Recognition_Project\logs\training.log',
        filemode='w',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 定义 CSV 文件路径
    metrics_csv_path = r'D:\Action_Recognition_Project\metrics\metrics.csv'

    # 定义 CSV 表头
    csv_headers = [
        'epoch',
        'train_loss',
        'train_accuracy',
        'train_precision',
        'train_recall',
        'train_f1',
        'val_loss',
        'val_accuracy',
        'val_precision',
        'val_recall',
        'val_f1',
        'val_map',
        'test_loss',
        'test_accuracy',
        'test_precision',
        'test_recall',
        'test_f1',
        'test_map'
    ]

    # 初始化 CSV 文件并写入表头
    with open(metrics_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_headers)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    print(f"使用设备: {device}")

    # 定义数据增强转换（主转换）
    main_transforms = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=15, p=0.5),
        A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # 定义额外的数据增强转换（针对少数类）
    minority_transforms = A.Compose([
        A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), translate_percent=(0.0, 0.1), shear=(-10, 10), p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
    ])

    
    minority_classes = ['retard', 'left', 'right', ' Accelerate']  # 替换为您的少数类名称列表

    # 加载训练集和验证集
    train_dataset = ActionRecognitionDataset(
        json_file=r'D:\Action_Recognition_Project\data\train_processed_data_valid.json',
        transform=main_transforms,
        minority_transform=minority_transforms,
        minority_classes=minority_classes
    )
    val_dataset = ActionRecognitionDataset(
        json_file=r'D:\Action_Recognition_Project\data\val_processed_data.json',
        transform=A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    )

    # 如果有测试集，定义测试集 Dataset 和 DataLoader
    test_dataset = ActionRecognitionDataset(
        json_file=r'D:\Action_Recognition_Project\data\test_processed_data.json',
        transform=A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    # 计算类别权重
    # 使用验证集标签来计算类别权重
    val_labels_np = extract_labels(val_dataset.data, val_dataset.label_mapping)
    class_weights = calculate_class_weights(val_labels_np)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    logging.info(f"类别权重: {class_weights}")
    print(f"类别权重: {class_weights}")

    # 计算每个样本的权重（用于 WeightedRandomSampler）
    # 使用训练集的标签来计算样本权重
    train_labels_np = extract_labels(train_dataset.data, train_dataset.label_mapping)
    sample_weights = train_labels_np.dot(class_weights)  # 每个样本的权重与其正标签数相关
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)

    # 创建采样器
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    logging.info(f"训练集大小: {len(train_dataset)}")
    logging.info(f"验证集大小: {len(val_dataset)}")
    logging.info(f"测试集大小: {len(test_dataset)}")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 初始化模型
    cnn_output_size = 2048  # ResNet50 的输出特征数量
    keypoints_size = 75      # BODY_25 模型，25 个关键点，每个关键点 3 个值 (x, y, confidence)
    lstm_hidden_size = 256
    num_classes = len(train_dataset.label_mapping)
    model = CNN_LSTM_Model(cnn_output_size, keypoints_size, lstm_hidden_size, num_classes)
    model = model.to(device)
    logging.info(f"模型架构: {model}")
    print(f"模型架构: {model}")

    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # 添加权重衰减
    logging.info("已定义损失函数和优化器")
    print("已定义损失函数和优化器")

    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir=r'D:\Action_Recognition_Project\logs\tensorboard')

    # 初始化 mAP 计算器
    map_calculator = MultilabelAveragePrecision(num_labels=num_classes).to(device)

    # 打印类别分布
    print_class_distribution(train_labels_np, train_dataset.label_mapping, "Train")
    print_class_distribution(val_labels_np, val_dataset.label_mapping, "Validation")
    test_labels_np = extract_labels(test_dataset.data, test_dataset.label_mapping)
    print_class_distribution(test_labels_np, test_dataset.label_mapping, "Test")

    # 训练循环
    num_epochs = 200
    best_val_map = 0.0

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        logging.info(f"=== Epoch {epoch + 1}/{num_epochs} ===")

        # 训练阶段
        train_loss, train_accuracy, train_precision, train_recall, train_f1 = train(
            model, train_loader, criterion, optimizer, device
        )
        print(f"训练 - Loss: {train_loss:.4f} - Accuracy: {train_accuracy:.4f} - Precision: {train_precision:.4f} - Recall: {train_recall:.4f} - F1 Score: {train_f1:.4f}")
        logging.info(f"训练 - Loss: {train_loss:.4f} - Accuracy: {train_accuracy:.4f} - Precision: {train_precision:.4f} - Recall: {train_recall:.4f} - F1 Score: {train_f1:.4f}")

        # 验证阶段
        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_map, ap_threshold_dict = validate(
            model, val_loader, criterion, device, map_calculator
        )
        print(f"验证 - Loss: {val_loss:.4f} - Accuracy: {val_accuracy:.4f} - Precision: {val_precision:.4f} - Recall: {val_recall:.4f} - F1 Score: {val_f1:.4f} - mAP: {val_map:.4f}")
        logging.info(f"验证 - Loss: {val_loss:.4f} - Accuracy: {val_accuracy:.4f} - Precision: {val_precision:.4f} - Recall: {val_recall:.4f} - F1 Score: {val_f1:.4f} - mAP: {val_map:.4f}")
        for key, value in ap_threshold_dict.items():
            print(f"验证 - {key}: {value:.4f}")
            logging.info(f"验证 - {key}: {value:.4f}")

        # 记录训练和验证指标到 CSV 文件
        with open(metrics_csv_path, mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)  # 修改变量名为 csv_writer
            csv_writer.writerow([
                epoch + 1,
                train_loss,
                train_accuracy,
                train_precision,
                train_recall,
                train_f1,
                val_loss,
                val_accuracy,
                val_precision,
                val_recall,
                val_f1,
                val_map,
                '', '', '', '', '',  # 测试集指标留空，稍后填充
            ])

        # 记录指标到 TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        writer.add_scalar('Precision/Train', train_precision, epoch)
        writer.add_scalar('Precision/Validation', val_precision, epoch)
        writer.add_scalar('Recall/Train', train_recall, epoch)
        writer.add_scalar('Recall/Validation', val_recall, epoch)
        writer.add_scalar('F1 Score/Train', train_f1, epoch)
        writer.add_scalar('F1 Score/Validation', val_f1, epoch)
        writer.add_scalar('mAP/Validation', val_map, epoch)
        for key, value in ap_threshold_dict.items():
            writer.add_scalar(key, value, epoch)

        # 保存每个 epoch 的模型
        model_save_path = os.path.join(r'D:\Action_Recognition_Project\models', f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), model_save_path)
        logging.info(f"保存模型到 {model_save_path}")
        print(f"保存模型到 {model_save_path}")

        # 更新最佳 mAP
        if val_map > best_val_map:
            best_val_map = val_map
            best_model_path = os.path.join(r'D:\Action_Recognition_Project\models', 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"保存最优模型到 {best_model_path}")
            print(f"保存最优模型到 {best_model_path}")

    # 训练完成后评估测试集并记录指标
    print("\n=== 开始评估测试集 ===")
    logging.info("=== 开始评估测试集 ===")
    test_loss, test_accuracy, test_precision, test_recall, test_f1, test_map, test_labels_np, test_preds = evaluate_test_set(
        model, test_loader, criterion, device
    )
    print(f"测试集 - Loss: {test_loss:.4f} - Accuracy: {test_accuracy:.4f} - Precision: {test_precision:.4f} - Recall: {test_recall:.4f} - F1 Score: {test_f1:.4f} - mAP: {test_map:.4f}")
    logging.info(f"测试集 - Loss: {test_loss:.4f} - Accuracy: {test_accuracy:.4f} - Precision: {test_precision:.4f} - Recall: {test_recall:.4f} - F1 Score: {test_f1:.4f} - mAP: {test_map:.4f}")

    # 将测试集指标写入 CSV 文件的最后一行
    with open(metrics_csv_path, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)  # 修改变量名为 csv_writer
        csv_writer.writerow([
            '', '', '', '', '', '',
            '', '', '', '', '',
            '',
            test_loss,
            test_accuracy,
            test_precision,
            test_recall,
            test_f1,
            test_map
        ])

    # 保存最终模型
    final_model_path = os.path.join(r'D:\Action_Recognition_Project\models', 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"保存最终模型到 {final_model_path}")
    print(f"保存最终模型到 {final_model_path}")

    # 保存测试集的真实标签和预测结果
    labels_save_path = r'D:\Action_Recognition_Project\data\test_processed_data_labels.npy'
    preds_save_path = r'D:\Action_Recognition_Project\models\final_model_preds.npy'
    np.save(labels_save_path, test_labels_np)
    np.save(preds_save_path, test_preds)
    logging.info(f"保存测试标签到 {labels_save_path} 和预测结果到 {preds_save_path}")
    print(f"保存测试标签到 {labels_save_path} 和预测结果到 {preds_save_path}")

    # 关闭 TensorBoard
    writer.close()
    print("训练完成，TensorBoard 已关闭。")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(f"训练过程中发生错误: {e}")
        print(f"训练过程中发生错误: {e}")
