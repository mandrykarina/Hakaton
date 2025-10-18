"""
Многошаговый скрипт для участия в хакатоне: мультимодальная (текст + изображение) классификация
Файл: multimodal_pipeline.py (адаптирован под CSV)

Инструкция: помести train.csv, test.csv и папку images/ в одну папку с этим скриптом.
Запуск: python multimodal_pipeline.py --data_dir ./data --out submission.csv --epochs 6

Требования: PyTorch, transformers, timm, pandas, scikit-learn, pillow, tqdm
Установка: pip install -r requirements.txt
"""

import argparse
import os
import random
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

class CFG:
    seed = 42
    img_size = 224
    bs = 32
    epochs = 6
    lr = 2e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone_img = 'resnet50'
    backbone_text = 'distilbert-base-uncased'
    text_max_len = 64
    num_workers = 4
    pretrained_text = True
    pretrained_img = True

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(CFG.seed)

class MultimodalDataset(Dataset):
    def __init__(self, df, images_dir, tokenizer, img_size=224, is_train=True):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.tokenizer = tokenizer
        self.img_size = img_size
        self.is_train = is_train

        if is_train:
            self.img_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.2,0.2,0.2,0.1)], p=0.3),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_ = row['id']
        img_path = self.images_dir / f"{id_}.webp"
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (self.img_size, self.img_size), (255,255,255))
        img = self.img_transform(img)

        text = str(row.get('description', '') or row.get('title', '') or '')
        text = text.strip()
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=CFG.text_max_len, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        out = {'id': id_, 'image': img, 'input_ids': input_ids, 'attention_mask': attention_mask}
        if 'class_label' in row.index and not pd.isna(row['class_label']):
            out['label'] = int(row['class_label'])
        return out

class MultimodalNet(nn.Module):
    def __init__(self, n_classes, img_backbone='resnet50', text_backbone='distilbert-base-uncased', pretrained_img=True, pretrained_text=True, fused_dim=512):
        super().__init__()
        self.img_model = timm.create_model(img_backbone, pretrained=pretrained_img, num_classes=0, global_pool='avg')
        img_feat_dim = self.img_model.num_features
        self.text_model = AutoModel.from_pretrained(text_backbone) if pretrained_text else AutoModel.from_config(AutoModel.from_pretrained(text_backbone).config)
        text_feat_dim = self.text_model.config.hidden_size
        self.img_proj = nn.Linear(img_feat_dim, fused_dim)
        self.text_proj = nn.Linear(text_feat_dim, fused_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim*2),
            nn.Dropout(0.2),
            nn.Linear(fused_dim*2, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fused_dim, n_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.img_model.forward_features(image) if hasattr(self.img_model, 'forward_features') else self.img_model(image)
        img_out = self.img_proj(img_feat)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = text_outputs.last_hidden_state[:,0,:]
        txt_out = self.text_proj(txt_feat)
        fused = torch.cat([img_out, txt_out], dim=1)
        out = self.head(fused)
        return out

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch in tqdm(loader, desc='train', leave=False):
        imgs = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(imgs, input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def valid_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='valid', leave=False):
            imgs = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(imgs, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

def predict(model, loader, device):
    model.eval()
    preds_all, ids_all = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='predict', leave=False):
            imgs = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(imgs, input_ids, attention_mask)
            preds = outputs.argmax(dim=1).cpu().numpy().tolist()
            preds_all.extend(preds)
            ids_all.extend(batch['id'])
    return ids_all, preds_all

def main(args):
    data_dir = Path(args.data_dir)
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    images_dir = data_dir / 'images'

    assert train_path.exists(), f"Не найден {train_path}"
    assert test_path.exists(), f"Не найден {test_path}"

    print('Загружаю данные...')
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    text_fields = [c for c in train_df.columns if 'desc' in c.lower() or 'title' in c.lower() or 'text' in c.lower()]
    if len(text_fields) > 0:
        print('Найдено текстовых полей в train:', text_fields)
        if 'description' not in train_df.columns:
            train_df['description'] = train_df[text_fields[0]].astype(str)
        if 'description' not in test_df.columns:
            test_df['description'] = test_df[text_fields[0]].astype(str)
    else:
        train_df['description'] = ''
        test_df['description'] = ''

    if 'class_label' not in train_df.columns:
        raise ValueError('В train.csv нет столбца class_label (ожидалось)')

    le = LabelEncoder()
    train_df['class_label'] = le.fit_transform(train_df['class_label'].astype(str))
    n_classes = len(le.classes_)
    print('Число классов:', n_classes)

    tokenizer = AutoTokenizer.from_pretrained(CFG.backbone_text)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.seed)
    tr_idx, val_idx = next(skf.split(train_df, train_df['class_label']))
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)

    train_ds = MultimodalDataset(tr_df, images_dir, tokenizer, img_size=CFG.img_size, is_train=True)
    val_ds = MultimodalDataset(val_df, images_dir, tokenizer, img_size=CFG.img_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=CFG.bs, shuffle=True, num_workers=CFG.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.bs, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)

    model = MultimodalNet(n_classes=n_classes, img_backbone=CFG.backbone_img, text_backbone=CFG.backbone_text, pretrained_img=CFG.pretrained_img, pretrained_text=CFG.pretrained_text)
    model.to(CFG.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f'Epoch {epoch}/{args.epochs}')
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, CFG.device)
        val_loss, val_acc = valid_one_epoch(model, val_loader, criterion, CFG.device)
        print(f' train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model_state_dict': model.state_dict(), 'label_encoder_classes': le.classes_}, 'best_model.pth')
            print(' Сохранил лучшую модель')

    print('Инференс на тестовой выборке')
    test_ds = MultimodalDataset(test_df, images_dir, tokenizer, img_size=CFG.img_size, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=CFG.bs, shuffle=False, num_workers=CFG.num_workers)

    ckpt = torch.load('best_model.pth', map_location=CFG.device)
    model.load_state_dict(ckpt['model_state_dict'])
    ids, preds = predict(model, test_loader, CFG.device)

    submission = pd.DataFrame({'id': ids, 'y_pred': preds})
    try:
        submission['y_pred'] = le.inverse_transform(submission['y_pred'].astype(int))
    except Exception:
        pass

    submission.to_csv(args.out, index=False)
    print('Сохранено', args.out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--out', type=str, default='submission.csv')
    parser.add_argument('--epochs', type=int, default=CFG.epochs)
    args = parser.parse_args()
    main(args)
