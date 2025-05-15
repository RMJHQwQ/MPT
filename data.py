from functools import partial
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from transformers import ViTModel
from torchvision.transforms import transforms

project_path = r"dataset/MVSA_Single/data/"
class TextDataset(Dataset):
    def __init__(self, sentences, labels, method_name, model_name):
        self.sentences = sentences
        self.labels = labels
        self.method_name = method_name
        self.model_name = model_name
        dataset = list()
        index = 0
        for data in sentences:
            tokens = data.split(' ')
            labels_id = labels[index]
            index += 1
            dataset.append((tokens, labels_id))
        self._dataset = dataset

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self.sentences)

class ImageDataset(Dataset):
    def __init__(self, address,label, transform=None):
        self.root_dir = ''
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 调整图像尺寸为 224x224
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
        self.file_list = address
        self.labels = label

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        label = self.labels[index]
        image = Image.open(file_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, image_transform):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        dataset = list()
        idx = 0
        for data in range(len(dataframe)):
            # if self.data.iloc[idx, 1] == 'str':
            #     idx+=1
            #     continue
            label = torch.tensor(self.data.iloc[idx, 0], dtype=torch.long)  # 获取标签
            print(str(self.data.iloc[idx, 2]))
            img_path = project_path + str(self.data.iloc[idx, 1])   # 获取图片路径
            text = self.data.iloc[idx, 2]  # 获取文本内容
            pos = self.data.iloc[idx,3]
            neu = self.data.iloc[idx,4]
            neg = self.data.iloc[idx,5]

            # 读取和转换图像
            # img_path = self.df.iloc[idx]["image_path"]
            try:
                image = Image.open(img_path).convert("RGB")
            except OSError as e:
                print(f"Error loading image {img_path}: {e}")
            # 如果图片损坏，返回一个空白图片（或其他处理方式）
                image = Image.new("RGB", (224, 224), (255, 255, 255))
                # image = Image.open(img_path).convert("RGB")
            image = self.image_transform(image)

            # 处理文本，转换为 BERT 输入格式
            text_inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=320, return_tensors="pt")
            text_inputs["input_ids"] = text_inputs["input_ids"].squeeze(0)
            text_inputs["token_type_ids"] = text_inputs["token_type_ids"].squeeze(0)
            text_inputs["attention_mask"] = text_inputs["attention_mask"].squeeze(0)
            idx += 1
            print(idx)

            pos_inputs = self.tokenizer(pos, padding="max_length", truncation=True, max_length=320,
                                         return_tensors="pt")
            pos_inputs["input_ids"] = pos_inputs["input_ids"].squeeze(0)
            pos_inputs["token_type_ids"] = pos_inputs["token_type_ids"].squeeze(0)
            pos_inputs["attention_mask"] = pos_inputs["attention_mask"].squeeze(0)

            neu_inputs = self.tokenizer(neu, padding="max_length", truncation=True, max_length=320,
                                         return_tensors="pt")
            neu_inputs["input_ids"] = neu_inputs["input_ids"].squeeze(0)
            neu_inputs["token_type_ids"] = neu_inputs["token_type_ids"].squeeze(0)
            neu_inputs["attention_mask"] = neu_inputs["attention_mask"].squeeze(0)

            neg_inputs = self.tokenizer(neg, padding="max_length", truncation=True, max_length=320,
                                         return_tensors="pt")
            neg_inputs["input_ids"] = neg_inputs["input_ids"].squeeze(0)
            neg_inputs["token_type_ids"] = neg_inputs["token_type_ids"].squeeze(0)
            neg_inputs["attention_mask"] = neg_inputs["attention_mask"].squeeze(0)
            dataset.append((image, text_inputs, pos_inputs, neu_inputs, neg_inputs, label))
        self._data = dataset


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self._data[idx]

def my_collate(batch, tokenizer):
    tokens, label_ids = map(list, zip(*batch))

    text_ids = tokenizer(tokens,
                         padding=True,
                         truncation=True,
                         max_length=320,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')
    return text_ids, torch.tensor(label_ids)

def load_both_dataset(tokenizer, train_batch_size, test_batch_size, model_name, method_name, workers):
    train_csv_file = r"mvsa_s_labels_lmm1.tsv"  # 修改为你的文件路径
    # test_csv_file = r"dataset/twitter2017/new_test.tsv"
    train_df = pd.read_csv(train_csv_file,sep="\t",encoding='gbk',encoding_errors='ignore')
    # test_df = pd.read_csv(test_csv_file,sep="\t",encoding="utf-8",encoding_errors='ignore')
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 统一图像大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = CustomDataset(train_df, tokenizer, image_transform)
    # test_dataset = CustomDataset(test_df, tokenizer, image_transform)
    train_size = int(len(train_df))  # 80% 训练
    # test_size = int(len(test_df))
    # train_dataset = random_split(dataset, [train_size, test_size])
    # test_dataset =
    tr_l, te_l = train_test_split(train_dataset, train_size=0.8, random_state=42)
    train_loader = DataLoader(tr_l, batch_size=train_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(te_l, batch_size=test_batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader

def load_text_dataset(tokenizer, train_batch_size, test_batch_size, model_name, method_name, workers):
    data = pd.read_csv('label.csv', sep=None, header=0, encoding='gbk', engine='python')
    len1 = int(len(list(data['labels'])) )
    labels = list(data['labels'])[0:len1]
    sentences = list(data['sentences'])[0:len1]
    # split train_set and test_set
    tr_sen, te_sen, tr_lab, te_lab = train_test_split(sentences, labels, train_size=0.8)
    # Dataset
    train_set = TextDataset(tr_sen, tr_lab, method_name, model_name)
    test_set = TextDataset(te_sen, te_lab, method_name, model_name)
    # DataLoader
    collate_fn = partial(my_collate, tokenizer=tokenizer)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=workers,
                              collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True, num_workers=workers,
                             collate_fn=collate_fn, pin_memory=True)
    return train_loader, test_loader

def load_dataset(train_batch_size, test_batch_size,workers=0):
    data = pd.read_csv('cv_address.csv',encoding='gbk', engine='python')
    len1 = int(len(list(data['labels'])))
    labels = list(data['labels'])[0:len1]
    sentences = list(data['sentences'])[0:len1]
    # split train_set and test_set
    tr_sen, te_sen, tr_lab, te_lab = train_test_split(sentences, labels, train_size=0.8)
    # Dataset
    train_set = ImageDataset(tr_sen, tr_lab)
    test_set = ImageDataset(te_sen, te_lab)
    # DataLoader
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=workers,
                               pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True, num_workers=workers,
                              pin_memory=True)
    return train_loader, test_loader

