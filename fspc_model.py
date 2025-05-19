import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer, BertConfig, AutoTokenizer, AutoModel
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score
import torch.nn.functional as F
import copy
import random
import os
import json
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from pypinyin import lazy_pinyin, Style
import zhconv
import ast
import torchvision
from PIL import Image

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

random_seed = 42


def seed_everything(seed=random_seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()

def normalize_audio_feature(file_name):
    audio_features = pd.read_csv(file_name)
    exclude_columns = ['pinyin']
    scaler = MinMaxScaler()
    columns_to_normalize = [col for col in audio_features.columns if col not in exclude_columns]
    df_normalized = audio_features.copy()
    df_normalized[columns_to_normalize] = scaler.fit_transform(df_normalized[columns_to_normalize])
    df_normalized.set_index('pinyin', inplace=True)
    audio_feature_dic = df_normalized.to_dict(orient='index')

    for key in audio_feature_dic.keys():
        audio_feature_dic[key] = np.array(list(audio_feature_dic[key].values()))

    return audio_feature_dic

padding_zero_feature = np.zeros(6373)


# Mandarin Audio
audio_feature_dic = normalize_audio_feature('audio_feature.csv')

audio_feature_without_tone_dic = {}
for key in audio_feature_dic.keys():
    if key[:-1] in audio_feature_without_tone_dic.keys():
        continue
    else:
        audio_feature_without_tone_dic[key[:-1]] = (audio_feature_dic[key[:-1]+'1'] +
                                                    audio_feature_dic[key[:-1]+'2'] +
                                                    audio_feature_dic[key[:-1]+'3'] +
                                                    audio_feature_dic[key[:-1]+'4'])/4

pu_special_cases = {
    ('zi', '子'): 'zi3',
    ('de', None): 'de2',
    ('tou', None): 'tou2',
    ('le', '了'): 'liao3',
    ('shang', '裳'): 'shang1',
    ('bo', '卜'): 'bu3',
    ('xu', '蓿'): 'xu4',
    ('zhe', '着'): 'zhuo2'
    }

def use_audio_feature(text_list, max_length=17):
    audio_feature = []
    attention_mask = []
    for text in text_list:
        one_pinyin_feature = []
        one_attention_mask = []
        tone = lazy_pinyin(text, style=Style.TONE3)
        for t, c in zip(tone, text):
            t = pu_special_cases.get((t, c)) or pu_special_cases.get((t, None)) or t
            try:
                one_pinyin_feature.append(audio_feature_dic[t])
                one_attention_mask.append(1)
            except:
                try:
                    one_pinyin_feature.append(audio_feature_without_tone_dic[t])
                    one_attention_mask.append(1)
                except:
                    one_pinyin_feature.append(padding_zero_feature)
                    one_attention_mask.append(0)
        while len(one_pinyin_feature) < max_length-1:
            one_pinyin_feature.append(padding_zero_feature)
            one_attention_mask.append(0)
        if len(one_pinyin_feature) > max_length-1:
            one_pinyin_feature = one_pinyin_feature[:max_length-1]
            one_attention_mask = one_attention_mask[:max_length-1]
        audio_feature.append(np.array(one_pinyin_feature))
        attention_mask.append(one_attention_mask)
    return np.array(audio_feature), attention_mask

# Cantonese Audio
yue_audio_feature_dic = normalize_audio_feature('yue_audio_feature.csv')

df = pd.read_csv('yuelist-20230104.tsv', sep="\t")
yuechar = np.array(df['CH'].tolist())
yuepinyin = np.array(df['JP'].tolist())
yuepinyin_dic = {}
for c, p in zip(yuechar, yuepinyin):
    if c not in yuepinyin_dic.keys():
        yuepinyin_dic[c] = p
yuepinyin_dic['茄'] = 'ke4'

def use_yue_audio_feature(name_list, max_length=17):
    audio_feature = []
    attention_mask = []
    for name in name_list:
        one_pinyin_feature = []
        one_attention_mask = []
        for c in name:
            t = yuepinyin_dic.get(c)
            try:
                one_pinyin_feature.append(yue_audio_feature_dic[t])
                one_attention_mask.append(1)
            except:
                print(c)
                one_pinyin_feature.append(padding_zero_feature)
                one_attention_mask.append(0)
        while len(one_pinyin_feature) < max_length-1:
            one_pinyin_feature.append(padding_zero_feature)
            one_attention_mask.append(0)
        if len(one_pinyin_feature) > max_length-1:
            one_pinyin_feature = one_pinyin_feature[:max_length-1]
            one_attention_mask = one_attention_mask[:max_length-1]
        audio_feature.append(np.array(one_pinyin_feature))
        attention_mask.append(one_attention_mask)
    return np.array(audio_feature), attention_mask

    
# Pre-generated visual features
with open('fspc_unet_feture_dict.pkl', 'rb') as file:
    fspc_unet_feture_dict = pickle.load(file)


class BertClassfication(nn.Module):
    def __init__(self):
        super(BertClassfication, self).__init__()

        # Classical Chinese Pretrained Model
        self.tokenizer = AutoTokenizer.from_pretrained("/public/gender_prediction/pinyin_gender/pretrain_model/guwenbert")
        self.model = AutoModel.from_pretrained("/public/gender_prediction/pinyin_gender/pretrain_model/guwenbert")

        # Audio Feature Extractor
        scale = 768 ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(768))
        self.positional_embedding = nn.Parameter(scale * torch.randn(17, 768))

        self.audio_config = BertConfig(hidden_size=768, num_hidden_layers=3, num_attention_heads=8)
        self.audio_model = BertEncoder(self.audio_config)
        self.fc = nn.Linear(6373, 768)

        # Classifier
        self.concat = nn.Linear(768*3, 768)
        self.cls_all = nn.Linear(768, 5)

        # Dialect Fusion
        self.attention = nn.MultiheadAttention(768, num_heads=2, batch_first=True)
        self.audio_concat = nn.Linear(768*2, 768)

        self.loss_ce = nn.CrossEntropyLoss()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # Vision Feature Extractor
        self.processor = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 768),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        

    def audio_feature_extractor(self, x, dialect=None):
        if dialect == 'yue':
            input_features, attention_mask = use_yue_audio_feature(x)
        elif dialect == 'pu':
            input_features, attention_mask = use_audio_feature(x)
        input_features = self.fc(torch.tensor(input_features).float().to(device))
        # Add class_embedding
        input_features = torch.cat([self.class_embedding + torch.zeros(input_features.shape[0], 1, 768, device=device), input_features], dim=1)
        attention_mask = torch.cat([torch.ones((input_features.shape[0], 1), device=device), torch.tensor(attention_mask).to(device)], dim=1)
        # Add positional_embedding
        input_features = input_features + self.positional_embedding
        audio_embedding = self.audio_model(input_features)[0]
        return audio_embedding

    def forward(self, x, y=None, epoch=None):

        # Text Feature
        batch_tokenized = self.tokenizer([pair[0] for pair in x], [pair[1] for pair in x], max_length=128, add_special_tokens=True, truncation=True, padding="max_length", return_tensors='pt')
        input_ids = batch_tokenized['input_ids'].to(device)
        attention_mask = batch_tokenized['attention_mask'].to(device)
        bert_embedding = self.model(input_ids, attention_mask=attention_mask)[1]

        # Dialect Feature
        audio_embedding_yue = self.audio_feature_extractor([pair[2] for pair in x], dialect='yue')
        audio_embedding_pu = self.audio_feature_extractor([pair[2] for pair in x], dialect='pu')
        
        # Cross Attention
        attn_output_pu, attn_weights = self.attention(audio_embedding_yue, audio_embedding_pu, audio_embedding_pu)
        attn_output_yue, attn_weights = self.attention(audio_embedding_pu, audio_embedding_yue, audio_embedding_yue)
        audio_embedding = self.audio_concat(torch.cat((attn_output_pu[:,0,:], attn_output_yue[:,0,:]), dim=1))

        # Visual Feature
        input_vision_feature = [fspc_unet_feture_dict[pair[0]] for pair in x]
        vision_embbeding = self.processor(torch.stack(input_vision_feature).squeeze(1).to(device))

        final_feature = self.concat(torch.cat((bert_embedding, audio_embedding, vision_embbeding), dim=1))
        output_all = self.cls_all(final_feature)

        if y is not None:
            # Multimodal Pretrain
            if epoch is not None and epoch <30:
                '''
                Multimodal Contrastive1 Representation Learning
                The code is from https://github.com/AndreyGuzhov/AudioCLIP
                '''
                audio_features = audio_embedding / audio_embedding.norm(dim=-1, keepdim=True)
                text_features = bert_embedding / bert_embedding.norm(dim=-1, keepdim=True)
                vision_features = vision_embbeding / vision_embbeding.norm(dim=-1, keepdim=True)
                logit_scale = self.logit_scale.exp()
                logits_audio_image = logit_scale * audio_features @ vision_features.t()
                logits_audio_text = logit_scale * audio_features @ text_features.t()
                logits_image_text  = logit_scale * vision_features  @ text_features.t()
                labels = torch.arange(len(logits_audio_image)).long().to(device)
                loss_logits_audio_image = self.loss_ce(logits_audio_image, labels)
                loss_logits_audio_text = self.loss_ce(logits_audio_text, labels)
                loss_logits_image_text = self.loss_ce(logits_image_text, labels)
                
                final_loss = loss_logits_audio_image + loss_logits_audio_text + loss_logits_image_text
                return output_all, final_loss
            else:
                final_loss = self.loss_ce(output_all, y)
            return output_all, final_loss
        else:
            return output_all


if __name__ == '__main__':

    def read_fspc_data(filenname):
        df = pd.read_csv(filenname, sep="\t")
        poem = np.array(df['text'].tolist())
        label = np.array(df['label'].tolist())-1
        return np.array(poem), label.tolist()
    
    raw_train_text, raw_train_label = read_fspc_data("/public/CCLUE/data/fspc/train.tsv")
    raw_test_text, raw_test_label = read_fspc_data("/public/CCLUE/data/fspc/test.tsv")
    raw_dev_text, raw_dev_label = read_fspc_data("/public/CCLUE/data/fspc/dev.tsv")

    # LLM Translation
    with open('/public/gender_prediction/use_audio_feature_in_sentiment_classification/poem_trans_dict.pkl', 'rb') as file:
        poem_trans_dict = pickle.load(file)


    train_text = [(x,poem_trans_dict[x],x) for x in raw_train_text]
    train_label = raw_train_label

    test_text = [(x,poem_trans_dict[x],x) for x in raw_test_text]
    dev_text = [(x,poem_trans_dict[x],x) for x in raw_dev_text]
    
    test_label = raw_test_label
    dev_label = raw_dev_label

    def load_data(dataset, label, batch_size):
        batch_count = int(len(dataset) / batch_size)
        batch_inputs, batch_targets = [], []
        for i in range(batch_count):
            batch_inputs.append(dataset[i * batch_size: (i + 1) * batch_size])
            batch_targets.append(label[i * batch_size: (i + 1) * batch_size])
        if batch_count * batch_size != len(dataset):
            batch_inputs.append(dataset[batch_count * batch_size:])
            batch_targets.append(label[batch_count * batch_size:])
            batch_count+=1
        print(batch_count)
        return batch_inputs, batch_targets, batch_count

    batch_train_inputs, batch_train_targets, train_batch_count = load_data(train_text, train_label, 64)
    batch_dev_inputs, batch_dev_targets, dev_batch_count = load_data(dev_text, dev_label, 64)
    batch_test_inputs, batch_test_targets, test_batch_count = load_data(test_text, test_label, 64)

    new_model = BertClassfication().to(device)
    optimizer = torch.optim.Adam(new_model.parameters(), lr=0.000001)
    epoch = 60
    print_every_batch = 10

    dev_acc = 0
    dev_final_loss = 99
    for e in range(epoch):
        new_model.train()
        print_avg_loss = 0
        for i in range(train_batch_count):
            inputs = np.array(batch_train_inputs[i])
            targets = torch.tensor(batch_train_targets[i]).to(device)
            optimizer.zero_grad()
            outputs, loss = new_model(inputs, targets, e)
            loss.backward()
            optimizer.step()
            print_avg_loss += loss.item()
            if i % print_every_batch == 0:
                print("Epoch: %d, Batch: %d, Loss: %.4f" % (e + 1, (i + 1), print_avg_loss / print_every_batch))
                print_avg_loss = 0
        new_model.eval()
        dev_pre = []
        dev_all_loss = 0
        with torch.no_grad():
            for i in range(dev_batch_count):
                if i % print_every_batch == 0:
                    print("dev_batch: %d" % (i + 1))
                dev_inputs = batch_dev_inputs[i]
                dev_targets = torch.tensor(batch_dev_targets[i]).to(device)
                dev_outputs, dev_loss = new_model(dev_inputs, dev_targets)
                dev_all_loss += dev_loss.item()
                dev_outputs = torch.argmax(dev_outputs, dim=1).cpu().numpy().tolist()
                dev_pre.extend(dev_outputs)

        temp_acc = accuracy_score(dev_label, dev_pre)
        temp_loss = dev_all_loss / len(dev_pre)
        print('dev_acc: %.4f, loss: %.4f' % (temp_acc, temp_loss))
        if temp_acc > dev_acc:
            dev_acc = temp_acc
            best_val_model_acc = copy.deepcopy(new_model.module) if hasattr(new_model, "module") else copy.deepcopy(new_model)
        if temp_loss < dev_final_loss:
            dev_final_loss = temp_loss
            best_val_model_loss = copy.deepcopy(new_model.module) if hasattr(new_model, "module") else copy.deepcopy(new_model)
    
    output_model_file = 'model/cclue_fspc_temp_acc_audio_trans_vision_combine.bin'
    torch.save(best_val_model_acc.state_dict(), output_model_file)
    output_model_file = 'model/cclue_fspc_temp_loss_audio_trans_vision_combine.bin'
    torch.save(best_val_model_loss.state_dict(), output_model_file)

    total = len(test_label)
    m_state_dict = torch.load('model/cclue_fspc_temp_acc_audio_trans_vision_combine_6485.bin')
    best_model = BertClassfication().to(device)
    best_model.load_state_dict(m_state_dict, strict=False)
    best_model.eval()
    test_pre = []
    test_pre_soft = []
    with torch.no_grad():
        for i in range(test_batch_count):
            test_inputs = batch_test_inputs[i]
            hidden_test_outputs = best_model(test_inputs)
            test_outputs = torch.argmax(hidden_test_outputs, dim=1).cpu().numpy().tolist()
            test_pre.extend(test_outputs)
            test_pre_soft.extend(hidden_test_outputs.cpu().numpy().tolist())
    print(classification_report(test_label, test_pre, digits=6))