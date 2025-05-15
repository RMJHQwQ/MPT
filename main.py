import pickle

import torch
import torch.nn as nn
from config import get_config
from models.MPT import MPT
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.img_model import ImgModel
from transformers import logging, AutoTokenizer, AutoModel
from matplotlib import pyplot as plt
from tqdm import tqdm
from data import load_both_dataset
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
import seaborn as sns
import os
import random



class EngineOn:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        # self.logger.info('> creating model {}'.format(args.model_name))
        if args.model_name == "chi_bert":
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
            self.text_vectorize = AutoModel.from_pretrained('bert-base-chinese')
        elif args.model_name == "bert":
            bert_path = "./bert/bert-base"
            self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
            self.text_vectorize = AutoModel.from_pretrained(bert_path)

        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()
        self.img_encoder = ImgModel(args.num_classes)
        self.Mymodel = MPT(args.train_batch_size, basemodel=self.text_vectorize, visual_model=self.img_encoder)
        self.w = torch.tensor([9.0, 2.7, 1.0])

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _tsne_plot(self, features, targets, filename):
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced = tsne.fit_transform(features)
        targets = torch.tensor(targets)

        plt.figure(figsize=(8, 6))
        palette = sns.color_palette("hsv", len(set(targets.numpy())))
        sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=targets.numpy(), palette=palette, s=15, legend='full')
        plt.title("t-SNE of Feature Embeddings")
        plt.savefig(filename)
        plt.close()

    def _train(self, dataloader, criterion, optimizer, temperature=0.07, contrastive_weight=0.5):
        train_loss, n_correct, n_train = 0, 0, 0
        all_preds, all_targets = [], []

        # Turn on the train mode
        self.Mymodel.train()
        self.Mymodel.to(self.args.device)
        scheduler = CosineAnnealingLR(optimizer, T_max=32)
        for image, text, pos, neu, neg, targets in tqdm(dataloader, disable=self.args.backend, ascii='>='):
            image = image.to(self.args.device)
            text = {k: v.to(self.args.device) for k, v in text.items()}
            targets = targets.to(self.args.device)
            pos = {k: v.to(self.args.device) for k, v in pos.items()}
            neu = {k: v.to(self.args.device) for k, v in neu.items()}
            neg = {k: v.to(self.args.device) for k, v in neg.items()}

            predicts, feature_embeddings = self.Mymodel(text, image, pos, neu, neg)
            loss = criterion(predicts, targets.long())
            feature_embeddings = F.normalize(feature_embeddings, dim=1)  # 归一化
            similarity_matrix = torch.matmul(feature_embeddings, feature_embeddings.T) / temperature  # 计算余弦相似度

            # 获取 batch 内的正样本索引
            mask = (targets.unsqueeze(1) == targets.unsqueeze(0)).float()  # shape: (batch_size, batch_size)

            # 计算 SupCon Loss
            exp_sim = torch.exp(similarity_matrix)  # softmax 分子
            exp_sim_sum = exp_sim.sum(dim=1, keepdim=True)  # softmax 分母

            # 仅对 **正样本** 计算 log-softmax
            log_probs = torch.log(exp_sim / exp_sim_sum)  # 计算 log p
            supcon_loss = -(log_probs * mask.float()).sum(dim=1) / mask.sum(dim=1)

            loss = (1 - contrastive_weight) * loss + contrastive_weight * supcon_loss.mean()
            torch.autograd.set_detect_anomaly(True)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
            n_train += targets.size(0)

            pred_labels = torch.argmax(predicts, dim=1)
            all_preds.extend(pred_labels.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        f1 = f1_score(all_targets, all_preds, average='macro')
        w_f1 = f1_score(all_targets, all_preds, average='weighted')
        return train_loss / n_train, n_correct / n_train, f1, w_f1

    def _test(self, dataloader, criterion, temperature=0.07, contrastive_weight=0.5):
        test_loss, n_correct, n_test = 0, 0, 0
        all_features = []
        all_preds, all_targets = [], []
        # Turn on the eval mode
        self.Mymodel.eval()
        with torch.no_grad():
            for image, text, pos, neu, neg, targets in tqdm(dataloader, disable=self.args.backend, ascii='>='):
                image = image.to(self.args.device)
                pos = {k: v.to(self.args.device) for k, v in pos.items()}
                neu = {k: v.to(self.args.device) for k, v in neu.items()}
                neg = {k: v.to(self.args.device) for k, v in neg.items()}
                text = {k: v.to(self.args.device) for k, v in text.items()}
                targets = targets.to(self.args.device)
                predicts, feature_embeddings = self.Mymodel(text, image, pos, neu, neg)
                loss = criterion(predicts, targets.long())
                feature_embeddings = F.normalize(feature_embeddings, dim=1)  # 归一化
                all_features.append(feature_embeddings.cpu())
                similarity_matrix = torch.matmul(feature_embeddings, feature_embeddings.T) / temperature  # 计算余弦相似度

                # 获取 batch 内的正样本索引
                mask = (targets.unsqueeze(1) == targets.unsqueeze(0)).float()  # shape: (batch_size, batch_size)

                # 计算 SupCon Loss
                exp_sim = torch.exp(similarity_matrix)  # softmax 分子
                exp_sim_sum = exp_sim.sum(dim=1, keepdim=True)  # softmax 分母

                # 仅对 **正样本** 计算 log-softmax
                log_probs = torch.log(exp_sim / exp_sim_sum)  # 计算 log p
                supcon_loss = -(log_probs * mask.float()).sum(dim=1) / mask.sum(dim=1)

                loss = (1 - contrastive_weight) * loss + contrastive_weight * supcon_loss.mean()
                test_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
                n_test += targets.size(0)
                pred_labels = torch.argmax(predicts, dim=1)
                all_preds.extend(pred_labels.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        f1 = f1_score(all_targets, all_preds, average='macro')
        w_f1 = f1_score(all_targets, all_preds, average='weighted')

        return test_loss / n_test, n_correct / n_test, f1, w_f1, torch.cat(all_features, dim=0), all_targets

    def run(self):
        # Print the parameters of model
        # for name, layer in self.Mymodel.named_parameters(recurse=True):
        # print(name, layer.shape, sep=" ")

        train_dataloader, test_dataloader = load_both_dataset(tokenizer=self.tokenizer,
                                                              train_batch_size=self.args.train_batch_size,
                                                              test_batch_size=self.args.test_batch_size,
                                                              model_name=self.args.model_name,
                                                              method_name=self.args.method_name,
                                                              workers=self.args.workers)
        _params = filter(lambda x: x.requires_grad, self.Mymodel.parameters())

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        criterion.to("cuda")
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.weight_decay)

        l_acc, l_trloss, l_teloss, l_epo = [], [], [], []
        # Get the best_loss and the best_acc
        best_loss, best_acc = 0, 0
        with open('bert-lstm.pkl', "wb") as file:
            for epoch in range(self.args.num_epoch):
                train_loss, train_acc, f1, wf1 = self._train(train_dataloader, criterion, optimizer)
                test_loss, test_acc, af1, awf1, features, targets = self._test(test_dataloader, criterion)
                self._tsne_plot(features, targets, f"plot/tsne_epoch{epoch}.png")
                l_epo.append(epoch), l_acc.append(test_acc), l_trloss.append(train_loss), l_teloss.append(test_loss)
                if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
                    best_acc, best_loss = test_acc, test_loss

                    # pickle.dump(self.Mymodel, file)
                self.logger.info(
                    '{}/{} - {:.2f}%'.format(epoch + 1, self.args.num_epoch, 100 * (epoch + 1) / self.args.num_epoch))
                self.logger.info('[train] loss: {:.4f}, acc: {:.2f}, f1:{:.2f}, weighted_F1:{:.4f}'.format(train_loss,
                                                                                                           train_acc * 100,
                                                                                                           f1, wf1))
                self.logger.info(
                    '[test] loss: {:.4f}, acc: {:.2f}, f1:{:.2f}, weighted_F1:{:.4f}'.format(test_loss, test_acc * 100,
                                                                                             af1, awf1))

        # with open('textcnn_lstm_attention.pkl', "rb") as file:
        #     model = pickle.load(file)
        #     model.eval()
        #     for inputs, targets in tqdm(test_dataloader, disable=self.args.backend, ascii=' >='):
        #         inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
        #         output = model(inputs,is_val='s')

        self.logger.info('best loss: {:.4f}, best acc: {:.2f}'.format(best_loss, best_acc * 100))
        self.logger.info('log saved: {}'.format(self.args.log_name))

        # Draw the training process
        plt.plot(l_epo, l_acc)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig('acc.png')

        plt.plot(l_epo, l_teloss)
        plt.ylabel('test-loss')
        plt.xlabel('epoch')
        plt.savefig('teloss.png')

        plt.plot(l_epo, l_trloss)
        plt.ylabel('train-loss')
        plt.xlabel('epoch')
        plt.savefig('trloss.png')
    # def run(self):
    #     x = torch.randn(1, 32, 512)
    #     img = torch.randn(1, 3, 224, 224)
    #     print(self.model(x,img))


if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    nb = EngineOn(args, logger)
    nb.run()

