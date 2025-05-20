import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import os


# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# 读取 entity2id 和 relation2id
def load_mappings(data_dir):
    entity2id = {}
    relation2id = {}
    with open(os.path.join(data_dir, "entity2id.txt")) as f:
        for line in f:
            if '\t' in line:
                e, idx = line.strip().split('\t')
                entity2id[e] = int(idx)
    with open(os.path.join(data_dir, "relation2id.txt")) as f:
        for line in f:
            if '\t' in line:
                r, idx = line.strip().split('\t')
                relation2id[r] = int(idx)
    return entity2id, relation2id


# 读取三元组文件
def load_triples(path, entity2id, relation2id):
    triples = []
    with open(path, 'r') as f:
        for line in f:
            h, t, r = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


class TransHModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=100, margin=1.0, norm=1):
        super(TransHModel, self).__init__()
        self.margin = margin
        self.norm = norm
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.norm_vector = nn.Embedding(num_relations, embedding_dim)  # 投影超平面法向量

        self.loss_fn = nn.MarginRankingLoss(margin=margin, reduction='mean')
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        nn.init.xavier_uniform_(self.norm_vector.weight.data)

    def _project(self, entity, norm):
        norm = F.normalize(norm, p=2, dim=-1)
        return entity - torch.sum(entity * norm, dim=-1, keepdim=True) * norm

    def _triple_score(self, triples):
        head = self.entity_embeddings(triples[:, 0])
        rel = self.relation_embeddings(triples[:, 1])
        tail = self.entity_embeddings(triples[:, 2])
        norm = self.norm_vector(triples[:, 1])

        head_proj = self._project(head, norm)
        tail_proj = self._project(tail, norm)
        return torch.norm(head_proj + rel - tail_proj, p=self.norm, dim=1)

    def forward(self, pos_triples, neg_triples):
        pos_scores = self._triple_score(pos_triples)
        neg_scores = self._triple_score(neg_triples)
        target = torch.tensor([-1], dtype=torch.long, device=pos_scores.device)
        return self.loss_fn(pos_scores, neg_scores, target)

    def get_entity_relation_embeddings(self):
        return self.entity_embeddings.weight.data, self.relation_embeddings.weight.data


def negative_sampling(triples, num_entities):
    neg_triples = []
    for h, r, t in triples:
        if random.random() < 0.5:
            h_ = random.randint(0, num_entities - 1)
            neg_triples.append((h_, r, t))
        else:
            t_ = random.randint(0, num_entities - 1)
            neg_triples.append((h, r, t_))
    return neg_triples


# 训练
def train(model, train_triples, entity_count, epochs=100, batch_size=128, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        random.shuffle(train_triples)
        losses = []
        for i in range(0, len(train_triples), batch_size):
            batch = train_triples[i:i+batch_size]
            neg_batch = negative_sampling(batch, entity_count)
            pos_tensor = torch.LongTensor(batch)
            neg_tensor = torch.LongTensor(neg_batch)

            loss = model(pos_tensor, neg_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {np.mean(losses):.4f}")


# 主程序
if __name__ == "__main__":
    set_seed()
    data_dir = "data/WN18"
    model_dir = "model_dict/wn18"

    entity2id, relation2id = load_mappings(data_dir)
    train_triples = load_triples(os.path.join(data_dir, "train.txt"), entity2id, relation2id)

    model = TransHModel(len(entity2id), len(relation2id), embedding_dim=100)
    train(model, train_triples, len(entity2id), epochs=100, batch_size=512)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), "model_dict/wn18/transh_model.pt")


