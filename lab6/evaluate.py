import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from tqdm import tqdm
from TransE import TransEModel, load_triples, load_mappings


# 从训练/验证/测试集构建已知三元组集合（用于filtered）
def build_known_triples(*triple_lists):
    known = set()
    for triples in triple_lists:
        for h, r, t in triples:
            known.add((h, r, t))
    return known


def load_n_to_n_triples(data_dir):
    path = os.path.join(data_dir, "n-n.txt")
    triples = []
    with open(path, 'r') as f:
        lines = f.readlines()[1:]  # 跳过第一行
        for line in lines:
            h, t, r = map(int, line.strip().split())
            triples.append((h, r, t))
    return triples


# 评估指标
def evaluate(model, test_triples, n_to_n_triples, entity2id, relation2id, dataset, all_triples=None, filtered=True):
    model.eval()
    entity_num = len(entity2id)
    hits_at_10 = []
    ranks = []
    hits_at_10_n_to_n_head = []
    ranks_n_to_n_head = []
    hits_at_10_n_to_n_tail = []
    ranks_n_to_n_tail = []

    known_triples = build_known_triples(all_triples) if filtered else set()

    with torch.no_grad():
        for h_id, r_id, t_id in tqdm(test_triples, desc='test episode:'):
            h_tensor = model.entity_embeddings.weight.data
            r_tensor = model.relation_embeddings.weight.data[r_id]
            t_tensor = model.entity_embeddings.weight.data

            # 替换头实体
            scores_head = torch.norm(h_tensor + r_tensor - model.entity_embeddings.weight.data[t_id], p=1, dim=1)
            if filtered:
                for e_id in range(entity_num):
                    if (e_id, r_id, t_id) in known_triples and e_id != h_id:
                        scores_head[e_id] = float('inf')
            rank_head = torch.argsort(scores_head).tolist().index(h_id) + 1
            ranks.append(rank_head)
            hits_at_10.append(1 if rank_head <= 10 else 0)

            # 替换尾实体
            scores_tail = torch.norm(model.entity_embeddings.weight.data[h_id] + r_tensor - t_tensor, p=1, dim=1)
            if filtered:
                for e_id in range(entity_num):
                    if (h_id, r_id, e_id) in known_triples and e_id != t_id:
                        scores_tail[e_id] = float('inf')
            rank_tail = torch.argsort(scores_tail).tolist().index(t_id) + 1
            ranks.append(rank_tail)
            hits_at_10.append(1 if rank_tail <= 10 else 0)

        if dataset == "fb15k":
            for h_id, r_id, t_id in tqdm(n_to_n_triples, desc='n to n test episode:'):
                h_tensor = model.entity_embeddings.weight.data
                r_tensor = model.relation_embeddings.weight.data[r_id]
                t_tensor = model.entity_embeddings.weight.data

                # 替换头实体
                scores_head = torch.norm(h_tensor + r_tensor - model.entity_embeddings.weight.data[t_id], p=1, dim=1)
                rank_head = torch.argsort(scores_head).tolist().index(h_id) + 1
                ranks_n_to_n_head.append(rank_head)
                hits_at_10_n_to_n_head.append(1 if rank_head <= 10 else 0)

                # 替换尾实体
                scores_tail = torch.norm(model.entity_embeddings.weight.data[h_id] + r_tensor - t_tensor, p=1, dim=1)
                rank_tail = torch.argsort(scores_tail).tolist().index(t_id) + 1
                ranks_n_to_n_tail.append(rank_tail)
                hits_at_10_n_to_n_tail.append(1 if rank_tail <= 10 else 0)

    mean_rank = np.mean(ranks)
    hits10 = np.mean(hits_at_10)
    print(f"Mean Rank: {mean_rank:.2f}")
    print(f"Hits@10: {hits10:.4f}")
    if dataset == "fb15k":
        mean_rank_n_to_n_head = np.mean(ranks_n_to_n_head)
        hits10_n_to_n_head = np.mean(hits_at_10_n_to_n_head)
        print(f"N to N Head Mean Rank: {mean_rank_n_to_n_head:.2f}")
        print(f"N to N Head Hits@10: {hits10_n_to_n_head:.4f}")
        mean_rank_n_to_n_tail = np.mean(ranks_n_to_n_tail)
        hits10_n_to_n_tail = np.mean(hits_at_10_n_to_n_tail)
        print(f"N to N Head Mean Rank: {mean_rank_n_to_n_tail:.2f}")
        print(f"N to N Head Hits@10: {hits10_n_to_n_tail:.4f}")
    return mean_rank, hits10


# 主程序
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="data/WN18", help="Dataset directory")
    parser.add_argument('--dataset', type=str, default="wn18", help="Dataset name")
    parser.add_argument('--file', type=str, default="test.txt", help="test.txt or valid.txt")
    parser.add_argument('--model_path', type=str, default="transe_model.pt")
    parser.add_argument('--filtered', action='store_true')
    args = parser.parse_args()

    entity2id, relation2id = load_mappings(args.data_dir)
    n_to_n_triples = load_n_to_n_triples(args.data_dir)
    test_triples = load_triples(os.path.join(args.data_dir, args.file), entity2id, relation2id)
    train_triples = load_triples(os.path.join(args.data_dir, "train.txt"), entity2id, relation2id)
    valid_triples = load_triples(os.path.join(args.data_dir, "valid.txt"), entity2id, relation2id)
    all_triples = train_triples + valid_triples + test_triples

    model = TransEModel(len(entity2id), len(relation2id), embedding_dim=100)
    model.load_state_dict(torch.load(os.path.join("model_dict", args.dataset, args.model_path)))

    evaluate(model, test_triples, n_to_n_triples, entity2id, relation2id, args.dataset,
             all_triples=all_triples, filtered=args.filtered)
