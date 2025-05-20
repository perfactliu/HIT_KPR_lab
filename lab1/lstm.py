import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
import jieba
from torch.utils.tensorboard import SummaryWriter


# ===================== #
# 1. 数据预处理
# ===================== #
class TextDataset:
    def __init__(self, file_path, seq_length=31, vocab_size=30000):
        with open(file_path, "r", errors='ignore', encoding="gb2312") as f:
            text = f.read().strip()

        # 只保留汉字、英文字母、数字和标点符号
        text = re.sub(r"[^\w\s,.!?;:，。！？；：]", "", text, flags=re.UNICODE)
        # print(text)

        # 使用 jieba 进行分词
        self.words = list(jieba.cut(text))
        # print(self.words)

        # 统计词频，构建词表（只保留前 30,000 个高频词）
        word_counts = {}
        for word in self.words:
            word_counts[word] = word_counts.get(word, 0) + 1
        sorted_vocab = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:vocab_size]
        self.vocab = [w for w, _ in sorted_vocab]

        # 构建词到索引的映射
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        # 将文本转换为索引序列（OOV 词用索引 0）
        self.data = [self.word2idx.get(word, 0) for word in self.words]
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        x = torch.tensor(self.data[index : index + self.seq_length], dtype=torch.long)
        y = torch.tensor(self.data[index + 1 : index + self.seq_length + 1], dtype=torch.long)
        return x, y


# ===================== #
# 2. LSTM 语言模型
# ===================== #
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size=30000, embedding_dim=64, hidden_dim=256):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=3, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        hidden = tuple(h.detach() for h in hidden)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size, device):
        return (torch.zeros(3, batch_size, 256).to(device),
                torch.zeros(3, batch_size, 256).to(device))


def generate_text(start_text, length=50):
    file_path = "./data/text.txt"
    vocab_size = 30000
    embedding_dim = 64
    seq_length = 31
    hidden_dim = 256
    dataset = TextDataset(file_path, seq_length, vocab_size)
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMLanguageModel(vocab_size, embedding_dim, hidden_dim).to(device)
    model.load_state_dict(torch.load("lstm_language_model.pth"))
    model.eval()

    words = start_text.lower().split()
    hidden = model.init_hidden(1, device)

    for _ in range(length):
        x = torch.tensor([[dataset.word2idx[w] for w in words[-30:]]], dtype=torch.long).to(device)
        output, hidden = model(x, hidden)
        next_word_idx = torch.argmax(output[0, -1]).item()
        words.append(dataset.idx2word[next_word_idx])

    return " ".join(words)


# ===================== #
# 3. 训练模型
# ===================== #
def train():
    file_path = "./data/text.txt"
    step = 1
    batch_size = 64
    seq_length = 31
    vocab_size = 30000
    embedding_dim = 64
    hidden_dim = 256
    num_epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    dataset = TextDataset(file_path, seq_length, vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = LSTMLanguageModel(vocab_size, embedding_dim, hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        hidden = model.init_hidden(batch_size, device)

        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output, hidden = model(x, hidden)
            # print(output.shape)
            # print(y.shape)

            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss', loss, step)
            step += 1

            if i % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "lstm_language_model.pth")
    print("模型已保存！")


if __name__ == "__main__":
    train()
    # generate_text("令狐冲")
