from sentence_transformers import SentenceTransformer


from sentence_transformers import SentenceTransformer as SBert

# 1. Load a pretrained Sentence Transformer model
model = SBert('your model path')
#paraphrase-multilingual-MiniLM-L12-v2解压的文件路径
# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
