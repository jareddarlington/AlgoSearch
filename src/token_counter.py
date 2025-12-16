from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct")
text = open("test.txt").read()
n_tokens = len(tokenizer.encode(text))
print(n_tokens)
