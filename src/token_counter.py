from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
text = open("test2.txt").read()
n_tokens = len(tokenizer.encode(text))
print(n_tokens)
