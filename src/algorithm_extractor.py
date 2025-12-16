from outlines.models.transformers import Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = ""

model = Transformers(
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto"),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)

from outlines.generate import choice
from typing import Literal

generator = choice(model, ["Positive", "Negative", "Neutral"])
sentiment = generator("Analyze: This product completely changed my life!")
print(sentiment)
