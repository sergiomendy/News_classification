from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


app = FastAPI()

class Input(BaseModel):
    text: str


model_name = 'serge-wilson/news_classification'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("text-classification", model = model,tokenizer = tokenizer)

def classify_article(article):
    result = classifier(article, truncation=True)[0]
    return result

@app.post("/classify")
def classify(article: Input):
    result = classify_article(article.text)
    return result
