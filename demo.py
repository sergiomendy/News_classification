import gradio as gr
from gradio.components import Textbox, Text, HTML
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

model_name = 'serge-wilson/news_classification'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)




def get_score_color(score):
    if score > 0.80:
        return "green"
    elif score > 0.50:
        return "orange"
    else:
        return "red"
    

def classify_article(article):
    pipe = pipeline("text-classification", model = model,tokenizer = tokenizer)
    result = pipe(article)[0]
    predicted_label = result.get("label")
    score = result.get("score")
    score_color = get_score_color(score)
    return predicted_label.capitalize(), f"Score : <span style='color: {score_color};'>{score:.2f}</span>"



main = gr.Interface(
    title="News Classification",
    fn=classify_article,
    inputs = Textbox(lines=10, label="Enter an article"),
    outputs = [Text(label="Prediction"), HTML(label="Score")]
)


main.launch()
