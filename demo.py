import gradio as gr
from gradio.components import Textbox, Text, HTML
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


model_name = 'serge-wilson/news_classification'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

#Creation d'une pipeline 
classifier = pipeline("text-classification", model = model,tokenizer = tokenizer)


def get_score_color(score):
    """
        Cette fonction est pour déterminer la couleur de l'output au niveau de l'interface gradio
    """
    if score > 0.80:
        return "green"
    elif score > 0.50:
        return "orange"
    else:
        return "red"
    

def classify_article(article):
    """
        Cette fonction est utiliser pour classifier l'article pris en argument et retourne le label prédit 
        et le score écrit sous format HTML.
    """
    result = classifier(article, truncation=True)[0]

    predicted_label = result.get("label")
    score = result.get("score")
    score_color = get_score_color(score)
    return predicted_label.capitalize(), f"Score : <span style='color: {score_color};'>{score:.2f}</span>"



demo = gr.Interface(
    title="News Classification",
    fn=classify_article,
    inputs = Textbox(lines=10, label="Enter an article"),
    outputs = [Text(label="Prediction"), HTML(label="Score")]
)


demo.launch()
