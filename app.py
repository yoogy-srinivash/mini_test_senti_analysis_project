import gradio as gr
from inference import predict_sentiment

interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Enter a sentence..."
    ),
    outputs="json",
    title="Sentiment Analysis using DistilBERT",
    description="Transformer-based sentiment analysis with confidence thresholding"
)

if __name__ == "__main__":
    interface.launch()