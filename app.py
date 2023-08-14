import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
model.load_state_dict(torch.load('//models/propmt_t5_model_state_dict.pth', map_location=torch.device('cpu')))
model.to(device)
model.eval()

# Set up the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Input(id='input-text', type='text', placeholder='Enter English text'),
    html.Button('Translate', id='translate-button', n_clicks=0),
    html.Div(id='output-gloss')
])

# Callback function for translation
@app.callback(
    Output('output-gloss', 'children'),
    Input('translate-button', 'n_clicks'),
    Input('input-text', 'value')
)
def translate(n_clicks, input_text):
    if n_clicks > 0 and input_text:
        prompt = "translate English to ASL gloss: "
        # Tokenize the input text
        input_data = tokenizer(prompt + input_text.upper(), return_tensors="pt").to(device)

        # Generate the translated output
        with torch.no_grad():
            output = model.generate(**input_data, max_length=512)

        # Decode the output and return the translation
        translation = tokenizer.decode(output[0], skip_special_tokens=True)
        return translation

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)