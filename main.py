import torch
from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
model.load_state_dict(torch.load('/Users/ay227/Desktop/SignKey/models/propmt_t5_model_state_dict.pth', map_location=torch.device('cpu')))
model.to(device)
model.eval()
app = Flask(__name__)


@app.route('/literal_translate', methods=['POST'])
def literal_translate():
    # Retrieve the text from the request
    input = request.json['text']
    glosses = dict()
    text = dict()
    glosses['HELLO'] = '[1, 2, 3]'
    glosses['NAME'] = '[1, 2, 4]'
    glosses['A'] = '[5]'
    glosses['H'] = '[6]'
    glosses['M'] = '[7]'
    glosses['E'] = '[8]'
    glosses['D'] = '[9]'
    text['Hello'] = 'HELLO'
    text['name'] = 'NAME'
    text['A'] = 'A'
    text['h'] = 'H'
    text['m'] = 'M'
    text['e'] = 'E'
    text['d'] = 'D'
    gloss_output = []
    pose_output = []
    parsed = []
    for txt in input.split(" "):
        try:
            gloss = text[txt]
            gloss_output.append(gloss)
            pose_output.append(glosses[gloss])
            parsed.append(txt)
        except:
            if txt not in parsed:
                for character in txt:
                    try:
                        gloss = text[character]
                        gloss_output.append(gloss)
                        pose_output.append(glosses[gloss])
                    except:
                        continue
            continue

    # Use your machine learning model to translate the text to pose or gloss
    gloss = " ".join(gloss_output)
    pose = " ".join(pose_output)
    # Return the output as a JSON response
    return jsonify({'Hello it is me': input,
                    'gloss_output': gloss,
                    'pose_output': pose
                    }
                   )

@app.route('/translate', methods=['POST'])
def translate():
    # Retrieve the text from the request
    input_text = request.json['text']
    prompt = "translate English to ASL gloss: "
    # Tokenize the input text
    input_data = tokenizer(prompt + input_text.upper(), return_tensors="pt").to(device)

    # Generate the translated output
    with torch.no_grad():
        output = model.generate(**input_data, max_length=512)

    # Decode the output and return the translation
    translation = tokenizer.decode(output[0], skip_special_tokens=True)
    # Return the output as a JSON response
    return jsonify({'Gloss Output': translation})


if __name__ == '__main__':
    app.run()
