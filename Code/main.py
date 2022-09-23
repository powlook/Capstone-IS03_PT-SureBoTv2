from flask import *
import glob, os, pprint, sys
from werkzeug.utils import secure_filename
from SureBoT_v2 import detect_text, executePipeline, configure_logger
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModel
import torch.nn as nn
# import warnings
from text_classifier import text_model
from VBInference import vb_model
from transformers import logging
logging.set_verbosity_warning()

# warnings.filterwarnings("ignore")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key/image_search.json"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_cpu = 'cpu'

chat = 0
surebot_logger = configure_logger(chat)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "upload")
# print("SureBoT is waking up..... Please give it a few minutes....")
print("* Running on http://localhost:5000/ (Press CTRL+C to quit)")

class Classifier(nn.Module):
    
    def __init__(self, dropout=0.5):

        super(Classifier, self).__init__()

        self.model = AutoModel.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre')
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,1)
        self.relu = nn.ReLU()
        self.softmax =nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids, visual_embeds,visual_attention_mask, visual_token_type_ids):

        # _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        for param in self.model.parameters():
            param.requires_grad = False
        
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids,output_hidden_states=True)
        # print(f"output hidden state tensor: {output.last_hidden_state}")
        # print(f"output hidden state shape:{output.last_hidden_state.shape}")
        # cls_hs = self.vbert(sent_id, attention_mask=mask)[0][:,0]         ### TO MODIFY AGAIN ####
        x = self.fc1(output.last_hidden_state)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.softmax(x)
        
        return x

txt_model, txt_tokenizer = text_model()
visualbert_model = vb_model()

############# Website Configuration #########################################
@app.route('/')
def home_form():

    return render_template('home.html')

@app.route('/image', methods = ['POST','GET'])
def upload():
    
    if request.method == 'POST':
        imagelist = glob.glob(os.path.join(os.getcwd(),"static","upload","*"))
        for image in imagelist:
            os.remove(image)

        f = request.files['File']
        filepath = os.path.join(app.config["UPLOAD_FOLDER"],secure_filename(f.filename))
        f.save(filepath)
        
        ############## Insert the codes to call out the different models ####################################
        ######### Return the results for the different models and replace the variables below ###############
        input_claim = detect_text(filepath)
        if input_claim == []:
            input_claim = '0'
        if (len(input_claim[0].split()) < 5):
            input_claim = ''
        
        result, vb_outcome, text_cls = executePipeline(input_claim, filepath, surebot_logger, txt_model, txt_tokenizer,visualbert_model)
              
        # img_doctoring = "TO BE FURTHER EDITED"

        all_scores = [result, vb_outcome, text_cls] #, img_doctoring]
        support, refute = 0, 0
        for score in all_scores:
            if "SUPPORTS" in score:
                support += 1
            elif "REFUTES" in score:
                refute += 1
        if support > refute:
            final_score = 'SUPPORTS'
        elif support < refute:
            final_score = 'REFUTES' 
        else:      
            final_score = "CANNOT BE DETERMINED"
        
    return render_template('image.html', filepath=filepath, vb_outcome=vb_outcome,
                            result=result, text_cls=text_cls, final=final_score)    

if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)
