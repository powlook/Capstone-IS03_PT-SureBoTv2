import os, pickle, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchinfo import summary
from transformers import BertTokenizer, AutoModel

from embedding_utils.text_embeddings import new_sumtext
from embedding_utils.visual_embeddings import img_visual_embeds, img2bgr

warnings.filterwarnings("ignore")

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

########################## Main ###############################

def vb_inference(img_path, text):

    # device = "cpu"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(torch.cuda.get_device_name(0))
    #print(torch.cuda.get_device_properties(0))
    model_file = 'pipeline_models/finalized_model.pkl'
    #model = AutoModel.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre')

    img_bgrlist = img2bgr(img_path)
    visual_embeds = img_visual_embeds(img_bgrlist, device)
    txt_embeds = new_sumtext(text, device)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer(txt_embeds, padding='max_length', max_length=50)
    input_ids = torch.tensor(tokens["input_ids"]).to(device)
    attention_mask = torch.tensor(tokens["attention_mask"]).to(device)
    token_type_ids = torch.tensor(tokens["token_type_ids"]).to(device)

    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(device)
    visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(device)

    ###### Load model
    loaded_model = pickle.load(open(model_file, 'rb')).to(device)
    #print(f"summary of model: {summary(loaded_model)}")
    predicted_class = np.nan
    outcome = {0:'REFUTES', 1:'SUPPORTS'}

    while not(predicted_class == 1 or predicted_class == 0):
        with torch.no_grad():
            
            preds = loaded_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids)
            preds = preds.detach().cpu().numpy()
            # print(preds)
            predicted_class = np.argmax(preds)
            # print(predicted_class)
            #if predicted_class == 0:
            #    print("Output: REFUTES")
            #elif predicted_class == 1:
            #    print("Output: SUPPORTS")
    
    return outcome[predicted_class]

if __name__ == "__main__":

    ###################### Parameters #############################
    #img_name = "0002.jpg"
    picture_folder = "D:/Capstone/surebot/images"
    text_file = pd.read_excel('ocr_text_220818.xlsx')
    select_image = 19
    img_name = text_file['filename'][select_image]
    text = text_file['ocr_text'][select_image]
    img_path = os.path.join(os.getcwd(), picture_folder, img_name)
    print(img_path)
    #text = "NATIONAL CENTRE FOR INFECTIOUS DISEASES The 42-year-old British man, who works as a flight attendant, is currently warded at NCID. The Straits Times"
    #text= "Fire breaks out at Kusu Island hilltop with 3 Malay shrines DEM WAKEY FFOR A fire yesterday engulfed a hilltop on Kusu Island where three Malay keramats or shrines are located. A group of campers on nearby Lazarus Island said they heard a loud explosion, followed by a few smaller ones, when the blaze started at about 6.20pm. The Singapore Civil Defence Force said it put out the fire with two water jets within an hour after it arrived. There were no reported injuries. "
    ###############################################################
    
    for i in range(5):
        result = vb_inference(img_path, text)
        print(result)