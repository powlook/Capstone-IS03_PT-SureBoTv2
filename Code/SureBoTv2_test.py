"""
Authors: Adriel Kuek, Chua Hao Zi, Lavanya, Francis Louis
Date Created: 25 March 2021
Version:
Email: hz29990@gmail.com, adrielkuek@gmail.com, francis.louis@gmail.com, lavanya2204@hotmail.com
Status: Development

Description:
SureBo(T) is an end to end automatic fact-checking BOT based on
TELEGRAM API that retrieves multi document inputs for fact
verification based on a single input query. The input query currently
takes the form of a text message that is dubious in content.

In fulfilment of the requirements for the Intelligent Reasoning Systems
project under the Master of Technology (Intelligent Systems)
- NUS Institute of System Sciences (AY2021 - Semester 2)

"""

import requests, time, os, re, io, cv2
import logging, warnings
import emoji
import validators
import numpy as np
import pandas as pd
from glob import glob
from pyfiglet import Figlet
from newspaper import fulltext
from spacy.lang.en import English
from celery.exceptions import SoftTimeLimitExceeded
import torch
import torch.nn as nn
from torchinfo import summary
from transformers import AutoModel   #BertTokenizer, 

from text_processing import process_ocr, process_text
from QueryImage import *
from GraphNetFC import graphNetFC
from EvidenceRetrieval import EvidenceRetrieval
from VBInference import vb_inference, vb_model
from text_classifier import text_classification, text_model

warnings.filterwarnings("ignore")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key/image_search.json"
#Use this at the top of your python code
from google.cloud import vision_v1

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
device_cpu = 'cpu'

# Params - ER
length_penalty = 1.5
topN = 5
max_length = 128
dist_thres = 0.3
# Params - GraphNET
feature_num = 768
evidence_num = 5
graph_layers = 2
num_class = 3
sequence_length = 128
# Aggregating method: top, max, mean, concat, att, sum
graph_pool = 'att'


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


def executePipeline(query, input_image, surebot_logger, txt_model, txt_tokenizer, visualbert_model):
    #####################################################
    # Initialization
    #####################################################
    try:
        print(f'DEVICE Used : {device}')
        surebot_logger.info(f'\n=============== NEW QUERY ===============')
        surebot_logger.info(f'DEVICE Used : {device}')
        start = time.time()
        cwd = os.path.dirname(os.path.realpath(__file__))
        print(f'INITIALISE EVIDENCE RETRIEVAL PIPELINE . . .')
        surebot_logger.info(f'INITIALISE EVIDENCE RETRIEVAL PIPELINE . . .')
        ER_pipeline = EvidenceRetrieval(cwd, device, surebot_logger)
        query_image = QueryImage(cwd)
        final_score = 'NO MATCHING ARTICLES FOUND'
        vb_outcome, text_cls = 'NONE', 'NONE'
        Filtered_Articles, Image_Articles = [], []
        word_count, matching_title = [], []

        # Get title from web-pages if ocr_text is []
        # We will accept a title for the image search if the length
        # of the title is > 10 words
        #print('query :', query)
        if query == '':
            print('\n*******UNABLE TO EXTRACT MEANINGFUL TEXT FROM IMAGES*******')
            print('*******TO GET TITLES OF WEB PAGES FOR TEXT EXTRACTION*******')
            _, match = query_image.detect_web(input_image)

            for i, item in enumerate(match):
                clean_title = process_title(item.page_title)
                print(clean_title)
                matching_title.append(clean_title)
                word_count.append(len(clean_title.split()))
            print('word_count :', word_count)
            if word_count == []:
                query = ''
            elif max(word_count) > 10:
                max_idx = word_count.index(max(word_count))
                query = matching_title[max_idx]
            else:
                query = ''
            print('web_title query', query)
            
        # To check if the ocr_text query is empty then return not detected
        sentenceToken = []
        querytext = ''
        if query == '':
            vb_outcome = 'NO RELEVANT OCR_TEXT DETECTED'
            final_score = 'NO RELEVANT OCR_TEXT DETECTED'
            text_cls = 'NO RELEVANT OCR_TEXT DETECTED'           
        else:
            # Query Preprocessing
            #querytext = query_preprocessing(query, surebot_logger)
            querytext = query[0]           # remove the list
            
            # Use SPACY to get number of tokens
            nlp = English()
            myDoc = nlp(querytext)
            for token in myDoc:
                sentenceToken.append(token.text)

            print(f'TOTAL NO. OF TOKENS FROM QUERY: {len(sentenceToken)}')
            vb_outcome = vb_inference(input_image, querytext, visualbert_model)
            text_cls = text_classification(querytext, txt_model, txt_tokenizer)

            # If tokens > 50 - Perform Abstractive Summary on Query
            # Else just skip and perform Doc Retrieval

            # Run ER pipeline    
            Filtered_Articles = []
            start_time = time.time()
            if len(sentenceToken) > 20:
                querytext = ER_pipeline.AbstractiveSummary(querytext, length_penalty) 
            Image_Articles = ER_pipeline.ReverseImageSearch(querytext, input_image, topN)
    
            Filtered_Articles = ER_pipeline.RetrieveArticles(querytext, topN)
    
            Filtered_Articles = Filtered_Articles + Image_Articles
    
            print(f'>>>>>>> TIME TAKEN - ER PIPELINE: {time.time() - start_time}')
            #surebot_logger.info(f'>>>>>>> TIME TAKEN - ER PIPELINE: {time.time() - start_time}')
            print('===== ARTICLES RETRIEVAL RESULTS =====')
            #surebot_logger.info(f'\n===== ARTICLES RETRIEVAL RESULTS =====')
            print('Number of Articles After Filtering: {len(Filtered_Articles)}')
            #surebot_logger.info(f'Number of Articles After Filtering: {len(Filtered_Articles)}')
    
            if len(Filtered_Articles) == 0:
                print('NO MATCHING ARTICLES FOUND')
                #surebot_logger.info(f'NO MATCHING ARTICLES FOUND')
            else:
                # Run Fact Verification - Graph NET
                graphNet = graphNetFC(cwd, device_cpu, feature_num, evidence_num, graph_layers,
                                      num_class, graph_pool, sequence_length, surebot_logger)
    
                FactVerification_List = []
                for i in range(len(Filtered_Articles)):
                    pred_dict, outputs, heatmap = graphNet.predict(querytext, Filtered_Articles[i][1])
    
                    FactVerification_List.append(pred_dict['predicted_label'])
                    print('graphNet prediction :', pred_dict)
                    #surebot_logger.info(pred_dict)
                    #print('[SUPPORTS, REFUTES, NOT ENOUGH INFO]')
                    #surebot_logger.info('[SUPPORTS, REFUTES, NOT ENOUGH INFO]')
                    #print((np.array(outputs.detach().cpu())))
                    #surebot_logger.info((np.array(outputs.detach().cpu())))
    
                maj_vote = 0
                for i in range(len(Filtered_Articles)):
                    print(f'ARTICLE: {Filtered_Articles[i][2]} - {FactVerification_List[i]}')
                    #surebot_logger.info(f'ARTICLE: {Filtered_Articles[i][2]} - {FactVerification_List[i]}')
                    if FactVerification_List[i] == 'SUPPORTS':
                        maj_vote += 1
    
                if (maj_vote / len(Filtered_Articles)) > 0.6:
                    final_score = 'SUPPORTS'
                    print('************** FINAL SCORE: SUPPORTS')
                    #surebot_logger.info(f'************** FINAL SCORE: SUPPORTS')
                elif (maj_vote / len(Filtered_Articles)) == 0.5:
                    final_score = 'NOT ENOUGH EVIDENCE'
                    print('************** FINAL SCORE: NOT ENOUGH SUPPORTING EVIDENCE')
                    #surebot_logger.info(f'************** FINAL SCORE: NOT ENOUGH SUPPORTING EVIDENCE')
                else:
                    final_score = 'REFUTES'
                    print('************** FINAL SCORE: REFUTES')
                    #surebot_logger.info(f'************** FINAL SCORE: REFUTES')
    

    except Exception as e:
        if isinstance(e, SoftTimeLimitExceeded):
            raise
        else:
            print('Error Type :', e)
    
    return final_score, vb_outcome, text_cls


def remove_emoji(text):
    return emoji.get_emoji_regexp().sub(u'', text)


def query_preprocessing(query, logger_handle):
    # Remove all EMOJI's from query
    query = query.encode('utf-16', 'surrogatepass').decode('utf-16')
    query = remove_emoji(query)

    # Extract all URL's and replace them with the text in the URL
    link_regex = re.compile('(?P<url>https?://[^\s]+)', re.DOTALL)
    links = re.findall(link_regex, query)

    for link in links:
        # Check URL Validity
        try:
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/14.0.835.163 Safari/535.1'}
            query_urlstatus = validators.url(link)
            if query_urlstatus:
                url_text = fulltext(requests.get(link, headers=headers).text)
            else:
                url_text = link
            query = query.replace(link, url_text)
        except:
            print('Exception when extracting full text from URL')
            logger_handle.info('Exception when extracting full text from URL')

    return query

def detect_text(path):
    """Detects text in the file."""
    client = vision_v1.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision_v1.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    
    text_list = []
    for text in texts[0:1]:
        raw_text = text.description
        clean_text = process_ocr(raw_text)
        text_list.append(clean_text)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return text_list

def configure_logger(chat):
    surebot_logger = logging.getLogger(chat)
    surebot_logger.setLevel(logging.INFO)
    logDir = './SurebotLog/'
    if not os.path.exists(logDir):
        os.makedirs(logDir)

    timing = time.asctime(time.localtime(time.time()))
    # logFile = logDir + '/' + timing.replace(' ','_') + '.log'
    logFile = logDir + '/chat_' + str(chat) + '.log'
    handler = logging.FileHandler(logFile)
    formatter = logging.Formatter('')
    handler.setFormatter(formatter)
    surebot_logger.addHandler(handler)

    return surebot_logger


if __name__ == "__main__":

    chat = 0
    txt_model, txt_tokenizer = text_model()
    visualbert_model = vb_model()
    surebot_logger = configure_logger(chat)
    picture_folder = '../val'
    files = glob(picture_folder+'/*.*')
    df = pd.DataFrame(columns=['filename', 'rev_img', 'vis_bert', 'text_cls', 'final_result', 'ground_truth', 'eval_res'])
    
    for i, img_filepath in enumerate(files[29:]):
        #img_filepath = os.path.join(picture_folder, file)
    
        file = img_filepath.split('\\')[1]
        print(file)
        if file[0] == '0':
            ground_truth = 'SUPPORTS'
        elif file[0] == '2':
            ground_truth = 'REFUTES'
        else: ground_truth = 'NONE'

        
        input_claim = detect_text(img_filepath)
        if input_claim == []:
            input_claim = '0'
        if (len(input_claim[0].split()) < 5):
            input_claim = ''
        result, vb_result, text_cls = executePipeline(input_claim, img_filepath, surebot_logger, txt_model, txt_tokenizer,visualbert_model)

        all_scores = [result, vb_result, text_cls] #, img_doctoring]
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

        if final_score == "CANNOT BE DETERMINED":
            eval_res = ''
        elif final_score == ground_truth:
            eval_res = 'CORRECT'
        else: eval_res = 'WRONG'

        print(i, file, result, vb_result, text_cls, final_score, ground_truth, eval_res)
        
        df.loc[i] = file, result, vb_result, text_cls, final_score, ground_truth, eval_res

    correct = df[df['eval_res'] == 'CORRECT']['eval_res'].value_counts()
    wrong   = df[df['eval_res'] == 'WRONG']['eval_res'].value_counts()
    print('Percentage correct :', round((correct[0] /(correct[0] + wrong[0]))*100,2), '%')
    df.to_excel('validation_29_62.xlsx')
    