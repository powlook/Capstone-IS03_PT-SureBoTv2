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
import torch
import numpy as np
import pandas as pd
from pyfiglet import Figlet
from newspaper import fulltext
from spacy.lang.en import English
from celery.exceptions import SoftTimeLimitExceeded
import torch.nn as nn
from torchinfo import summary
from transformers import BertTokenizer, AutoModel

from text_processing import *
from QueryImage import *
from GraphNetFC import graphNetFC
from EvidenceRetrieval import EvidenceRetrieval
from VBInference import vb_inference

warnings.filterwarnings("ignore")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key/image_search.json"
#Use this at the top of your python code
from google.cloud import vision_v1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_cpu = 'cpu'

# Params - ER
length_penalty = 1.5
topN = 5
max_length = 128
dist_thres = 0.4
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


def executePipeline(query, input_image, surebot_logger):
    #####################################################
    # Initialization
    #####################################################
    try:
        print(f'DEVICE Available: {device}')
        surebot_logger.info(f'\n=============== NEW QUERY ===============')
        surebot_logger.info(f'DEVICE Available: {device}')
        start = time.time()
        cwd = os.path.dirname(os.path.realpath(__file__))
        print(f'INITIALISE EVIDENCE RETRIEVAL PIPELINE . . .')
        surebot_logger.info(f'INITIALISE EVIDENCE RETRIEVAL PIPELINE . . .')
        ER_pipeline = EvidenceRetrieval(cwd, device, surebot_logger)
        query_image = QueryImage(cwd)
        final_score = 'NO MATCHING ARTICLES FOUND'
        Filtered_Articles, Image_Articles = [], []
        word_count, matching_title = [], []

        # Get title from web-pages if ocr_text is []
        # We will accept a title for the image search if the length
        # of the title is > 10 words
        if query == '':
            _, match = query_image.detect_web(input_image)

            for i, item in enumerate(match):
                clean_title = process_title(item.page_title)
                print(clean_title)
                matching_title.append(clean_title)
                word_count.append(len(clean_title.split()))
            print('word_count :', word_count)
            if word_count == '':
                query = ''
            elif max(word_count) > 10:
                max_idx = word_count.index(max(word_count))
                query = matching_title[max_idx]
            else:
                query = ''

        # Query Preprocessing
        querytext = query_preprocessing(query, surebot_logger)
        print('query     :', query)
        print('querytext :', querytext)
        # Use SPACY to get number of tokens
        nlp = English()
        myDoc = nlp(querytext)
        sentenceToken = []
        for token in myDoc:
            sentenceToken.append(token.text)

        vb_outcome = vb_inference(input_image, querytext)

        print(f'TOTAL NO. OF TOKENS FROM QUERY: {len(sentenceToken)}')
        surebot_logger.info(f'TOTAL NO. OF TOKENS FROM QUERY: {len(sentenceToken)}')
        print(sentenceToken)
        surebot_logger.info(sentenceToken)

        # If tokens > 50 - Perform Abstractive Summary on Query
        # Else just skip and perform Doc Retrieval
        if len(sentenceToken) > 50:
            querytext = ER_pipeline.AbstractiveSummary(querytext, length_penalty)

        # Run ER pipeline
        start_time = time.time()

        Filtered_Articles = ER_pipeline.RetrieveArticles(querytext, topN)
        Image_Articles = ER_pipeline.ReverseImageSearch(querytext, input_image, topN)
        Filtered_Articles = Filtered_Articles + Image_Articles

        print(f'>>>>>>> TIME TAKEN - ER PIPELINE: {time.time() - start_time}')
        surebot_logger.info(f'>>>>>>> TIME TAKEN - ER PIPELINE: {time.time() - start_time}')

        print(f'===== ARTICLES RETRIEVAL RESULTS =====')
        surebot_logger.info(f'\n===== ARTICLES RETRIEVAL RESULTS =====')
        print(f'Number of Articles After Filtering: {len(Filtered_Articles)}')
        surebot_logger.info(f'Number of Articles After Filtering: {len(Filtered_Articles)}')

        output_message = ""
        #output_message = "===== FACT CHECK RESULTS ====="
        #output_message += "\nTime-Taken: {} seconds".format(int(time.time() - start))
        #output_message += "\nQuery Input: {}".format(query)

        if len(Filtered_Articles) == 0:
            output_message += 'NO MATCHING ARTICLES FOUND'
            print(f'NO MATCHING ARTICLES FOUND')
            surebot_logger.info(f'NO MATCHING ARTICLES FOUND')
        else:
            # Run Fact Verification - Graph NET
            graphNet = graphNetFC(cwd, device_cpu, feature_num, evidence_num, graph_layers,
                                  num_class, graph_pool, sequence_length, surebot_logger)

            FactVerification_List = []
            for i in range(len(Filtered_Articles)):
                pred_dict, outputs, heatmap = graphNet.predict(querytext, Filtered_Articles[i][1])

                FactVerification_List.append(pred_dict['predicted_label'])
                print(pred_dict)
                surebot_logger.info(pred_dict)
                print('[SUPPORTS, REFUTES, NOT ENOUGH INFO]')
                surebot_logger.info('[SUPPORTS, REFUTES, NOT ENOUGH INFO]')
                print((np.array(outputs.detach().cpu())))
                surebot_logger.info((np.array(outputs.detach().cpu())))

                # Plot Attention Heat map to visualize
                # ax = sns.heatmap(heatmap, linewidth=1.0, cmap="YlGnBu")
                # plt.show()
                # plt.clf()

            maj_vote = 0
            for i in range(len(Filtered_Articles)):
                print(f'ARTICLE: {Filtered_Articles[i][2]} - {FactVerification_List[i]}')
                surebot_logger.info(f'ARTICLE: {Filtered_Articles[i][2]} - {FactVerification_List[i]}')
                if FactVerification_List[i] == 'SUPPORTS':
                    maj_vote += 1

            if (maj_vote / len(Filtered_Articles)) > 0.6:
                final_score = 'SUPPORTS'
                print(f'************** FINAL SCORE: SUPPORTS')
                surebot_logger.info(f'************** FINAL SCORE: SUPPORTS')
            elif (maj_vote / len(Filtered_Articles)) == 0.5:
                final_score = 'NOT ENOUGH EVIDENCE'
                print(f'************** FINAL SCORE: NOT ENOUGH SUPPORTING EVIDENCE')
                surebot_logger.info(f'************** FINAL SCORE: NOT ENOUGH SUPPORTING EVIDENCE')
            else:
                final_score = 'REFUTES'
                print(f'************** FINAL SCORE: REFUTES')
                surebot_logger.info(f'************** FINAL SCORE: REFUTES')

            output_message += "\n\n----- Total Articles Found: {} -----".format(len(Filtered_Articles))
            #for i in range(len(Filtered_Articles)):
            #    output_message += "\n\nURL {}".format(i+1)
            #    output_message += '\n' + Filtered_Articles[i][2] + '\n*[Summary]* '
            #    for j in range(len(Filtered_Articles[i][1])):
            #        output_message += Filtered_Articles[i][1][j]
            output_message += '\n\nFINAL SCORE: ' + final_score

    except Exception as e:
        if isinstance(e, SoftTimeLimitExceeded):
            raise
        else:
            output_message = 'Exception occurred in pipeline'
            print(e)
    
    return final_score, vb_outcome    #, final_score, len(Filtered_Articles)


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

    LogDir = logDir
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
    surebot_logger = configure_logger(chat)
    custom_fig = Figlet(font='slant')
    surebot_banner = custom_fig.renderText("SureBoTv2")
    print('\n')
    print(surebot_banner + "version 2.0\n")
    surebot_logger.info('\n')
    surebot_logger.info(surebot_banner + "version 2.0\n")
    picture_folder = '../images'
    while True:
        #print(f'\n\nSureBoT: Input a claim that you would like to fact-check!')
        surebot_logger.info(f'SureBoTv2: Input a picture (filename) that you will like to fact-check!')
        file_name = str(input("Filename: "))
        img_filepath = os.path.join(picture_folder, file_name)
        img = cv2.imread(img_filepath, cv2.IMREAD_ANYCOLOR)
        cv2.imshow('Picture', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        input_claim = detect_text(img_filepath)[0]
        if (len(input_claim.split()) < 5):
            input_claim = ''
        print(f'\n\nProcessing your claim......', file_name)
        surebot_logger.info(input_claim)
        result, vb_result = executePipeline(input_claim, img_filepath, surebot_logger)
        #result = result.encode('utf-16', 'surrogatepass').decode('utf-16')
        print('Reverse Image Search Results :', result)
        print('Visual Bert Comparison Results :', vb_result)
