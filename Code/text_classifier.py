import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification 
#from transformers import BertTokenizer, BertForSequenceClassification
#from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
import numpy as np
import datetime
import warnings
from transformers import logging
logging.set_verbosity_error()

warnings.filterwarnings('ignore')

# sentences_test = "Singaporeans: Ben Davis is no longer one of us. He renounced his citizenship to siam NS and play football. Ben Davis: *Doesn't include the Sg flag on his Instagram after becoming a successful Fulham player. Singaporeans: MAKEUP Wake Up, Singapore 21 hrs. You can't disown Ben Davis then later claim him as 'one of us' when he starts succeeding in football. www.wakeupsg.com ale 393 Like Most Relevant 101 Comments 74 Shares Comment Share Rayner Yang Good luck, you made the right decision Like Reply 20h Ruak Redniham It is sad yhat we do not recognise, support nor appreciate our home grown talent by taking them under our wing. Instead we so readily let them go and write them off. Like Reply 14h 1 Reply Victor Lau Good move, Ben, and well done... Like Reply 6h Jil Khoo Nailed it. All the best Ben! Write a comment... 25 o 6 1 B"

#######################################################################
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

####################################################################

def text_model():
    # print("\n******* TEXT CLASSIFIER MODEL LOADED *******")
    # device = torch.device("cpu")
    # model_name = "roberta"     # bert / distilbert / roberta

##    if model_name == "bert":
##        #print('Loading Bert tokenizer...')
##        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
##        model = BertForSequenceClassification.from_pretrained(
##        "bert-base-uncased",            # Use the 12-layer BERT model, with an uncased vocab.
##        num_labels = 2,                 # The number of output labels--2 for binary classification. You can increase this for multi-class tasks.   
##        output_attentions = False,      # Whether the model returns attentions weights.
##        output_hidden_states = False,   # Whether the model returns all hidden-states.
##        )
##        output_file = "./pipeline_models/trained_model/model_bert.pth"
##        
##    elif model_name == "distilbert":
##        #print('Loading DistilBert tokenizer...')
##        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
##        model = DistilBertForSequenceClassification.from_pretrained(
##        "distilbert-base-uncased",
##        num_labels = 2, # The number of output labels--2 for binary classification. 
##        output_attentions = False, # Whether the model returns attentions weights.
##        output_hidden_states = False, # Whether the model returns all hidden-states.
##        )
##        output_file = "./pipeline_models/trained_model/model_distilbert.pth"
        
##    elif model_name == "roberta":
        #print('Loading Roberta Base tokenizer...')
    
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    model = AutoModelForSequenceClassification.from_pretrained(
    'xlm-roberta-base',
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False
    )
    output_file = "./pipeline_models/trained_model/model_roberta_base.pth"
        
    optimizer = AdamW(model.parameters(),
                    lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )


    ################ Load Model ###################
    checkpoint = torch.load(output_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, tokenizer

    ################ Evaluation ###################

def text_classification(input_text, model, tokenizer):
    # # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    encoded_dict = tokenizer.encode_plus(
                        input_text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 512,           # Pad & truncate all sentences.
                        truncation = True,
                        padding = 'max_length',
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                    )

    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    model.eval()

    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(input_ids, attention_mask=attention_masks)

    logits = outputs[0]

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    #print(logits)

    pred_labels = np.argmax(logits, axis=1).flatten()


    if pred_labels[0] == 1:
        output_label = "SUPPORTS"
    elif pred_labels[0] == 0:
        output_label = "REFUTES"
        
    print(f'\nText Classfication Results : {output_label}\n')
    
    return output_label
