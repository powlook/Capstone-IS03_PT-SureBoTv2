from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from spacy.lang.en import English

def check_tokens(input_text):
    nlp = English()
    myDoc = nlp(input_text)
    sentenceToken = []
    for token in myDoc:
        sentenceToken.append(token.text)
        
    return len(sentenceToken)

def summarizer(input_text, device):
    PegasusModel_dir = "google/pegasus-cnn_dailymail"
    tokenizer = PegasusTokenizer.from_pretrained(PegasusModel_dir)
    PegasusModel = PegasusForConditionalGeneration.from_pretrained(PegasusModel_dir).to(device)
    batch = tokenizer(input_text, padding='longest', return_tensors="pt").to(device)
    translated = PegasusModel.generate(**batch, max_new_tokens=512)
    new_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return new_text   

def new_sumtext(input_text, device):
    txt_list = []
    text = input_text
    token_length = check_tokens(text)
    if token_length > 400:          # BERT tokenizer max length only 512
        text = summarizer(text, device)[0]
    txt_list.append(text)
    
    return txt_list

    