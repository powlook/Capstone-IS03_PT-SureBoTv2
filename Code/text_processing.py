import re
import nltk
import string
nltk.download('words')
english_words = set(nltk.corpus.words.words())
ascii = set(string.printable)


def remove_non_ascii(s):
    return filter(lambda x: x in ascii, s)


def process_title(text):
    
    text = text.replace('\n', ' ')
    text = text.replace('&#39;', '')
    text = text.replace('&quot;', '')
    text = text.replace('&amp;', '')    
    text = text.replace('<b>', '')
    text = text.replace('</b>', '')
    
    text = text.split()
    text = [word for word in text if len(word) < 15]
    #text = [word for word in text if word.lower().isalnum()]
    text = [word for word in text if word.lower() in english_words and word.lower().isalnum()]
    text = ' '.join([x for x in text])    
    
    return text


def process_ocr(text):
    
    text = text.replace('\n', ' ')
    text = re.sub(r'[^\x00-\x7F]', '', text)
    text = text.split()
    #text = [word for word in text if len(word) < 20]
    #text = [word for word in text if word.lower() in english_words and word.lower().isalnum()]
    text = ' '.join([x for x in text])
    
    return text


def process_text(text, domain):
    
    text = [x for x in text if x != '']
    text = [x for x in text if x != '\xa0']
    text = [x for x in text if len(x.split())>5]
    text = [x.replace('\n', '') for x in text]
    text = [x.replace('\xa0', ' ') for x in text]
    
    if domain == 'blackdotresearch.sg':
        clean_text = process_text_blackdotresearch(text)

    elif domain == 'todayonline.com':
        clean_text = process_text_todayonline(text)     
        
    elif domain == 'singaporeuncensored.com':
        clean_text = process_text_singaporeuncensored(text)   
        
    elif domain == 'mustsharenews.com':
        clean_text = process_text_mustsharenews(text)    
        
    elif domain == 'straitstimes.com':
        clean_text = process_text_straitstimes(text)     
        
    elif domain == 'tnp.sg':
        clean_text = process_text_tnp(text)      
        
    elif domain == 'getindianews.com':
        clean_text = process_text_getindianews(text)
        
    elif domain == 'theindependent.sg':
        clean_text = process_text_theindependent(text)
        
    elif domain == 'cnsnews.com':
        clean_text = process_text_cnsnews(text)
        
    elif domain == 'reuters.com':
        clean_text = process_text_reuters(text)
        
    else:
        clean_text = process_text_general(text)
        
    clean_text = ' '.join(clean_text.split()[0:500])    
    
    return clean_text


def process_text_general(clean_text):
    
    clean_text = ' '.join([x for x in clean_text])
    clean_text = clean_text.replace('\n ', ' ')
    clean_text = clean_text.strip()
    clean_text = clean_text[:clean_text.find("### Full Video Loading")]
    #clean_text = clean_text.replace('#','') 
    clean_text = ' '.join(clean_text.split())
    
    return clean_text


def process_text_blackdotresearch(clean_text):
    
    ending = 'Save my name'
    clean_text = ' '.join([x for x in clean_text])
    clean_text = clean_text.replace('\n', '')
    clean_text = clean_text[:clean_text.find("###")]
    clean_text = clean_text.replace('#','')
    clean_text = clean_text[:clean_text.find(ending)]    
    clean_text = ' '.join(clean_text.split())
    
    return clean_text


def process_text_todayonline(clean_text):
    
    clean_text = ' '.join([x for x in clean_text])
    clean_text = clean_text.replace('\n \n', '<para>')
    clean_text = clean_text.replace('\n', '')
    clean_text = clean_text.replace("\'","")
    clean_text = clean_text[:clean_text.find("##  Stay in the know.")]
    clean_text = clean_text.replace('<para>', '')    
    clean_text = ' '.join(clean_text.split())

    return clean_text

def process_text_singaporeuncensored(clean_text):
    
    clean_text = ' '.join([x for x in clean_text])
    clean_text = clean_text.replace('\n \n', '')
    clean_text = clean_text.strip()
    clean_text = clean_text[:clean_text.find("### Full Video Loading")]
    clean_text = ' '.join(clean_text.split())
    
    return clean_text

def process_text_mustsharenews(clean_text):
    
    clean_text = ' '.join([x for x in clean_text])
    clean_text = clean_text.replace('\n \n', '<para>')
    clean_text = clean_text.replace('\n', '')
    clean_text = clean_text.replace("\'","")
    clean_text = clean_text.replace("## ![Must Share News – Independent News For", '')
    clean_text = clean_text.replace('<para>', '')
    clean_text = clean_text[:clean_text.find("_Have news you must share?")]
    clean_text = ' '.join(clean_text.split())

    return clean_text


def process_text_straitstimes(clean_text):
    
    heading = "The Straits Times Toggle navigation Best News Website or Mobile Service # The Straits Times Best News Website or Mobile Service # The Straits Times The Straits Times Toggle navigation #"
    clean_text = ' '.join([x for x in clean_text])
    #clean_text = clean_text.replace('\n \n', '')
    clean_text = clean_text.replace('\n ', '')
    clean_text = clean_text.replace("\'","")
    clean_text = clean_text.replace(heading, '')
    clean_text = clean_text.replace('<para>', '')
    clean_text = clean_text[:clean_text.find("Already have an account?")]
    clean_text = clean_text[:clean_text.find("Join STs Telegram channel")]    
    clean_text = ' '.join(clean_text.split())

    return clean_text


def process_text_tnp(clean_text):
    ending = 'Get The New Paper on your phone with the free TNP app.'
    clean_text = ' '.join([x for x in clean_text])
    clean_text = clean_text.replace('\n' , '')
    clean_text = clean_text.strip()
    clean_text = clean_text[:clean_text.find(ending)]
    #clean_text = clean_text.replace('#','') 
    clean_text = ' '.join(clean_text.split())
    
    return clean_text


def process_text_getindianews(clean_text):
    heading = 'Home __News __'
    ending = 'Previous article'
    clean_text = ' '.join([x for x in clean_text])
    clean_text = clean_text.replace('\n' , ' ')
    clean_text = clean_text.strip()
    clean_text = clean_text[clean_text.find(heading):]
    clean_text = clean_text[:clean_text.find(ending)]
    #clean_text = clean_text.replace('#','') 
    clean_text = ' '.join(clean_text.split())
    
    return clean_text


def process_text_theindependent(clean_text):
    heading = 'Home News __Featured News __'
    ending = 'Follow us on Social Media'
    clean_text = ' '.join([x for x in clean_text])
    clean_text = clean_text.replace('\n' , ' ')
    clean_text = clean_text.strip()
    clean_text = clean_text[clean_text.find(heading):]
    clean_text = clean_text[:clean_text.find(ending)]
    #clean_text = clean_text.replace('#','') 
    clean_text = ' '.join(clean_text.split())
    
    return clean_text


def process_text_cnsnews(clean_text):

    ending = 'Copyright 1998-2022 '
    clean_text = ' '.join([x for x in clean_text])
    clean_text = clean_text.replace('\n' , ' ')
    clean_text = clean_text.strip()
    clean_text = clean_text[:clean_text.find(ending)]
    #clean_text = clean_text.replace('#','') 
    clean_text = ' '.join(clean_text.split())
    
    return clean_text


def process_text_reuters(clean_text):

    ending = 'This article was'
    clean_text = ' '.join([x for x in clean_text])
    clean_text = clean_text.replace('\n' , ' ')
    clean_text = clean_text.strip()
    clean_text = clean_text[:clean_text.find(ending)]
    clean_text = ' '.join(clean_text.split())
    
    return clean_text


def process_text_nytimes(clean_text):

    ending = 'This article was'
    clean_text = ' '.join([x for x in clean_text])
    clean_text = clean_text.replace('\n' , ' ')
    clean_text = clean_text.strip()
    clean_text = clean_text[:clean_text.find(ending)]
    clean_text = ' '.join(clean_text.split())
    
    return clean_text


