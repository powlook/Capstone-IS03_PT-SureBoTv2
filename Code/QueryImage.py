import io, os, re, cv2
import json
import pandas as pd
from glob import glob
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
#from summarizer_pegasus import summarise_text
from text_processing import *

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key/image_search.json"
#Use this at the top of your python code

from google.cloud import vision_v1

class QueryImage(object):
    
    def __init__(self, filepath):

        self.filepath = filepath
        self.vision = vision_v1
        self.client = vision_v1.ImageAnnotatorClient() 

    def detect_web(self, path):
        """Detects web annotations given an image."""

        #client = vision_v1.ImageAnnotatorClient()   
        with io.open(path, 'rb') as image_file:
            content = image_file.read()

        image = self.vision.Image(content=content)
        response = self.client.web_detection(image=image)
        annotations = response.web_detection

        full_matching_images = []
        matching_images = []
        #partial_matching_images = []
        #visually_similar_images = []
        
        if annotations.full_matching_images:
            for page in annotations.full_matching_images:
                full_matching_images.append(page)  
        
        if annotations.pages_with_matching_images:
            for page in annotations.pages_with_matching_images:
                matching_images.append(page)


        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))
        
        return full_matching_images, matching_images


    def detect_text(self, path):
        """Detects text in the file."""

        #client = vision_v1.ImageAnnotatorClient()

        with io.open(path, 'rb') as image_file:
            content = image_file.read()

        image = self.vision.Image(content=content)

        response = self.client.text_detection(image=image)
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


    def extract_text_from_html(self, output_file, url):

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.10) Chrome/41.0.2228.0 firefox/71.0'}    
        req = Request(url, headers=headers)
        
        try:
            html_page = urlopen(req)
            soup = BeautifulSoup(html_page, "html.parser")
            results = soup.find_all(['p'])
            text = [result.text for result in results]
            return text
        except:
            return []

        #with open(output_file, "w", encoding='utf-8') as f:
        #    f.writelines(text)
        #f.close()


    def get_whitelist(self, whitelist):

        df = pd.read_excel(whitelist)
        wh_list = df['domain']
        
        return list(wh_list)


    def get_image_from_web(self, path):
        import requests
        from requests import ConnectionError, ConnectTimeout, HTTPError
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.10) Chrome/41.0.2228.0 firefox/71.0'}
        #try:
        img_data = requests.get(path, headers=headers).content
        try:
            with open('image_name.jpg', 'wb') as handler:
                handler.write(img_data)
        except(ConnectionError, HTTPError, ConnectTimeout):
            return None

def reverse_image_search(select_file):

    # The script below process a selected image
    cwd = os.path.dirname(os.path.realpath(__file__))
    query_image = QueryImage(cwd)
    whitelist_file = 'whitelist.xlsx'
    html_text = 'html_text.txt'
    regex = '^(?:https?:\\/\\/)?(?:[^@\\/\\n]+@)?(?:www\\.)?([^:\\/\\n]+)'
    clean_text_list = []

    whitelist = query_image.get_whitelist(os.path.join(cwd, whitelist_file))
    _, match = query_image.detect_web(select_file)

    for i, item in enumerate(match):
        domain = re.findall(regex, item.url)[0]          
        if domain in whitelist:                
            clean_title = process_title(item.page_title)
            print('clean_title :', clean_title)
            text = query_image.extract_text_from_html(html_text, item.url)
            #print('extract_text :', text)
            clean_text = process_text(text, domain)
            print(clean_text)
            if len(clean_text.split()) > 50:
                clean_text_list.append(clean_text)
    
    return clean_text_list
                

def main(select_file):
    # The script below process a selected image
    cwd = os.path.dirname(os.path.realpath(__file__))
    query_image = QueryImage(cwd)
    whitelist_file = 'whitelist.xlsx'
    html_text = 'html_text.txt'
    regex = '^(?:https?:\\/\\/)?(?:[^@\\/\\n]+@)?(?:www\\.)?([^:\\/\\n]+)'
    count = 1
    whitelist = query_image.get_whitelist(os.path.join(cwd, whitelist_file))

    ocr_text = query_image.detect_text(select_file)

    full, match = query_image.detect_web(select_file)
           
    for i, item in enumerate(match):
        link_info = {}
        domain = re.findall(regex, item.url)[0]          
        if domain in whitelist:
            print(f'Count : {count}')            
            print('item_url :', item.url)            
            print('domain name :', domain)
            print('item_title :', item.page_title)
            clean_title = process_title(item.page_title)
            print('clean_title :', clean_title)
            text = query_image.extract_text_from_html(html_text, item.url)
            #print('extract_text :', text)
            clean_text = process_text(text, domain)
            
            #summary = summarise_text(clean_text)
            #link_info['summary'] = summary
            #matching_links.append([link_info])
            print('clean text :', clean_text)
            #print('number of words :', len(clean_text.split()))        
            #print('summary :', summary)
            print('\n')
            count += 1
        #else:
            #print('url not in whitelist')
            #print('item_url :', item.url)
            #print('\n')             
            #print('domain name :', domain)
            #print('item_title :', item.page_title)
            #clean_title = process_title(item.page_title)
            #print('clean_title :', clean_title)
            #text = query_image.extract_text_from_html(html_text, item.url)
            #print('extract_text :', text)
            #clean_text = process_text(text)
    

0
if __name__ == "__main__":

    picture_folder = "D:/Capstone/surebot/images"

    while True:
        file_name = str(input("Filename: "))
        select_file = os.path.join(picture_folder, file_name)
        print('\nfilename :', select_file)
        img = cv2.imread(select_file, cv2.IMREAD_ANYCOLOR)
        cv2.imshow('Picture', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        results = reverse_image_search(select_file)
        #main(select_file)























'''

# The script below processes all the images in the folder
regex = '^(?:https?:\\/\\/)?(?:[^@\\/\\n]+@)?(?:www\\.)?([^:\\/\\n]+)'
filepath = "D:/Capstone/webscraping"
files = glob(os.path.join(filepath, 'images/*.*'))
whitelist = get_whitelist()
locations = []
df = pd.DataFrame(columns=['filename', 'ocr_text', 'word_count', 'summary'])

for num, file in enumerate(files[0:50]):
    location = []
    full_matching_url = []
    matching_image_url = []
    matching_title = [] 
    summary = ''
    
    ocr_text = detect_text(file)
 
    print(num,' ', file)
    file_name = file.split('\\')[-1]
    if ocr_text == []:
        ocr_text = 'None'
        word_count = 0
    else:
        word_count = len(ocr_text[0].split())        
    
    if ocr_text == 'None':
        summary = 'None'
    else:
        summary = summarise_text(ocr_text[0])
        
    df.loc[num] = [file_name, ocr_text[0], word_count, summary]
    
df.to_excel('ocr_text_220818_model2.xlsx') 

# In[5]
# The script below processes all the images in the folder
regex = '^(?:https?:\\/\\/)?(?:[^@\\/\\n]+@)?(?:www\\.)?([^:\\/\\n]+)'
pwd = "D:/Capstone/webscraping"
temp_image = 'image_name.jpg'
html_text = 'html_text.txt'
files = glob(os.path.join(pwd, 'images_fake/*.*'))
whitelist = get_whitelist()
locations = []
df = pd.DataFrame(columns=['filename', 'full_matching_url', 'matching_url', 'matching_title', 'ocr_text', 'summary'])

for num, file in enumerate(files):
    location = []
    full_matching_url = []
    matching_image_url = []
    matching_title = []
    summary = ''
    
    ocr_text = detect_text(file)
    
    file_name = file.split('\\')[-1]
    full, match = detect_web(file)

    if full:
        for i, item in enumerate(full):         
            full_matching_url.append(item.url)
            try:
                get_image_from_web(item.url)
                if detect_landmarks_local(temp_image) != []:
                    locations.append(detect_landmarks_loca, mathl(temp_image))
            except:
                continue
    location.append(locations[0:1])
    
    # This is the best match to use to scrap for any information
    
    if match:

        for i, item in enumerate(match):
            domain = re.findall(regex, item.url)[0]  
            matching_image_url.append(item.url+'\n')
            clean_title = process_title(item.page_title)
            matching_title.append(clean_title+'\n')
            
            if domain in whitelist:
                text = extract_text_from_html(html_text, item.url)
                clean_text = process_text(text)

                if domain == 'blackdotresearch.sg':
                    clean_text = process_text_blackdotresearch(clean_text)
        
                elif domain == 'todayonline.com':
                    clean_text = process_text_todayonline(clean_text)     
                    
                elif domain == 'singaporeuncensored.com':
                    clean_text = process_text_singaporeuncensored(clean_text)   
                    
                elif domain == 'mustsharenews.com':
                    clean_text = process_text_mustsharenews(clean_text)    
                    
                elif domain == 'straitstimes.com':
                    clean_text = process_text_straitstimes(clean_text)     
                    
                elif domain == 'tnp.sg':
                    clean_text = process_text_tnp(clean_text)      
                    
                elif domain == 'getindianews.com':
                    clean_text = process_text_getindianews(clean_text)
                    
                elif domain == 'theindependent.sg':
                    clean_text = process_text_theindependent(clean_text)
                    
                elif domain == 'cnsnews.com':
                    clean_text = process_text_cnsnews(clean_text)
                    
                elif domain == 'reuters.com':
                    clean_text = process_text_reuters(clean_text)
                    
                else:
                    clean_text = process_text_general(clean_text)
                    
                clean_text = ' '.join(clean_text.split()[0:350])
                #summary = summarise_text(clean_text)
                
    else:
        summary = 'None'


    df.loc[num] = [file_name, full_matching_url, matching_image_url, matching_title, ocr_text, summary]

df.to_csv('image_titles_summary.csv')
'''
