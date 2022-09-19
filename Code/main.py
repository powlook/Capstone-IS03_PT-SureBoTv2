from flask import *
import glob, os, pprint, sys
from werkzeug.utils import secure_filename
from pathlib import Path
from PIL import Image

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "upload")
print(app.config["UPLOAD_FOLDER"])

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
        
        rev_image = "NO MATCHING ARTICLES"
        vb_outcome = "SUPPORT"
        img_doctoring = "REFUTES"
        text_cls = "SUPPORT"
        
        final = "SUPPORT"
        
    return render_template('image.html', filepath=filepath, vb_outcome=vb_outcome, rev_image=rev_image,img_doctoring=img_doctoring, text_cls=text_cls, final=final)    

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
