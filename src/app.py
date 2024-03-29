from flask import Flask
from flask.helpers import send_file
from flask import request, render_template
from LoadModel import GeneratorModule, gpuAvailable
from FaceRecognize import FaceIdentifier
from datetime import datetime
from flask import jsonify
import os
import subprocess
import shutil
import random

app = Flask(__name__) 
gen = GeneratorModule()
fi = FaceIdentifier()
if not gpuAvailable:
    print("--> Loading faces for CPU")
    fi.loadFaces()

attrib_d = {}
attributesList = gen.getAttributesOrder()

try:
    os.makedirs("static")
except:
    pass

for i in attributesList:
    attrib_d[i.lower()] = None

for i in range(7):
    attrib_d["rv" + str(i)] = None

@app.route('/')
def home():
    return render_template("index.html", project_title="FIGSI")

@app.route('/get_image')
def get_image():
    global gen
    global attrib_d
    new_d = dict(attrib_d)

    randV = None

    try:
        for i in new_d.keys():
            new_d[i] = float(request.args.get(i))
        randV = new_d
    except:
        pass

    print("APT hit at ", datetime.now().strftime("%d %B, %Y (%H:%M:%S)"))
    # gen.getImage(new_d, randomVector=[-0.39174175, 0.39174175, -0.39174175, -0.39174175, -0.39174175, -0.39174175, 1.39174175 ], autoSave = True, imageName = "test_fromflask")
    gen.getImage(new_d, randomVector=randV, autoSave = True, imageName = "test_fromflask")
    # gen.getImage(new_d, autoSave = True, imageName = "test_fromflask")
  
    filename = gen.getOutputImagePath()
    return send_file(filename, mimetype='image/png')

@app.route('/refresh_dataset')
def refresh_dataset():
    fi.loadFaces()
    return "done"

@app.route('/get_latest_details')
def get_image_details():
    outputImagePath = gen.getOutputImagePath()
    faceName = random.randint(10000,99999)
    shutil.copy(outputImagePath, f"static/{faceName}.png")
    print("===>>", outputImagePath)

    faceDetails = None
    if gpuAvailable:
        print(">>> Running on GPU")
        out = subprocess.Popen(['python', 'FaceRecognize.py', os.path.join(os.getcwd(), outputImagePath)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        _id0, _error = out.communicate()
        print("--->>", _id0)
        faceDetails = fi.getDetails(_id0.decode("utf-8").strip())
    else:     
        # MEMORY FULL ERROR CAN BE USED ON HIGHER END GPU(S) OR IF OTHER ALTERNATIVE IS POSSIBLE
        print(">>> Running on CPU")
        _id = fi.getFaceID(outputImagePath)
        print("--->>", _id)
        faceDetails = fi.getDetails(_id[0])

    print("===>>", faceDetails)

    if "Error" in faceDetails:
        return render_template("suspect.html", 
        image=f"{faceName}.png",
        name="No Match Found", 
        dob="-", 
        married="-", 
        county="-", 
        state="-", 
        address="-"
        ) 
    else:
        return render_template("suspect.html", 
        image=f"{faceName}.png",
        name=f"{faceDetails['First Name']} {faceDetails['Middle Name']} {faceDetails['Last Name']}", 
        dob=faceDetails['DOB'], 
        married=faceDetails['Married'], 
        county=faceDetails['Country'], 
        state=faceDetails['State'], 
        address=f"{faceDetails['Address']}\nContact: {faceDetails['Contact Number']}") 


if __name__ == '__main__':
    app.run(debug=False,port=80)

# http://127.0.0.1/get_image?5_o_clock_shadow=0&bags_under_eyes=0&big_lips=0&big_nose=0&chubby=0&double_chin=0&goatee=0&heavy_makeup=0.1&high_cheekbones=0.1&male=0&mustache=0&narrow_eyes=0&no_beard=0.1&oval_face=0.1&pale_skin=0&pointy_nose=0.1&rosy_cheeks=0&sideburns=0.1&smiling=0.1&straight_hair=0.1&wavy_hair=0&young=0.1&hair_color=0.075&hair_size=0.1&combine_eyebrow=0&rv0=17.917&rv1=18.143&rv2=8.966&rv3=29.206&rv4=13.653&rv5=13.822&rv6=20.095
# http://127.0.0.1/get_image?5_o_clock_shadow=0&bags_under_eyes=0&big_lips=0&big_nose=0&chubby=0&double_chin=0&goatee=0&heavy_makeup=1&high_cheekbones=1&male=0&mustache=0&narrow_eyes=0&no_beard=1&oval_face=1&pale_skin=0&pointy_nose=1&rosy_cheeks=0&sideburns=1&smiling=1&straight_hair=1&wavy_hair=0&young=1&hair_color=0.75&hair_size=1&combine_eyebrow=0&rv0=-1.5520749999999999&rv1=-1.546425&rv2=-1.77585&rv3=-1.26985&rv4=-1.6586750000000001&rv5=-1.65445&rv6=-1.497625
