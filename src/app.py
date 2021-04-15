from flask import Flask
from flask.helpers import send_file
from flask import request
from LoadModel import GeneratorModule
from FaceRecognize import FaceIdentifier
from datetime import datetime
from flask import jsonify
import os
import subprocess

app = Flask(__name__) 
gen = GeneratorModule()
fi = FaceIdentifier()
attrib_d = {}
attributesList = gen.getAttributesOrder()

for i in attributesList:
    attrib_d[i.lower()] = None

for i in range(7):
    attrib_d["rv" + str(i)] = None

@app.route('/')
def home():
    return "FIGSI"

@app.route('/get_image')
def get_image():
    global gen
    global attrib_d
    new_d = dict(attrib_d)

    for i in new_d.keys():
        new_d[i] = float(request.args.get(i))

    print("APT hit at ", datetime.now().strftime("%d %B, %Y (%H:%M:%S)"))
    # gen.getImage(new_d, randomVector=[-0.39174175, 0.39174175, -0.39174175, -0.39174175, -0.39174175, -0.39174175, 1.39174175 ], autoSave = True, imageName = "test_fromflask")
    gen.getImage(new_d, randomVector=new_d, autoSave = True, imageName = "test_fromflask")
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
    print("===>>", outputImagePath)

    # MEMORY FULL ERROR CAN BE USED ON HIGHER END GPU(S) OR IF OTHER ALTERNATIVE IS POSSIBLE
    # _id = fi.getFaceID(outputImagePath)
    # print("--->>", _id)
    # faceDetails = fi.getDetails(_id[0])

    out = subprocess.Popen(['python', 'FaceRecognize.py', os.path.join(os.getcwd(), outputImagePath)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    _id0, _error = out.communicate()
    print("--->>", _id0)
    faceDetails = fi.getDetails(_id0.decode("utf-8").strip())

    print("===>>", faceDetails)

    return jsonify(faceDetails)

if __name__ == '__main__':
    app.run(debug=False,port=80)

# http://127.0.0.1/get_image?5_o_clock_shadow=0&bags_under_eyes=0&big_lips=0&big_nose=0&chubby=0&double_chin=0&goatee=0&heavy_makeup=0.1&high_cheekbones=0.1&male=0&mustache=0&narrow_eyes=0&no_beard=0.1&oval_face=0.1&pale_skin=0&pointy_nose=0.1&rosy_cheeks=0&sideburns=0.1&smiling=0.1&straight_hair=0.1&wavy_hair=0&young=0.1&hair_color=0.075&hair_size=0.1&combine_eyebrow=0&rv0=17.917&rv1=18.143&rv2=8.966&rv3=29.206&rv4=13.653&rv5=13.822&rv6=20.095
# http://127.0.0.1/get_image?5_o_clock_shadow=0&bags_under_eyes=0&big_lips=0&big_nose=0&chubby=0&double_chin=0&goatee=0&heavy_makeup=1&high_cheekbones=1&male=0&mustache=0&narrow_eyes=0&no_beard=1&oval_face=1&pale_skin=0&pointy_nose=1&rosy_cheeks=0&sideburns=1&smiling=1&straight_hair=1&wavy_hair=0&young=1&hair_color=0.75&hair_size=1&combine_eyebrow=0&rv0=-1.5520749999999999&rv1=-1.546425&rv2=-1.77585&rv3=-1.26985&rv4=-1.6586750000000001&rv5=-1.65445&rv6=-1.497625
