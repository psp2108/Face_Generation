from flask import Flask
from flask.helpers import send_file
from flask.wrappers import Request
from flask import request
from LoadModel import GeneratorModule

app = Flask(__name__) 
gen = GeneratorModule()
attrib_d = {}
attributesList = gen.getAttributesOrder()

for i in range(len(attributesList)):
    attrib_d[attributesList[i].lower()] = None

for i in range(7):
    attrib_d["rv" + str(i)] = None

@app.route('/')
def home():
    return "FIGSI"

@app.route('/get_image/')
def get_image():
    global gen
    global attrib_d
    new_d = dict(attrib_d)

    for i in new_d.keys():
        new_d[i] = float(request.args.get(i))

    print(new_d)

    gen.getImage(new_d, autoSave = True, imageName = "test_fromflask")
  
    filename = gen.getOutputImagePath()
    return send_file(filename, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True,port=80)

# http://127.0.0.1/get_image?5_o_clock_shadow=0&bags_under_eyes=0&big_lips=0&big_nose=0&chubby=0&double_chin=0&goatee=0&heavy_makeup=0.1&high_cheekbones=0.1&male=0&mustache=0&narrow_eyes=0&no_beard=0.1&oval_face=0.1&pale_skin=0&pointy_nose=0.1&rosy_cheeks=0&sideburns=0.1&smiling=0.1&straight_hair=0.1&wavy_hair=0&young=0.1&hair_color=0.075&hair_size=0.1&combine_eyebrow=0&rv0=17.917&rv1=18.143&rv2=8.966&rv3=29.206&rv4=13.653&rv5=13.822&rv6=20.095
