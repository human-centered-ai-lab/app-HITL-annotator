from flask import Flask, request, make_response, send_from_directory, jsonify
import concurrent.futures
app = Flask(__name__)
import uuid
import time
from skimage.io import imsave
import os
import skimage
from waitress import serve

annotators = {}
session = None

def run_web_app(port):
    serve(app, host='0.0.0.0', port=port)

class Ui:
    def __init__(self) -> None:
        global annotators
        annotators = {}
     
    def start(self, port):
        executor = concurrent.futures.ThreadPoolExecutor()
        executor.submit(run_web_app, port)
        global session
        session = str(uuid.uuid4)
    
    def register(self, id):
        global session
        global annotators
        if id in annotators: return None
        annotators[id] = []
        return session
    
    def request_annotation(self, annotator_id, image, labels, label):
        global annotators
        if not annotator_id in annotators: raise "Annotator not found"
        id = str(uuid.uuid4())
        if not os.path.exists('framework/tmp'): os.makedirs('framework/tmp', exist_ok=True)
        ubyte_img = skimage.util.img_as_ubyte(image)
        imsave(f"framework/tmp/{annotator_id}_{id}.png", ubyte_img)
        entry = {
            "id": id,
            "image": f'{annotator_id}_{id}.png',
            "additional_info": None,
            "result": None,
            "labels": labels,
            'label': label
        }
        annotators[annotator_id].append(entry)
        while True:
            updated = get_entry(annotator_id, id)
            if updated is None: raise "Entry not found"
            if updated['result'] is not None:
                 return updated['result']
            time.sleep(1)


def get_entry(aid, eid):
     global annotators
     if not aid in annotators: return None
     entries = annotators[aid]
     for entry in entries:
          if entry['id'] == eid:
               return entry
     return None


def set_entry_result(aid, eid, result):
     global annotators
     if not aid in annotators: return None
     entries = annotators[aid]
     for entry in entries:
          if entry['id'] == eid:
               entry['result'] = result
               os.remove(f"framework/tmp/{aid}_{eid}.png")
               return True
     return False


@app.route("/api/register", methods = ['POST'])
def register_client():
    global annotators
    data = request.json
    annotator_id = data.get('id')
    #print(annotators)
    if annotator_id not in annotators:
         return "Annotator not found", 404
    res = make_response()
    res.set_cookie("session", value=session)
    res.set_cookie("annotator", value=annotator_id)
    return res

@app.route("/api/image/<path>", methods = ['GET'])
def get_image(path):
     return send_from_directory('tmp', path)

@app.route("/react/<path:path>", methods = ['GET'])
def get_webapp(path):
     return send_from_directory('react', path)

@app.route("/api/sample", methods = ['GET'])
def get_new_sample():
    global session
    global annotators
    cookies = request.cookies
    if 'annotator' not in cookies or 'session' not in cookies: return jsonify({'registered': False})
    annotator_id = cookies["annotator"]
    session_id = cookies["session"]
    if not annotator_id in annotators or session_id != session:
         return "Not found", 404
    user = annotators[annotator_id]
    #print(user)
    for entry in user:
         if entry['result'] is None:
              data = {
                   'id': entry['id'],
                   'labels': entry['labels'].tolist(),
                   'image': f'/api/image/{entry["image"]}',
                   'suggestion': entry['label']
              }
              #print(data)
              return jsonify(data)
    return jsonify(None)

@app.route("/api/registered", methods = ['GET'])
def registered():
     global session
     global annotators
     cookies = request.cookies
     if 'annotator' not in cookies or 'session' not in cookies: return jsonify({'registered': False})
     annotator_id = cookies["annotator"]
     session_id = cookies["session"]
     if not annotator_id in annotators or session_id != session:
          return jsonify({'registered': False})
     return jsonify({'registered': True})


@app.route("/api/entry/<entry_id>", methods = ['POST'])
def vote(entry_id):
    global session
    global annotators
    cookies = request.cookies
    if 'annotator' not in cookies or 'session' not in cookies: return jsonify({'registered': False})
    annotator_id = cookies["annotator"]
    session_id = cookies["session"]
    if not annotator_id in annotators or session_id != session:
         return "Not found", 404
    data = request.json
    result = data.get('result')
    user = annotators[annotator_id]
    for entry in user:
         if entry['id'] == entry_id:
              if result not in entry['labels']:
                   return "Label not possible", 400
              entry['result'] = result
              return entry_id, 200
    
