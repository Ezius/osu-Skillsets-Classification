import requests
import time
import json
import torch
import models
import osu_beatmap_parser as obp
import models_utils
from flask import Flask, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

API_URL = "http://localhost:20727/json"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using Device:",device)
torch.set_default_device(device)

SKILLSET_LABELS = ["AIM", "STREAM", "ALT", "TECH", "SPEED", "RHYTHM"]
NUM_CLASSES = len(SKILLSET_LABELS)
INPUT_DIM = 12
MAX_SEQ_LEN = 124
BATCH_SIZE = 16

model = models.SkillsetClassifier_v1_1(MAX_SEQ_LEN, NUM_CLASSES)

#loads model weights
PATH = "model_weights/osu_skills_v2.pth"
model.load_state_dict(torch.load(PATH,weights_only=True))
model.eval()

latest_data = None
def fetch_beatmap_data():
    response = requests.get(API_URL, timeout=3)
    #print("SteamCompanion Fetching Data", response.status_code)
    try:
        data = json.loads(response.content.decode("utf-8-sig"))
        return data
    except Exception as e:
        print("Fetch error:", e)
    return None


def update_loop(poll_interval=2):
    previous_location, previous_mods = None, None
    global latest_data
    print("Starting osu! beatmap fetcher...")
    while True:
        data = fetch_beatmap_data()
        if data:            
            location = data.get('osuFileLocation')
            mods = data.get('mods')
            
            if location and (previous_location != location or previous_mods != mods):
                try:
                    beatmap = obp.Beatmap.file_to_beatmap(location)
                    latest_data = models_utils.process_data(model, beatmap, True if "DT" in data.get('mods') else False)
                    previous_location, previous_mods = location, mods
                except Exception as e:
                    print("Failed to Process Beatmap", e)
                    print(e.with_traceback())
        else:
            print("No map loaded.")
        time.sleep(poll_interval)

@app.route("/skillsets.json")
def get_skillsets():
    if latest_data is None:
        return jsonify({"status": "no map loaded"})
    #print("Sending:",jsonify(latest_data).data)
    return jsonify(latest_data)



if __name__ == "__main__":
    from threading import Thread
    t = Thread(target=update_loop, daemon=True)
    t.start()
    app.run(port=7272, debug=False)
