import requests
import time
import json
import torch
import models
import osu_beatmap_parser as obp
import models_utils

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

def fetch_beatmap_data():
    try:
        response = requests.get(API_URL)
        print(response.status_code)
        #print(response.text)
        try:
            data = json.loads(response.content.decode("utf-8-sig"))
        except Exception as e:
            print(e)
            print("Failed to JSONIFY the Data")
        #print(data)
        return data
    except requests.exceptions.RequestException:
        return None
    except ValueError:
        return None

def main(poll_interval=1):
    old_location = None
    print("Starting osu! beatmap fetcher...")
    try:
        while True:
            data = fetch_beatmap_data()
            #print(data)
            if data:
                #print(f"Title: {data.get('mapArtistTitle')}")
                #print(f"Artist: {data.get('artistRoman')}")
                #print(f"Difficulty: {data.get('mapDiff')}")
                #print(f"Beatmap ID: {data.get('mapid')}")
                #print(f"Mapset ID: {data.get('mapsetid')}")
                #print(f"File Location: {data.get('osuFileLocation')}")
                #print(f"Mods: {data.get('mods')}")
                #print(f"Time left: {data.get('timeLeft')}")
                #print("------")
                #try:
                
                location = data.get('osuFileLocation')
                if old_location != location:
                    print(old_location, location)
                    beatmap = obp.Beatmap.file_to_beatmap(location)
                    models_utils.visualize_beatmap_skillsets_live(model, beatmap, True if "DT" in data.get('mods') else False)
                    old_location = location
                #except Exception as e:
                #    print(e)

            else:
                print("No map loaded.")
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("Fetcher stopped.")

if __name__ == "__main__":
    main()
