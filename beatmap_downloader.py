import time
import requests
from pathlib import Path
import pandas as pd

def download_missing_beatmaps():
    skillsets = pd.read_csv("used_beatmaps/skillsets.csv")
    missing_beatmaps_ids = []
    for beatmap_id in skillsets.ID:
        output_path = Path(f"data/{beatmap_id}.osu")
        if not output_path.exists():
            missing_beatmaps_ids.append(beatmap_id)
    missing_beatmaps_ids


    with requests.Session() as session:
        for i,beatmap_id in enumerate(missing_beatmaps_ids):
            url = f"https://osu.ppy.sh/osu/{beatmap_id}"
            response = session.get(url)

            if response.status_code == 200:
                # Optional: normalize to CRLF for Windows tools
                with open(f"data/{beatmap_id}.osu", "wb") as f:
                    f.write(response.text.replace('\r\n', '\n').replace('\r', '').encode('utf-8'))

                print("File Saved")
            else:
                print("Failed to fetch the .osu file.")
            time.sleep(2)

    print("Finished")