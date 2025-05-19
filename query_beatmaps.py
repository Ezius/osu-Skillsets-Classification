from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import chain
import time
import os
import pickle
import osu_beatmap_parser as obp
import re

SONG_PATH = "E:\\osu!\\Songs\\"

def process_beatmapdata(process_idx, skills, set_ids_dict, beatmap_ids_dict):
    seen_ids = []
    found_maps = []
    folders = os.listdir(SONG_PATH)

    folder_chunk = len(folders)//os.cpu_count()

    start_idx = process_idx*folder_chunk
    end_idx = (process_idx+1)*folder_chunk if process_idx < os.cpu_count()-1 else len(folders)

    process_folders = folders[start_idx:end_idx]

    for folder in process_folders:
        for skill in skills:
                for set_id in set_ids_dict[skill]:
                    if set_id in folder:
                        full_folder_path = os.path.join(SONG_PATH, folder)
                        for file in os.listdir(full_folder_path):
                            if file.endswith(".osu"):
                                file_path = os.path.join(full_folder_path, file)
                                with open(file_path, encoding="utf-8", errors="ignore") as f:
                                    contents = f.read()
                                    # Look for beatmap ID inside the file
                                    for beatmap_id in beatmap_ids_dict[skill]:
                                        if beatmap_id not in seen_ids:
                                            if f"BeatmapID:{beatmap_id}" in contents:
                                                try:
                                                    data, extra_data = obp.Beatmap.file_to_beatmap(file_path).beatmap_to_data(normalize=True)
                                                    if len(data)>=350:
                                                        found_maps.append((set_id,beatmap_id,data,extra_data,skill))
                                                        seen_ids.append(beatmap_id)
                                                except Exception as e:
                                                    print(f"Error with Beatmap set {set_id}, id: {beatmap_id} error: {e}")

    return found_maps


def parse_collection_from_txt(string):
        return re.findall(r"https://osu\.ppy\.sh/s/(\d+)", string), re.findall(r"★\((\d+)\)", string)

def get_beatmapdata():
    skills = ["AIM", "STREAM", "ALT", "TECH", "SPEED", "READING", "RHYTHM"]
    skill_strings = []
    for skill in skills:
        with open(f"chiv_collections/{skill}.txt", encoding="utf-8", errors="ignore") as f:
            skill_strings.append(''.join(f.readlines()))


    set_ids_dict = {}
    beatmap_ids_dict = {}
    for i,skill in enumerate(skill_strings):
        set_ids_dict[skills[i]], beatmap_ids_dict[skills[i]] = parse_collection_from_txt(skill)

    with ProcessPoolExecutor() as executor:
        map_lists = [executor.submit(process_beatmapdata, cpu, skills, set_ids_dict, beatmap_ids_dict) for cpu in range(os.cpu_count())]

    maps = [map_list.result() for map_list in as_completed(map_lists)]
    maps = list(chain.from_iterable(maps))
    return maps

def main():
    start = time.perf_counter()

    found_maps = get_beatmapdata()
    with open("data/beatmap_data.pkl", "wb") as pkl:
        pickle.dump(found_maps, pkl)

    end = time.perf_counter()
    print(end-start)

if __name__ == '__main__':
    main()