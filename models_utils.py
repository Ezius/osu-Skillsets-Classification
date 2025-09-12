
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import requests
import numpy as np
import random
import osu_beatmap_parser as obp
import io
import time

SKILLSET_LABELS = ["AIM", "STREAM", "ALT", "TECH", "SPEED", "RHYTHM"]
NUM_CLASSES = len(SKILLSET_LABELS)
INPUT_DIM = 12
MAX_SEQ_LEN = 124
BATCH_SIZE = 16


def one_hot_labels(label):
    empty = np.zeros(len(SKILLSET_LABELS))
    if label in SKILLSET_LABELS:
        idx = SKILLSET_LABELS.index(label)
        empty[idx] = 1
        return empty
    else:
        empty[-1] = 1
        return empty

def apply_dt(beatmap_sequence, extra_data):
    beatmap_sequence[:,5] *= 1.5
    beatmap_sequence[3:5] /= 1.5
    if extra_data[0] >= 5:
        extra_data[0] = min((extra_data[0]*2 + 13)/3,    11)
    else:
        extra_data[0] = max(extra_data[0]*400/750 + 5,   0)

def apply_ht(beatmap_sequence, extra_data):
    beatmap_sequence[:,5] *= 0.75
    beatmap_sequence[3:5] /= 0.75
    if extra_data[0] >= 5:
        extra_data[0] = min((extra_data[0]*2 + 13)/3,    11) #todo
    else:
        extra_data[0] = max(extra_data[0]*400/750 + 5,   0) #todo

def apply_hr(beatmap_sequence, extra_data):
    beatmap_sequence[:,1] = 1-beatmap_sequence[:,1]
    extra_data[0] = min(extra_data[0]*1.4, 10)

def apply_ez(beatmap_sequence, extra_data):
    extra_data[0] = extra_data[0]/2

def decode_output(tensor):
    return SKILLSET_LABELS[tensor.argmax()]

def create_beatmaps_batch(beatmaps_data, extra_data, labels, batch_size): #beatmaps_data and extra_data must match
    max_length = len(beatmaps_data)-1
    ids = [random.randint(0,max_length) for _ in range(batch_size)]
    swapx = random.random()
    swapy = random.random()
    swapx_ = False if swapx<0.5 else True
    swapy_ = False if swapy<0.5 else True
    batch = []
    y = []
    extra = []
    for id in ids:
        y.append(one_hot_labels(labels[id]))
        extra.append(extra_data[id])
        sequence_length = beatmaps_data[id].shape[0]-1
        start = random.randint(0,sequence_length-MAX_SEQ_LEN)
        beatmap = beatmaps_data[id][start:start+MAX_SEQ_LEN]

        if swapx_:
            beatmap[:,0] = 1-beatmap[:,0]
        swapx_ = not swapx_
        if swapy_:
            beatmap[:,1] = 1-beatmap[:,1]
        swapy_ = not swapy_
        
        batch.append(beatmap)
    return np.array(batch), np.array(extra), np.array(y)


def get_beatmap_from_website(id):
    url = f"https://osu.ppy.sh/osu/{id}"
    response = requests.get(url)
    if response.status_code == 200:
        text_file = io.StringIO(response.text.replace('\r', '')).readlines()
        beatmap = obp.Beatmap.str_to_beatmap(text_file)
    else:
        print("Could not get Beatmap with id: " + id)
        return None

    return beatmap




def visualize_beatmap_skillsets(model, beatmap:obp.Beatmap, dt:bool=False):
    def median_filter_columns(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        pad = kernel_size // 2
        
        # Expand to 3D: (1, rows, columns)
        x_3d = x.unsqueeze(0)
        # Pad the rows (dim=1), no padding on columns (dim=2)
        x_padded = f.pad(x_3d, (0, 0, pad, pad), mode='reflect')
        x_padded = x_padded.squeeze(0)  # Back to 2D

        # Unfold rows
        windows = x_padded.unfold(dimension=0, size=kernel_size, step=1)
        return windows.median(dim=2).values

    beatmap_sequence, extra = beatmap.beatmap_to_data()
    if dt:
        apply_dt(beatmap_sequence, extra)
            
    beatmap_sequence = torch.tensor(beatmap_sequence, dtype=torch.float32)
    extra = torch.tensor(extra, dtype=torch.float32)
    #print(extra)
    skillspread = []
    step_size = MAX_SEQ_LEN//20
    skillvals = torch.zeros(((beatmap_sequence.shape[0]-MAX_SEQ_LEN)//step_size+1,len(SKILLSET_LABELS)))
    for i,j in enumerate(range(0,beatmap_sequence.shape[0]-MAX_SEQ_LEN-1,step_size)):
        #print(beatmap_sequence[None,j:j+MAX_SEQ_LEN,:].shape)
        prediction = model(beatmap_sequence[None,None,j:j+MAX_SEQ_LEN,:], extra[None,:])
        skillvals[i] = prediction
        skillspread.append(decode_output(prediction))

    
    sm = nn.Softmax(dim=-1)
    kernel_size=5

    skillvals_filtered = median_filter_columns(skillvals, kernel_size)
    skillvals_filtered = torch.round(skillvals_filtered, decimals=1)
    skillvals_filtered = sm(skillvals_filtered)

    #time = [b.hit_object.time for b in beatmap.beatmap_objects]
    max_time = max(map(lambda b: b.hit_object.time, beatmap.beatmap_objects))
    min_time = min(map(lambda b: b.hit_object.time, beatmap.beatmap_objects))
    b_time = np.linspace(min_time,max_time, len(skillvals_filtered))
    b_time = b_time/1000/60

    skillvaluespread = skillvals_filtered.cpu().detach().numpy()
    bottom = np.zeros(len(b_time))

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(SKILLSET_LABELS))]

    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 3])
    fig.suptitle(beatmap.metadata.title)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.grid(color="gray", linestyle="--", alpha=0.2)
    ax1.margins(x=0, y=0)
    ax1.set_ylim(0,1)

    for i,la in enumerate(SKILLSET_LABELS):
        ax1.plot(b_time,skillvaluespread[:,i], label=la)
        ax1.fill_between(b_time, skillvaluespread[:,i],bottom, alpha=0.2)
    ax1.set_xticks((np.linspace(min(b_time),max(b_time),15)).round(2))
    ax1.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.stackplot(b_time,
                       skillvaluespread[:,0],skillvaluespread[:,1],skillvaluespread[:,2],skillvaluespread[:,3],
                       skillvaluespread[:,4],skillvaluespread[:,5],
                       labels=SKILLSET_LABELS, baseline="zero", alpha=0.9, edgecolor="black", linewidth=1)
    ax2.set_xticks((np.linspace(min(b_time),max(b_time),15)).round(2))
    ax2.grid(color="gray", linestyle="--", alpha=0.1)
    ax2.margins(x=0, y=0)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    ax3 = fig.add_subplot(gs[:, 0])
    ax3.barh(y=SKILLSET_LABELS, width=np.sum(skillvaluespread, axis=0)/np.sum(skillvaluespread), color=colors, edgecolor="black")
    ax3.invert_yaxis()
    ax3.invert_xaxis()
    yticklabels = ax3.get_yticklabels()
    for label, color in zip(yticklabels, colors):
        label.set_color(color)
        label.set_fontsize('x-large')
    #ax3.yaxis.set_label_position("right")
    #ax3.yaxis.tick_right()

    plt.tight_layout()
    plt.show()
    time.sleep(1)

fig_dict = {}
def visualize_beatmap_skillsets_live(model, beatmap:obp.Beatmap, dt: bool = False, update: bool = True):
    """
    Visualize skillsets for a beatmap. If `update=True`, it updates an existing plot instead of creating a new one.
    """
    # --- Helper ---
    def median_filter_columns(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        pad = kernel_size // 2
        x_3d = x.unsqueeze(0)
        x_padded = f.pad(x_3d, (0, 0, pad, pad), mode='reflect')
        x_padded = x_padded.squeeze(0)
        windows = x_padded.unfold(dimension=0, size=kernel_size, step=1)
        return windows.median(dim=2).values

    # --- Data processing ---
    beatmap_sequence, extra = beatmap.beatmap_to_data()
    if dt:
        apply_dt(beatmap_sequence, extra)
            
    beatmap_sequence = torch.tensor(beatmap_sequence, dtype=torch.float32)
    extra = torch.tensor(extra, dtype=torch.float32)

    step_size = MAX_SEQ_LEN // 20
    skillvals = torch.zeros(((beatmap_sequence.shape[0]-MAX_SEQ_LEN)//step_size+1,len(SKILLSET_LABELS)))
    
    for i,j in enumerate(range(0, beatmap_sequence.shape[0]-MAX_SEQ_LEN-1, step_size)):
        prediction = model(beatmap_sequence[None,None,j:j+MAX_SEQ_LEN,:], extra[None,:])
        skillvals[i] = prediction

    sm = nn.Softmax(dim=-1)
    skillvals_filtered = median_filter_columns(skillvals, kernel_size=5)
    skillvals_filtered = torch.round(skillvals_filtered, decimals=1)
    skillvals_filtered = sm(skillvals_filtered)
    skillvaluespread = skillvals_filtered.cpu().detach().numpy()

    # --- Time axis ---
    max_time = max(b.hit_object.time for b in beatmap.beatmap_objects)
    min_time = min(b.hit_object.time for b in beatmap.beatmap_objects)
    b_time = np.linspace(min_time, max_time, len(skillvaluespread)) / 1000 / 60  # minutes

    colors = [plt.get_cmap("tab10")(i) for i in range(len(SKILLSET_LABELS))]

    # --- Create figure once, then update ---
    if 'fig' not in fig_dict or not update:
        plt.ion()
        fig = plt.figure(figsize=(18, 5))
        fig_dict['fig'] = fig
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 3])
        fig_dict['gs'] = gs
        fig.suptitle(beatmap.metadata.title)

        # Axes
        fig_dict['ax1'] = fig.add_subplot(gs[0, 1])
        fig_dict['ax2'] = fig.add_subplot(gs[1, 1])
        fig_dict['ax3'] = fig.add_subplot(gs[:, 0])
    else:
        fig = fig_dict['fig']
    
    ax1 = fig_dict['ax1']
    ax2 = fig_dict['ax2']
    ax3 = fig_dict['ax3']

    # --- Clear previous plots ---
    ax1.cla()
    ax2.cla()
    ax3.cla()

    # --- Line + fill plot ---
    ax1.grid(color="gray", linestyle="--", alpha=0.2)
    ax1.margins(x=0, y=0)
    ax1.set_ylim(0,1)
    for i, la in enumerate(SKILLSET_LABELS):
        ax1.plot(b_time, skillvaluespread[:,i], label=la)
        ax1.fill_between(b_time, skillvaluespread[:,i], 0, alpha=0.2)
    ax1.set_xticks(np.linspace(min(b_time), max(b_time), 15).round(2))
    ax1.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()

    # --- Stackplot ---
    ax2.stackplot(b_time, *[skillvaluespread[:,i] for i in range(len(SKILLSET_LABELS))],
                  labels=SKILLSET_LABELS, baseline="zero", alpha=0.9, edgecolor="black", linewidth=1)
    ax2.set_xticks(np.linspace(min(b_time), max(b_time), 15).round(2))
    ax2.grid(color="gray", linestyle="--", alpha=0.1)
    ax2.margins(x=0, y=0)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    # --- Horizontal bar plot ---
    ax3.barh(y=SKILLSET_LABELS, width=np.sum(skillvaluespread, axis=0)/np.sum(skillvaluespread),
             color=colors, edgecolor="black")
    ax3.invert_yaxis()
    ax3.invert_xaxis()
    yticklabels = ax3.get_yticklabels()
    for label, color in zip(yticklabels, colors):
        label.set_color(color)
        label.set_fontsize('x-large')

    plt.tight_layout()
    plt.draw()
    plt.pause(0.2)  # small pause to update the plot