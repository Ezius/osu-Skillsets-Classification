from abc import ABC
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

PLAYFIELD_SIZE = (512, 384)

class HitObject(ABC):
    def __init__(self, x, y, time, type):
        #self.string = ""
        self.x = x
        self.y = y
        self.time = time
        self.type = type
        #self.hit_sound = hit_sound
        #self.hit_sample = hit_sample

    def __repr__(self):
        return str(self.__dict__)
    
    @staticmethod
    def str_to_hit_object(string):
        hit_object_elems = string.split(',')
        if len(hit_object_elems)==6:
            return Circle(string)
        elif len(hit_object_elems)==7:
            return Spinner(string)
        else:
            return Slider(string)

class Circle(HitObject):
    def __init__(self, circle_string:str):
        self.string = circle_string
        circle_elems = circle_string.split(',')
        super().__init__(int(circle_elems[0]), int(circle_elems[1]), int(circle_elems[2]), int(circle_elems[3]))
    def __str__(self):
        return f"{self.string}"

class Slider(HitObject):
    def __init__(self,slider_string):
        self.string = slider_string
        slider_elems = slider_string.split(",") 
        super().__init__(int(slider_elems[0]), int(slider_elems[1]), int(slider_elems[2]), int(slider_elems[3]))
        type_points = slider_elems[5].split('|')
        self.curve_type = type_points[0]
        self.curve_points = [(self.x,self.y)] + [tuple(map(int, t.split(","))) for t in [sub.replace(":",",") for sub in slider_elems[5].split('|')[1:]]]
        self.repeats = int(slider_elems[6])
        self.length = float(slider_elems[7])
        #self.edge_sounds = slider_elems[8]
        #self.edge_sets = slider_elems[9]

    def __str__(self):
        return f"{self.string}"
        
    def slider_parts(self):
        slider_f = []
        slider_parts = []
        for i,p in enumerate(self.curve_points):
            if i==len(self.curve_points)-1:
                slider_f = [p] + slider_f 
                slider_parts = [slider_f] + slider_parts 
                return slider_parts
            slider_f = [p] + slider_f 
            if self.curve_points[i] == self.curve_points[i+1]:
                slider_parts = [slider_f] + slider_parts 
                slider_f = []
    
    def draw_slider(self):
        values = self.get_slider_path()
        plt.xlim(-10, PLAYFIELD_SIZE[0]+10)
        plt.ylim(PLAYFIELD_SIZE[1]+10, -10)
        plt.plot(values[:,0], values[:,1])
    
    def get_slider_path(self):
        if self.curve_type=="P":
            return np.array(self.perfect_curve(self.curve_points))
        elif self.curve_type == "L":
            return np.linspace(self.curve_points[0], self.curve_points[1], 64)
        else:
            return np.array([self.bernstein_bezier(part) for part in self.slider_parts()]).reshape(-1,2)[::-1]
        
    def bernstein_bezier(self, points, cells=150):
        x, y = zip(*points)
        x = np.array(x)
        y = np.array(y)
        n = len(points) - 1
        t = np.linspace(0, 1, cells)

        bezier = np.zeros((cells,2))

        # Precompute binomial coefficients and reuse basis functions
        binomials = [math.comb(n, i) for i in range(n + 1)]
        for i in range(n + 1):
            basis = binomials[i] * (t ** i) * ((1 - t) ** (n - i))
            bezier[:,0] += basis * x[i]
            bezier[:,1] += basis * y[i]
        return bezier
    
    def get_circle_center(self, a, b, c, epsilon=1e-10):
        """Compute the center of the circle through points a, b, and c."""
        A = b - a
        B = c - a
        A_perp = np.array([-A[1], A[0]])
        B_perp = np.array([-B[1], B[0]])

        mid_ab = (a + b) / 2
        mid_ac = (a + c) / 2

        # Solve for intersection of the perpendicular bisectors
        mat = np.column_stack((A_perp, -B_perp))
        rhs = mid_ac - mid_ab
        if abs(np.linalg.det(mat)) < epsilon:
            # Points are colinear or too close to calculate a unique circle
            return None
        t = np.linalg.solve(mat, rhs)
        center = mid_ab + t[0] * A_perp
        return center

    def perfect_curve(self, points, num_points=150):
        """Generate arc points from a to c via b (assuming circular arc)."""
        a, b, c = [np.array(p) for p in points]
        center = self.get_circle_center(a, b, c)

        if center is None:
            return np.linspace(a,c,num_points)

        r = np.linalg.norm(a - center)

        def angle(p):
            return np.arctan2(p[1] - center[1], p[0] - center[0])

        angle_a = angle(a)
        angle_b = angle(b)
        angle_c = angle(c)

        # Ensure the arc goes through b
        angles = np.linspace(angle_a, angle_c, num_points)
        if not (angle_b > min(angle_a, angle_c) and angle_b < max(angle_a, angle_c)):
            if angle_c < angle_a: angle_c += 2 * np.pi
            else: angle_a += 2 * np.pi
            angles = np.linspace(angle_a, angle_c, num_points)

        arc_points = np.array([    [center[0] + r * np.cos(theta), center[1] + r * np.sin(theta)]    for theta in angles])
        return np.array(arc_points)
        
class Spinner(HitObject):
    def __init__(self, spinner_string:str):
        self.string = spinner_string
        spinner_elems = spinner_string.split(',')
        super().__init__(int(spinner_elems[0]), int(spinner_elems[1]), int(spinner_elems[2]), int(spinner_elems[3]))
        self.end_time = int(spinner_elems[5])

    def __str__(self):
        return f"{self.string}"
    
class Difficulty:
    def __init__(self, hp:float, cs:float, od:float, ar:float, slider_velocity:float, slider_tick_rate:float):
        self.hp = hp
        self.cs = cs
        self.od = od
        self.ar = ar
        self.slider_velocity = slider_velocity
        self.slider_tick_rate = slider_tick_rate

    def __repr__(self):
        return str(self.__dict__)

class TimingPoint:
    def __init__(self, timing_point_str:str):
        self.string = timing_point_str
        timing_data = timing_point_str.split(',')
        self.time = int(float(timing_data[0]))
        self.beat_length = float(timing_data[1])
        # self.meter = int(timing_data[2])
        # self.sample_set = int(timing_data[3])
        # self.sample_index = int(timing_data[4])
        # self.volume = int(timing_data[5])
        self.uninherited = int(timing_data[6])
        # self.effects = int(timing_data[7])
    def __repr__(self):
        return str(self.__dict__)
    def __str__(self):
        return f"{self.string}"
    
class BeatMapObject:
    def __init__(self, hit_object:HitObject, timing_point:TimingPoint, beat_duration:float):
        self.beat_duration = beat_duration
        self.hit_object = hit_object
        self.timing_point = timing_point

    def __repr__(self):
        return str(self.__dict__)
        
    def get_bpm(self):
        return 1 / self.beat_duration * 1000 * 60
    
    def get_sv(self):
        if self.timing_point.uninherited==0:
            return abs(100/self.timing_point.beat_length)
        else:
            return 1

class Metadata:
    def __init__(self, title, artist, creator, dif_name, beatmap_id, set_id):
        self.title = title
        self.artist = artist
        self.creator = creator
        self.dif_name = dif_name
        self.beatmap_id = int(beatmap_id)
        self.set_id = int(set_id)
    def __repr__(self):
        return str(self.__dict__)

class Beatmap:
    def __init__(self, metadata:Metadata, difficulty:Difficulty, beatmap_objects:list[BeatMapObject]):
        self.metadata = metadata
        self.beatmap_objects = beatmap_objects
        self.difficulty = difficulty
    def __repr__(self):
        return str(self.__dict__)
    
    def create_datapoint(self, x, y, time, bpm, datatype: int):
        '''
        0:pos_x
        1:pos_y
        2:distance_to_previous_object
        3:time
        4:deltatime
        5:bpm

        0:circle   --one hot from here with previous_index+typeindex
        1:slider
        2:slider point
        3:slider repeat #* i had the most amount of cancer trying to implement this and the perfect curve slider but it hopefully works now
        4:slider end
        5:spinner
        6:spinner end
        '''
        features = np.zeros(13)
        features[0] = x
        features[1] = y
        #features[2] = 0
        features[3] = time
        #features[4] = 0
        features[5] = bpm
        features[datatype + 6] = 1
        return features
    
    def create_extra_info(self):
        return np.array([self.difficulty.ar])

    def beatmap_to_data(self, normalize=True):
        data_points=[]
        for ob in self.beatmap_objects:
            hit_object = ob.hit_object
            bpm = ob.get_bpm()

            if isinstance(hit_object, Circle):
                data_points.append(self.create_datapoint(hit_object.x, hit_object.y,  hit_object.time, bpm, 0))

            elif isinstance(hit_object, Slider):
                data_points.append(self.create_datapoint(hit_object.x, hit_object.y, hit_object.time, bpm, 1))
                ticks, end = self.get_slider_ticks(ob)
                repeats = hit_object.repeats
                duration_per_repeat = self.slider_duration(ob)
                previous_time = hit_object.time
                total_difference = hit_object.time
                rounds = 1
                while (repeats >= 1):
                    for tick in ticks:
                        current_time = tick[2]
                        #print(f"previous_time1 {previous_time:.2f}, current_time1 {current_time:.2f}, DIFFERENCE:  {np.abs(current_time-previous_time):.2f}")
                        total_difference = total_difference + np.abs(tick[2]+-previous_time)
                        data_points.append(self.create_datapoint(tick[0], tick[1], total_difference, bpm, 2))
                        previous_time = current_time
                    if repeats<=1:
                        if hit_object.repeats % 2 == 0:                            
                            current_time = hit_object.time
                            total_difference = total_difference + np.abs(current_time-previous_time)
                            data_points.append(self.create_datapoint(hit_object.x, hit_object.y, total_difference, bpm, 4))
                            previous_time = current_time
                        else:                            
                            current_time = hit_object.time+duration_per_repeat
                            total_difference = total_difference + np.abs(current_time-previous_time)
                            data_points.append(self.create_datapoint(end[0], end[1], total_difference, bpm, 4))
                            previous_time = current_time
                    else:
                        if repeats % 2 == 1 - (hit_object.repeats % 2):
                            current_time = hit_object.time
                            total_difference = total_difference + np.abs(current_time-previous_time)
                            data_points.append(self.create_datapoint(hit_object.x, hit_object.y, total_difference, bpm, 3))
                            previous_time = current_time
                        else:
                            current_time = hit_object.time+duration_per_repeat
                            total_difference = total_difference + np.abs(current_time-previous_time)
                            data_points.append(self.create_datapoint(end[0], end[1], total_difference, bpm, 3))
                            previous_time = current_time

                    rounds += 1
                    repeats -= 1
                    ticks = ticks[::-1]

            elif isinstance(hit_object, Spinner):
                data_points.append(self.create_datapoint(hit_object.x, hit_object.y, hit_object.time, bpm, 5))
                data_points.append(self.create_datapoint(hit_object.x, hit_object.y, hit_object.end_time, bpm, 6))

        data_points = np.array(data_points)
        extra_data = self.create_extra_info()

        if normalize:
            self.normalize_data(data_points, extra_data)

        diffs = data_points[:,:2][1:] - data_points[:,:2][:-1]
        data_points[1:,2] = np.linalg.norm(diffs, axis=1)

        dtime = np.abs(data_points[:,3][1:] - data_points[:,3][:-1])
        data_points[1:,4] = dtime
        return data_points, extra_data
    
    def normalize_data(self, data, extra_data):
        data[:,0] /= PLAYFIELD_SIZE[0]
        data[:,1] /= PLAYFIELD_SIZE[1]
        data[:,3] -= data[:,3].min()
        data[:,3:5] /= 180000 #3 minutes

        extra_data[0] /= 10
        

    
    def slider_duration(self, beat_object:BeatMapObject):
        hit_object:Slider = beat_object.hit_object
        return hit_object.length/(self.difficulty.slider_velocity*100*beat_object.get_sv())*beat_object.beat_duration


    def get_slider_ticks(self, beat_object:BeatMapObject):
        hit_object:Slider = beat_object.hit_object
        sd = self.slider_duration(beat_object)
        tick_amount = sd / (beat_object.beat_duration/self.difficulty.slider_tick_rate)
        tick_length = hit_object.length / tick_amount
        tick_count = 1
        ticks = []
        points = hit_object.get_slider_path()
        total_length = 0
        diffs = points[1:] - points[:-1]
        delta_length = np.linalg.norm(diffs,axis=1)
        for i in range(len(points)-1):
            if total_length >= tick_length * tick_count:
                tick_count += 1
                ticks.append((points[i][0], points[i][1], total_length/hit_object.length*sd + hit_object.time))
            total_length = total_length + delta_length[i]
            if total_length>sd:
                break
        end = (points[i][0], points[i][1], total_length/hit_object.length*sd + hit_object.time)
        return np.array(ticks), np.array(end)
        
    @staticmethod
    def str_to_beatmap(string):

        metadata_keys = {
            'title': 'Title:',
            'artist': 'Artist:',
            'creator': 'Creator:',
            'dif_name': 'Version:',
            'beatmap_id': 'BeatmapID:',
            'set_id': 'BeatmapSetID:'
        }

        difficulty_keys = {
            'hp': 'HPDrainRate:',
            'cs': 'CircleSize:',
            'od': 'OverallDifficulty:',
            'ar': 'ApproachRate:',
            'slider_velocity': 'SliderMultiplier:',
            'slider_tick_rate': 'SliderTickRate:'
        }

        raw_map = string
        timing_points_index = raw_map.index("[TimingPoints]\n")
        objects_index = raw_map.index('[HitObjects]\n')
        metadata_index = raw_map.index('[Metadata]\n')
        difficulty_index = raw_map.index('[Difficulty]\n')

        metadata_values = {key: next((line.split(prefix)[-1].strip() for line in raw_map[metadata_index + 1:difficulty_index] if line.startswith(prefix)), None)
                            for key, prefix in metadata_keys.items()}
        difficulty_values = {key: float(next((line.split(prefix)[-1].strip() for line in raw_map[difficulty_index + 1:] if line.startswith(prefix)), None))
                                for key, prefix in difficulty_keys.items()}
        metadata = Metadata(**metadata_values)
        difficulty = Difficulty(**difficulty_values)

        timing_points:list[TimingPoint] = []
        for tp in raw_map[timing_points_index+1:]:
            if tp=="\n": break
            timing_points.append(TimingPoint(tp))
        beat_duration = timing_points[0].beat_length
        tp_idx = 0
        beatmap_objects = []
        for hit_obj in raw_map[objects_index+1:]:
            if hit_obj == "\n": break
            hit_object = HitObject.str_to_hit_object(hit_obj)
            while (tp_idx + 1 < len(timing_points) and timing_points[tp_idx].time <= hit_object.time):
                if timing_points[tp_idx].beat_length >= 0:
                    beat_duration = timing_points[tp_idx].beat_length
                tp_idx += 1

            beatmap_objects.append(BeatMapObject(hit_object, timing_points[tp_idx], beat_duration))
        return Beatmap(metadata, difficulty, beatmap_objects)

    @staticmethod
    def file_to_beatmap(file_location):
        raw_map = open(file_location, "r", encoding="utf-8", errors="ignore").readlines()
        return Beatmap.str_to_beatmap(raw_map)