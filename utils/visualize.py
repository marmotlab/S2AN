#!/usr/bin/env python3
import yaml
# matplotlib.use("Agg")
from matplotlib.patches import Circle, Rectangle, Arrow
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import argparse
import sys, os
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, ".."))

from map_generator import MapLoader

#log: change to moving ai map and scene save style.
# original saving style is not allowed.
# schedule use the yaml as usually. 


Colors = ['orange']#, 'blue', 'green']

# add interface to moving ai scen and map
def load_movingai2dict(map_fname, scen_fname, num_agents):
  loader = MapLoader()
  obtacles = loader.read_map(map_fname).astype(int)
  start_locs, goal_locs = loader.read_scen(scen_fname, num_agents)
  start_locs = start_locs.astype(int)
  goal_locs = goal_locs.astype(int)
  map = dict()
  map["map"] = dict()
  map["map"]["dimensions"] = obtacles.shape
  obs_locs = np.stack(np.where(obtacles == -1)).T
  map["map"]["obstacles"] = obs_locs
  map["agents"]= [{
    "goal": goal_locs[i, :], 
    "name":"agent"+str(i), "start": start_locs[i, :]} 
    for i in range(num_agents)]
  return map

class  Scene:
  def __init__(self, map, schedule, scale=1) :
    self.map = map
    self.schedule = schedule

    aspect = map["map"]["dimensions"][0] / map["map"]["dimensions"][1]

    self.fig = plt.figure(frameon=False, figsize=(4 * aspect, 4))
    self.ax = self.fig.add_subplot(111, aspect='equal')
    self.fig.subplots_adjust(left=0,right=1,bottom=0,top=1, wspace=None, hspace=None)
    # self.ax.set_frame_on(False)

    self.patches = []
    self.artists = []
    self.agents = dict()
    self.start_text = dict()
    self.goal_text = dict()
    # create boundary patch
    xmin = -0.5
    ymin = -0.5
    xmax = int(map["map"]["dimensions"][0]/scale) - 0.5
    ymax = int(map["map"]["dimensions"][1]/scale) - 0.5

    # self.ax.relim()
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    # self.ax.set_xticks([])
    # self.ax.set_yticks([])
    # plt.axis('off')
    # self.ax.axis('tight')
    # self.ax.axis('off')

    self.patches.append(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor='none', edgecolor='red'))
    for o in map["map"]["obstacles"]:
      x, y = int(o[0]/scale), int(o[1]/scale)
      self.patches.append(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='red', edgecolor='red'))

    # create agents:
    self.T = 0
    # draw goals first
    for d, i in zip(map["agents"], range(0, len(map["agents"]))):
      if "goal" in d:
        goals = [d["goal"]]
      if "potentialGoals" in d:
        goals = [goal for goal in d["potentialGoals"]]
      for goal in goals:
        self.patches.append(Rectangle((int(goal[0]/scale) - 0.5,int(goal[1]/scale) - 0.5), 1, 1, facecolor=Colors[i%len(Colors)], edgecolor='black', alpha=0.5))

    for d, i in zip(map["agents"], range(0, len(map["agents"]))):
      name = d["name"]
      self.agents[name] = Circle((int(d["start"][0]/scale), int(d["start"][1]/scale)), 0.5, facecolor=Colors[i%len(Colors)], edgecolor='black')
      self.agents[name].original_face_color = Colors[i%len(Colors)]
      self.patches.append(self.agents[name])
      self.T = 1

      self.start_text[name] = self.ax.text(int(d["start"][0]/scale), int(d["start"][1]/scale), name.replace('agent', ''))
      self.start_text[name].set_horizontalalignment('center')
      self.start_text[name].set_verticalalignment('center')
      self.artists.append(self.start_text[name])

      self.goal_text[name] = self.ax.text(int(d["goal"][0]/scale), int(d["goal"][1]/scale), name.replace('agent', ''))
      self.goal_text[name].set_horizontalalignment('center')
      self.goal_text[name].set_verticalalignment('center')
      self.artists.append(self.goal_text[name])

    for patch in self.patches:
        self.ax.add_patch(patch)
    plt.show()
    print('hold on')

class Animation:
  def __init__(self, map, schedule):
    self.map = map
    self.schedule = schedule

    aspect = map["map"]["dimensions"][0] / map["map"]["dimensions"][1]

    self.fig = plt.figure(frameon=False, figsize=(4 * aspect, 4))
    self.ax = self.fig.add_subplot(111, aspect='equal')
    self.fig.subplots_adjust(left=0,right=1,bottom=0,top=1, wspace=None, hspace=None)
    # self.ax.set_frame_on(False)

    self.patches = []
    self.artists = []
    self.agents = dict()
    self.agent_names = dict()
    # create boundary patch
    xmin = -0.5
    ymin = -0.5
    xmax = (map["map"]["dimensions"][0]) - 0.5
    ymax = (map["map"]["dimensions"][1]) - 0.5

    # self.ax.relim()
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    # self.ax.set_xticks([])
    # self.ax.set_yticks([])
    # plt.axis('off')
    # self.ax.axis('tight')
    # self.ax.axis('off')

    self.patches.append(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor='none', edgecolor='red'))
    for o in map["map"]["obstacles"]:
      x, y = o[0], o[1]
      self.patches.append(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='red', edgecolor='red'))

    # create agents:
    self.T = 0
    # draw goals first
    for d, i in zip(map["agents"], range(0, len(map["agents"]))):
      if "goal" in d:
        goals = [d["goal"]]
      if "potentialGoals" in d:
        goals = [goal for goal in d["potentialGoals"]]
      for goal in goals:
        self.patches.append(Rectangle((goal[0] - 0.25, goal[1] - 0.25), 0.5, 0.5, facecolor=Colors[i%len(Colors)], edgecolor='black', alpha=0.5))

    for d, i in zip(map["agents"], range(0, len(map["agents"]))):
      name = d["name"]
      self.agents[name] = Circle((d["start"][0], d["start"][1]), 0.3, facecolor=Colors[i%len(Colors)], edgecolor='black')
      self.agents[name].original_face_color = Colors[i%len(Colors)]
      self.patches.append(self.agents[name])
      self.T = max(self.T, schedule["schedule"][name][-1]["t"])
      self.agent_names[name] = self.ax.text(d["start"][0], d["start"][1], name.replace('agent', ''))
      self.agent_names[name].set_horizontalalignment('center')
      self.agent_names[name].set_verticalalignment('center')
      self.artists.append(self.agent_names[name])

    # self.ax.set_axis_off()
    # self.fig.axes[0].set_visible(False)
    # self.fig.axes.get_yaxis().set_visible(False)

    # self.fig.tight_layout()

    self.anim = animation.FuncAnimation(self.fig, self.animate_func,
                               init_func=self.init_func,
                               frames=int(self.T+1) * 4,
                               interval=100,
                               blit=True)

  def save(self, file_name, speed):
    self.anim.save(
      file_name,
      "ffmpeg",
      fps=10 * speed,
      dpi=200),
      # savefig_kwargs={"pad_inches": 0, "bbox_inches": "tight"})

  def show(self):
    plt.show()

  def init_func(self):
    for p in self.patches:
      self.ax.add_patch(p)
    for a in self.artists:
      self.ax.add_artist(a)
    return self.patches + self.artists

  def animate_func(self, i):
    for agent_name in self.schedule["schedule"]:
      agent = schedule["schedule"][agent_name]
      pos = self.getState(i / 2, agent)
      p = (pos[0], pos[1])
      self.agents[agent_name].center = p
      self.agent_names[agent_name].set_position(p)

    # reset all colors
    for _,agent in self.agents.items():
      agent.set_facecolor(agent.original_face_color)

    # check drive-drive collisions
    agents_array = [agent for _,agent in self.agents.items()]
    for i in range(0, len(agents_array)):
      for j in range(i+1, len(agents_array)):
        d1 = agents_array[i]
        d2 = agents_array[j]
        pos1 = np.array(d1.center)
        pos2 = np.array(d2.center)
        if np.linalg.norm(pos1 - pos2) < 0.7:
          d1.set_facecolor('red')
          d2.set_facecolor('red')
          print("COLLISION! (agent-agent) ({}, {})".format(i, j))

    return self.patches + self.artists


  def getState(self, t, d):
    idx = 0
    while idx < len(d) and d[idx]["t"] < t:
      idx += 1
    if idx == 0:
      return np.array([float(d[0]["x"]), float(d[0]["y"])])
    elif idx < len(d):
      posLast = np.array([float(d[idx-1]["x"]), float(d[idx-1]["y"])])
      posNext = np.array([float(d[idx]["x"]), float(d[idx]["y"])])
    else:
      return np.array([float(d[-1]["x"]), float(d[-1]["y"])])
    dt = d[idx]["t"] - d[idx-1]["t"]
    t = (t - d[idx-1]["t"]) / dt
    pos = (posNext - posLast) * t + posLast
    return pos



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--map", '-m', help="input file containing map", required=True)
  parser.add_argument("--scene", '-v', help="moving ai scene name", required=True)
  parser.add_argument("--num_agents", '-n', type=int, help="number of angents to load", required=True)

  parser.add_argument("--schedule", '-s',help="schedule for agents")

  parser.add_argument('--video', dest='video', default=None, help="output video file (or leave empty to show on screen)")
  parser.add_argument("--speed", type=int, default=1, help="speedup-factor")
  parser.add_argument("--scale", type=float, default=1, help="scale_size")
  args = parser.parse_args()
  
  num_agents = args.num_agents

  map = load_movingai2dict(args.map, args.scene, num_agents)


  # arg.schedule not input. visualize the problem instead.
  if not args.schedule:
    gs = args.scale  # grid size
    map_sizex = int(map['map']['dimensions'][0]//gs) + 1
    map_sizey = int(map['map']['dimensions'][1]//gs) + 1
    scene = np.ones([map_sizex, map_sizey, 3])
    obs = map['map']['obstacles']
    for ob in obs: 
      x = int(ob[0] //gs)
      y = int(ob[1] //gs)      
      scene[x, y, :] = 255
    agents = map['agents']
    # for agent in agents:
    animation = Scene(map, 1,gs )

  else:
    with open(args.schedule) as states_file:
      schedule = yaml.safe_load(states_file)

    animation = Animation(map, schedule)

    if args.video:
      animation.save(args.video, args.speed)
    else:
      animation.show()
