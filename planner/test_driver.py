import os
import ray
import torch
import yaml
import numpy as np
import heapq

from .model import PolicyNet
from .env import Env
from .agent import Agent
from .utils import *
from .parameter import *
from .sensor import sensor_work
from mapinpaint.networks import Generator
from mapinpaint.evaluator import Evaluator

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

# Other configuration settings in parameter.py
NUM_TEST = 100
NUM_META_AGENT = 4  # number of parallel tests, NUM_TEST % NUM_META_AGENT should be 0
SAFE_MODE = True
SAVE_GIFS = True

# ===== Multi-robot settings =====
NUM_ROBOTS = 3
COMM_RANGE = 32.0

# ===== Global planner behavior =====
GLOBAL_REPLAN_EVERY = 1  # connected -> replan every k steps

# NOTE: we repurpose these as "current round budget" in terms of number of remaining cells
NEXT_TARGET_WEIGHT_PER_ROBOT = 8000
NEXT_MAX_BFS_LAYER = 10000

MAX_JUMP_FACTOR = 3.0

# ===== Round completion / RV reaching =====
RV_REACHED_DIST = 5.0 * NODE_RESOLUTION

# ===== Stuck guard =====
STUCK_EPS = 0.08 * NODE_RESOLUTION
STUCK_LIMIT = 12

# ===== Frontier -> standoff free selection =====
STANDOFF_NEIGHBORS_8 = True

if SAVE_GIFS:
    os.makedirs(gif_path, exist_ok=True)


def run_test():
    device = torch.device('cpu')
    global_network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)

    print(f"Testing on {device}, model: {model_path}, num of tests: {NUM_TEST}, num of samples: {N_GEN_SAMPLE}")
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(f'{model_path}/checkpoint.pth', weights_only=True, map_location=device)
    global_network.load_state_dict(checkpoint['policy_model'])

    meta_agents = [Runner.remote(i) for i in range(NUM_META_AGENT)]
    weights = global_network.state_dict()
    curr_test = 0

    travel_dist = []
    explored_rate = []
    success_rate = []
    sr_room = []
    sr_tunnel = []
    sr_outdoor = []
    td_room = []
    td_tunnel = []
    td_outdoor = []

    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        job_list.append(meta_agent.job.remote(weights, curr_test))
        curr_test += 1

    try:
        while len(travel_dist) < curr_test:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)

            for job in done_jobs:
                metrics, info = job
                travel_dist.append(metrics['travel_dist'])
                explored_rate.append(metrics['explored_rate'])
                success_rate.append(metrics['success_rate'])

                if 'room' in info['map_path']:
                    sr_room.append(metrics['success_rate'])
                    td_room.append(metrics['travel_dist'])
                elif 'tunnel' in info['map_path']:
                    sr_tunnel.append(metrics['success_rate'])
                    td_tunnel.append(metrics['travel_dist'])
                elif 'outdoor' in info['map_path']:
                    sr_outdoor.append(metrics['success_rate'])
                    td_outdoor.append(metrics['travel_dist'])

                if curr_test < NUM_TEST:
                    job_list.append(meta_agents[info['id']].job.remote(weights, curr_test))
                    curr_test += 1

        print('=====================================')
        print('| Test:', FOLDER_NAME)
        print('| Total test: {} with {} predictions'.format(NUM_TEST, N_GEN_SAMPLE))
        print('| Average success rate:', np.array(success_rate).mean())
        print('| Average travel distance:', np.array(travel_dist).mean())
        print('| Average explored rate:', np.array(explored_rate).mean())
        print('| Room success rate: {}, travel distance: {:.2f} ± {:.2f}'.format(
            np.mean(sr_room) if len(sr_room) > 0 else np.nan,
            np.mean(td_room) if len(td_room) > 0 else np.nan,
            np.std(td_room) if len(td_room) > 0 else np.nan))
        print('| Tunnel success rate: {}, travel distance: {:.2f} ± {:.2f}'.format(
            np.mean(sr_tunnel) if len(sr_tunnel) > 0 else np.nan,
            np.mean(td_tunnel) if len(td_tunnel) > 0 else np.nan,
            np.std(td_tunnel) if len(td_tunnel) > 0 else np.nan))
        print('| Outdoor success rate: {}, travel distance: {:.2f} ± {:.2f}'.format(
            np.mean(sr_outdoor) if len(sr_outdoor) > 0 else np.nan,
            np.mean(td_outdoor) if len(td_outdoor) > 0 else np.nan,
            np.std(td_outdoor) if len(td_outdoor) > 0 else np.nan))

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)

class MultiTestWorker:

    MODE_EXPLORE = 0
    MODE_GO_RV = 1

    PHASE_CONNECTED = "CONNECTED"
    PHASE_SILENT = "SILENT"

    def __init__(self, meta_agent_id, policy_net, predictor, global_step,
                 num_robots=NUM_ROBOTS, comm_range=COMM_RANGE,
                 device='cpu', save_image=False):

        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.num_robots = int(num_robots)
        self.comm_range = float(comm_range)

        # ---------- knobs ----------
        self.GLOBAL_REPLAN_EVERY = int(max(1, GLOBAL_REPLAN_EVERY))
        self.COMM_EPS = 0.5  # reduce flicker

        self.STUCK_EPS = STUCK_EPS
        self.STUCK_LIMIT = STUCK_LIMIT

        # In SILENT phase, a robot switches to GO_RV if its region has no valid target for K steps
        self.REGION_DONE_STREAK = 3
        self._region_done_streak = [0] * self.num_robots

        # Global planner budget
        self.USE_ACTIVE_BUDGET = True
        self.NEXT_TARGET_WEIGHT_PER_ROBOT = int(NEXT_TARGET_WEIGHT_PER_ROBOT)
        self.NEXT_MAX_BFS_LAYER = int(NEXT_MAX_BFS_LAYER)

        self.SLACK_RATIO = 0.08
        self.TRAVEL_W = 0.15

        # nav thresholds
        self.GOAL_REACHED_DIST = 0.55 * NODE_RESOLUTION
        self.WP_REACHED_DIST = 0.40 * NODE_RESOLUTION

        # ---------- env init ----------
        base_env = Env(global_step, plot=save_image, test=True)
        base_start = np.array(base_env.robot_location.copy(), dtype=float)

        node_coords, _ = get_updating_node_coords(
            base_start, base_env.ground_truth_info, check_connectivity=True
        )
        dists = np.linalg.norm(node_coords - base_start, axis=1)
        base_node = node_coords[int(np.argmin(dists))]

        # close start nodes
        start_nodes = [base_node]
        rng = np.random.default_rng(seed=global_step + meta_agent_id)
        sep_min = 0.6 * NODE_RESOLUTION

        # candidates close to base_node
        d2_all = np.sum((node_coords - base_node[None, :]) ** 2, axis=1)
        near_candidates = node_coords[d2_all <= (max(2.0 * NODE_RESOLUTION, 0.30 * self.comm_range) ** 2)]
        if near_candidates.shape[0] == 0:
            idx = np.argsort(d2_all)
            near_candidates = node_coords[idx[:min(400, len(node_coords))]]

        for _ in range(self.num_robots - 1):
            chosen = None
            for _try in range(8000):
                cand = near_candidates[rng.integers(len(near_candidates))]
                if any(np.linalg.norm(cand - p) < sep_min for p in start_nodes):
                    continue
                chosen = cand
                break
            if chosen is None:
                chosen = base_node
            start_nodes.append(chosen)

        # build envs
        self.envs = []
        for k in range(self.num_robots):
            if k == 0:
                env = base_env
            else:
                env = Env(global_step, plot=save_image, test=True)
                env.ground_truth = base_env.ground_truth
                env.map_path = base_env.map_path

            env.belief_origin_x = base_env.belief_origin_x
            env.belief_origin_y = base_env.belief_origin_y
            env.ground_truth_info = base_env.ground_truth_info

            env.robot_location = np.array(start_nodes[k], dtype=float)
            env.robot_cell = get_cell_position_from_coords(env.robot_location, base_env.ground_truth_info)

            env.robot_belief = np.ones_like(base_env.ground_truth, dtype=np.uint8) * UNKNOWN
            env.robot_belief = sensor_work(
                env.robot_cell,
                env.sensor_range / env.cell_size,
                env.robot_belief,
                env.ground_truth
            )
            env.belief_info = MapInfo(env.robot_belief, env.belief_origin_x, env.belief_origin_y, env.cell_size)

            if save_image:
                env.trajectory_x = [env.robot_location[0]]
                env.trajectory_y = [env.robot_location[1]]

            self.envs.append(env)

        self.env = self.envs[0]

        # ---------- agents ----------
        self.robots = [Agent(policy_net, predictor, self.device, self.save_image) for _ in range(self.num_robots)]

        # ---------- state ----------
        self.team_phase = self.PHASE_CONNECTED
        self.global_round = 0
        self._last_global_plan_step = -10**9

        self.region_masks = [None] * self.num_robots
        self.global_rendezvous = None

        self.robot_mode = [self.MODE_EXPLORE] * self.num_robots
        self.current_targets = [None] * self.num_robots

        self.path_cache = [deque() for _ in range(self.num_robots)]
        self._stuck_cnt = [0] * self.num_robots

        # blacklist unreachable explore targets (avoid infinite retry)
        self.BLACKLIST_TTL = 60
        self._black = [dict() for _ in range(self.num_robots)]  # key->expire_step

        # debug caches
        self._global_pred_info = None
        self._pred_free_cache = None
        self._remaining_cache = None
        self._current_active_cache = None
        self._next_remaining_cache = None

        self.perf_metrics = dict()
        self.frame_files = [] if self.save_image else None

    # ==========================================================
    # Connectivity + multi-hop sync
    # ==========================================================
    def _comm_connected(self):
        n = self.num_robots
        if n <= 1:
            return True
        thr2 = (self.comm_range + self.COMM_EPS) ** 2
        visited = {0}
        q = [0]
        while q:
            i = q.pop(0)
            pi = self.envs[i].robot_location
            for j in range(n):
                if j == i or j in visited:
                    continue
                pj = self.envs[j].robot_location
                if float(np.sum((pi - pj) ** 2)) <= thr2:
                    visited.add(j)
                    q.append(j)
        return len(visited) == n

    def _sync_beliefs_multi_hop(self):
        n = self.num_robots
        if n <= 1:
            return

        thr2 = (self.comm_range + self.COMM_EPS) ** 2
        adj = [[] for _ in range(n)]
        for i in range(n):
            pi = self.envs[i].robot_location
            for j in range(i + 1, n):
                pj = self.envs[j].robot_location
                if float(np.sum((pi - pj) ** 2)) <= thr2:
                    adj[i].append(j)
                    adj[j].append(i)

        visited = [False] * n
        for s in range(n):
            if visited[s]:
                continue
            comp = []
            qq = [s]
            visited[s] = True
            while qq:
                u = qq.pop(0)
                comp.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        qq.append(v)

            b0 = self.envs[comp[0]].robot_belief
            merged = np.ones_like(b0, dtype=np.uint8) * UNKNOWN
            occ = np.zeros_like(b0, dtype=bool)
            free = np.zeros_like(b0, dtype=bool)
            for idx in comp:
                b = self.envs[idx].robot_belief
                occ |= (b == OCCUPIED)
                free |= (b == FREE)
            merged[occ] = OCCUPIED
            merged[~occ & free] = FREE

            for idx in comp:
                self.envs[idx].robot_belief = merged.copy()
                self.envs[idx].belief_info.map = self.envs[idx].robot_belief

    def _segment_collision(self, map_info, start_xy, end_xy, unknown_is_collision=False):
        start_xy = np.array(start_xy, dtype=float).reshape(2)
        end_xy = np.array(end_xy, dtype=float).reshape(2)

        x_min = map_info.map_origin_x
        y_min = map_info.map_origin_y
        x_max = map_info.map_origin_x + map_info.cell_size * (map_info.map.shape[1] - 1)
        y_max = map_info.map_origin_y + map_info.cell_size * (map_info.map.shape[0] - 1)

        start_xy[0] = float(np.clip(start_xy[0], x_min, x_max))
        start_xy[1] = float(np.clip(start_xy[1], y_min, y_max))
        end_xy[0] = float(np.clip(end_xy[0], x_min, x_max))
        end_xy[1] = float(np.clip(end_xy[1], y_min, y_max))

        s_cell = get_cell_position_from_coords(start_xy, map_info, check_negative=False)
        e_cell = get_cell_position_from_coords(end_xy, map_info, check_negative=False)
        x0, y0 = int(s_cell[0]), int(s_cell[1])
        x1, y1 = int(e_cell[0]), int(e_cell[1])

        grid = map_info.map
        H, W = grid.shape

        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        err = dx - dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        dx2, dy2 = 2 * dx, 2 * dy

        while 0 <= x < W and 0 <= y < H:
            v = int(grid[y, x])
            if v == OCCUPIED:
                return True
            if unknown_is_collision and (v == UNKNOWN):
                return True
            if x == x1 and y == y1:
                break
            if err > 0:
                x += x_inc
                err -= dy2
            else:
                y += y_inc
                err += dx2

        return False

    # ==========================================================
    # Node snap + canonicalization
    # ==========================================================
    def _snap_to_node(self, env, world_xy):
        xy = np.array(world_xy, dtype=float).reshape(2)
        xy = np.round(xy / NODE_RESOLUTION) * NODE_RESOLUTION
        xy = np.round(xy, 1)

        info = env.ground_truth_info
        x_min = info.map_origin_x
        y_min = info.map_origin_y
        x_max = info.map_origin_x + (info.map.shape[1] - 1) * info.cell_size
        y_max = info.map_origin_y + (info.map.shape[0] - 1) * info.cell_size

        xy[0] = float(np.clip(xy[0], x_min, x_max))
        xy[1] = float(np.clip(xy[1], y_min, y_max))
        return np.round(xy, 1)

    def _node_is_free_gt(self, env, node_xy):
        info = env.ground_truth_info
        cell = get_cell_position_from_coords(np.array(node_xy, dtype=float), info, check_negative=False)
        cx, cy = int(cell[0]), int(cell[1])
        if cy < 0 or cy >= info.map.shape[0] or cx < 0 or cx >= info.map.shape[1]:
            return False
        return int(info.map[cy, cx]) == FREE

    def _canonicalize_free_node_gt(self, env, world_xy, max_r=4):
        base = self._snap_to_node(env, world_xy)
        if self._node_is_free_gt(env, base):
            return base

        # search in node-grid offsets
        step = float(NODE_RESOLUTION)
        best = None
        best_d2 = 1e18
        for r in range(1, max_r + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    cand = base + np.array([dx * step, dy * step], dtype=float)
                    cand = self._snap_to_node(env, cand)
                    if not self._node_is_free_gt(env, cand):
                        continue
                    d2 = float(np.sum((cand - base) ** 2))
                    if d2 < best_d2:
                        best_d2 = d2
                        best = cand
            if best is not None:
                return best
        return None

    def _astar_nodegrid_gt(self, env, start_xy, goal_xy, max_expand=250000):
        info = env.ground_truth_info

        s = self._canonicalize_free_node_gt(env, start_xy)
        g = self._canonicalize_free_node_gt(env, goal_xy)
        if s is None or g is None:
            return []

        sk = (float(s[0]), float(s[1]))
        gk = (float(g[0]), float(g[1]))
        if sk == gk:
            return []

        step = float(NODE_RESOLUTION)
        nbrs = [(step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)]

        def h(k):
            return (abs(k[0] - gk[0]) + abs(k[1] - gk[1])) / step

        open_pq = []
        heapq.heappush(open_pq, (h(sk), 0.0, sk))
        came = {sk: None}
        gscore = {sk: 0.0}

        expanded = 0
        while open_pq:
            _, gcur, cur = heapq.heappop(open_pq)
            if cur == gk:
                path = []
                p = cur
                while p is not None and p != sk:
                    path.append(p)
                    p = came[p]
                path.reverse()
                return [np.array([p[0], p[1]], dtype=float) for p in path]

            expanded += 1
            if expanded > max_expand:
                return []

            if gcur > gscore.get(cur, 1e18):
                continue

            cur_xy = np.array([cur[0], cur[1]], dtype=float)
            for dx, dy in nbrs:
                nxt = (float(np.round(cur[0] + dx, 1)), float(np.round(cur[1] + dy, 1)))
                nxt_xy = np.array([nxt[0], nxt[1]], dtype=float)

                # node must be free in GT
                if not self._node_is_free_gt(env, nxt_xy):
                    continue

                # edge must not collide in GT
                if self._segment_collision(info, cur_xy, nxt_xy, unknown_is_collision=False):
                    continue

                ng = gcur + 1.0
                if ng < gscore.get(nxt, 1e18):
                    gscore[nxt] = ng
                    came[nxt] = cur
                    heapq.heappush(open_pq, (ng + h(nxt), ng, nxt))

        return []

    def _next_from_cache(self, rid, curr):
        dq = self.path_cache[rid]
        while dq:
            wp = dq[0]
            if np.linalg.norm(curr - wp) <= self.WP_REACHED_DIST:
                dq.popleft()
            else:
                break
        if not dq:
            return curr
        return np.array(dq[0], dtype=float)

    # ==========================================================
    # Agent graph update (belief utility nodes)
    # ==========================================================
    def _update_agent_graphs(self):
        for rid in range(self.num_robots):
            env = self.envs[rid]
            ag = self.robots[rid]

            env.belief_info.map = env.robot_belief.astype(np.uint8)
            ag.update_map(env.belief_info)

            ag.location = np.array(env.robot_location, dtype=float)
            ag.update_updating_map(ag.location)
            ag.update_frontiers()

            ag.node_manager.update_graph(
                ag.location,
                ag.frontier,
                ag.updating_map_info,
                ag.map_info
            )
            ag.update_location(ag.location)

    # ==========================================================
    # Blacklist helpers (for unreachable explore targets)
    # ==========================================================
    def _black_prune(self, rid, step):
        dead = [k for k, exp in self._black[rid].items() if exp <= step]
        for k in dead:
            self._black[rid].pop(k, None)

    def _black_has(self, rid, xy):
        if xy is None:
            return True
        key = (round(float(xy[0]), 1), round(float(xy[1]), 1))
        return key in self._black[rid]

    def _black_add(self, rid, xy, step):
        if xy is None:
            return
        key = (round(float(xy[0]), 1), round(float(xy[1]), 1))
        self._black[rid][key] = step + self.BLACKLIST_TTL

    # ==========================================================
    # Explore target selection (utility nodes inside region)
    # ==========================================================
    def _choose_region_utility_node(self, rid, step):
        env = self.envs[rid]
        ag = self.robots[rid]
        mask = self.region_masks[rid]
        if mask is None:
            return None

        self._black_prune(rid, step)

        H, W = mask.shape
        curr = np.array(env.robot_location, dtype=float)

        best = None
        best_d2 = 1e18
        for q in ag.node_manager.nodes_dict.__iter__():
            node = q.data
            if getattr(node, "visited", 0) == 1:
                continue
            if float(getattr(node, "utility", 0.0)) <= 0.0:
                continue

            xy = np.array(node.coords, dtype=float)

            # region constraint
            cell = get_cell_position_from_coords(xy, env.belief_info, check_negative=False)
            cx, cy = int(cell[0]), int(cell[1])
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            if not mask[cy, cx]:
                continue

            if self._black_has(rid, xy):
                continue

            d2 = float(np.sum((xy - curr) ** 2))
            if d2 < best_d2:
                best_d2 = d2
                best = xy

        return best

    # ==========================================================
    # Global planner pieces (predicted partition + RV)
    # ==========================================================
    def _refresh_prediction_for_global(self):
        env0 = self.envs[0]
        ag0 = self.robots[0]
        ag0.update_map(env0.belief_info)
        ag0.update_predict_map()
        self._global_pred_info = ag0.pred_mean_map_info

    def _pred_map_to_free_mask(self, pred_map, union_free=None, union_occ=None):
        pm = np.array(pred_map, dtype=float)
        mx = float(np.nanmax(pm)) if pm.size > 0 else 255.0
        hi, lo = (0.60, 0.40) if mx <= 1.5 else (200.0, 50.0)

        free_high = True
        if union_free is not None and union_occ is not None:
            fv = pm[union_free]
            ov = pm[union_occ]
            if fv.size >= 50 and ov.size >= 50:
                free_high = (float(np.nanmean(fv)) >= float(np.nanmean(ov)))

        if free_high:
            pred_free = (pm >= hi)
            pred_occ = (pm <= lo)
        else:
            pred_free = (pm <= lo)
            pred_occ = (pm >= hi)

        pred_unk = (~pred_free) & (~pred_occ)
        return pred_free.astype(bool), pred_occ.astype(bool), pred_unk.astype(bool)

    def _multi_source_bfs_dist(self, free_mask, seed_cells, max_layer=None):
        H, W = free_mask.shape
        dist = np.full((H, W), 1e9, dtype=np.int32)
        q = deque()
        for (sx, sy) in seed_cells:
            if 0 <= sx < W and 0 <= sy < H and free_mask[sy, sx]:
                dist[sy, sx] = 0
                q.append((sx, sy))
        dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while q:
            x, y = q.popleft()
            d0 = dist[y, x]
            if max_layer is not None and d0 >= max_layer:
                continue
            for dx, dy in dirs4:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and free_mask[ny, nx]:
                    nd = d0 + 1
                    if nd < dist[ny, nx]:
                        dist[ny, nx] = nd
                        q.append((nx, ny))
        return dist

    def _neighbors_mask(self, mask):
        H, W = mask.shape
        out = np.zeros_like(mask, dtype=bool)
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    out[ny, nx] = True
        return out

    def _balanced_partition(self, allowed_mask, seed_cells, work_mask):
        allowed = allowed_mask.astype(bool)
        work = (work_mask.astype(bool) & allowed)

        H, W = allowed.shape
        R = self.num_robots
        owner = np.full((H, W), -1, dtype=np.int32)

        total_allowed = int(np.sum(allowed))
        total_work = int(np.sum(work))
        if total_allowed == 0:
            return owner

        # quotas (with slack)
        # work quota balances exploration workload; total quota prevents gigantic "corridor" regions
        quota_work = max(1, int(np.ceil(max(1, total_work) / float(R))))
        quota_tot  = max(1, int(np.ceil(total_allowed / float(R))))

        work_hi = int(np.ceil((1.0 + self.SLACK_RATIO) * quota_work))
        work_lo = max(1, int(np.floor((1.0 - self.SLACK_RATIO) * quota_work)))
        tot_hi  = int(np.ceil((1.0 + self.SLACK_RATIO) * quota_tot))

        dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def inb(x, y):
            return 0 <= x < W and 0 <= y < H

        # project seed to nearest allowed cell
        def project_to_allowed(sx, sy):
            sx = int(np.clip(int(sx), 0, W - 1))
            sy = int(np.clip(int(sy), 0, H - 1))
            if allowed[sy, sx]:
                return (sx, sy)
            for r in [1, 2, 3, 5, 8, 12, 18, 25]:
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        nx, ny = sx + dx, sy + dy
                        if inb(nx, ny) and allowed[ny, nx]:
                            return (nx, ny)
            ys, xs = np.where(allowed)
            return (int(xs[0]), int(ys[0])) if xs.size > 0 else (sx, sy)

        # seeds (deduplicate)
        seeds = []
        used = set()
        for rid, (sx, sy) in enumerate(seed_cells):
            p = project_to_allowed(sx, sy)
            if p in used:
                # find nearby unused allowed
                found = None
                for r in [1, 2, 3, 5, 8]:
                    for dx in range(-r, r + 1):
                        for dy in range(-r, r + 1):
                            nx, ny = p[0] + dx, p[1] + dy
                            if inb(nx, ny) and allowed[ny, nx] and (nx, ny) not in used:
                                found = (nx, ny)
                                break
                        if found is not None:
                            break
                    if found is not None:
                        break
                p = found if found is not None else p
            used.add(p)
            seeds.append(p)

        # region frontiers (connected growth)
        frontier = [deque() for _ in range(R)]
        cnt_work = [0] * R
        cnt_tot = [0] * R

        # init assign seeds
        for rid in range(R):
            sx, sy = seeds[rid]
            if owner[sy, sx] == -1 and allowed[sy, sx]:
                owner[sy, sx] = rid
                frontier[rid].append((sx, sy))
                cnt_tot[rid] += 1
                if work[sy, sx]:
                    cnt_work[rid] += 1

        # region-level heap: expand region with smallest work load first
        heap = [(cnt_work[rid], cnt_tot[rid], rid) for rid in range(R)]
        heapq.heapify(heap)
        inactive = [False] * R

        def everyone_reached_lo():
            return min(cnt_work) >= work_lo

        while heap:
            wcnt, tcnt, rid = heapq.heappop(heap)
            if inactive[rid]:
                continue
            if wcnt != cnt_work[rid] or tcnt != cnt_tot[rid]:
                continue  # stale

            fq = frontier[rid]
            expanded = False

            # try to expand one step from boundary
            while fq:
                x, y = fq[0]

                cand_work = None
                cand_any = None

                for dx, dy in dirs4:
                    nx, ny = x + dx, y + dy
                    if not inb(nx, ny):
                        continue
                    if not allowed[ny, nx]:
                        continue
                    if owner[ny, nx] != -1:
                        continue

                    if work[ny, nx]:
                        cand_work = (nx, ny)
                        break
                    if cand_any is None:
                        cand_any = (nx, ny)

                pick = None
                if cand_work is not None:
                    # don't allow taking more work if already above hi AND others still below lo
                    if (cnt_work[rid] < work_hi) or everyone_reached_lo():
                        pick = cand_work
                    else:
                        # defer work grab; maybe expand via travel to keep connectivity
                        pick = cand_any
                else:
                    pick = cand_any

                if pick is None:
                    fq.popleft()
                    continue

                # soft cap total size to avoid gigantic regions
                if cnt_tot[rid] >= tot_hi and (not work[pick[1], pick[0]]):
                    # if we are over total cap and this is just travel, skip it
                    fq.popleft()
                    continue

                nx, ny = pick
                owner[ny, nx] = rid
                cnt_tot[rid] += 1
                if work[ny, nx]:
                    cnt_work[rid] += 1

                fq.append((nx, ny))
                fq.rotate(-1)
                expanded = True
                break

            if not expanded:
                inactive[rid] = True
                continue

            heapq.heappush(heap, (cnt_work[rid], cnt_tot[rid], rid))

        # final fill: assign any leftover allowed cells to nearest existing owner (keeps regions covering walkable area)
        unassigned = allowed & (owner == -1)
        if int(np.sum(unassigned)) > 0:
            q = deque()
            ys, xs = np.where(owner != -1)
            for x, y in zip(xs.tolist(), ys.tolist()):
                q.append((x, y))

            while q:
                x, y = q.popleft()
                rid = int(owner[y, x])
                for dx, dy in dirs4:
                    nx, ny = x + dx, y + dy
                    if not inb(nx, ny):
                        continue
                    if not allowed[ny, nx]:
                        continue
                    if owner[ny, nx] != -1:
                        continue
                    owner[ny, nx] = rid
                    q.append((nx, ny))

        return owner


    def _pick_rv(self, env0, union_free, remaining, next_remaining):
        H, W = union_free.shape
        mean_world = np.mean(np.array([e.robot_location for e in self.envs], dtype=float), axis=0)
        mean_cell = get_cell_position_from_coords(mean_world, env0.belief_info, check_negative=False)
        mx, my = int(mean_cell[0]), int(mean_cell[1])
        mx = max(0, min(W - 1, mx))
        my = max(0, min(H - 1, my))

        cand = union_free & self._neighbors_mask(next_remaining)
        if int(np.sum(cand)) == 0:
            cand = union_free & self._neighbors_mask(remaining)
        if int(np.sum(cand)) == 0:
            cand = union_free.copy()

        ys, xs = np.where(cand)
        d2 = (xs - mx) ** 2 + (ys - my) ** 2
        k = int(np.argmin(d2))
        cx, cy = int(xs[k]), int(ys[k])

        wx = env0.belief_origin_x + float(cx) * env0.cell_size
        wy = env0.belief_origin_y + float(cy) * env0.cell_size
        rv = np.array([wx, wy], dtype=float)

        rv_can = self._canonicalize_free_node_gt(env0, rv)
        return rv_can if rv_can is not None else self._snap_to_node(env0, rv)

    def _global_plan(self, step_idx, replan_only=False):
        env0 = self.envs[0]
        H, W = env0.robot_belief.shape

        union_free = np.zeros((H, W), dtype=bool)
        union_occ = np.zeros((H, W), dtype=bool)
        for e in self.envs:
            union_free |= (e.robot_belief == FREE)
            union_occ |= (e.robot_belief == OCCUPIED)

        self._refresh_prediction_for_global()
        if self._global_pred_info is None:
            pred_free = union_free.copy()
        else:
            pred_free, _, _ = self._pred_map_to_free_mask(self._global_pred_info.map,
                                                         union_free=union_free, union_occ=union_occ)

        remaining = pred_free & (~union_free)

        seed_cells = []
        for rid in range(self.num_robots):
            cell = get_cell_position_from_coords(self.envs[rid].robot_location, env0.belief_info, check_negative=False)
            seed_cells.append((int(cell[0]), int(cell[1])))

        dist = self._multi_source_bfs_dist(pred_free, seed_cells, max_layer=self.NEXT_MAX_BFS_LAYER)

        if not self.USE_ACTIVE_BUDGET:
            current_active = pred_free.copy()
        else:
            rem_dists = dist[remaining]
            if rem_dists.size == 0:
                current_active = pred_free.copy()
            else:
                budget = int(max(1, self.NEXT_TARGET_WEIGHT_PER_ROBOT) * self.num_robots)
                if rem_dists.size <= budget:
                    thr = int(min(int(rem_dists.max()), self.NEXT_MAX_BFS_LAYER))
                else:
                    k = min(rem_dists.size - 1, budget - 1)
                    kth = np.partition(rem_dists, k)[k]
                    thr = int(min(int(kth), self.NEXT_MAX_BFS_LAYER))
                current_active = pred_free & (dist <= thr)

        next_remaining = remaining & (~current_active)

        rv = self._pick_rv(env0, union_free=union_free, remaining=remaining, next_remaining=next_remaining)

        work_mask = remaining & current_active
        owner = self._balanced_partition(current_active, seed_cells, work_mask=work_mask)

        self.region_masks = [(owner == rid) for rid in range(self.num_robots)]
        self.global_rendezvous = rv

        self._pred_free_cache = pred_free
        self._remaining_cache = remaining
        self._current_active_cache = current_active
        self._next_remaining_cache = next_remaining

        if not replan_only:
            self.robot_mode = [self.MODE_EXPLORE] * self.num_robots
            self.current_targets = [None] * self.num_robots
            self.path_cache = [deque() for _ in range(self.num_robots)]
            self._stuck_cnt = [0] * self.num_robots
            self._region_done_streak = [0] * self.num_robots
            self._black = [dict() for _ in range(self.num_robots)]

    # ==========================================================
    # If RV becomes unreachable (should be rare), pick a safe fallback RV
    # ==========================================================
    def _fallback_rv_gt(self):
        env0 = self.envs[0]
        rv = self._canonicalize_free_node_gt(env0, env0.robot_location)
        return rv if rv is not None else self._snap_to_node(env0, env0.robot_location)

    # ==========================================================
    # Episode loop
    # ==========================================================
    def run_episode(self):
        self.team_phase = self.PHASE_CONNECTED
        self.global_round = 0
        self._last_global_plan_step = -10**9

        # initial sync & graphs
        self._sync_beliefs_multi_hop()
        self._update_agent_graphs()

        # initial global plan if connected
        if self._comm_connected():
            self._global_plan(step_idx=0, replan_only=False)

        if self.save_image:
            self.plot_multi_env(0)

        done = False
        for step in range(MAX_EPISODE_STEP):
            connected_pre = self._comm_connected()

            # CONNECTED -> SILENT if lost at start of step
            if self.team_phase == self.PHASE_CONNECTED and (not connected_pre):
                self.team_phase = self.PHASE_SILENT

            # CONNECTED phase: global replan (connected only)
            if self.team_phase == self.PHASE_CONNECTED and connected_pre:
                if (step - self._last_global_plan_step) >= self.GLOBAL_REPLAN_EVERY:
                    self._last_global_plan_step = step
                    self._sync_beliefs_multi_hop()
                    self._update_agent_graphs()
                    self._global_plan(step_idx=step, replan_only=True)

            # always update local graphs for utility targets
            self._update_agent_graphs()

            # ---------- choose actions ----------
            next_locs = []
            for rid in range(self.num_robots):
                env = self.envs[rid]
                curr = np.array(env.robot_location, dtype=float)

                if self.robot_mode[rid] == self.MODE_EXPLORE:
                    tgt = self._choose_region_utility_node(rid, step)

                    if tgt is None:
                        # no valid target in region
                        if self.team_phase == self.PHASE_SILENT:
                            self._region_done_streak[rid] += 1
                        else:
                            self._region_done_streak[rid] = 0

                        # only in SILENT do we switch to GO_RV
                        if self.team_phase == self.PHASE_SILENT and self._region_done_streak[rid] >= self.REGION_DONE_STREAK:
                            self.robot_mode[rid] = self.MODE_GO_RV
                            self.current_targets[rid] = None
                            self.path_cache[rid].clear()
                            next_locs.append(curr.copy())
                            continue

                        next_locs.append(curr.copy())
                        continue

                    # have a target
                    self._region_done_streak[rid] = 0
                    self.current_targets[rid] = tgt

                    if len(self.path_cache[rid]) == 0:
                        path = self._astar_nodegrid_gt(env, curr, tgt)
                        if len(path) == 0:
                            # unreachable on GT at node-res -> blacklist and stay this step
                            self._black_add(rid, tgt, step)
                            next_locs.append(curr.copy())
                            continue
                        self.path_cache[rid] = deque(path)

                    nxt = self._next_from_cache(rid, curr)
                    nxt = self._snap_to_node(env, nxt)
                    next_locs.append(nxt)

                else:
                    # GO_TO_RV
                    rv = self.global_rendezvous
                    if rv is None:
                        next_locs.append(curr.copy())
                        continue

                    self.current_targets[rid] = np.array(rv, dtype=float)

                    if len(self.path_cache[rid]) == 0:
                        path = self._astar_nodegrid_gt(env, curr, rv)
                        if len(path) == 0:
                            # RV unreachable at node-res -> choose a safe fallback RV
                            self.global_rendezvous = self._fallback_rv_gt()
                            rv = self.global_rendezvous
                            self.current_targets[rid] = np.array(rv, dtype=float)
                            path = self._astar_nodegrid_gt(env, curr, rv)

                        self.path_cache[rid] = deque(path)

                    nxt = self._next_from_cache(rid, curr)
                    nxt = self._snap_to_node(env, nxt)
                    next_locs.append(nxt)

            # vertex conflict (simple)
            for i in range(self.num_robots):
                for j in range(i + 1, self.num_robots):
                    if np.linalg.norm(next_locs[i] - next_locs[j]) < 0.5 * NODE_RESOLUTION:
                        next_locs[j] = np.array(self.envs[j].robot_location, dtype=float)

            for rid in range(self.num_robots):
                env = self.envs[rid]
                curr = np.array(env.robot_location, dtype=float)
                nxt = np.array(next_locs[rid], dtype=float)

                if self._segment_collision(env.ground_truth_info, curr, nxt, unknown_is_collision=False):
                    nxt = curr.copy()
                    self.path_cache[rid].clear()

                prev = curr.copy()
                env.step(nxt)
                env.robot_location = self._snap_to_node(env, env.robot_location)

                now = np.array(env.robot_location, dtype=float)
                if np.linalg.norm(now - prev) < self.STUCK_EPS:
                    self._stuck_cnt[rid] += 1
                else:
                    self._stuck_cnt[rid] = 0

                if self._stuck_cnt[rid] >= self.STUCK_LIMIT:
                    # clear and retry
                    self.path_cache[rid].clear()
                    self.current_targets[rid] = None
                    self._stuck_cnt[rid] = 0

                if self.save_image:
                    env.trajectory_x.append(env.robot_location[0])
                    env.trajectory_y.append(env.robot_location[1])

            # ---------- sync after move ----------
            self._sync_beliefs_multi_hop()

            # SILENT -> CONNECTED: instantaneous full reconnect + all robots GO_RV
            connected_post = self._comm_connected()
            if self.team_phase == self.PHASE_SILENT and connected_post:
                if all(m == self.MODE_GO_RV for m in self.robot_mode):
                    self.team_phase = self.PHASE_CONNECTED
                    self.global_round += 1
                    self._last_global_plan_step = step
                    self._sync_beliefs_multi_hop()
                    self._update_agent_graphs()
                    self._global_plan(step_idx=step, replan_only=False)

            # termination: union frontier empty
            union_frontiers = set()
            for env in self.envs:
                union_frontiers |= get_frontier_in_map(env.belief_info)
            if len(union_frontiers) == 0:
                done = True

            if self.save_image:
                self.plot_multi_env(step + 1)
            if done:
                break

        self.perf_metrics = {
            'travel_dist': sum(e.travel_dist for e in self.envs),
            'explored_rate': float(np.mean([e.explored_rate for e in self.envs])),
            'success_rate': bool(done),
            'travel_dist_each': [e.travel_dist for e in self.envs]
        }

        if self.save_image and self.frame_files:
            make_gif(
                gif_path, self.global_step, self.frame_files,
                float(np.mean([e.explored_rate for e in self.envs]))
            )

    # ==========================================================
    #visualization
    # ==========================================================
    def _world_to_cell_xy(self, env0, world_xy):
        wx, wy = float(world_xy[0]), float(world_xy[1])
        cx = (wx - env0.belief_origin_x) / env0.cell_size
        cy = (wy - env0.belief_origin_y) / env0.cell_size
        return cx, cy

    def _ensure_pred_maps_for_vis(self):
        pred_infos = [None] * self.num_robots
        for rid in range(self.num_robots):
            env = self.envs[rid]
            ag = self.robots[rid]
            try:
                ag.update_map(env.belief_info)
                ag.update_predict_map()
                pred_infos[rid] = getattr(ag, "pred_mean_map_info", None)
            except Exception:
                pred_infos[rid] = None
        return pred_infos
    def plot_multi_env(self, step):
        if not self.save_image:
            return

        plt.switch_backend('agg')

        env0 = self.envs[0]

        # Pred maps for each agent (optional; might be None if predictor fails)
        pred_infos = self._ensure_pred_maps_for_vis()

        R = self.num_robots
        fig = plt.figure(figsize=(12, 3.4 * R))
        gs = gridspec.GridSpec(R, 2, figure=fig, wspace=0.05, hspace=0.15)

        cmap_regions = plt.get_cmap('tab10', R)

        # global RV and connection flag
        rv = self.global_rendezvous
        conn_flag = int(self._comm_connected())

        for rid in range(R):
            env = self.envs[rid]
            c = cmap_regions(rid)

            # ---------- LEFT: belief map ----------
            axL = fig.add_subplot(gs[rid, 0])
            bg = env.robot_belief.copy()
            axL.imshow(bg, cmap='gray', origin='upper', vmin=0, vmax=255)

            # region overlay on belief (optional)
            if self.region_masks[rid] is not None:
                rm = self.region_masks[rid].astype(float)
                axL.imshow(rm, origin='upper', alpha=0.18)

            # trajectory
            if hasattr(env, "trajectory_x") and len(env.trajectory_x) > 1:
                traj = np.stack([env.trajectory_x, env.trajectory_y], axis=1)
                xs = (traj[:, 0] - env0.belief_origin_x) / env0.cell_size
                ys = (traj[:, 1] - env0.belief_origin_y) / env0.cell_size
                axL.plot(xs, ys, linewidth=2.0)

            # current position
            x, y = self._world_to_cell_xy(env0, env.robot_location)
            axL.plot([x], [y], marker='o', markersize=8,
                    markeredgecolor='k', markeredgewidth=1.0, color=c)

            # current target
            if self.current_targets[rid] is not None:
                tx, ty = self._world_to_cell_xy(env0, self.current_targets[rid])
                axL.plot([tx], [ty], marker='x', markersize=10, mew=2.0, color=c)
                axL.plot([x, tx], [y, ty], linestyle='--', linewidth=1.5)

            # RV
            if rv is not None:
                gx, gy = self._world_to_cell_xy(env0, rv)
                axL.scatter([gx], [gy], s=160, marker='*', facecolors='gold',
                            edgecolors='k', linewidths=1.2, zorder=6)

            axL.set_title(f'Robot {rid} | BELIEF | mode={"EXP" if self.robot_mode[rid]==self.MODE_EXPLORE else "GO_RV"}')
            axL.set_axis_off()

            axR = fig.add_subplot(gs[rid, 1])

            pred_info = pred_infos[rid]
            if pred_info is None or getattr(pred_info, "map", None) is None:
                # fallback: show belief if pred not available
                axR.imshow(bg, cmap='gray', origin='upper', vmin=0, vmax=255)
                axR.set_title(f'Robot {rid} | PRED (unavailable) -> BELIEF fallback')
            else:
                pm = np.array(pred_info.map, dtype=float)

                # robust normalize for visualization
                finite = np.isfinite(pm)
                if np.any(finite):
                    vmin = float(np.nanpercentile(pm[finite], 2))
                    vmax = float(np.nanpercentile(pm[finite], 98))
                    if vmax <= vmin + 1e-6:
                        vmax = vmin + 1.0
                else:
                    vmin, vmax = 0.0, 1.0

                axR.imshow(pm, origin='upper', vmin=vmin, vmax=vmax)

                # region mask overlay (stronger alpha here)
                if self.region_masks[rid] is not None:
                    rm = self.region_masks[rid].astype(float)
                    axR.imshow(rm, origin='upper', alpha=0.28)

                axR.set_title(f'Robot {rid} | PRED mean map + REGION')

            # overlay trajectory and markers on predicted view as well
            if hasattr(env, "trajectory_x") and len(env.trajectory_x) > 1:
                traj = np.stack([env.trajectory_x, env.trajectory_y], axis=1)
                xs = (traj[:, 0] - env0.belief_origin_x) / env0.cell_size
                ys = (traj[:, 1] - env0.belief_origin_y) / env0.cell_size
                axR.plot(xs, ys, linewidth=2.0)

            axR.plot([x], [y], marker='o', markersize=8,
                    markeredgecolor='k', markeredgewidth=1.0, color=c)

            if self.current_targets[rid] is not None:
                tx, ty = self._world_to_cell_xy(env0, self.current_targets[rid])
                axR.plot([tx], [ty], marker='x', markersize=10, mew=2.0, color=c)
                axR.plot([x, tx], [y, ty], linestyle='--', linewidth=1.5)

            if rv is not None:
                gx, gy = self._world_to_cell_xy(env0, rv)
                axR.scatter([gx], [gy], s=160, marker='*', facecolors='gold',
                            edgecolors='k', linewidths=1.2, zorder=6)

            axR.set_axis_off()

        # global super title
        fig.suptitle(f'phase={self.team_phase} | round={self.global_round} | conn={conn_flag} | step={step}',
                    y=0.995, fontsize=14)

        frame_path = f'{gif_path}/{self.global_step}_{step}_panel.png'
        plt.tight_layout(rect=[0, 0, 1, 0.985])
        plt.savefig(frame_path, dpi=170)
        plt.close(fig)
        self.frame_files.append(frame_path)


@ray.remote(num_cpus=1)
class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.worker = None
        self.network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
        self.network.to(self.device)
        self.predictor = self.load_predictor()

    def load_predictor(self):
        config_path = f'{generator_path}/config.yaml'
        checkpoint_path = os.path.join(
            generator_path,
            [f for f in os.listdir(generator_path)
             if f.startswith('gen') and f.endswith('.pt')][0]
        )
        with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.SafeLoader)
        generator = Generator(config['netG'], USE_GPU)
        generator.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        predictor = Evaluator(config, generator, USE_GPU, N_GEN_SAMPLE)
        print("Map predictor loaded from {}".format(checkpoint_path))
        return predictor

    def set_weights(self, weights):
        self.network.load_state_dict(weights)

    def do_job(self, episode_number):
        if NUM_ROBOTS == 1:
            self.worker = TestWorker(
                self.meta_agent_id,
                self.network,
                self.predictor,
                episode_number,
                device=self.device,
                save_image=SAVE_GIFS
            )
        else:
            self.worker = MultiTestWorker(
                self.meta_agent_id,
                self.network,
                self.predictor,
                episode_number,
                num_robots=NUM_ROBOTS,
                comm_range=COMM_RANGE,
                device=self.device,
                save_image=SAVE_GIFS
            )

        self.worker.run_episode()
        return self.worker.perf_metrics

    def job(self, weights, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        self.set_weights(weights)
        metrics = self.do_job(episode_number)
        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
            "map_path": self.worker.env.map_path,
        }
        return metrics, info


if __name__ == '__main__':
    ray.init()
    run_test()   