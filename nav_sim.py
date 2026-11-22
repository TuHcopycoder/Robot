"""
Mo phong robot ve sinh tu dong dang vi sai trong can ho:
  - A*            : lap duong di toan cuc tren ban do luoi chiem cho
  - Pure Pursuit  : bo bam quy dao hinh hoc
  - DWA           : Dynamic Window Approach - bam quy dao + tranh vat can cuc bo
Robot don lan luot cac khu vuc ban (zone cleaning) roi quay ve tram sac.

Tac gia: Tran Tri Thanh - Hoc phan Ky thuat Robot.
Toan bo thuat toan duoc tu cai dat bang NumPy/SciPy.
"""
import heapq
import time
import numpy as np
from scipy.ndimage import distance_transform_edt

# ------------------------------------------------------------------
# Thong so robot va mo phong
# ------------------------------------------------------------------
RES = 0.05          # do phan giai ban do [m/o]
R_ROBOT = 0.17      # ban kinh robot [m]
INFLATE = 0.10      # khoang an toan them khi lap ke hoach A* [m]
V_MAX = 0.30        # van toc dai toi da [m/s]
W_MAX = 1.50        # van toc goc toi da [rad/s]
A_V = 0.50          # gia toc dai toi da [m/s^2]
A_W = 3.00          # gia toc goc toi da [rad/s^2]
DT = 0.10           # chu ky dieu khien / buoc mo phong [s]
GOAL_TOL = 0.10     # sai so cho phep khi den diem dich [m]
SENSE_R = 2.0       # tam cam bien LiDAR mo phong [m]

# Pure Pursuit
PP_L = 0.35         # khoang nhin truoc [m]

# DWA (Fox, Burgard, Thrun 1997)
DWA_T = 1.5         # thoi gian du bao quy dao [s]
DWA_NV, DWA_NW = 7, 21   # so mau van toc trong cua so dong
DWA_W = dict(heading=0.6, path=0.6, clear=0.05, vel=2.0)  # trong so ham chi phi
DWA_LOCAL = 1.2     # khoang cach diem dich cuc bo tren duong A* [m]
DWA_DMAX = 0.25    # khoang ho tu do tro len thi chi phi vat can = 0 [m]
DWA_MARGIN = 0.02   # khoang ho toi thieu de quy dao duoc chap nhan [m]


# ------------------------------------------------------------------
# 1. Moi truong
# ------------------------------------------------------------------
class World:
    """Ban do can ho: tuong + noi that (da biet), vat can la (chua co trong
    ban do) va vat can dong (vat nuoi)."""

    def __init__(self, w, h, rects, unknown=(), movers=(), name=""):
        self.w, self.h, self.name = w, h, name
        self.rects = list(rects)            # vat can co trong ban do
        self.unknown = list(unknown)        # vat can la (robot khong biet truoc)
        self.movers = list(movers)          # vat can dong
        self.nx, self.ny = int(round(w / RES)), int(round(h / RES))
        occ = np.zeros((self.ny, self.nx), bool)
        occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
        for (x0, y0, x1, y1) in self.rects:
            occ[int(round(y0 / RES)):int(round(y1 / RES)),
                int(round(x0 / RES)):int(round(x1 / RES))] = True
        self.occ = occ
        # ban do khoang cach (EDT) tu moi o toi vat can gan nhat [m]
        self.edt = distance_transform_edt(~occ) * RES

    # khoang ho chinh xac tu tam robot toi cac vat can (da tru ban kinh)
    def clearance(self, x, y, t=0.0, include_unknown=True, include_movers=True):
        x, y = np.asarray(x, float), np.asarray(y, float)
        d = np.minimum.reduce([x - RES, y - RES, self.w - RES - x, self.h - RES - y])
        rects = self.rects + (self.unknown if include_unknown else [])
        for (x0, y0, x1, y1) in rects:
            dx = np.maximum.reduce([x0 - x, np.zeros_like(x), x - x1])
            dy = np.maximum.reduce([y0 - y, np.zeros_like(y), y - y1])
            d = np.minimum(d, np.hypot(dx, dy))
        if include_movers:
            for mv in self.movers:
                px, py = mv.pos(t)
                d = np.minimum(d, np.hypot(x - px, y - py) - mv.r)
        return d - R_ROBOT


class Mover:
    """Vat can dong: di chuyen qua lai tren doan thang P0-P1 voi toc do v."""

    def __init__(self, p0, p1, v, r=0.15, phase=0.0):
        self.p0, self.p1 = np.array(p0, float), np.array(p1, float)
        self.v, self.r, self.phase = v, r, phase
        self.L = np.linalg.norm(self.p1 - self.p0)

    def pos(self, t):
        s = (self.v * t + self.phase * 2 * self.L) % (2 * self.L)
        s = s if s <= self.L else 2 * self.L - s
        return self.p0 + (self.p1 - self.p0) * s / self.L

    def vel(self, t, h=0.05):
        return (self.pos(t + h) - self.pos(t - h)) / (2 * h)


# ------------------------------------------------------------------
# 2. A* tren luoi chiem cho
# ------------------------------------------------------------------
N8 = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
      (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (-1, -1, np.sqrt(2))]


def inflated_grid(world, inflate=INFLATE):
    """Khong gian cau hinh: o bi chan neu tam robot dat tai do cach vat can
    nho hon R_ROBOT + inflate."""
    return world.edt <= R_ROBOT + inflate


def astar(blocked, start, goal, heuristic="euclid"):
    """A* 8 lan can, khong cat goc vat can.
    heuristic = 'euclid' (A*) hoac 'zero' (tuong duong Dijkstra).
    Tra ve (duong di theo toa do [m], so nut da mo rong)."""
    s = (int(start[1] / RES), int(start[0] / RES))
    g = (int(goal[1] / RES), int(goal[0] / RES))
    if blocked[s] or blocked[g]:
        raise ValueError("diem dau/cuoi nam trong vung cam")
    if heuristic == "euclid":
        h = lambda n: np.hypot(n[0] - g[0], n[1] - g[1])
    else:
        h = lambda n: 0.0
    ny, nx = blocked.shape
    gcost = {s: 0.0}
    parent = {s: None}
    closed = np.zeros_like(blocked)
    openq = [(h(s), 0.0, s)]
    expanded = 0
    while openq:
        _, gc, u = heapq.heappop(openq)
        if closed[u]:
            continue
        closed[u] = True
        expanded += 1
        if u == g:
            break
        for dr, dc, c in N8:
            v = (u[0] + dr, u[1] + dc)
            if not (0 <= v[0] < ny and 0 <= v[1] < nx) or blocked[v] or closed[v]:
                continue
            if dr and dc and (blocked[u[0] + dr, u[1]] or blocked[u[0], u[1] + dc]):
                continue          # khong cat qua goc vat can
            ng = gc + c
            if ng < gcost.get(v, np.inf):
                gcost[v], parent[v] = ng, u
                heapq.heappush(openq, (ng + h(v), ng, v))
    path, n = [], g
    while n is not None:
        path.append(((n[1] + 0.5) * RES, (n[0] + 0.5) * RES))
        n = parent[n]
    path = np.array(path[::-1])
    path[0], path[-1] = start, goal
    return path, expanded, closed


def line_free(blocked, p, q):
    n = max(2, int(np.linalg.norm(np.subtract(q, p)) / (RES / 2)))
    for s in np.linspace(0, 1, n):
        x, y = p[0] + s * (q[0] - p[0]), p[1] + s * (q[1] - p[1])
        if blocked[int(y / RES), int(x / RES)]:
            return False
    return True


def smooth_path(blocked, path):
    """Rut gon duong A* bang kiem tra tam nhin thang (line-of-sight)."""
    out, k = [path[0]], 0
    while k < len(path) - 1:
        j = len(path) - 1
        while j > k + 1 and not line_free(blocked, path[k], path[j]):
            j -= 1
        out.append(path[j]); k = j
    return np.array(out)


def path_length(p):
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def densify(p, step=0.02):
    out = [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(out)


def plan_route(world, points, inflate=INFLATE, heuristic="euclid", smooth=True):
    """Lap duong A* noi lan luot cac diem: tram sac -> cac khu vuc -> tram sac."""
    blocked = inflated_grid(world, inflate)
    legs, stats = [], []
    for a, b in zip(points[:-1], points[1:]):
        t0 = time.perf_counter()
        raw, nexp, closed = astar(blocked, a, b, heuristic)
        dt = time.perf_counter() - t0
        p = smooth_path(blocked, raw) if smooth else raw
        legs.append(p)
        stats.append(dict(raw_len=path_length(raw), len=path_length(p), expanded=int(nexp),
                          time_ms=1000 * dt, n_wp=len(p), closed=closed))
    return legs, stats


# ------------------------------------------------------------------
# 3. Mo hinh robot vi sai
# ------------------------------------------------------------------
def unicycle(state, v, w, dt=DT):
    x, y, th = state
    return np.array([x + v * np.cos(th) * dt, y + v * np.sin(th) * dt,
                     (th + w * dt + np.pi) % (2 * np.pi) - np.pi])


def actuator(v_cmd, w_cmd, v, w):
    """Gioi han van toc va gia toc cua dong co (ap dung cho moi bo dieu khien)."""
    v_new = np.clip(v_cmd, max(-V_MAX, v - A_V * DT), min(V_MAX, v + A_V * DT))
    w_new = np.clip(w_cmd, max(-W_MAX, w - A_W * DT), min(W_MAX, w + A_W * DT))
    return v_new, w_new


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class PathTracker:
    """Luu duong tham chieu va tim diem gan nhat (chi tim tien ve phia truoc)."""

    def __init__(self, path):
        self.ref = densify(path)
        self.s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.ref, axis=0), axis=1))]
        self.idx = 0

    def update(self, x, y):
        hi = np.searchsorted(self.s, self.s[self.idx] + 1.0)
        seg = self.ref[self.idx:hi + 1]
        self.idx += int(np.argmin(np.hypot(seg[:, 0] - x, seg[:, 1] - y)))
        return float(np.hypot(*(self.ref[self.idx] - (x, y))))   # sai so ngang

    def point_ahead(self, dist):
        k = min(np.searchsorted(self.s, self.s[self.idx] + dist), len(self.ref) - 1)
        return self.ref[k]


# ------------------------------------------------------------------
# 4. Pure Pursuit
# ------------------------------------------------------------------
def pure_pursuit(state, tracker, L=PP_L):
    x, y, th = state
    gx, gy = tracker.point_ahead(L)
    alpha = wrap(np.arctan2(gy - y, gx - x) - th)
    Ld = max(np.hypot(gx - x, gy - y), 1e-3)
    v = V_MAX
    w = 2 * v * np.sin(alpha) / Ld          # w = v * kappa, kappa = 2 sin(a)/Ld
    if abs(w) > W_MAX:                      # giu nguyen do cong khi bao hoa
        v *= W_MAX / abs(w); w = np.sign(w) * W_MAX
    d_goal = np.hypot(*(tracker.ref[-1] - (x, y)))
    v = min(v, np.sqrt(2 * A_V * 0.5 * d_goal))   # giam toc khi gan dich
    return v, w


# ------------------------------------------------------------------
# 5. Dynamic Window Approach
# ------------------------------------------------------------------
class DWA:
    def __init__(self, world, weights=None, T=DWA_T, nv=DWA_NV, nw=DWA_NW,
                 predict_movers=False):
        self.world = world
        self.wt = dict(DWA_W, **(weights or {}))
        self.T, self.nv, self.nw = T, nv, nw
        self.predict_movers = predict_movers
        self.n_steps = int(round(T / DT))
        self.last = None   # luu quy dao ung vien de ve hinh
        self.stuck, self.escape_dir = 0, 1.0

    def obstacle_clearance(self, X, Y, t, rx, ry):
        """Khoang ho cua cac diem du bao: ban do tinh (EDT) + vat can do duoc
        bang LiDAR trong ban kinh SENSE_R (vat can la, vat nuoi)."""
        wd = self.world
        r = np.clip((Y / RES).astype(int), 0, wd.ny - 1)
        c = np.clip((X / RES).astype(int), 0, wd.nx - 1)
        d = wd.edt[r, c] - RES / 2 - R_ROBOT
        for (x0, y0, x1, y1) in wd.unknown:
            cx, cy = np.clip(rx, x0, x1), np.clip(ry, y0, y1)
            if np.hypot(cx - rx, cy - ry) > SENSE_R:
                continue        # ngoai tam cam bien
            dx = np.maximum(np.maximum(x0 - X, 0), X - x1)
            dy = np.maximum(np.maximum(y0 - Y, 0), Y - y1)
            d = np.minimum(d, np.hypot(dx, dy) - R_ROBOT)
        for mv in wd.movers:
            px, py = mv.pos(t)
            if np.hypot(px - rx, py - ry) - mv.r > SENSE_R:
                continue
            if self.predict_movers:     # du bao vat can di thang deu
                vx, vy = mv.vel(t)
                tt = np.arange(1, self.n_steps + 1) * DT
                px, py = px + vx * tt, py + vy * tt
            d = np.minimum(d, np.hypot(X - px, Y - py) - mv.r - R_ROBOT)
        return d

    def control(self, state, v, w, tracker, t):
        x, y, th = state
        # (1) cua so dong: van toc dat duoc trong mot chu ky
        vs = np.linspace(max(0.0, v - A_V * DT), min(V_MAX, v + A_V * DT), self.nv)
        ws = np.linspace(max(-W_MAX, w - A_W * DT), min(W_MAX, w + A_W * DT), self.nw)
        V, W = [a.ravel() for a in np.meshgrid(vs, ws)]
        # (2) du bao quy dao voi (v, w) khong doi trong T giay (tich phan Euler
        #     giong het mo hinh robot): theta truoc buoc j la th + w*j*DT
        k = np.arange(self.n_steps) * DT
        TH0 = th + np.outer(W, k)                 # huong o dau moi buoc
        X = x + np.cumsum(V[:, None] * np.cos(TH0) * DT, axis=1)
        Y = y + np.cumsum(V[:, None] * np.sin(TH0) * DT, axis=1)
        TH = TH0 + W[:, None] * DT                # huong o cuoi moi buoc
        # (3) loai quy dao va cham va quy dao khong kip phanh
        clr = self.obstacle_clearance(X, Y, t, x, y).min(axis=1)
        brake_ok = V <= np.sqrt(2 * A_V * np.maximum(clr, 0)) + 1e-9
        ok = (clr > DWA_MARGIN) & brake_ok
        # (4) ham chi phi (cang nho cang tot)
        gx, gy = tracker.point_ahead(DWA_LOCAL)
        near_goal = np.hypot(*(tracker.ref[-1] - (x, y))) < DWA_LOCAL
        rows = np.arange(len(V))
        if near_goal:   # gan dich: danh gia tai diem quy dao gan dich nhat
            j = np.argmin(np.hypot(gx - X, gy - Y), axis=1)
        else:           # Fox et al.: danh gia tai cuoi quy dao
            j = np.full(len(V), self.n_steps - 1)
        ang = np.abs(wrap(np.arctan2(gy - Y[rows, j], gx - X[rows, j]) - TH[rows, j]))
        reach = np.hypot(gx - X[rows, j], gy - Y[rows, j]) < GOAL_TOL
        c_head = np.where(reach, 0.0, ang / np.pi)                       # [0, 1]
        ref = tracker.ref[tracker.idx:np.searchsorted(tracker.s, tracker.s[tracker.idx] + 2.0)]
        dpath = np.min(np.hypot(X[:, -1, None] - ref[None, :, 0], Y[:, -1, None] - ref[None, :, 1]), axis=1)
        c_path = np.minimum(dpath, 1.0)                                  # [0, 1] (m)
        c_obs = np.maximum(0.0, 1.0 / np.maximum(clr, 1e-3) - 1.0 / DWA_DMAX)  # tang nhanh khi gan vat can
        c_vel = (V_MAX - V) / V_MAX                                       # [0, 1]
        J = (self.wt["heading"] * c_head + self.wt["path"] * c_path
             + self.wt["clear"] * c_obs + self.wt["vel"] * c_vel)
        J[~ok] = np.inf
        self.last = dict(X=X, Y=Y, ok=ok, J=J, V=V, W=W)
        if not ok.any():                    # khong co quy dao an toan: dung, quay tai cho
            return 0.0, np.sign(wrap(np.arctan2(gy - y, gx - x) - th)) * W_MAX * 0.5
        i = int(np.argmin(J))
        # chong ket: robot dung yen ma lenh tot nhat cung gan bang 0 -> quay tai cho
        if V[i] < 0.01 and abs(v) < 0.01 and abs(W[i]) < 0.05 and not near_goal:
            self.stuck += 1
        else:
            self.stuck = 0
        if self.stuck > 10:
            return 0.0, self.escape_dir * W_MAX
        return float(V[i]), float(W[i])


# ------------------------------------------------------------------
# 6. Mo phong nhiem vu don dep
# ------------------------------------------------------------------
def run_mission(world, legs, controller="PP", pp_L=PP_L, dwa=None, t_max=900.0,
                stop_on_collision=True, record=True):
    """Robot di lan luot cac chang A*. Tai moi diem dich robot quay tai cho
    huong theo chang tiep theo (giong nhau cho ca hai bo dieu khien)."""
    p0 = legs[0][0]
    d0 = legs[0][1] - legs[0][0]
    st = np.array([p0[0], p0[1], np.arctan2(d0[1], d0[0])])
    v = w = 0.0
    t, dist = 0.0, 0.0
    log = dict(t=[], x=[], y=[], th=[], v=[], w=[], cte=[], clr=[], leg=[])
    comp, n_coll, in_coll, collided_at = [], 0, False, None
    reached = 0
    if controller == "DWA" and dwa is None:
        dwa = DWA(world)
    for li, leg in enumerate(legs):
        tr = PathTracker(leg)
        # quay tai cho huong ve doan dau cua chang
        d = leg[1] - leg[0]
        while abs(wrap(np.arctan2(d[1], d[0]) - st[2])) > np.deg2rad(5) and t < t_max:
            e = wrap(np.arctan2(d[1], d[0]) - st[2])
            v, w = actuator(0.0, np.clip(3 * e, -W_MAX, W_MAX), v, w)
            st = unicycle(st, v, w); t += DT
            _log(log, t, st, v, w, 0.0, world.clearance(st[0], st[1], t), li) if record else None
        while t < t_max:
            cte = tr.update(st[0], st[1])
            if np.hypot(*(leg[-1] - st[:2])) < GOAL_TOL:
                reached += 1
                break
            c0 = time.perf_counter()
            if controller == "PP":
                vc, wc = pure_pursuit(st, tr, pp_L)
            else:
                vc, wc = dwa.control(st, v, w, tr, t)
            comp.append(time.perf_counter() - c0)
            v, w = actuator(vc, wc, v, w)
            st = unicycle(st, v, w); t += DT; dist += abs(v) * DT
            clr = float(world.clearance(st[0], st[1], t))
            if record:
                _log(log, t, st, v, w, cte, clr, li)
            if clr < 0 and not in_coll:
                n_coll += 1
                if collided_at is None:
                    collided_at = (float(st[0]), float(st[1]), t)
            in_coll = clr < 0
            if clr < 0 and stop_on_collision:
                break
        if collided_at is not None and stop_on_collision:
            break
        if t >= t_max:
            break
    for k in log:
        log[k] = np.array(log[k])
    moving = log["v"] > 1e-3 if record else None
    res = dict(success=(reached == len(legs) and n_coll == 0), reached=reached, n_legs=len(legs),
               time=t, dist=dist, collisions=n_coll, collided_at=collided_at,
               comp_ms=1000 * float(np.mean(comp)) if comp else 0.0,
               comp_max_ms=1000 * float(np.max(comp)) if comp else 0.0, log=log)
    if record and len(log["t"]):
        res.update(cte_rms=float(np.sqrt(np.mean(log["cte"] ** 2))), cte_max=float(log["cte"].max()),
                   min_clear=float(log["clr"].min()),
                   w_rough=float(np.mean(np.abs(np.diff(log["w"]))) / DT),  # do em: |dw/dt| tb
                   v_mean=float(log["v"][moving].mean()) if moving.any() else 0.0)
    return res


def _log(log, t, st, v, w, cte, clr, li):
    for k, val in zip(("t", "x", "y", "th", "v", "w", "cte", "clr", "leg"),
                      (t, st[0], st[1], st[2], v, w, cte, float(clr), li)):
        log[k].append(val)


# ------------------------------------------------------------------
# 7. Kich ban: can ho 2 phong 8 x 6 m
# ------------------------------------------------------------------
FURNITURE = [
    (4.6, 0.0, 4.75, 2.3), (4.6, 3.3, 4.75, 6.0),   # tuong ngan, cua rong 1 m
    (0.3, 4.6, 2.5, 5.7),    # sofa
    (1.4, 2.2, 2.6, 3.0),    # ban tra
    (3.4, 5.2, 4.6, 6.0),    # ke TV
    (5.6, 3.6, 7.7, 5.7),    # giuong
    (4.75, 0.0, 5.8, 0.6),   # tu quan ao
    (6.9, 0.3, 7.7, 1.1),    # ban lam viec
]
DOCK = (0.40, 0.40)
ZONES = {                     # cac khu vuc can don (zone cleaning)
    "Z1 - Truoc sofa": (1.20, 4.10),
    "Z2 - Goc ke TV": (3.90, 4.70),
    "Z3 - Canh giuong": (6.60, 3.05),
    "Z4 - Ban lam viec": (6.20, 1.50),
}
UNKNOWN = [                   # vat can la: do dac bi de sai cho, chua co trong ban do
    (0.80, 1.85, 1.10, 2.15),     # gio do choi
    (5.35, 2.62, 5.70, 2.92),     # thung carton trong phong ngu
    (2.35, 1.00, 2.75, 1.40),     # ghe bi keo ra
]
PET = dict(p0=(3.0, 0.5), p1=(3.0, 3.1), v=0.25, r=0.15)   # vat nuoi chay qua lai


def route_points():
    return [DOCK] + list(ZONES.values()) + [DOCK]


def scenarios(pet_phase=0.0):
    kb1 = World(8.0, 6.0, FURNITURE, name="KB1")
    kb2 = World(8.0, 6.0, FURNITURE, unknown=UNKNOWN, name="KB2")
    kb3 = World(8.0, 6.0, FURNITURE,
                movers=[Mover(PET["p0"], PET["p1"], PET["v"], PET["r"], pet_phase)], name="KB3")
    return kb1, kb2, kb3
