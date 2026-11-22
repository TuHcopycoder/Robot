"""Chay toan bo thi nghiem (TN1-TN6), luu results.json va ve hinh vao figs/."""
import json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import nav_sim as ns

from matplotlib import font_manager
FONT = "Times New Roman"
if not any(f.name == FONT for f in font_manager.fontManager.ttflist):
    FONT = "Liberation Serif"   # may khong co Times New Roman (vd. Linux)
plt.rcParams.update({"font.family": FONT, "font.size": 11, "axes.titlesize": 11, "mathtext.fontset": "stix",
                     "savefig.dpi": 220, "savefig.bbox": "tight"})
os.makedirs("figs", exist_ok=True)
C_PP, C_DWA, C_DWAP, C_REF = "#c0392b", "#1f6fb0", "#1f8a5b", "#555555"
N_PHASE = 20                  # so lan chay KB3 (lech pha chuyen dong vat nuoi)


def clean(d):
    return {k: v for k, v in d.items() if k not in ("log", "closed")}


def draw_world(ax, w, unknown=True, legs=None, zones=True):
    ax.add_patch(Rectangle((0, 0), w.w, w.h, fill=False, lw=2, ec="k"))
    for (x0, y0, x1, y1) in w.rects:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#3d3d3d", ec="none"))
    if unknown:
        for (x0, y0, x1, y1) in w.unknown:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#f0a030", ec="#a05a00", hatch="///"))
    if legs is not None:
        for k, l in enumerate(legs):
            ax.plot(l[:, 0], l[:, 1], "--", color=C_REF, lw=1, label="Đường A*" if k == 0 else None)
    if zones:
        for name, p in ns.ZONES.items():
            ax.plot(*p, "*", color="#8e44ad", ms=11, zorder=6)
            ax.text(p[0] + 0.12, p[1] + 0.08, name.split(" ")[0], fontsize=9, color="#8e44ad", zorder=6, clip_on=True)
    ax.plot(*ns.DOCK, "s", color="#c0392b", ms=8, zorder=6)
    ax.set_xlim(0, w.w); ax.set_ylim(0, w.h); ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")


def main():
    t_start = time.time()
    kb1, kb2, kb3 = ns.scenarios()
    pts = ns.route_points()
    R = {}

    # ---------------- TN1: A* ----------------
    legs, st_a = ns.plan_route(kb1, pts)
    _, st_d = ns.plan_route(kb1, pts, heuristic="zero")
    R["astar"] = [dict(leg=i + 1, raw_len=a["raw_len"], len=a["len"], exp_astar=a["expanded"],
                       exp_dij=d["expanded"], len_dij=d["raw_len"], t_astar=a["time_ms"], t_dij=d["time_ms"],
                       n_wp=a["n_wp"]) for i, (a, d) in enumerate(zip(st_a, st_d))]
    infl = []
    for m in (0.0, 0.05, 0.10, 0.15):
        lg, st = ns.plan_route(kb1, pts, inflate=m)
        r = ns.run_mission(kb1, lg, "PP")
        infl.append(dict(inflate=m, len=sum(s["len"] for s in st), pp_min_clear=r["min_clear"],
                         pp_success=r["success"], pp_time=r["time"]))
    R["inflate"] = infl
    print("TN1 xong", round(time.time() - t_start, 1))

    # ---------------- TN2: KB1 PP vs DWA ----------------
    runs = {}
    for c in ("PP", "DWA"):
        runs[("KB1", c)] = ns.run_mission(kb1, legs, c)
    R["kb1"] = {c: clean(runs[("KB1", c)]) for c in ("PP", "DWA")}

    # TN3: khao sat L cua Pure Pursuit
    ppL = {}
    for L in (0.20, 0.35, 0.60):
        r = ns.run_mission(kb1, legs, "PP", pp_L=L)
        ppL[L] = r
    R["pp_L"] = [dict(L=L, **{k: v for k, v in clean(r).items()}) for L, r in ppL.items()]
    print("TN2-3 xong", round(time.time() - t_start, 1))

    # ---------------- TN4: KB2 vat can la ----------------
    for c in ("PP", "DWA"):
        runs[("KB2", c)] = ns.run_mission(kb2, legs, c)
    R["kb2"] = {c: clean(runs[("KB2", c)]) for c in ("PP", "DWA")}
    single = []
    names = ["Giỏ đồ chơi", "Thùng carton", "Ghế bị kéo ra"]
    for k, ob in enumerate(ns.UNKNOWN):
        w1 = ns.World(8.0, 6.0, ns.FURNITURE, unknown=[ob])
        rp, rd = ns.run_mission(w1, legs, "PP"), ns.run_mission(w1, legs, "DWA")
        single.append(dict(obstacle=names[k], pp_success=rp["success"], pp_collided=rp["collided_at"],
                           pp_min_clear=rp["min_clear"], dwa_success=rd["success"], dwa_min_clear=rd["min_clear"],
                           dwa_time=rd["time"]))
    R["kb2_single"] = single
    # TN6: anh huong thoi gian du bao T cua DWA (KB2)
    hor = []
    for T in (1.0, 1.5, 2.0, 3.0):
        r = ns.run_mission(kb2, legs, "DWA", dwa=ns.DWA(kb2, T=T), t_max=300)
        hor.append(dict(T=T, **clean(r)))
    R["dwa_T"] = hor
    print("TN4,6 xong", round(time.time() - t_start, 1))

    # ---------------- TN5: KB3 vat nuoi di chuyen ----------------
    ph_res = []
    for ph in np.arange(N_PHASE) / N_PHASE:
        w3 = ns.scenarios(ph)[2]
        row = dict(phase=float(ph))
        for name, kw in [("PP", dict(controller="PP")),
                         ("DWA", dict(controller="DWA", dwa=ns.DWA(w3))),
                         ("DWA+P", dict(controller="DWA", dwa=ns.DWA(w3, predict_movers=True)))]:
            r = ns.run_mission(w3, legs, t_max=300, **kw)
            stopped = None
            if r["collided_at"] is not None:
                stopped = bool(r["log"]["v"][-1] < 0.01)
            row[name] = dict(success=r["success"], time=r["time"], min_clear=r["min_clear"],
                             collided_at=r["collided_at"], robot_stopped_at_collision=stopped)
            if abs(ph - 0.5) < 1e-9:
                runs[("KB3", name)] = r
        ph_res.append(row)
    R["kb3"] = ph_res
    summ = {}
    for name in ("PP", "DWA", "DWA+P"):
        s = [p[name] for p in ph_res]
        ok = [x for x in s if x["success"]]
        summ[name] = dict(success=len(ok), n=len(s),
                          time_mean=float(np.mean([x["time"] for x in ok])) if ok else None,
                          clear_mean=float(np.mean([x["min_clear"] for x in ok])) if ok else None,
                          stopped_coll=sum(1 for x in s if x["robot_stopped_at_collision"]))
    R["kb3_summary"] = summ
    print("TN5 xong", round(time.time() - t_start, 1))

    json.dump(R, open("results.json", "w"), indent=1, ensure_ascii=False, default=float)
    figures(kb1, kb2, legs, runs, ppL, R)
    print("Tong thoi gian", round(time.time() - t_start, 1), "s")


def figures(kb1, kb2, legs, runs, ppL, R):
    # Hinh: kich ban
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    w3 = ns.scenarios(0.0)[2]
    for ax, w, ttl in zip(axs, (kb1, kb2, w3), ("KB1 – Bản đồ đúng, môi trường tĩnh",
                                               "KB2 – Có 3 vật cản lạ (chưa có trong bản đồ)",
                                               "KB3 – Có vật nuôi di chuyển")):
        draw_world(ax, w, unknown=(w is kb2), legs=legs)
        ax.set_title(ttl)
    p0, p1 = np.array(ns.PET["p0"]), np.array(ns.PET["p1"])
    axs[2].annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<->", color="#8b4513", lw=1.5))
    axs[2].add_patch(Circle(w3.movers[0].pos(0), ns.PET["r"], fc="#d2a679", ec="#8b4513"))
    axs[2].text(3.12, 0.6, "vật nuôi\n0,25 m/s", fontsize=9, color="#8b4513")
    axs[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig("figs/fig_scenarios.png"); plt.close(fig)

    # Hinh: A* vs Dijkstra (chang 5)
    blocked = ns.inflated_grid(kb1)
    a, b = ns.route_points()[4], ns.route_points()[5]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, h, ttl in zip(axs, ("zero", "euclid"), ("Dijkstra (h = 0)", "A* (h = khoảng cách Euclid)")):
        raw, nexp, closed = ns.astar(blocked, a, b, h)
        draw_world(ax, kb1, unknown=False, zones=False)
        ax.imshow(np.where(blocked & ~kb1.occ, 1, np.nan), origin="lower", extent=(0, 8, 0, 6),
                  cmap="Greys", vmin=0, vmax=4, alpha=0.6, interpolation="nearest")
        ax.imshow(np.where(closed, 1.0, np.nan), origin="lower", extent=(0, 8, 0, 6), cmap="Blues",
                  vmin=0, vmax=1.6, alpha=0.55, interpolation="nearest")
        sm = ns.smooth_path(blocked, raw)
        ax.plot(raw[:, 0], raw[:, 1], "-", color="#c0392b", lw=1.6, label="Đường A* gốc")
        ax.plot(sm[:, 0], sm[:, 1], "--", color="k", lw=1.2, label="Sau khi rút gọn")
        ax.plot(*a, "o", color="#8e44ad", ms=7); ax.plot(*b, "s", color="#c0392b", ms=7)
        ax.set_title(f"{ttl}: {nexp} nút được mở rộng")
    axs[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig("figs/fig_astar.png"); plt.close(fig)

    # Hinh: KB1 quy dao PP vs DWA
    rp, rd = runs[("KB1", "PP")], runs[("KB1", "DWA")]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.35, 1]))
    draw_world(axs[0], kb1, legs=legs)
    axs[0].plot(rp["log"]["x"], rp["log"]["y"], color=C_PP, lw=1.3, label="Pure Pursuit")
    axs[0].plot(rd["log"]["x"], rd["log"]["y"], color=C_DWA, lw=1.3, label="DWA")
    axs[0].legend(loc="lower right", fontsize=8); axs[0].set_title("Toàn bộ lộ trình KB1")
    axs[0].add_patch(Rectangle((3.4, 2.5), 2.4, 2.5, fill=False, ec="k", lw=0.8, ls=":"))
    draw_world(axs[1], kb1, legs=legs)
    axs[1].plot(rp["log"]["x"], rp["log"]["y"], color=C_PP, lw=1.6)
    axs[1].plot(rd["log"]["x"], rd["log"]["y"], color=C_DWA, lw=1.6)
    axs[1].set_xlim(3.4, 5.8); axs[1].set_ylim(2.5, 5.0); axs[1].set_title("Phóng to: khúc rẽ qua cửa phòng ngủ")
    fig.tight_layout(); fig.savefig("figs/fig_kb1_paths.png"); plt.close(fig)

    # Hinh: v, w theo thoi gian KB1
    fig, axs = plt.subplots(3, 1, figsize=(11, 6.4), sharex=True)
    for r, c, lb in ((rp, C_PP, "Pure Pursuit"), (rd, C_DWA, "DWA")):
        L = r["log"]
        axs[0].plot(L["t"], L["v"], color=c, lw=1, label=lb)
        axs[1].plot(L["t"], L["w"], color=c, lw=1)
        axs[2].plot(L["t"], 100 * L["cte"], color=c, lw=1)
    axs[0].set_ylabel("v [m/s]"); axs[1].set_ylabel("ω [rad/s]"); axs[2].set_ylabel("Sai số ngang [cm]")
    axs[2].set_xlabel("Thời gian [s]"); axs[0].legend(fontsize=8, ncol=2, loc="lower right")
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("figs/fig_kb1_signals.png"); plt.close(fig)

    # Hinh: anh huong L cua PP
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    draw_world(ax, kb1, legs=legs, zones=False)
    for (L, r), c in zip(ppL.items(), ("#2a9d8f", C_PP, "#6a4c93")):
        ax.plot(r["log"]["x"], r["log"]["y"], color=c, lw=1.5, label=f"L = {L:.2f} m".replace(".", ","))
    ax.set_xlim(3.3, 5.2); ax.set_ylim(2.5, 5.0); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Pure Pursuit với các khoảng nhìn trước L")
    fig.savefig("figs/fig_pp_L.png"); plt.close(fig)

    # Hinh: KB2
    rp2, rd2 = runs[("KB2", "PP")], runs[("KB2", "DWA")]
    fig, ax0 = plt.subplots(figsize=(8.5, 6.3))
    draw_world(ax0, kb2, legs=legs)
    ax0.plot(rd2["log"]["x"], rd2["log"]["y"], color=C_DWA, lw=1.3, label="DWA")
    ax0.plot(rp2["log"]["x"], rp2["log"]["y"], color=C_PP, lw=2, label="Pure Pursuit")
    cx, cy, _ = rp2["collided_at"]
    ax0.plot(cx, cy, "X", color=C_PP, ms=12, mec="k", zorder=8, label="PP va chạm, dừng")
    ax0.legend(loc="lower right", fontsize=8)
    fig.savefig("figs/fig_kb2.png"); plt.close(fig)
    zooms = [((0.1, 1.6), (1.1, 2.6)), ((4.8, 6.3), (2.2, 3.7)), ((1.8, 3.3), (0.7, 2.2))]
    names = ["Giỏ đồ chơi", "Thùng carton", "Ghế bị kéo ra"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
    for k, (ax, ((x0, x1), (y0, y1))) in enumerate(zip(axs, zooms)):
        draw_world(ax, kb2, legs=legs, zones=False)
        ax.plot(rd2["log"]["x"], rd2["log"]["y"], color=C_DWA, lw=1.6, label="DWA")
        for c in np.arange(0, len(rd2["log"]["x"]), 12):
            if x0 <= rd2["log"]["x"][c] <= x1 and y0 <= rd2["log"]["y"][c] <= y1:
                ax.add_patch(Circle((rd2["log"]["x"][c], rd2["log"]["y"][c]), ns.R_ROBOT, fill=False, ec=C_DWA, lw=0.5, alpha=0.6))
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_title(f"DWA tránh: {names[k]}")
    axs[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig("figs/fig_kb2_zoom.png"); plt.close(fig)

    # Hinh: KB3 - pha 0.25: mot pha dien hinh
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1, 1.3]))
    w3 = ns.scenarios(0.5)[2]
    draw_world(axs[0], w3, unknown=False, legs=legs, zones=False)
    for name, c in (("PP", C_PP), ("DWA", C_DWA), ("DWA+P", C_DWAP)):
        r = runs[("KB3", name)]
        lab = {"PP": "Pure Pursuit", "DWA": "DWA", "DWA+P": "DWA + dự báo vật cản động"}[name]
        axs[0].plot(r["log"]["x"], r["log"]["y"], color=c, lw=1.5, label=lab)
        if r["collided_at"] is not None:
            axs[0].plot(r["collided_at"][0], r["collided_at"][1], "X", color=c, ms=11, mec="k", zorder=9)
        axs[1].plot(r["log"]["t"], 100 * r["log"]["clr"], color=c, lw=1.1, label=lab)
    mv = w3.movers[0]
    tt = runs[("KB3", "DWA+P")]["log"]["t"]
    for t in np.arange(52, 64, 2.0):
        axs[0].add_patch(Circle(mv.pos(t), mv.r, fc="#d2a679", ec="#8b4513", alpha=0.35))
    axs[0].set_xlim(1.8, 4.8); axs[0].set_ylim(0.3, 3.3)
    axs[0].legend(fontsize=8, loc="upper left"); axs[0].set_title("KB3: chặng Z4 → trạm sạc (vật nuôi: vòng tròn nâu)")
    axs[1].axhline(0, color="k", lw=0.8); axs[1].set_xlim(40, 75)
    axs[1].set_xlabel("Thời gian [s]"); axs[1].set_ylabel("Khoảng hở tới vật cản [cm]"); axs[1].grid(alpha=0.3)
    axs[1].legend(fontsize=8); axs[1].set_title("Khoảng hở d(t); dưới 0 là va chạm")
    fig.tight_layout(); fig.savefig("figs/fig_kb3.png"); plt.close(fig)

    # Hinh: tong hop ty le thanh cong
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    names = ["Pure Pursuit", "DWA", "DWA + dự báo"]
    kb1s = [R["kb1"]["PP"]["success"], R["kb1"]["DWA"]["success"], None]
    kb2s = [R["kb2"]["PP"]["success"], R["kb2"]["DWA"]["success"], None]
    s3 = [R["kb3_summary"][k]["success"] / R["kb3_summary"][k]["n"] for k in ("PP", "DWA", "DWA+P")]
    x = np.arange(3); wb = 0.25
    vals = [[100 * float(kb1s[0]), 100 * float(kb2s[0]), 100 * s3[0]],
            [100 * float(kb1s[1]), 100 * float(kb2s[1]), 100 * s3[1]],
            [np.nan, np.nan, 100 * s3[2]]]
    for k, (v, c) in enumerate(zip(vals, (C_PP, C_DWA, C_DWAP))):
        bars = axs[0].bar(x + (k - 1) * wb, v, wb, color=c, label=names[k])
        for b_, val in zip(bars, v):
            if not np.isnan(val):
                axs[0].text(b_.get_x() + b_.get_width() / 2, val + 2, f"{val:.0f}", ha="center", fontsize=8)
    axs[0].set_xticks(x); axs[0].set_xticklabels(["KB1", "KB2", f"KB3 ({N_PHASE} lần)"])
    axs[0].set_ylabel("Tỷ lệ hoàn thành [%]"); axs[0].set_ylim(0, 130); axs[0].legend(fontsize=8, loc="upper left", ncol=3)
    axs[0].set_title("Tỷ lệ hoàn thành nhiệm vụ không va chạm")
    ms = [R["kb1"]["PP"]["comp_ms"], R["kb1"]["DWA"]["comp_ms"]]
    axs[1].bar([0, 1], ms, color=[C_PP, C_DWA], width=0.5)
    axs[1].set_yscale("log"); axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(["Pure Pursuit", "DWA"])
    for k, m in enumerate(ms):
        axs[1].text(k, m * 1.3, f"{m:.3f} ms".replace(".", ","), ha="center", fontsize=9)
    axs[1].set_ylabel("Thời gian tính / chu kỳ [ms]"); axs[1].set_title("Chi phí tính toán (KB1)")
    axs[1].set_ylim(min(ms) / 3, max(ms) * 5)
    fig.tight_layout(); fig.savefig("figs/fig_summary.png"); plt.close(fig)

    # Hinh: minh hoa cua so dong cua DWA (tai luc gap gio do choi)
    d = ns.DWA(kb2)
    tr = ns.PathTracker(legs[0])
    L = runs[("KB2", "DWA")]["log"]
    k = int(np.argmin(np.abs(L["y"] - 1.45) + 10 * (L["leg"] != 0)))
    st = np.array([L["x"][k], L["y"][k], L["th"][k]])
    for j in range(k + 1):
        tr.update(L["x"][j], L["y"][j])
    vv, ww = d.control(st, L["v"][k], L["w"][k], tr, L["t"][k])
    D = d.last
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axs[0]
    ax.add_patch(Rectangle((0, -ns.W_MAX), ns.V_MAX, 2 * ns.W_MAX, fc="#f2f2f2", ec="k", lw=0.8))
    ax.text(0.01, ns.W_MAX - 0.18, "V_s: giới hạn của robot", fontsize=9)
    v0, w0 = L["v"][k], L["w"][k]
    ax.add_patch(Rectangle((max(0, v0 - ns.A_V * ns.DT), w0 - ns.A_W * ns.DT), min(ns.V_MAX, v0 + ns.A_V * ns.DT) - max(0, v0 - ns.A_V * ns.DT),
                           2 * ns.A_W * ns.DT, fc="#dde9f6", ec=C_DWA, lw=1.2))
    sc = ax.scatter(D["V"][D["ok"]], D["W"][D["ok"]], c=D["J"][D["ok"]], cmap="viridis_r", s=18, zorder=5)
    ax.scatter(D["V"][~D["ok"]], D["W"][~D["ok"]], marker="x", color=C_PP, s=18, zorder=5, label="Không chấp nhận (va chạm/không kịp phanh)")
    ax.plot(vv, ww, "o", ms=11, mfc="none", mec="k", mew=1.8, zorder=6, label="Lệnh được chọn")
    ax.plot(v0, w0, "+", color="k", ms=10, label="(v, ω) hiện tại")
    fig.colorbar(sc, ax=ax, label="Chi phí J")
    ax.set_xlim(-0.02, 0.34); ax.set_ylim(-1.6, 1.6); ax.set_xlabel("v [m/s]"); ax.set_ylabel("ω [rad/s]")
    ax.legend(fontsize=7.5, loc="lower left"); ax.set_title("Không gian vận tốc: cửa sổ động V_d (khung xanh)")
    ax = axs[1]
    draw_world(ax, kb2, legs=legs, zones=False)
    for i in range(len(D["V"])):
        ax.plot(D["X"][i], D["Y"][i], color=("#9bbbd8" if D["ok"][i] else "#e6a5a0"), lw=0.6)
    i = int(np.argmin(D["J"]))
    ax.plot(D["X"][i], D["Y"][i], color="k", lw=2, label="Quỹ đạo tốt nhất")
    ax.add_patch(Circle(st[:2], ns.R_ROBOT, fill=False, ec=C_DWA, lw=1.2))
    ax.set_xlim(0.0, 1.6); ax.set_ylim(0.8, 2.6); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Các quỹ đạo dự báo (xanh: chấp nhận, đỏ: loại)")
    fig.tight_layout(); fig.savefig("figs/fig_dwa_window.png"); plt.close(fig)
    print("Ve hinh xong")


if __name__ == "__main__":
    main()
