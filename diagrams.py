"""Ve cac so do minh hoa: mo hinh robot, so do khoi he thong, luu do DWA."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrowPatch, Arc

FONT = "Times New Roman"
if not any(f.name == FONT for f in font_manager.fontManager.ttflist):
    FONT = "Liberation Serif"
plt.rcParams.update({"font.family": FONT, "font.size": 11, "savefig.dpi": 220, "savefig.bbox": "tight", "mathtext.fontset": "stix"})


def arrow(ax, p, q, c="k", cs="arc3"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.2, color=c,
                                 shrinkA=0, shrinkB=0, connectionstyle=cs))


# 1. Mo hinh robot vi sai
fig, ax = plt.subplots(figsize=(6.2, 4.6))
th = np.deg2rad(30); x0, y0 = 3.0, 2.0; R = 1.0
ax.add_patch(Circle((x0, y0), R, fc="#eef3f8", ec="k", lw=1.5))
u = np.array([np.cos(th), np.sin(th)]); n = np.array([-np.sin(th), np.cos(th)])
for s in (1, -1):
    c = np.array([x0, y0]) + s * 0.85 * n
    ax.add_patch(Rectangle((c[0] - 0.3, c[1] - 0.09), 0.6, 0.18, angle=np.rad2deg(th), rotation_point="center", fc="#333"))
ax.plot(x0, y0, "ko", ms=4)
arrow(ax, (x0, y0), (x0 + 1.9 * u[0], y0 + 1.9 * u[1])); ax.text(x0 + 2.0 * u[0], y0 + 2.0 * u[1], r"$v$", fontsize=13)
ax.plot([x0, x0 + 1.6], [y0, y0], "k--", lw=0.8)
ax.add_patch(Arc((x0, y0), 1.6, 1.6, theta1=0, theta2=30, lw=1)); ax.text(x0 + 0.9, y0 + 0.15, r"$\theta$", fontsize=13)
ax.add_patch(Arc((x0, y0), 0.7, 0.7, theta1=60, theta2=330, lw=1, color="#c0392b"))
ax.text(x0 - 0.55, y0 + 0.45, r"$\omega$", color="#c0392b", fontsize=13)
cl = np.array([x0, y0]) + 0.85 * n; cr = np.array([x0, y0]) - 0.85 * n
arrow(ax, cl, cl + 0.9 * u, c="#2a6fb0"); ax.text(*(cl + 0.95 * u + [0, 0.05]), r"$v_L$", color="#2a6fb0", fontsize=12)
arrow(ax, cr, cr + 1.3 * u, c="#2a6fb0"); ax.text(*(cr + 1.35 * u), r"$v_R$", color="#2a6fb0", fontsize=12)
ax.annotate("", xy=cl, xytext=cr, arrowprops=dict(arrowstyle="<->", lw=0.8))
ax.text(*((cl + cr) / 2 + [-0.45, -0.35]), r"$b$", fontsize=12)
arrow(ax, (0.2, 0.2), (1.4, 0.2)); arrow(ax, (0.2, 0.2), (0.2, 1.4))
ax.text(1.45, 0.1, r"$x$"); ax.text(0.05, 1.45, r"$y$"); ax.text(x0 + 0.08, y0 - 0.25, r"$(x, y)$", fontsize=11)
ax.set_xlim(0, 6); ax.set_ylim(0, 4.6); ax.set_aspect("equal"); ax.axis("off")
fig.savefig("figs/fig_robot_model.png"); plt.close(fig)


def box(ax, x, y, w, h, t, fc="#eef3f8", fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", fc=fc, ec="k", lw=1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)


# 2. So do khoi he thong
fig, ax = plt.subplots(figsize=(11.5, 3.9)); ax.axis("off"); ax.set_xlim(0, 11.5); ax.set_ylim(0, 3.9)
box(ax, 0.1, 2.5, 1.8, 1.0, "Bản đồ lưới\nchiếm chỗ (5 cm)", "#f4efe3")
box(ax, 2.3, 2.5, 1.9, 1.0, "Trạm sạc +\ncác khu vực cần dọn", "#f4efe3")
box(ax, 4.6, 2.5, 2.1, 1.0, "A*: lập đường\ntoàn cục từng chặng\n+ rút gọn đường", "#fff3d6")
box(ax, 7.1, 2.9, 2.0, 0.85, "Pure Pursuit", "#fde3e0")
box(ax, 7.1, 1.75, 2.0, 0.85, "DWA\n(tránh vật cản cục bộ)", "#dfeaf6")
box(ax, 9.6, 2.3, 1.8, 1.0, "Giới hạn động cơ\n(v, ω, gia tốc)", "#e6f2ea")
box(ax, 7.1, 0.2, 2.0, 0.95, "Mô hình động học\nrobot vi sai", "#e6f2ea")
box(ax, 4.6, 0.2, 2.1, 0.95, "LiDAR mô phỏng:\nvật cản lạ, vật nuôi", "#fbeaea")
box(ax, 0.1, 0.2, 2.6, 0.95, "Chỉ số đánh giá: thành công,\nthời gian, sai số, khoảng hở", "#fbeaea")
arrow(ax, (1.9, 3.0), (2.3, 3.0)); arrow(ax, (4.2, 3.0), (4.6, 3.0))
arrow(ax, (6.7, 3.15), (7.1, 3.3)); arrow(ax, (6.7, 2.85), (7.1, 2.2))
ax.text(4.65, 2.2, "đường tham chiếu", fontsize=8.5)
arrow(ax, (9.1, 3.3), (9.6, 3.0)); arrow(ax, (9.1, 2.2), (9.6, 2.6)); ax.text(9.15, 3.4, "v, ω", fontsize=9)
arrow(ax, (10.5, 2.3), (9.1, 0.7), cs="arc3,rad=-0.2")
arrow(ax, (7.1, 0.45), (6.7, 0.45)); ax.text(6.75, 0.6, "(x, y, θ)", fontsize=8.5)
arrow(ax, (6.7, 0.95), (7.6, 1.75), cs="arc3,rad=-0.25"); ax.text(6.2, 1.35, "vật cản đo được", fontsize=8.5)
arrow(ax, (4.6, 0.67), (2.7, 0.67))
ax.text(8.1, 2.66, "hoặc", fontsize=9, style="italic", color="#555", ha="center")
fig.savefig("figs/fig_block.png"); plt.close(fig)

# 3. Luu do DWA
fig, ax = plt.subplots(figsize=(7.6, 9.4)); ax.axis("off"); ax.set_xlim(0, 7.6); ax.set_ylim(0.2, 9.6)
def node(x, y, t, kind="box", w=4.6, h=0.7):
    if kind == "dia":
        ax.add_patch(plt.Polygon([[x, y + h / 2], [x + w / 2, y], [x, y - h / 2], [x - w / 2, y]], fc="#fff6e0", ec="k"))
    elif kind == "term":
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.3", fc="#e6f2ea", ec="k"))
    else:
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc="#eef3f8", ec="k"))
    ax.text(x, y, t, ha="center", va="center", fontsize=9.5)
X = 3.1
node(X, 9.15, "Mỗi chu kỳ Δt = 0,1 s: đọc trạng thái (x, y, θ, v, ω),\nvật cản đo được, đường A*", "term", 5.4, 0.75)
node(X, 8.05, "Tính cửa sổ động $V_d = V_s \\cap [v \\pm a_v\\Delta t] \\times [\\omega \\pm a_\\omega\\Delta t]$;\nlấy mẫu 7 × 21 cặp (v, ω)", "box", 5.6, 0.75)
node(X, 6.95, "Với mỗi cặp (v, ω): dự báo quỹ đạo trong T = 1,5 s\nbằng mô hình động học (2.4)", "box", 5.6, 0.75)
node(X, 5.85, "Loại quỹ đạo va chạm hoặc không kịp phanh:\n$d_{min} \\leq$ 2 cm hoặc $v > \\sqrt{2 a_v d_{min}}$", "box", 5.6, 0.75)
node(X, 4.55, "Còn quỹ đạo\nchấp nhận được?", "dia", 3.2, 1.25)
node(X, 3.15, "Tính chi phí $J = k_h C_h + k_p C_p + k_o C_o + k_v C_v$;\nchọn $(v^*, \\omega^*)$ có J nhỏ nhất", "box", 4.9, 0.75)
node(X, 1.95, "Robot đứng yên quá 1 s? → quay tại chỗ (chống kẹt)", "box", 5.6, 0.6)
node(X, 0.85, "Gửi $(v^*, \\omega^*)$ xuống động cơ, lặp chu kỳ sau", "term", 5.0, 0.6)
node(6.75, 3.15, "Dừng,\nquay về\nhướng đích", "box", 1.3, 1.0)
for a, b in [((X, 8.77), (X, 8.43)), ((X, 7.67), (X, 7.33)), ((X, 6.57), (X, 6.23)), ((X, 5.47), (X, 5.18)),
             ((X, 3.92), (X, 3.53)), ((X, 2.77), (X, 2.25)), ((X, 1.65), (X, 1.15))]:
    arrow(ax, a, b)
ax.text(X + 0.1, 3.7, "Có", fontsize=9)
ax.plot([X + 1.6, 6.75], [4.55, 4.55], "k", lw=1.2); arrow(ax, (6.75, 4.55), (6.75, 3.65)); ax.text(X + 1.7, 4.65, "Không", fontsize=9)
ax.plot([6.75, 6.75], [2.65, 0.85], "k", lw=1.2); arrow(ax, (6.75, 0.85), (X + 2.5, 0.85))
fig.savefig("figs/fig_dwa_flow.png"); plt.close(fig)
print("ok")
