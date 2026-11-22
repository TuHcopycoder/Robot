
# Robot vệ sinh tự động: A* + Pure Pursuit + DWA

Mã nguồn mô phỏng kèm báo cáo cuối kỳ học phần Kỹ thuật Robot.

| Tệp | Chức năng |
|---|---|
| `nav_sim.py` | Bản đồ căn hộ, A* (và Dijkstra), Pure Pursuit, DWA, mô hình robot vi sai, mô phỏng nhiệm vụ |
| `run_experiments.py` | Chạy thí nghiệm TN1–TN6, ghi `results.json`, vẽ hình vào `figs/` |
| `diagrams.py` | Vẽ sơ đồ: mô hình robot, sơ đồ khối, lưu đồ DWA |

Cài thư viện: `pip install numpy scipy matplotlib`

Chạy (Spyder: đặt thư mục làm việc là thư mục này, mở file và bấm F5):
```
python run_experiments.py   # ~20 s
python diagrams.py
```
Hình được lưu vào thư mục `figs/`, số liệu các bảng nằm trong `results.json`.
Thông số robot và thuật toán nằm ở đầu tệp `nav_sim.py`.
