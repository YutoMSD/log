import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ==========================================
# 1. バンド構造計算エンジン ($N+M$ 層対応)
# ==========================================
def calculate_bands(theta_deg, N_top, M_bot, delta_V, cutoff_N=2, k_density=60):
    """
    上層 N_top 層 (AB積層) + 下層 M_bot 層 (AB積層) のねじれ多層グラフェンのバンド計算
    """
    # 基本物理定数
    gamma1 = 380.0       # AB-stacking interlayer hopping [meV]
    # omega = 110.7        # Moiré interlayer hopping parameter [meV]
    w_AA = 79.7   # Intrasublattice moiré potential [meV]
    w_AB = 97.5   # Intersublattice moiré potential [meV]
    d = 1.420            # Carbon-carbon distance [Angstrom]
    hv = 1.5 * d * 2970  # Fermi velocity [meV * Angstrom]
    valley = 1           # K valley

    theta = theta_deg * np.pi / 180.0
    I = complex(0, 1)
    ei120 = np.cos(2*np.pi/3) + valley*I*np.sin(2*np.pi/3)
    ei240 = np.cos(2*np.pi/3) - valley*I*np.sin(2*np.pi/3)

    # 逆格子ベクトルとディラック点
    b1m = 8*np.pi*np.sin(theta/2)/3/d * np.array([0.5, -np.sqrt(3)/2])
    b2m = 8*np.pi*np.sin(theta/2)/3/d * np.array([0.5, np.sqrt(3)/2])
    qb  = 8*np.pi*np.sin(theta/2)/3/np.sqrt(3)/d * np.array([0, -1])
    K1  = 8*np.pi*np.sin(theta/2)/3/np.sqrt(3)/d * np.array([-np.sqrt(3)/2, -0.5])
    K2  = 8*np.pi*np.sin(theta/2)/3/np.sqrt(3)/d * np.array([-np.sqrt(3)/2, 0.5])


    Tqb  = np.array([[w_AA, w_AB], [w_AB, w_AA]], dtype=complex)
    Tqtr = np.array([[w_AA*ei120, w_AB], [w_AB*ei240, w_AA*ei120]], dtype=complex)
    Tqtl = np.array([[w_AA*ei240, w_AB], [w_AB*ei120, w_AA*ei240]], dtype=complex)
    TqbD, TqtrD, TqtlD = Tqb.conj().T, Tqtr.conj().T, Tqtl.conj().T

    # k空間グリッド構築
    L_grid = []
    invL = np.zeros((2*cutoff_N+1, 2*cutoff_N+1), int)
    count = 0
    for i in range(-cutoff_N, cutoff_N+1):
        for j in range(-cutoff_N, cutoff_N+1):
            L_grid.append([i, j])
            invL[i+cutoff_N, j+cutoff_N] = count
            count += 1
    L_grid = np.array(L_grid)
    siteN = (2*cutoff_N+1)**2

    total_layers = N_top + M_bot
    dim = 2 * total_layers * siteN
    H_base = np.zeros((dim, dim), dtype=complex)

# (A) 各ブロック内の AB 積層 (Bernal stacking) ホッピング (\gamma_1)
    
    # 上層ブロック内 (Layer 0 to N_top - 2)
    for l in range(N_top - 1):
        for i in range(siteN):
            if l % 2 == 0:
                idx_l1 = 2 * (l * siteN + i) + 1         # Layer l, Sub B
                idx_l2 = 2 * ((l + 1) * siteN + i)       # Layer l+1, Sub A
            else:
                idx_l1 = 2 * (l * siteN + i)             # Layer l, Sub A
                idx_l2 = 2 * ((l + 1) * siteN + i) + 1   # Layer l+1, Sub B
                
            H_base[idx_l2, idx_l1] = gamma1
            H_base[idx_l1, idx_l2] = gamma1

    # 下層ブロック内 (Layer N_top to total_layers - 2)
    for l in range(N_top, total_layers - 1):
        for i in range(siteN):
            # 下層ブロックの1層目を基準として偶奇を判定する
            if (l - N_top) % 2 == 0:
                idx_l1 = 2 * (l * siteN + i) + 1         # Layer l, Sub B
                idx_l2 = 2 * ((l + 1) * siteN + i)       # Layer l+1, Sub A
            else:
                idx_l1 = 2 * (l * siteN + i)             # Layer l, Sub A
                idx_l2 = 2 * ((l + 1) * siteN + i) + 1   # Layer l+1, Sub B
                
            H_base[idx_l2, idx_l1] = gamma1
            H_base[idx_l1, idx_l2] = gamma1

    # (B) 上層と下層の接合面におけるモアレホッピング
    # 上層最下層 (Layer N_top - 1) <-> 下層最上層 (Layer N_top)
    top_interface = N_top - 1
    bot_interface = N_top

    for i in range(siteN):
        ix, iy = L_grid[i]
        idx_top = top_interface * siteN + i
        bot_base = bot_interface * siteN

        # TqbD (Top -> Bot)
        j = bot_base + i
        H_base[2*j:2*j+2, 2*idx_top:2*idx_top+2] = TqbD

        if iy != valley * cutoff_N:
            j = bot_base + invL[ix + cutoff_N, iy + valley*1 + cutoff_N]
            H_base[2*j:2*j+2, 2*idx_top:2*idx_top+2] = TqtrD

        if ix != -valley * cutoff_N:
            j = bot_base + invL[ix - valley*1 + cutoff_N, iy + cutoff_N]
            H_base[2*j:2*j+2, 2*idx_top:2*idx_top+2] = TqtlD

    for i in range(siteN):
        ix, iy = L_grid[i]
        idx_bot = (bot_interface * siteN) + i
        top_base = top_interface * siteN

        # Tqb (Bot -> Top)
        j = top_base + i
        H_base[2*j:2*j+2, 2*idx_bot:2*idx_bot+2] = Tqb

        if iy != -valley * cutoff_N:
            j = top_base + invL[ix + cutoff_N, iy - valley*1 + cutoff_N]
            H_base[2*j:2*j+2, 2*idx_bot:2*idx_bot+2] = Tqtr

        if ix != valley * cutoff_N:
            j = top_base + invL[ix + valley*1 + cutoff_N, iy + cutoff_N]
            H_base[2*j:2*j+2, 2*idx_bot:2*idx_bot+2] = Tqtl

    # (C) 垂直電場による各層のポテンシャル勾配
    # 全体での電位差 delta_V を全層に線形に分配
    if total_layers > 1:
        v_potentials = np.linspace(delta_V / 2.0, -delta_V / 2.0, total_layers)
        for l in range(total_layers):
            v_val = v_potentials[l]
            for i in range(siteN):
                idx = 2 * (l * siteN + i)
                H_base[idx, idx] += v_val
                H_base[idx+1, idx+1] += v_val

    # (D) k依存の対角成分 (ディラック項) 計算関数
    def get_eigenvalues(kx, ky):
        H = H_base.copy()
        
        # 上層グループ (回転角 +theta/2)
        for l in range(N_top):
            for i in range(siteN):
                ix, iy = L_grid[i]
                ax = kx - valley*K1[0] + ix*b1m[0] + iy*b2m[0]
                ay = ky - valley*K1[1] + ix*b1m[1] + iy*b2m[1]
                qx = np.cos(theta/2)*ax + np.sin(theta/2)*ay
                qy = -np.sin(theta/2)*ax + np.cos(theta/2)*ay
                
                val = hv * (valley*qx - I*qy)
                idx = 2 * (l * siteN + i)
                H[idx, idx+1] = val
                H[idx+1, idx] = np.conj(val)

        # 下層グループ (回転角 -theta/2)
        for l in range(N_top, total_layers):
            for i in range(siteN):
                ix, iy = L_grid[i]
                ax = kx - valley*K2[0] + ix*b1m[0] + iy*b2m[0]
                ay = ky - valley*K2[1] + ix*b1m[1] + iy*b2m[1]
                qx = np.cos(theta/2)*ax - np.sin(theta/2)*ay
                qy = np.sin(theta/2)*ax + np.cos(theta/2)*ay

                val = hv * (valley*qx - I*qy)
                idx = 2 * (l * siteN + i)
                H[idx, idx+1] = val
                H[idx+1, idx] = np.conj(val)

        return np.linalg.eigh(H)[0]

    # k経路の設定
    kD = -qb[1]
    KtoKp = np.arange(-1/2, 1/2, 1/k_density)
    KptoG = np.arange(1/2, 0, -1/2/k_density)
    GtoM  = np.arange(0, np.sqrt(3)/2, 1/k_density)
    MtoK  = np.arange(-np.sqrt(3)/4, -np.sqrt(3)/2, -1/k_density)

    k_path = []
    for k in KtoKp: k_path.append((np.sqrt(3)/2*kD, k*kD))
    for k in KptoG: k_path.append((np.sqrt(3)*k*kD, k*kD))
    for k in GtoM:  k_path.append((-1/2.0*k*kD, -np.sqrt(3)/2*k*kD))
    for k in MtoK:  k_path.append((k*kD, (-1/np.sqrt(3)*k-1.0)*kD))

    # 計算実行
    E = np.array([get_eigenvalues(kx, ky) for kx, ky in k_path])
    ticks = [0, len(KtoKp), len(KtoKp)+len(KptoG), len(KtoKp)+len(KptoG)+len(GtoM), len(k_path)-1]

    return E, ticks


# ==========================================
# 2. GUI アプリケーション部
# ==========================================
class MoiréBandApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Twisted Multilayer Graphene Band Simulator")
        self.root.geometry("1100x700")

        # UI レイアウト設定
        self.setup_ui()

    def setup_ui(self):
        # 左パネル: 入力コントロール
        control_frame = ttk.LabelFrame(self.root, text=" Parameters ", padding=15)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Twist Angle
        ttk.Label(control_frame, text="Twist Angle θ (°):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.entry_theta = ttk.Entry(control_frame, width=12)
        self.entry_theta.insert(0, "1.25")
        self.entry_theta.grid(row=0, column=1, pady=5)

        # Top Layers (N)
        ttk.Label(control_frame, text="Top Stack Layers (N):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.spin_N = ttk.Spinbox(control_frame, from_=1, to=5, width=10)
        self.spin_N.set(2)
        self.spin_N.grid(row=1, column=1, pady=5)

        # Bottom Layers (M)
        ttk.Label(control_frame, text="Bottom Stack Layers (M):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.spin_M = ttk.Spinbox(control_frame, from_=1, to=5, width=10)
        self.spin_M.set(2)
        self.spin_M.grid(row=2, column=1, pady=5)

        # Electric Field / Potential Difference
        ttk.Label(control_frame, text="Potential Bias ΔV (meV):").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.entry_deltaV = ttk.Entry(control_frame, width=12)
        self.entry_deltaV.insert(0, "0.0")
        self.entry_deltaV.grid(row=3, column=1, pady=5)

        # Energy Range Y-limit
        ttk.Label(control_frame, text="E Range (meV): ±").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.entry_erange = ttk.Entry(control_frame, width=12)
        self.entry_erange.insert(0, "50")
        self.entry_erange.grid(row=4, column=1, pady=5)

        # K-space Resolution
        ttk.Label(control_frame, text="K-Path Resolution:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.combo_kdens = ttk.Combobox(control_frame, values=["Low (40)", "Medium (60)", "High (100)"], state="readonly", width=11)
        self.combo_kdens.current(1)
        self.combo_kdens.grid(row=5, column=1, pady=5)

        # Run Button
        self.btn_calc = ttk.Button(control_frame, text="Plot Band Structure", command=self.start_thread)
        self.btn_calc.grid(row=6, column=0, columnspan=2, pady=20, sticky="ew")

        # Status Label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var, foreground="blue", wraplength=200)
        self.status_label.grid(row=7, column=0, columnspan=2, pady=10)

        # 右パネル: Matplotlib グラフ描画エリア
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ツールバーの追加 (拡大・保存機能など)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()

        # 初期描画の案内
        self.ax.text(0.5, 0.5, "Set parameters and click 'Plot Band Structure'", ha='center', va='center')
        self.canvas.draw()

    def start_thread(self):
        # 計算中にUIをブロックしないようにマルチスレッド化
        self.btn_calc.config(state=tk.DISABLED)
        self.status_var.set("Calculating... Please wait.")
        threading.Thread(target=self.run_simulation, daemon=True).start()

    def run_simulation(self):
        try:
            theta = float(self.entry_theta.get())
            N = int(self.spin_N.get())
            M = int(self.spin_M.get())
            delta_V = float(self.entry_deltaV.get())
            erange = float(self.entry_erange.get())
            
            k_map = {"Low (40)": 40, "Medium (60)": 60, "High (100)": 100}
            k_density = k_map[self.combo_kdens.get()]

            # バックグラウンドで計算を実行
            E, ticks = calculate_bands(theta, N, M, delta_V, cutoff_N=2, k_density=k_density)

            # メインスレッドで描画更新
            self.root.after(0, lambda: self.update_plot(E, ticks, theta, N, M, delta_V, erange))

        except ValueError:
            self.root.after(0, lambda: messagebox.showerror("Input Error", "Please enter valid numerical values."))
            self.root.after(0, lambda: self.reset_ui())
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.reset_ui())

    def update_plot(self, E, ticks, theta, N, M, delta_V, erange):
        self.ax.clear()
        
        num_k, num_bands = E.shape
        for j in range(num_bands):
            self.ax.plot(np.arange(num_k), E[:, j], color='royalblue', linewidth=1.2)

        self.ax.set_title(f"Twisted {N}+{M} Layer Graphene (θ={theta:.2f}°, ΔV={delta_V} meV)", fontsize=13)
        self.ax.set_xlim(0, num_k - 1)
        self.ax.set_ylim(-erange, erange)

        xticks_labels = ('K', "K$'$", '$\Gamma$', 'M', 'K')
        self.ax.set_xticks(ticks)
        self.ax.set_xticklabels(xticks_labels, fontsize=12)

        for pos in ticks:
            self.ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.8)

        self.ax.set_ylabel("Energy (meV)", fontsize=13)
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

        self.reset_ui()

    def reset_ui(self):
        self.btn_calc.config(state=tk.NORMAL)
        self.status_var.set("Ready")

if __name__ == "__main__":
    root = tk.Tk()
    app = MoiréBandApp(root)
    root.mainloop()