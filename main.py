from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import scipy.signal as sig
from scipy import interpolate
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei', 'WenQuanYi Zen Hei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import uvicorn

# 设置 matplotlib 中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

app = FastAPI(title="PLL 开环分析 API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== 请求参数模型 =====
class PLLParams(BaseModel):
    Ip_mA: float = 15.0
    Kvco_MHz: float = 187.2
    N: float = 140.0
    C1_nF: float = 0.56
    C2_nF: float = 330.0
    C3_nF: float = 1.57
    C4_nF: float = 0.0
    R2: float = 47.0
    R3: float = 15.0
    R4: float = 0.0

# ===== 核心计算函数 =====
def calculate_pll(params: PLLParams):
    Ip = params.Ip_mA * 1e-3
    Kvco = params.Kvco_MHz * 1e6 * 2 * np.pi
    C1 = params.C1_nF * 1e-9
    C2 = params.C2_nF * 1e-9
    C3 = params.C3_nF * 1e-9
    C4 = params.C4_nF * 1e-9
    R2 = params.R2
    R3 = params.R3
    R4 = params.R4

    A0 = C1 + C2 + C3 + C4
    A1 = (C2 * R2 * (C1 + C3 + C4) + 
          R3 * (C1 + C2) * (C3 + C4) + 
          C4 * R4 * (C1 + C2 + C3))
    A2 = (C1 * C2 * R2 * R3 * (C3 + C4) + 
          C4 * R4 * (C2 * C3 * R3 + C1 * C3 * R3 + C1 * C2 * R2 + C2 * C3 * R2))
    A3 = C1 * C2 * C3 * C4 * R2 * R3 * R4

    K = Ip / (2 * np.pi) * Kvco / params.N
    num_G = [K * R2 * C2, K]
    den_G = [A3, A2, A1, A0, 0, 0]

    # 修复：tf2zpk 返回三个值
    zeros, poles, _ = sig.tf2zpk(num_G, den_G)
    zeros_positive = [abs(z) for z in zeros if abs(z) > 1e-6]
    poles_positive = [abs(p) for p in poles if abs(p) > 1e-6]
    zeros_positive.sort()
    poles_positive.sort()

    zeros_khz = [f"{z/(2*np.pi*1e3):.3f}" for z in zeros_positive]
    zeros_text = ", ".join(zeros_khz) if zeros_khz else "无"

    # 极点显示
    poles_mhz = [p/(2*np.pi*1e6) for p in poles_positive]
    poles_text_list = []
    for i, p in enumerate(poles_mhz, start=1):
        # 从 i=1 开始，显示为极点3、极点4...
        poles_text_list.append(f"极点{i+2}: {p:.3f} MHz")
    poles_text = "\n".join(poles_text_list) if poles_text_list else "无"  

    freq_Hz = np.logspace(2, 8, 5000)
    w = 2 * np.pi * freq_Hz
    G = sig.TransferFunction(num_G, den_G)
    w_out, mag, phase = sig.bode(G, w=w)
    mag_db = mag
    phase_margin_deg = 180 + phase
    freq_khz = w_out / (2 * np.pi) / 1000

    valid_idx = np.where(np.isfinite(mag_db))[0]
    if len(valid_idx) > 1:
        f_interp_func = interpolate.interp1d(mag_db[valid_idx], w_out[valid_idx], 
                                              kind='linear', fill_value='extrapolate')
        f_cross_rad = float(f_interp_func(0))
        f_cross_khz = f_cross_rad / (2 * np.pi) / 1000
        phase_interp_func = interpolate.interp1d(w_out[valid_idx], phase_margin_deg[valid_idx],
                                                  kind='linear', fill_value='extrapolate')
        Pm = float(phase_interp_func(f_cross_rad))
    else:
        idx = np.argmin(np.abs(mag_db))
        f_cross_khz = w_out[idx] / (2 * np.pi) / 1000
        Pm = phase_margin_deg[idx]

    # 生成波特图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    # 幅频图
    ax1.semilogx(freq_khz, mag_db, 'b', linewidth=1.5)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1)
    # 标注环路带宽（穿越频率）
    if f_cross_khz > 0:
        ax1.plot(f_cross_khz, 0, 'ro', markersize=8)
        ax1.text(f_cross_khz, 5, f'{f_cross_khz:.3f} kHz', 
                 ha='center', va='bottom', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1, 1e5])
    ax1.set_ylabel("幅值 (dB)")
    ax1.set_title("开环幅频曲线")

    # 相频图
    ax2.semilogx(freq_khz, phase_margin_deg, 'r', linewidth=1.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([1, 1e5])
    ax2.set_xlabel("频率 (kHz)")
    ax2.set_ylabel("相位裕量 (deg)")
    ax2.set_title("开环相频曲线")
    # 标注相位裕量
    if f_cross_khz > 0 and not np.isnan(Pm):
        ax2.plot(f_cross_khz, Pm, 'ro', markersize=8)
        ax2.text(f_cross_khz, Pm - 8, f'PM = {Pm:.2f}°', 
                 ha='center', va='top', fontsize=9)
    
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)

    return {
        "bandwidth": round(f_cross_khz, 3),
        "phase_margin": round(Pm, 2),
        "zeros_text": zeros_text,
        "poles_text": poles_text,
        "plot_image": img_base64
    }

# ===== API 端点 =====
@app.get("/")
def root():
    return {"message": "PLL 开环分析 API 已启动", "docs": "/docs"}

@app.post("/api/calculate")
def calculate(params: PLLParams):
    try:
        result = calculate_pll(params)
        return JSONResponse(content={"success": True, "data": result})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=400)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
