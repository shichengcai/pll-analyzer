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
import random

# 设置 matplotlib 中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

app = FastAPI(title="PLL开环分析 API")

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

# ===== 简化版计算函数（只返回带宽和相位裕量）=====
def calculate_pll_simple(params: PLLParams):
    """简化版计算，只返回带宽和相位裕量"""
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

    return f_cross_khz, Pm

# ===== 自动设计函数 =====
def design_pll(Ip_mA, Kvco_MHz, N, target_bw, target_pm, order=4, max_iter=10000):
    """根据目标带宽和相位裕量，自动搜索元件值"""
    
    # 根据阶数初始化元件值
    if order == 2:
        C1, C2, R2 = 1.0, 100.0, 50.0
        C3, C4, R3, R4 = 0, 0, 0, 0
        step_C1, step_C2, step_R2 = 0.1, 10.0, 1.0
    elif order == 3:
        C1, C2, C3 = 1.0, 100.0, 1.0
        R2, R3 = 50.0, 10.0
        C4, R4 = 0, 0
        step_C1, step_C2, step_C3 = 0.1, 10.0, 0.1
        step_R2, step_R3 = 1.0, 1.0
    else:  # order == 4
        C1, C2, C3, C4 = 1.0, 100.0, 1.0, 0.1
        R2, R3, R4 = 50.0, 10.0, 1.0
        step_C1, step_C2, step_C3, step_C4 = 0.1, 10.0, 0.1, 0.01
        step_R2, step_R3, step_R4 = 1.0, 1.0, 0.1

    best_err = float('inf')
    best_components = None
    best_bw = 0
    best_pm = 0

    for i in range(max_iter):
        # 构建临时 PLLParams 对象进行计算
        temp_params = PLLParams(
            Ip_mA=Ip_mA, Kvco_MHz=Kvco_MHz, N=N,
            C1_nF=C1, C2_nF=C2, C3_nF=C3, C4_nF=C4,
            R2=R2, R3=R3, R4=R4
        )
        try:
            bw, pm = calculate_pll_simple(temp_params)
        except:
            bw, pm = 0, 0

        if bw > 0 and pm > 0:
            err = abs(bw - target_bw) / target_bw + abs(pm - target_pm) / target_pm
            if err < best_err:
                best_err = err
                best_components = (C1, C2, C3, C4, R2, R3, R4)
                best_bw = bw
                best_pm = pm
                if err < 0.02:
                    break

        # 随机调整
        C1 += step_C1 * (random.random() - 0.5) * 2
        C2 += step_C2 * (random.random() - 0.5) * 2
        if order >= 3:
            C3 += step_C3 * (random.random() - 0.5) * 2
            R3 += step_R3 * (random.random() - 0.5) * 2
        if order == 4:
            C4 += step_C4 * (random.random() - 0.5) * 2
            R4 += step_R4 * (random.random() - 0.5) * 2
        R2 += step_R2 * (random.random() - 0.5) * 2

        # 限制范围
        if order == 2:
            C1, C2, R2 = max(0.01, C1), max(0.1, C2), max(0.1, R2)
        elif order == 3:
            C1, C2, C3 = max(0.01, C1), max(0.1, C2), max(0.01, C3)
            R2, R3 = max(0.1, R2), max(0.1, R3)
        else:
            C1, C2, C3, C4 = max(0.01, C1), max(0.1, C2), max(0.01, C3), max(0.001, C4)
            R2, R3, R4 = max(0.1, R2), max(0.1, R3), max(0.01, R4)

    if best_components is None:
        return None

    C1, C2, C3, C4, R2, R3, R4 = best_components
    if order == 2:
        return {"C1_nF": C1, "C2_nF": C2, "C3_nF": 0, "C4_nF": 0, "R2": R2, "R3": 0, "R4": 0, "bw": best_bw, "pm": best_pm}
    elif order == 3:
        return {"C1_nF": C1, "C2_nF": C2, "C3_nF": C3, "C4_nF": 0, "R2": R2, "R3": R3, "R4": 0, "bw": best_bw, "pm": best_pm}
    else:
        return {"C1_nF": C1, "C2_nF": C2, "C3_nF": C3, "C4_nF": C4, "R2": R2, "R3": R3, "R4": R4, "bw": best_bw, "pm": best_pm}

# ===== 完整计算函数 =====
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

    zeros, poles, _ = sig.tf2zpk(num_G, den_G)
    zeros_positive = [abs(z) for z in zeros if abs(z) > 1e-6]
    poles_positive = [abs(p) for p in poles if abs(p) > 1e-6]
    zeros_positive.sort()
    poles_positive.sort()

    zeros_khz = [f"{z/(2*np.pi*1e3):.3f}" for z in zeros_positive]
    zeros_text = ", ".join(zeros_khz) if zeros_khz else "无"

    poles_mhz = [p/(2*np.pi*1e6) for p in poles_positive]
    poles_text_list = []
    for i, p in enumerate(poles_mhz, start=1):
        poles_text_list.append(f"极点{i+2}: {p:.3f}")
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
    
    ax1.semilogx(freq_khz, mag_db, 'b', linewidth=1.5)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1)
    if f_cross_khz > 0:
        ax1.plot(f_cross_khz, 0, 'ro', markersize=8)
        ax1.text(f_cross_khz, 5, f'{f_cross_khz:.3f} kHz', 
                 ha='center', va='bottom', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1, 1e5])
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_title("Open-loop Magnitude")

    ax2.semilogx(freq_khz, phase_margin_deg, 'r', linewidth=1.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([1, 1e5])
    ax2.set_xlabel("Frequency (kHz)")
    ax2.set_ylabel("Phase Margin (deg)")
    ax2.set_title("Open-loop Phase")
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

# ===== 设计接口参数模型 =====
class DesignParams(BaseModel):
    Ip_mA: float = 15.0
    Kvco_MHz: float = 187.2
    N: float = 140.0
    target_bw: float = 150.0
    target_pm: float = 60.0
    order: int = 4

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

@app.post("/api/design")
def design(params: DesignParams):
    try:
        result = design_pll(
            params.Ip_mA, params.Kvco_MHz, params.N,
            params.target_bw, params.target_pm,
            params.order, max_iter=10000
        )
        if result is None:
            return JSONResponse(content={"success": False, "error": "设计失败，请调整参数"}, status_code=400)
        
        analysis_params = PLLParams(
            Ip_mA=params.Ip_mA,
            Kvco_MHz=params.Kvco_MHz,
            N=params.N,
            C1_nF=result["C1_nF"],
            C2_nF=result["C2_nF"],
            C3_nF=result["C3_nF"],
            C4_nF=result["C4_nF"],
            R2=result["R2"],
            R3=result["R3"],
            R4=result["R4"]
        )
        analysis_result = calculate_pll(analysis_params)
        
        return JSONResponse(content={
            "success": True,
            "components": {
                "C1_nF": round(result["C1_nF"], 3),
                "C2_nF": round(result["C2_nF"], 3),
                "C3_nF": round(result["C3_nF"], 3),
                "C4_nF": round(result["C4_nF"], 3),
                "R2": round(result["R2"], 1),
                "R3": round(result["R3"], 1),
                "R4": round(result["R4"], 1),
            },
            "analysis": analysis_result
        })
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=400)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
