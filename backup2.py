# ============================================================================
# IMPORT LIBRARIES
# ============================================================================
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================================
# KONFIGURASI GLOBAL - MULTI-DOMAIN EXPERT SYSTEM
# ============================================================================
# --- Pump Standard Thresholds ---
PUMP_STANDARDS = {
    "API 610": {
        "velocity_limits": {
            "Zone A (Good)": 2.3,
            "Zone B (Acceptable)": 3.6,
            "Zone C (Unacceptable)": 5.7,
            "Zone D (Danger)": 9.0
        },
        "temp_limits": {
            "normal_max": 70,
            "elevated_max": 75,
            "warning_max": 85,
            "critical_min": 85
        },
        "bearing_life_hours": 25000,
        "severity_multiplier": 1.2,
        "description": "Heavy-duty untuk Oil & Gas, critical service"
    },
    "ISO 10816": {
        "velocity_limits": {
            "Zone A (Good)": 2.8,
            "Zone B (Acceptable)": 4.5,
            "Zone C (Unacceptable)": 7.1,
            "Zone D (Danger)": 11.0
        },
        "temp_limits": {
            "normal_max": 70,
            "elevated_max": 80,
            "warning_max": 90,
            "critical_min": 90
        },
        "bearing_life_hours": 17500,
        "severity_multiplier": 1.0,
        "description": "General purpose industrial, balanced reliability"
    },
    "ANSI/HI": {
        "velocity_limits": {
            "Zone A (Good)": 3.5,
            "Zone B (Acceptable)": 5.6,
            "Zone C (Unacceptable)": 8.9,
            "Zone D (Danger)": 14.0
        },
        "temp_limits": {
            "normal_max": 75,
            "elevated_max": 85,
            "warning_max": 95,
            "critical_min": 95
        },
        "bearing_life_hours": 10000,
        "severity_multiplier": 0.8,
        "description": "Light-duty untuk general service, cost-effective"
    }
}

# --- Mechanical Vibration Limits (Default ISO 10816) ---
ISO_LIMITS_VELOCITY = {
    "Zone A (Good)": 2.8,
    "Zone B (Acceptable)": 4.5,
    "Zone C (Unacceptable)": 7.1,
    "Zone D (Danger)": 11.0
}

ACCEL_BASELINE = {
    "Band1 (0.5-1.5kHz)": 0.3,
    "Band2 (1.5-5kHz)": 0.2,
    "Band3 (5-16kHz)": 0.15
}

# --- Bearing Temperature Thresholds (Default ISO) ---
BEARING_TEMP_LIMITS = {
    "normal_max": 70,
    "elevated_min": 70,
    "elevated_max": 80,
    "warning_min": 80,
    "warning_max": 90,
    "critical_min": 90,
    "delta_threshold": 15,
    "ambient_reference": 30
}

# --- Hydraulic Fluid Properties (BBM Specific - Pertamina) ---
FLUID_PROPERTIES = {
    "Pertalite (RON 90)": {
        "sg": 0.73,
        "vapor_pressure_kpa_38C": 52,
        "viscosity_cst_40C": 0.6,
        "flash_point_C": -43,
        "risk_level": "High"
    },
    "Pertamax (RON 92)": {
        "sg": 0.74,
        "vapor_pressure_kpa_38C": 42,
        "viscosity_cst_40C": 0.6,
        "flash_point_C": -43,
        "risk_level": "High"
    },
    "Diesel / Solar": {
        "sg": 0.84,
        "vapor_pressure_kpa_38C": 0.5,
        "viscosity_cst_40C": 3.0,
        "flash_point_C": 52,
        "risk_level": "Moderate"
    }
}

# --- Electrical Thresholds (IEC 60034-1 & Practical Limits) ---
ELECTRICAL_LIMITS = {
    "voltage_unbalance_warning": 1.0,
    "voltage_unbalance_critical": 2.0,
    "current_unbalance_warning": 5.0,
    "current_unbalance_critical": 8.0,
    "voltage_tolerance_low": 90,
    "voltage_tolerance_high": 110,
    "current_load_warning": 90,
    "current_load_critical": 100,
    "service_factor": 1.0
}

# ============================================================================
# PHYSICAL BOUNDS - FALSE POSITIVE FILTERING
# ============================================================================
PHYSICAL_BOUNDS = {
    "velocity_mm_s":        {"min": 0.0,   "max": 45.0,  "soft_max": 30.0},
    "acceleration_g":       {"min": 0.0,   "max": 50.0,  "soft_max": 20.0},
    "bearing_temp_c":       {"min": 20.0,  "max": 150.0, "soft_max": 110.0},
    "rpm":                  {"min": 500,   "max": 4000,  "soft_max": 3600},
    "pressure_bar":         {"min": -1.0,  "max": 50.0,  "soft_max": 30.0},
    "flow_m3h":             {"min": 0.0,   "max": 5000.0,"soft_max": 2000.0},
    "motor_power_kw":       {"min": 0.1,   "max": 5000.0,"soft_max": 1000.0},
    "npsh_margin_m":        {"min": -5.0,  "max": 30.0,  "soft_max": 25.0},
    "efficiency_percent":   {"min": 0.0,   "max": 100.0, "soft_max": 95.0},
    "voltage_v":            {"min": 100.0, "max": 700.0, "soft_max": 500.0},
    "current_a":            {"min": 0.0,   "max": 2000.0,"soft_max": 1000.0},
}

# ============================================================================
# FAULT-DOMAIN MAPPING - WEIGHTED CONFIDENCE SCORING
# ============================================================================
FAULT_DOMAIN_MAP = {
    "UNBALANCE":            {"mechanical": 0.45, "temperature": 0.15, "hydraulic": 0.10, "electrical": 0.30},
    "MISALIGNMENT":         {"mechanical": 0.40, "temperature": 0.20, "hydraulic": 0.15, "electrical": 0.25},
    "LOOSENESS":            {"mechanical": 0.50, "temperature": 0.10, "hydraulic": 0.15, "electrical": 0.25},
    "BEARING_EARLY":        {"mechanical": 0.35, "temperature": 0.35, "hydraulic": 0.10, "electrical": 0.20},
    "BEARING_DEVELOPED":    {"mechanical": 0.35, "temperature": 0.35, "hydraulic": 0.15, "electrical": 0.15},
    "BEARING_SEVERE":       {"mechanical": 0.35, "temperature": 0.35, "hydraulic": 0.15, "electrical": 0.15},
    "CAVITATION":           {"mechanical": 0.25, "temperature": 0.10, "hydraulic": 0.50, "electrical": 0.15},
    "IMPELLER_WEAR":        {"mechanical": 0.20, "temperature": 0.15, "hydraulic": 0.50, "electrical": 0.15},
    "EFFICIENCY_DROP":      {"mechanical": 0.15, "temperature": 0.15, "hydraulic": 0.50, "electrical": 0.20},
    "VOLTAGE_UNBALANCE":    {"mechanical": 0.20, "temperature": 0.15, "hydraulic": 0.15, "electrical": 0.50},
    "CURRENT_UNBALANCE":    {"mechanical": 0.20, "temperature": 0.15, "hydraulic": 0.15, "electrical": 0.50},
    "Normal":               {"mechanical": 0.34, "temperature": 0.22, "hydraulic": 0.22, "electrical": 0.22},
}

CROSS_CONFIRM_BONUS = {1: 0, 2: 10, 3: 18, 4: 25}

# ============================================================================
# DIFFERENTIAL DIAGNOSIS PAIRS
# ============================================================================
DIFFERENTIAL_PAIRS = {
    ("UNBALANCE", "LOOSENESS"): {
        "distinguisher": "FFT pattern",
        "UNBALANCE_evidence": [
            "Dominant 1x RPM di semua arah (H, V, A) secara konsisten",
            "Rasio 1x >> 2x >> 3x (menurun tajam)",
            "Vibrasi relatif seragam antar titik pengukuran",
        ],
        "LOOSENESS_evidence": [
            "Banyak harmonik: 1x, 2x, 3x RPM semuanya signifikan",
            "Dominan satu arah saja (umumnya Vertical atau Axial)",
            "Variasi besar antar titik pengukuran",
        ],
        "tie_breaker": "LOOSENESS"
    },
    ("MISALIGNMENT", "BEARING_DEVELOPED"): {
        "distinguisher": "Directionality & temperature pattern",
        "MISALIGNMENT_evidence": [
            "Tinggi di arah Axial pada titik DE pump DAN motor",
            "Komponen 2x RPM dominan di arah Axial",
            "ΔT Pump_DE vs Motor_DE >10°C",
        ],
        "BEARING_DEVELOPED_evidence": [
            "Band 2 dan Band 3 acceleration keduanya tinggi",
            "Lokalisasi pada satu titik (bukan simetris DE-NDE)",
            "ΔT DE-NDE >15°C pada titik yang sama",
        ],
        "tie_breaker": "BEARING_DEVELOPED"
    },
    ("CAVITATION", "BEARING_DEVELOPED"): {
        "distinguisher": "NPSH margin & frequency band pattern",
        "CAVITATION_evidence": [
            "NPSH margin < 0.5 m",
            "Semua Band (1, 2, 3) acceleration tinggi secara merata",
            "Ada anomali head atau flow aktual vs desain",
        ],
        "BEARING_DEVELOPED_evidence": [
            "NPSH margin aman (>0.5 m)",
            "Band 3 acceleration paling tinggi (lokalisasi frekuensi tinggi)",
            "ΔT DE-NDE >15°C, performa hidrolik masih normal",
        ],
        "tie_breaker": "CAVITATION"
    },
    ("VOLTAGE_UNBALANCE", "MISALIGNMENT"): {
        "distinguisher": "Origin domain & 2x RPM presence",
        "VOLTAGE_UNBALANCE_evidence": [
            "Voltage unbalance >1% terukur langsung",
            "Current unbalance mengikuti (>5%)",
            "Vibrasi 2x RPM muncul setelah ada perubahan supply",
        ],
        "MISALIGNMENT_evidence": [
            "Voltage unbalance <1% (electrical normal)",
            "Axial vibration di coupling point (pump DE & motor DE) tinggi bersamaan",
            "2x RPM muncul konsisten di arah Axial",
        ],
        "tie_breaker": "MISALIGNMENT"
    },
}

# ============================================================================
# FUNGSI HELPER - PUMP STANDARD THRESHOLD ADJUSTMENT
# ============================================================================
def get_standard_thresholds(pump_standard):
    standard_config = PUMP_STANDARDS.get(pump_standard, PUMP_STANDARDS["ISO 10816"])
    return {
        "velocity_limits": standard_config["velocity_limits"],
        "temp_limits": standard_config["temp_limits"],
        "severity_multiplier": standard_config["severity_multiplier"],
        "bearing_life_hours": standard_config["bearing_life_hours"]
    }

def adjust_severity_by_standard(severity, pump_standard):
    multiplier = PUMP_STANDARDS.get(pump_standard, PUMP_STANDARDS["ISO 10816"])["severity_multiplier"]
    if pump_standard == "API 610":
        if severity == "Medium":
            return "High"
        elif severity == "Low":
            return "Medium"
    elif pump_standard == "ANSI/HI":
        if severity == "Medium":
            return "Low"
    return severity

# ============================================================================
# FITUR BARU 1: FALSE POSITIVE FILTER - VALIDASI INPUT DATA
# ============================================================================
def validate_input_data(vel_data: dict, bands_data: dict, temp_data: dict,
                       suction_pressure: float, discharge_pressure: float,
                       flow_rate: float, motor_power: float,
                       v_l1l2: float, v_l2l3: float, v_l3l1: float,
                       i_l1: float, i_l2: float, i_l3: float,
                       rpm: int) -> dict:
    hard_errors = []
    soft_warnings = []
    consistency_warnings = []
    
    b = PHYSICAL_BOUNDS
    
    if rpm < b["rpm"]["min"] or rpm > b["rpm"]["max"]:
        hard_errors.append(f"❌ RPM {rpm} di luar batas fisik ({b['rpm']['min']}–{b['rpm']['max']}). "
                         "Pastikan mesin beroperasi normal saat pengukuran.")
    
    for point, vel in vel_data.items():
        if vel < b["velocity_mm_s"]["min"]:
            hard_errors.append(f"❌ Velocity {point}: {vel} mm/s tidak boleh negatif.")
        elif vel > b["velocity_mm_s"]["max"]:
            hard_errors.append(f"❌ Velocity {point}: {vel} mm/s melampaui batas fisik maksimum "
                             f"({b['velocity_mm_s']['max']} mm/s). Periksa alat ukur.")
        elif vel > b["velocity_mm_s"]["soft_max"]:
            soft_warnings.append(f"⚠️ Velocity {point}: {vel} mm/s sangat ekstrem. "
                               "Verifikasi data sebelum lanjut.")
    
    for point, bands in bands_data.items():
        for band_name, val in bands.items():
            if val < b["acceleration_g"]["min"]:
                hard_errors.append(f"❌ {band_name} di {point}: {val} g tidak boleh negatif.")
            elif val > b["acceleration_g"]["max"]:
                hard_errors.append(f"❌ {band_name} di {point}: {val} g melampaui batas fisik "
                                 f"({b['acceleration_g']['max']} g).")
            elif val > b["acceleration_g"]["soft_max"]:
                soft_warnings.append(f"⚠️ {band_name} di {point}: {val} g sangat tinggi. Verifikasi sensor.")
    
    if temp_data:
        for loc, temp in temp_data.items():
            if temp is not None and temp > 0:
                if temp < b["bearing_temp_c"]["min"]:
                    hard_errors.append(f"❌ Bearing temp {loc}: {temp}°C terlalu rendah. "
                                     "Pastikan nilai bukan nol akibat sensor error.")
                elif temp > b["bearing_temp_c"]["max"]:
                    hard_errors.append(f"❌ Bearing temp {loc}: {temp}°C melampaui batas fisik "
                                     f"({b['bearing_temp_c']['max']}°C). Periksa sensor.")
                elif temp > b["bearing_temp_c"]["soft_max"]:
                    soft_warnings.append(f"⚠️ Bearing temp {loc}: {temp}°C sangat tinggi. "
                                       "Verifikasi sebelum operasi dilanjutkan.")
    
    if discharge_pressure <= suction_pressure:
        hard_errors.append(f"❌ Discharge pressure ({discharge_pressure} bar) ≤ Suction pressure "
                         f"({suction_pressure} bar). Tidak mungkin secara fisik untuk pompa yang beroperasi. "
                         "Periksa pembacaan pressure gauge.")
    
    if suction_pressure < b["pressure_bar"]["min"]:
        hard_errors.append(f"❌ Suction pressure {suction_pressure} bar terlalu rendah. "
                         "Kemungkinan sensor error atau vakum ekstrem.")
    
    if discharge_pressure > b["pressure_bar"]["max"]:
        soft_warnings.append(f"⚠️ Discharge pressure {discharge_pressure} bar sangat tinggi. "
                           "Verifikasi range alat ukur pressure.")
    
    if flow_rate < b["flow_m3h"]["min"]:
        hard_errors.append(f"❌ Flow rate {flow_rate} m³/h tidak boleh negatif.")
    
    if flow_rate > b["flow_m3h"]["soft_max"]:
        soft_warnings.append(f"⚠️ Flow rate {flow_rate} m³/h sangat besar. Periksa satuan pengukuran.")
    
    if motor_power < b["motor_power_kw"]["min"]:
        hard_errors.append(f"❌ Motor power {motor_power} kW tidak valid. Mesin harus beroperasi saat inspeksi.")
    
    if motor_power > b["motor_power_kw"]["soft_max"]:
        soft_warnings.append(f"⚠️ Motor power {motor_power} kW sangat besar. Verifikasi data nameplate.")
    
    for phase, v in [("L1-L2", v_l1l2), ("L2-L3", v_l2l3), ("L3-L1", v_l3l1)]:
        if v < b["voltage_v"]["min"]:
            hard_errors.append(f"❌ Voltage {phase}: {v} V terlalu rendah. "
                             "Mesin tidak mungkin beroperasi normal.")
        elif v > b["voltage_v"]["max"]:
            hard_errors.append(f"❌ Voltage {phase}: {v} V melampaui batas fisik. Cek alat ukur.")
        elif v > b["voltage_v"]["soft_max"]:
            soft_warnings.append(f"⚠️ Voltage {phase}: {v} V sangat tinggi. Verifikasi transformer tap.")
    
    for phase, i in [("L1", i_l1), ("L2", i_l2), ("L3", i_l3)]:
        if i < b["current_a"]["min"]:
            hard_errors.append(f"❌ Current {phase}: {i} A tidak boleh negatif.")
        elif i > b["current_a"]["soft_max"]:
            soft_warnings.append(f"⚠️ Current {phase}: {i} A sangat besar. Verifikasi FLA motor.")
    
    if vel_data:
        vel_values = list(vel_data.values())
        if len(set(vel_values)) == 1 and vel_values[0] > 0 and len(vel_values) > 3:
            consistency_warnings.append(
                "⚠️ KONSISTENSI: Semua nilai velocity identik persis. "
                "Kemungkinan copy-paste error. Verifikasi data lapangan."
            )
    
    for point, bands in bands_data.items():
        b1 = bands.get("Band1", 0)
        b3 = bands.get("Band3", 0)
        if b3 > b1 * 3 and b3 > 1.0:
            consistency_warnings.append(
                f"⚠️ KONSISTENSI: Band 3 di {point} ({b3:.2f}g) jauh > Band 1 ({b1:.2f}g). "
                "Pola tidak tipikal — verifikasi data acceleration."
            )
    
    for point, vel in vel_data.items():
        bands = bands_data.get(point, {})
        b3 = bands.get("Band3", 0)
        if vel > 7.0 and b3 < 0.05:
            consistency_warnings.append(
                f"⚠️ KONSISTENSI: Velocity tinggi ({vel:.1f} mm/s) di {point} "
                f"tapi Band 3 sangat rendah ({b3:.3f}g). "
                "Kemungkinan pengukuran di kondisi yang berbeda."
            )
    
    # === MOTOR POWER VS CURRENT CHECK ===
    i_avg_check = (i_l1 + i_l2 + i_l3) / 3 if all([i_l1, i_l2, i_l3]) else 0
    v_avg_check = (v_l1l2 + v_l2l3 + v_l3l1) / 3 if all([v_l1l2, v_l2l3, v_l3l1]) else 0
    
    if i_avg_check > 0 and v_avg_check > 0 and motor_power > 0:
        estimated_power_kw = (v_avg_check * i_avg_check * 1.732 * 0.85) / 1000
        ratio = motor_power / estimated_power_kw if estimated_power_kw > 0 else 1
        
        if ratio > 10.0 or ratio < 0.5:
            consistency_warnings.append(
                f"⚠️ KONSISTENSI: Motor power input ({motor_power:.1f} kW) tidak sesuai dengan "
                f"estimasi dari voltage/current ({estimated_power_kw:.1f} kW). "
                "Verifikasi data electrical atau power input."
            )
    
    if hard_errors:
        status = "REJECT"
    elif soft_warnings or consistency_warnings:
        status = "WARNING"
    else:
        status = "OK"
    
    return {
        "status": status,
        "hard_errors": hard_errors,
        "soft_warnings": soft_warnings,
        "consistency_warnings": consistency_warnings,
        "total_issues": len(hard_errors) + len(soft_warnings) + len(consistency_warnings)
    }

# ============================================================================
# FITUR BARU 2: WEIGHTED CONFIDENCE SCORING
# ============================================================================
def weighted_confidence_score(mech_result: dict, hyd_result: dict, elec_result: dict,
                             temp_data: dict, temp_adjustment: int,
                             primary_fault: str, pump_standard: str = "ISO 10816") -> dict:
    domain_weights = FAULT_DOMAIN_MAP.get(primary_fault, FAULT_DOMAIN_MAP["Normal"])
    
    mech_conf   = mech_result.get("confidence", 0)
    hyd_conf    = hyd_result.get("confidence", 0)
    elec_conf   = elec_result.get("confidence", 0)
    temp_conf   = 70 + temp_adjustment if temp_data else 0
    
    mech_fault  = mech_result.get("fault_type", "normal")
    hyd_fault   = hyd_result.get("fault_type", "normal")
    elec_fault  = elec_result.get("fault_type", "normal")
    
    weighted_score = 0.0
    denominator = 0.0
    
    weighted_score += domain_weights["mechanical"] * mech_conf
    denominator += domain_weights["mechanical"]
    
    weighted_score += domain_weights["hydraulic"] * hyd_conf
    denominator += domain_weights["hydraulic"]
    
    weighted_score += domain_weights["electrical"] * elec_conf
    denominator += domain_weights["electrical"]
    
    if temp_data and temp_conf > 0:
        weighted_score += domain_weights["temperature"] * temp_conf
        denominator += domain_weights["temperature"]
    
    base_confidence = (weighted_score / denominator) if denominator > 0 else 0
    
    confirming_domains = 0
    confirmation_detail = []
    
    if mech_fault not in ["normal", None]:
        confirming_domains += 1
        confirmation_detail.append(f"Mechanical ({mech_fault})")
    
    if hyd_fault not in ["normal", None]:
        confirming_domains += 1
        confirmation_detail.append(f"Hydraulic ({hyd_fault})")
    
    if elec_fault not in ["normal", None]:
        confirming_domains += 1
        confirmation_detail.append(f"Electrical ({elec_fault})")
    
    if temp_data:
        any_abnormal_temp = any(t is not None and t > 70 for t in temp_data.values())
        if any_abnormal_temp:
            confirming_domains += 1
            confirmation_detail.append("Temperature (elevated)")
    
    cross_bonus = CROSS_CONFIRM_BONUS.get(confirming_domains, 0)
    
    sev_mult = PUMP_STANDARDS.get(pump_standard, PUMP_STANDARDS["ISO 10816"])["severity_multiplier"]
    severity_adj = 0
    if sev_mult > 1.0 and mech_result.get("severity") in ["Medium", "High"]:
        severity_adj = 3
    elif sev_mult < 1.0:
        severity_adj = -3
    
    final_confidence = int(min(95, base_confidence + cross_bonus + severity_adj))
    
    return {
        "final_confidence": final_confidence,
        "base_confidence": round(base_confidence, 1),
        "cross_bonus": cross_bonus,
        "severity_adj": severity_adj,
        "confirming_domains": confirming_domains,
        "confirmation_detail": confirmation_detail,
        "domain_weights_used": domain_weights,
        "breakdown": {
            "mechanical": round(domain_weights["mechanical"] * mech_conf, 1),
            "hydraulic": round(domain_weights["hydraulic"] * hyd_conf, 1),
            "electrical": round(domain_weights["electrical"] * elec_conf, 1),
            "temperature": round(domain_weights["temperature"] * temp_conf, 1) if temp_data else 0,
        }
    }

# ============================================================================
# FITUR BARU 3: DIFFERENTIAL DIAGNOSIS
# ============================================================================
def perform_differential_diagnosis(mech_result: dict, hyd_result: dict, elec_result: dict,
                                  temp_data: dict, vel_data: dict, bands_data: dict,
                                  fft_data_dict: dict, rpm_hz: float) -> dict:
    mech_diag = mech_result.get("diagnosis", "Normal")
    hyd_diag  = hyd_result.get("diagnosis", "NORMAL_OPERATION")
    elec_diag = elec_result.get("diagnosis", "NORMAL_ELECTRICAL")
    
    active_pair = None
    for pair_key in DIFFERENTIAL_PAIRS.keys():
        fault_a, fault_b = pair_key
        all_diags = [mech_diag, hyd_diag, elec_diag]
        if fault_a in all_diags or fault_b in all_diags:
            active_pair = pair_key
            break
    
    if active_pair is None:
        return {"applicable": False, "pair": None, "winner": mech_diag,
                "loser": None, "winner_score": 0, "loser_score": 0,
                "winner_evidence": [], "loser_evidence": [], "reasoning": ""}
    
    fault_a, fault_b = active_pair
    pair_info = DIFFERENTIAL_PAIRS[active_pair]
    
    score_a = 0
    score_b = 0
    evidence_a_found = []
    evidence_b_found = []
    
    if fault_a == "UNBALANCE":
        for point, fft_pts in fft_data_dict.items():
            if fft_pts:
                amp_1x = next((p[1] for p in fft_pts if abs(p[0] - rpm_hz) < 0.05 * rpm_hz), 0)
                amp_2x = next((p[1] for p in fft_pts if abs(p[0] - 2*rpm_hz) < 0.05*rpm_hz), 0)
                amp_3x = next((p[1] for p in fft_pts if abs(p[0] - 3*rpm_hz) < 0.05*rpm_hz), 0)
                
                if amp_1x > 0:
                    total = amp_1x + amp_2x + amp_3x
                    if total > 0 and amp_1x / total > 0.65:
                        score_a += 2
                        evidence_a_found.append(f"1x RPM mendominasi {amp_1x/(total+1e-9)*100:.0f}% spektrum di {point}")
        
        h_vels = [v for k, v in vel_data.items() if "Horizontal" in k]
        v_vels = [v for k, v in vel_data.items() if "Vertical" in k]
        if h_vels and v_vels and max(h_vels) > 0 and max(v_vels) > 0:
            ratio_hv = min(max(h_vels), max(v_vels)) / max(max(h_vels), max(v_vels))
            if ratio_hv > 0.6:
                score_a += 1
                evidence_a_found.append(f"Vibrasi H dan V relatif seimbang (rasio {ratio_hv:.2f}) → ciri khas unbalance")
    
    if fault_a == "MISALIGNMENT":
        for point, fft_pts in fft_data_dict.items():
            if "Axial" in point and "DE" in point and fft_pts:
                amp_1x = next((p[1] for p in fft_pts if abs(p[0]-rpm_hz) < 0.05*rpm_hz), 0)
                amp_2x = next((p[1] for p in fft_pts if abs(p[0]-2*rpm_hz) < 0.05*rpm_hz), 0)
                
                if amp_2x > 0.4 * amp_1x and amp_1x > 0:
                    score_a += 2
                    evidence_a_found.append(f"2x RPM signifikan di arah Axial {point} → pola misalignment")
        
        pump_de_ax = vel_data.get("Pump DE Axial", 0)
        motor_de_ax = vel_data.get("Motor DE Axial", 0)
        thresholds = get_standard_thresholds("ISO 10816")
        lim_b = thresholds["velocity_limits"]["Zone B (Acceptable)"]
        
        if pump_de_ax > lim_b and motor_de_ax > lim_b:
            score_a += 2
            evidence_a_found.append(f"Pump DE Axial ({pump_de_ax:.1f}) DAN Motor DE Axial ({motor_de_ax:.1f}) keduanya > Zone B → axial misalignment")
        
        if temp_data:
            pump_de_t = temp_data.get("Pump_DE", 0) or 0
            motor_de_t = temp_data.get("Motor_DE", 0) or 0
            if abs(pump_de_t - motor_de_t) > 10 and pump_de_t > 0 and motor_de_t > 0:
                score_a += 1
                evidence_a_found.append(f"ΔT Pump_DE vs Motor_DE = {abs(pump_de_t-motor_de_t):.0f}°C → pola thermal misalignment")
    
    if fault_a == "CAVITATION":
        npsh_m = hyd_result.get("details", {}).get("npsh_margin_m", 99)
        if npsh_m < 0.5:
            score_a += 3
            evidence_a_found.append(f"NPSH margin = {npsh_m:.2f} m (<0.5 m) → risiko cavitation kuat")
        
        all_bands_high = 0
        for point, bands in bands_data.items():
            b1 = bands.get("Band1", 0)
            b2 = bands.get("Band2", 0)
            b3 = bands.get("Band3", 0)
            if (b1 > 0.3 * ACCEL_BASELINE["Band1 (0.5-1.5kHz)"] * 2 and 
                b2 > 0.3 * ACCEL_BASELINE["Band2 (1.5-5kHz)"] * 2 and 
                b3 > 0.3 * ACCEL_BASELINE["Band3 (5-16kHz)"] * 2):
                all_bands_high += 1
        
        if all_bands_high > 0:
            score_a += 1
            evidence_a_found.append(f"Semua Band (1,2,3) tinggi di {all_bands_high} titik → pola kavitasi merata")
    
    if fault_a == "VOLTAGE_UNBALANCE":
        elec_details = elec_result.get("details", {})
        v_unb = elec_details.get("voltage_unbalance", 0)
        i_unb = elec_details.get("current_unbalance", 0)
        
        if v_unb > 1.0:
            score_a += 2
            evidence_a_found.append(f"Voltage unbalance terukur {v_unb:.2f}% (>1%) — electrical origin terkonfirmasi")
        
        if i_unb > 5.0:
            score_a += 1
            evidence_a_found.append(f"Current unbalance {i_unb:.2f}% mengikuti voltage unbalance")
    
    if fault_b == "LOOSENESS":
        harmonics_count = 0
        for point, fft_pts in fft_data_dict.items():
            if fft_pts:
                amp_1x = next((p[1] for p in fft_pts if abs(p[0]-rpm_hz) < 0.05*rpm_hz), 0)
                amp_2x = next((p[1] for p in fft_pts if abs(p[0]-2*rpm_hz) < 0.05*rpm_hz), 0)
                amp_3x = next((p[1] for p in fft_pts if abs(p[0]-3*rpm_hz) < 0.05*rpm_hz), 0)
                
                if amp_2x > 0.3 * amp_1x and amp_3x > 0.2 * amp_1x:
                    harmonics_count += 1
                    evidence_b_found.append(f"Harmonik 2x dan 3x RPM signifikan di {point} → pola looseness")
        
        if harmonics_count > 0:
            score_b += 2
    
    if fault_b == "BEARING_DEVELOPED":
        for point, bands in bands_data.items():
            b2 = bands.get("Band2", 0)
            b3 = bands.get("Band3", 0)
            if b2 > 2.0 * ACCEL_BASELINE["Band2 (1.5-5kHz)"] and b3 > 1.5 * ACCEL_BASELINE["Band3 (5-16kHz)"]:
                score_b += 2
                evidence_b_found.append(f"Band 2 ({b2:.2f}g) & Band 3 ({b3:.2f}g) tinggi di {point} → bearing fault signature")
        
        if temp_data:
            pump_de_t = temp_data.get("Pump_DE", 0) or 0
            pump_nde_t = temp_data.get("Pump_NDE", 0) or 0
            if abs(pump_de_t - pump_nde_t) > 15 and pump_de_t > 0:
                score_b += 2
                evidence_b_found.append(f"Pump DE-NDE ΔT = {abs(pump_de_t-pump_nde_t):.0f}°C (>15°C) → localized bearing fault")
        
        hyd_perf_normal = hyd_diag in ["NORMAL_OPERATION", "EFFICIENCY_DROP"]
        if hyd_perf_normal:
            score_b += 1
            evidence_b_found.append("Performa hidrolik relatif normal → bukan cavitation origin")
    
    if fault_b == "MISALIGNMENT":
        pump_de_ax = vel_data.get("Pump DE Axial", 0)
        motor_de_ax = vel_data.get("Motor DE Axial", 0)
        thresholds = get_standard_thresholds("ISO 10816")
        lim_b = thresholds["velocity_limits"]["Zone B (Acceptable)"]
        
        if pump_de_ax > lim_b and motor_de_ax > lim_b:
            score_b += 2
            evidence_b_found.append(f"Axial vibration tinggi di coupling point → mechanical misalignment")
        
        for point, fft_pts in fft_data_dict.items():
            if "Axial" in point and fft_pts:
                amp_1x = next((p[1] for p in fft_pts if abs(p[0]-rpm_hz) < 0.05*rpm_hz), 0)
                amp_2x = next((p[1] for p in fft_pts if abs(p[0]-2*rpm_hz) < 0.05*rpm_hz), 0)
                
                if amp_2x > 0.4 * amp_1x and amp_1x > 0:
                    score_b += 1
                    evidence_b_found.append(f"2x RPM di Axial {point}")
    
    if score_a > score_b:
        winner, loser = fault_a, fault_b
        w_score, l_score = score_a, score_b
        w_ev, l_ev = evidence_a_found, evidence_b_found
    elif score_b > score_a:
        winner, loser = fault_b, fault_a
        w_score, l_score = score_b, score_a
        w_ev, l_ev = evidence_b_found, evidence_a_found
    else:
        winner = pair_info["tie_breaker"]
        loser = fault_a if winner == fault_b else fault_b
        w_score, l_score = score_a, score_b
        w_ev = evidence_a_found if winner == fault_a else evidence_b_found
        l_ev = evidence_b_found if winner == fault_a else evidence_a_found
    
    a_label = pair_info.get(f"{fault_a}_evidence", [])
    b_label = pair_info.get(f"{fault_b}_evidence", [])
    
    if not w_ev:
        w_ev = a_label if winner == fault_a else b_label
    if not l_ev:
        l_ev = b_label if loser == fault_b else a_label
    
    reasoning = (
        f"Sistem memilih **{winner}** (skor {w_score}) dibanding **{loser}** (skor {l_score}) "
        f"berdasarkan pembeda utama: *{pair_info['distinguisher']}*."
    )
    if w_score == l_score:
        reasoning += f" Karena skor sama, sistem memilih {winner} sebagai fault yang lebih berisiko (tie-breaker)."
    
    return {
        "applicable": True,
        "pair": active_pair,
        "winner": winner,
        "loser": loser,
        "winner_score": w_score,
        "loser_score": l_score,
        "winner_evidence": w_ev,
        "loser_evidence": l_ev,
        "reasoning": reasoning
    }

# ============================================================================
# FUNGSI REKOMENDASI - MULTI-DOMAIN (UPDATED WITH STANDARD)
# ============================================================================
def get_mechanical_recommendation(diagnosis: str, location: str, severity: str = "Medium",
                                 pump_standard: str = "ISO 10816") -> str:
    standard_note = f"({pump_standard} Standard)"
    
    rec_map = {
        "UNBALANCE": (
            f"🔧 **{location} - Unbalance** {standard_note}\n"
            f"• Lakukan single/dual plane balancing pada rotor\n"
            f"• Periksa: material buildup pada impeller, korosi blade, keyway wear\n"
            f"• Target residual unbalance: < 4W/N (g·mm) per ISO 1940-1\n"
            f"• Severity: {severity} → {'Segera jadwalkan balancing' if severity != 'Low' else 'Monitor trend'}"
        ),
        "MISALIGNMENT": (
            f"🔧 **{location} - Misalignment** {standard_note}\n"
            f"• Lakukan laser alignment pump-motor coupling\n"
            f"• Toleransi target: < 0.05 mm offset, < 0.05 mm/m angular\n"
            f"• Periksa: pipe strain, soft foot, coupling wear\n"
            f"• Severity: {severity} → {'Stop & align segera' if severity == 'High' else 'Jadwalkan alignment'}"
        ),
        "LOOSENESS": (
            f"🔧 **{location} - Mechanical Looseness** {standard_note}\n"
            f"• Torque check semua baut: foundation, bearing housing, baseplate\n"
            f"• Periksa: crack pada struktur, worn dowel pins, grout deterioration\n"
            f"• Gunakan torque wrench sesuai spec manufacturer\n"
            f"• Severity: {severity} → {'Amankan sebelum operasi' if severity == 'High' else 'Jadwalkan tightening'}"
        ),
        "BEARING_EARLY": (
            f"🔧 **{location} - Early Bearing Fault / Lubrication** {standard_note}\n"
            f"• Cek lubrication: jenis grease, interval, quantity\n"
            f"• Ambil oil sample jika applicable (particle count, viscosity)\n"
            f"• Monitor trend Band 3 mingguan\n"
            f"• Severity: {severity} → {'Ganti grease & monitor ketat' if severity != 'Low' else 'Lanjutkan monitoring'}"
        ),
        "BEARING_DEVELOPED": (
            f"🔧 **{location} - Developed Bearing Fault** {standard_note}\n"
            f"• Jadwalkan bearing replacement dalam 1-3 bulan\n"
            f"• Siapkan spare bearing (pastikan clearance & fit sesuai spec)\n"
            f"• Monitor weekly: jika Band 1 naik drastis → percepat jadwal\n"
            f"• Severity: {severity} → {'Plan shutdown segera' if severity == 'High' else 'Siapkan work order'}"
        ),
        "BEARING_SEVERE": (
            f"🔴 **{location} - Severe Bearing Damage** {standard_note}\n"
            f"• RISK OF CATASTROPHIC FAILURE - Pertimbangkan immediate shutdown\n"
            f"• Jika continue operasi: monitor hourly, siapkan emergency replacement\n"
            f"• Investigasi root cause: lubrication, installation, loading?\n"
            f"• Severity: HIGH → Action required dalam 24 jam"
        ),
        "Tidak Terdiagnosa": (
            f"⚠️ **Pola Tidak Konsisten** {standard_note}\n"
            f"• Data tidak match dengan rule mekanikal standar\n"
            f"• Kemungkinan: multi-fault interaction, measurement error, atau fault non-rutin\n"
            f"• Rekomendasi: Analisis manual oleh Vibration Analyst Level II+ dengan full spectrum review"
        )
    }
    return rec_map.get(diagnosis, rec_map["Tidak Terdiagnosa"])

def get_hydraulic_recommendation(diagnosis: str, fluid_type: str, severity: str = "Medium",
                                pump_standard: str = "ISO 10816") -> str:
    standard_note = f"({pump_standard} Standard)"
    
    rec_map = {
        "CAVITATION": (
            f"💧 **{fluid_type} - Cavitation Risk** {standard_note}\n"
            f"• Tingkatkan suction pressure atau turunkan fluid temperature\n"
            f"• Cek: strainer clogged, valve posisi, NPSH margin\n"
            f"• Target NPSH margin: > 0.5 m untuk {fluid_type}\n"
            f"• Severity: {severity} → {'Evaluasi immediate shutdown jika NPSH margin <0.3m' if severity == 'High' else 'Monitor intensif'}"
        ),
        "IMPELLER_WEAR": (
            f"💧 **{fluid_type} - Impeller Wear / Internal Clearance** {standard_note}\n"
            f"• Jadwalkan inspection impeller & wear ring\n"
            f"• Ukur internal clearance vs spec OEM\n"
            f"• Pertimbangkan: fluid viscosity effect pada slip loss\n"
            f"• Severity: {severity} → {'Siapkan spare impeller' if severity != 'Low' else 'Monitor trend efisiensi'}"
        ),
        "SYSTEM_RESISTANCE_HIGH": (
            f"💧 **{fluid_type} - System Resistance Higher Than Design** {standard_note}\n"
            f"• Cek valve discharge position, clogged line, atau filter pressure drop\n"
            f"• Verifikasi P&ID vs as-built condition\n"
            f"• Evaluasi: apakah operating point masih dalam acceptable range?\n"
            f"• Severity: {severity} → {'Adjust valve / clean line segera' if severity == 'High' else 'Jadwalkan system review'}"
        ),
        "EFFICIENCY_DROP": (
            f"💧 **{fluid_type} - Efficiency Degradation** {standard_note}\n"
            f"• Investigasi: mechanical loss vs hydraulic loss vs fluid property mismatch\n"
            f"• Severity: {severity} → {'Plan overhaul dalam 1-3 bulan' if severity != 'Low' else 'Monitor monthly'}"
        ),
        "NORMAL_OPERATION": (
            f"✅ **{fluid_type} - Normal Operation** {standard_note}\n"
            f"• Semua parameter dalam batas acceptable (±5% dari design)\n"
            f"• Rekam data ini sebagai baseline untuk trend monitoring\n"
            f"• Severity: Low → Continue routine monitoring"
        ),
        "Tidak Terdiagnosa": (
            f"⚠️ **Pola Tidak Konsisten** {standard_note}\n"
            f"• Data hydraulic tidak match dengan rule standar\n"
            f"• Rekomendasi: Verifikasi data lapangan + cross-check dengan electrical/mechanical data"
        )
    }
    return rec_map.get(diagnosis, rec_map["Tidak Terdiagnosa"])

def get_electrical_recommendation(diagnosis: str, severity: str = "Medium",
                                 pump_standard: str = "ISO 10816") -> str:
    standard_note = f"({pump_standard} Standard)"
    
    rec_map = {
        "UNDER_VOLTAGE": (
            f"⚡ **Under Voltage Condition** {standard_note}\n"
            f"• Cek supply voltage di MCC: possible transformer tap / cable voltage drop\n"
            f"• Verify: motor rated voltage vs actual operating voltage\n"
            f"• Severity: {severity} → {'Coordinate dengan electrical team segera' if severity == 'High' else 'Monitor voltage trend'}"
        ),
        "OVER_VOLTAGE": (
            f"⚡ **Over Voltage Condition** {standard_note}\n"
            f"• Cek supply voltage di MCC: possible transformer tap issue\n"
            f"• Verify: motor rated voltage vs actual operating voltage\n"
            f"• Severity: {severity} → {'Coordinate dengan electrical team segera' if severity == 'High' else 'Monitor voltage trend'}"
        ),
        "VOLTAGE_UNBALANCE": (
            f"⚡ **Voltage Unbalance Detected** {standard_note}\n"
            f"• Cek 3-phase supply balance di source: possible single-phase loading\n"
            f"• Inspect: loose connection, corroded terminal, faulty breaker\n"
            f"• Severity: {severity} → {'Balance supply sebelum mechanical damage' if severity != 'Low' else 'Monitor monthly'}"
        ),
        "CURRENT_UNBALANCE": (
            f"⚡ **Current Unbalance Detected** {standard_note}\n"
            f"• Investigasi: winding fault, rotor bar issue, atau supply problem\n"
            f"• Cek insulation resistance & winding resistance balance\n"
            f"• Severity: {severity} → {'Schedule electrical inspection' if severity != 'Low' else 'Continue monitoring'}"
        ),
        "OVER_LOAD": (
            f"⚡ **Over Load Condition** {standard_note}\n"
            f"• Motor operating above FLA rating\n"
            f"• Verify: process load, mechanical binding, or electrical issue\n"
            f"• Severity: {severity} → {'Reduce load immediately' if severity == 'High' else 'Monitor trend closely'}"
        ),
        "UNDER_LOAD": (
            f"⚡ **Under Load Condition** {standard_note}\n"
            f"• Motor operating below 50% FLA\n"
            f"• Verify: process demand, pump sizing, or system resistance\n"
            f"• Severity: Low → Review operating point vs BEP"
        ),
        "NORMAL_ELECTRICAL": (
            f"✅ **Normal Electrical Condition** {standard_note}\n"
            f"• Voltage balance <2%, current balance <5%, within rated limits\n"
            f"• Severity: Low → Continue routine electrical monitoring"
        ),
        "Tidak Terdiagnosa": (
            f"⚠️ **Pola Tidak Konsisten** {standard_note}\n"
            f"• Data electrical tidak match dengan rule standar\n"
            f"• Rekomendasi: Verifikasi dengan power quality analyzer + cross-check domain lain"
        )
    }
    return rec_map.get(diagnosis, rec_map["Tidak Terdiagnosa"])

# ============================================================================
# FUNGSI TEMPERATURE ANALYSIS (UPDATED WITH STANDARD)
# ============================================================================
def get_temperature_status(temp_celsius, pump_standard="ISO 10816"):
    temp_limits = PUMP_STANDARDS.get(pump_standard, PUMP_STANDARDS["ISO 10816"])["temp_limits"]
    
    if temp_celsius is None or temp_celsius == 0:
        return "N/A", "⚪", 0
    
    if temp_celsius < temp_limits["normal_max"]:
        return "Normal", "🟢", 0
    elif temp_celsius < temp_limits["elevated_max"]:
        return "Elevated", "🟡", 0
    elif temp_celsius < temp_limits["warning_max"]:
        return "Warning", "🟠", 1
    else:
        return "Critical", "🔴", 2

def calculate_temperature_confidence_adjustment(temp_dict, diagnosis_consistent, pump_standard="ISO 10816"):
    temp_limits = PUMP_STANDARDS.get(pump_standard, PUMP_STANDARDS["ISO 10816"])["temp_limits"]
    adjustment = 0
    notes = []
    
    for location, temp in temp_dict.items():
        if temp is None or temp == 0:
            continue
        
        status, color, sev_level = get_temperature_status(temp, pump_standard)
        
        if status == "Critical":
            if diagnosis_consistent:
                adjustment += 20
                notes.append(f"⚠️ {location}: {temp}°C (Critical) - Strong thermal confirmation")
            else:
                adjustment -= 10
                notes.append(f"⚠️ {location}: {temp}°C (Critical) - Review required")
        elif status == "Warning":
            if diagnosis_consistent:
                adjustment += 15
                notes.append(f"⚠️ {location}: {temp}°C (Warning) - Thermal confirmation")
            else:
                adjustment -= 5
                notes.append(f"⚠️ {location}: {temp}°C (Warning) - Monitor closely")
        elif status == "Elevated":
            if diagnosis_consistent:
                adjustment += 10
                notes.append(f"📈 {location}: {temp}°C (Elevated) - Early thermal indication")
            else:
                notes.append(f"📈 {location}: {temp}°C (Elevated) - Monitor trend")
    
    if temp_dict.get("Pump_DE") and temp_dict.get("Pump_NDE"):
        delta_pump = abs(temp_dict["Pump_DE"] - temp_dict["Pump_NDE"])
        if delta_pump > 15:
            adjustment += 5
            notes.append(f"🔍 Pump DE-NDE ΔT: {delta_pump}°C (>15°C) - Localized fault indicated")
    
    if temp_dict.get("Motor_DE") and temp_dict.get("Motor_NDE"):
        delta_motor = abs(temp_dict["Motor_DE"] - temp_dict["Motor_NDE"])
        if delta_motor > 15:
            adjustment += 5
            notes.append(f"🔍 Motor DE-NDE ΔT: {delta_motor}°C (>15°C) - Localized fault indicated")
    
    if temp_dict.get("Motor_DE") and temp_dict.get("Pump_DE"):
        if temp_dict["Motor_DE"] > temp_dict["Pump_DE"] + 10:
            notes.append("⚡ Motor DE > Pump DE - Possible electrical origin")
    
    return min(20, max(-10, adjustment)), notes

# ============================================================================
# FUNGSI PERHITUNGAN - HYDRAULIC DOMAIN (UPDATED WITH MOTOR EFFICIENCY)
# ============================================================================
def calculate_hydraulic_parameters(suction_pressure, discharge_pressure, flow_rate,
                                   motor_power, sg, motor_efficiency=0.90, fluid_temp_c=40):
    """
    Calculate hydraulic parameters with motor efficiency consideration
    
    Args:
        suction_pressure: Suction pressure in bar
        discharge_pressure: Discharge pressure in bar
        flow_rate: Flow rate in m³/h
        motor_power: Motor nameplate power in kW
        sg: Specific gravity
        motor_efficiency: Motor efficiency (default 0.90 or 90%)
        fluid_temp_c: Fluid temperature in Celsius
    
    Returns:
        Dictionary with hydraulic parameters
    """
    delta_p = discharge_pressure - suction_pressure
    head = delta_p * 10.2 / sg if sg > 0 else 0
    hydraulic_power = (flow_rate * head * sg * 9.81) / 3600 if flow_rate > 0 and head > 0 else 0
    
    # Calculate shaft power (power delivered to pump shaft)
    shaft_power = motor_power * motor_efficiency if motor_power > 0 else 0
    
    # Calculate pump efficiency (hydraulic power / shaft power)
    efficiency = (hydraulic_power / shaft_power * 100) if shaft_power > 0 else 0
    
    return {
        "delta_p_bar": delta_p,
        "head_m": head,
        "hydraulic_power_kw": hydraulic_power,
        "shaft_power_kw": shaft_power,
        "efficiency_percent": efficiency,
        "motor_efficiency": motor_efficiency
    }

def classify_hydraulic_performance(head_aktual, head_design, efficiency_aktual,
                                  efficiency_bep, flow_aktual, flow_design):
    dev_head = ((head_aktual - head_design) / head_design * 100) if head_design > 0 else 0
    dev_eff = ((efficiency_aktual - efficiency_bep) / efficiency_bep * 100) if efficiency_bep > 0 else 0
    dev_flow = ((flow_aktual - flow_design) / flow_design * 100) if flow_design > 0 else 0
    
    if dev_head < -5 and dev_eff < -5:
        return "UNDER_PERFORMANCE", {"head_dev": dev_head, "eff_dev": dev_eff}
    elif dev_head > 5 and dev_flow < -5:
        return "OVER_RESISTANCE", {"head_dev": dev_head, "flow_dev": dev_flow}
    elif dev_eff < -10 and abs(dev_head) <= 5:
        return "EFFICIENCY_DROP", {"eff_dev": dev_eff}
    elif abs(dev_head) <= 5 and abs(dev_eff) <= 5 and abs(dev_flow) <= 5:
        return "NORMAL", {"head_dev": dev_head, "eff_dev": dev_eff, "flow_dev": dev_flow}
    else:
        return "MIXED_DEVIATION", {"head_dev": dev_head, "eff_dev": dev_eff, "flow_dev": dev_flow}

# ============================================================================
# FUNGSI PERHITUNGAN - ELECTRICAL DOMAIN
# ============================================================================
def calculate_electrical_parameters(v_l1l2, v_l2l3, v_l3l1, i_l1, i_l2, i_l3,
                                   rated_voltage, fla):
    v_avg = (v_l1l2 + v_l2l3 + v_l3l1) / 3
    i_avg = (i_l1 + i_l2 + i_l3) / 3
    
    v_deviations = [abs(v - v_avg) for v in [v_l1l2, v_l2l3, v_l3l1]]
    voltage_unbalance = (max(v_deviations) / v_avg * 100) if v_avg > 0 else 0
    
    i_deviations = [abs(i - i_avg) for i in [i_l1, i_l2, i_l3]]
    current_unbalance = (max(i_deviations) / i_avg * 100) if i_avg > 0 else 0
    
    load_estimate = (i_avg / fla * 100) if fla > 0 else 0
    
    voltage_within_tolerance = (ELECTRICAL_LIMITS["voltage_tolerance_low"] <= 
                               (v_avg / rated_voltage * 100) <= 
                               ELECTRICAL_LIMITS["voltage_tolerance_high"])
    
    return {
        "v_avg": v_avg,
        "i_avg": i_avg,
        "voltage_unbalance_percent": voltage_unbalance,
        "current_unbalance_percent": current_unbalance,
        "load_estimate_percent": load_estimate,
        "voltage_within_tolerance": voltage_within_tolerance
    }

def diagnose_electrical_condition(electrical_calc, motor_specs):
    result = {
        "diagnosis": "NORMAL_ELECTRICAL",
        "confidence": 95,
        "severity": "Low",
        "fault_type": "normal",
        "domain": "electrical",
        "details": {}
    }
    
    voltage_unbalance = electrical_calc.get("voltage_unbalance_percent", 0)
    current_unbalance = electrical_calc.get("current_unbalance_percent", 0)
    load_estimate = electrical_calc.get("load_estimate_percent", 0)
    voltage_within_tolerance = electrical_calc.get("voltage_within_tolerance", True)
    v_avg = electrical_calc.get("v_avg", 0)
    
    rated_voltage = motor_specs.get("rated_voltage", 400)
    
    if not voltage_within_tolerance:
        if v_avg < rated_voltage * 0.9:
            result["diagnosis"] = "UNDER_VOLTAGE"
            result["confidence"] = 70
            result["severity"] = "High" if load_estimate > 80 else "Medium"
            result["fault_type"] = "voltage"
        elif v_avg > rated_voltage * 1.1:
            result["diagnosis"] = "OVER_VOLTAGE"
            result["confidence"] = 70
            result["severity"] = "Medium"
            result["fault_type"] = "voltage"
        
        result["details"] = {
            "voltage_unbalance": voltage_unbalance,
            "current_unbalance": current_unbalance,
            "load_estimate": load_estimate
        }
        return result
    
    if voltage_unbalance > ELECTRICAL_LIMITS["voltage_unbalance_warning"]:
        result["diagnosis"] = "VOLTAGE_UNBALANCE"
        calculated_conf = 60 + int((voltage_unbalance - ELECTRICAL_LIMITS["voltage_unbalance_warning"]) * 15)
        result["confidence"] = min(95, calculated_conf)
        result["severity"] = "High" if voltage_unbalance > ELECTRICAL_LIMITS["voltage_unbalance_critical"] else "Medium"
        result["fault_type"] = "voltage"
    
    elif current_unbalance > ELECTRICAL_LIMITS["current_unbalance_warning"]:
        result["diagnosis"] = "CURRENT_UNBALANCE"
        calculated_conf = 60 + int((current_unbalance - ELECTRICAL_LIMITS["current_unbalance_warning"]) * 5)
        result["confidence"] = min(95, calculated_conf)
        result["severity"] = "High" if current_unbalance > ELECTRICAL_LIMITS["current_unbalance_critical"] else "Medium"
        result["fault_type"] = "current"
    
    else:
        if load_estimate > ELECTRICAL_LIMITS["current_load_critical"]:
            result["diagnosis"] = "OVER_LOAD"
            result["confidence"] = min(95, 55 + int(load_estimate - 100))
            result["severity"] = "Medium"
            result["fault_type"] = "load"
        elif load_estimate < 50:
            result["diagnosis"] = "UNDER_LOAD"
            result["confidence"] = 50
            result["severity"] = "Low"
            result["fault_type"] = "load"
    
    result["details"] = {
        "voltage_unbalance": voltage_unbalance,
        "current_unbalance": current_unbalance,
        "load_estimate": load_estimate
    }
    
    return result

# ============================================================================
# FUNGSI DIAGNOSA - MECHANICAL DOMAIN (UPDATED WITH STANDARD)
# ============================================================================
def diagnose_mechanical_system(vel_data, bands_data, fft_data_dict, rpm_hz, temp_data,
                              pump_standard="ISO 10816"):
    thresholds = get_standard_thresholds(pump_standard)
    velocity_limits = thresholds["velocity_limits"]
    severity_multiplier = thresholds["severity_multiplier"]
    
    result = {
        "diagnosis": "Normal",
        "confidence": 99,
        "severity": "Low",
        "fault_type": "normal",
        "domain": "mechanical",
        "champion_points": [],
        "temperature_notes": [],
        "point_diagnoses": {},
        "pump_standard": pump_standard
    }
    
    limit_warning = velocity_limits["Zone B (Acceptable)"]
    limit_danger = velocity_limits["Zone C (Unacceptable)"]
    
    worst_bearing_severity = "Low"
    bearing_diag = "Normal"
    
    base3 = ACCEL_BASELINE["Band3 (5-16kHz)"]
    base2 = ACCEL_BASELINE["Band2 (1.5-5kHz)"]
    base1 = ACCEL_BASELINE["Band1 (0.5-1.5kHz)"]
    
    problematic_points = []
    
    for point, bands in bands_data.items():
        b3 = bands.get("Band3", 0)
        b2 = bands.get("Band2", 0)
        b1 = bands.get("Band1", 0)
        vel = vel_data.get(point, 0)
        
        point_diagnosis = {
            "velocity": vel,
            "bands": bands,
            "fault_type": "normal",
            "severity": "Low"
        }
        
        # High frequency analysis (bearing faults)
        if b1 > 2.5 * base1 and b2 > 1.5 * base2:
            point_diagnosis["fault_type"] = "BEARING_SEVERE"
            point_diagnosis["severity"] = "High"
            worst_bearing_severity = "High"
            bearing_diag = "BEARING_SEVERE"
            problematic_points.append(point)
        elif b2 > 2.0 * base2 and b3 > 1.5 * base3:
            point_diagnosis["fault_type"] = "BEARING_DEVELOPED"
            point_diagnosis["severity"] = "High" if b2 > 3*base2 else "Medium"
            if point_diagnosis["severity"] == "High":
                worst_bearing_severity = "High"
            bearing_diag = "BEARING_DEVELOPED"
            problematic_points.append(point)
        elif b3 > 2.0 * base3:
            if worst_bearing_severity == "Low":
                worst_bearing_severity = "Medium"
            point_diagnosis["fault_type"] = "BEARING_EARLY"
            point_diagnosis["severity"] = "Medium"
            bearing_diag = "BEARING_EARLY"
            problematic_points.append(point)
        
        # Low frequency analysis (unbalance, misalignment, looseness)
        if vel > limit_warning:
            low_freq_severity = "High" if vel > limit_danger else "Medium"
            
            parts = point.split()
            if len(parts) >= 3:
                machine = parts[0]
                end = parts[1]
                direction = parts[2]
            else:
                machine, end, direction = "Pump", "DE", "Horizontal"
            
            fft_champ_data = fft_data_dict.get(point, [(rpm_hz, 0.1), (2*rpm_hz, 0.05)])
            amp_1x = next((p[1] for p in fft_champ_data if abs(p[0]-rpm_hz) < 0.05*rpm_hz), 0)
            amp_2x = next((p[1] for p in fft_champ_data if abs(p[0]-2*rpm_hz) < 0.05*rpm_hz), 0)
            
            low_freq_diag = None
            
            if direction == "Axial" and end == "DE":
                opp_machine = "Pump" if machine == "Motor" else "Motor"
                opp_point = f"{opp_machine} DE Axial"
                opp_vel = vel_data.get(opp_point, 0)
                
                if amp_2x > 0.5 * amp_1x or opp_vel > limit_warning:
                    low_freq_diag = "MISALIGNMENT"
            
            elif direction == "Horizontal":
                opp_end = "NDE" if end == "DE" else "DE"
                opp_point = f"{machine} {opp_end} Horizontal"
                opp_vel = vel_data.get(opp_point, 0)
                
                total_fft = sum(p[1] for p in fft_champ_data) if fft_champ_data else 1
                if amp_1x > 0.7 * total_fft or opp_vel > limit_warning:
                    low_freq_diag = "UNBALANCE"
            
            elif direction == "Vertical":
                high_verts = sum(1 for p, v in vel_data.items() if "Vertical" in p and v > limit_warning)
                if high_verts >= 2 or (amp_2x > 0.1 and amp_1x > 0.1):
                    low_freq_diag = "LOOSENESS"
            
            if low_freq_diag:
                point_diagnosis["fault_type"] = low_freq_diag
                point_diagnosis["severity"] = adjust_severity_by_standard(low_freq_severity, pump_standard)
                problematic_points.append(point)
        
        result["point_diagnoses"][point] = point_diagnosis
    
    result["champion_points"] = problematic_points if problematic_points else ["Tidak Ada (Normal)"]
    
    if any(p["severity"] == "High" for p in result["point_diagnoses"].values()):
        result["severity"] = "High"
    elif any(p["severity"] == "Medium" for p in result["point_diagnoses"].values()):
        result["severity"] = "Medium"
    
    high_freq_faults = ["BEARING_SEVERE", "BEARING_DEVELOPED", "BEARING_EARLY"]
    low_freq_faults = ["UNBALANCE", "MISALIGNMENT", "LOOSENESS"]
    
    for fault in high_freq_faults:
        if any(p["fault_type"] == fault for p in result["point_diagnoses"].values()):
            result["diagnosis"] = fault
            result["fault_type"] = "high_freq"
            result["confidence"] = 85
            break
    else:
        for fault in low_freq_faults:
            if any(p["fault_type"] == fault for p in result["point_diagnoses"].values()):
                result["diagnosis"] = fault
                result["fault_type"] = "low_freq"
                result["confidence"] = 75
                break
    
    return result

# ============================================================================
# FUNGSI DIAGNOSA - HYDRAULIC DOMAIN
# ============================================================================
def diagnose_hydraulic_single_point(hydraulic_calc, design_params, fluid_props, context):
    result = {
        "diagnosis": "NORMAL_OPERATION",
        "confidence": 95,
        "severity": "Low",
        "fault_type": "normal",
        "domain": "hydraulic",
        "details": {}
    }
    
    head_aktual = hydraulic_calc.get("head_m", 0)
    eff_aktual = hydraulic_calc.get("efficiency_percent", 0)
    head_design = design_params.get("rated_head_m", 0)
    eff_bep = design_params.get("bep_efficiency", 0)
    flow_design = design_params.get("rated_flow_m3h", 0)
    flow_aktual = context.get("flow_aktual", 0)
    
    pattern, deviations = classify_hydraulic_performance(
        head_aktual, head_design, eff_aktual, eff_bep, flow_aktual, flow_design
    )
    result["details"]["deviations"] = deviations
    
    suction_pressure_bar = context.get("suction_pressure_bar", 0)
    vapor_pressure_kpa = fluid_props.get("vapor_pressure_kpa_38C", 0)
    sg = fluid_props.get("sg", 0.84)
    
    p_suction_abs_kpa = (suction_pressure_bar + 1.013) * 100
    npsha_estimated = (p_suction_abs_kpa - vapor_pressure_kpa) / (sg * 9.81) if sg > 0 else 0
    npshr = design_params.get("npsh_required_m", 0)
    npsh_margin = npsha_estimated - npshr
    
    result["details"]["npsh_margin_m"] = npsh_margin
    
    if npsh_margin < 0.5:
        result["diagnosis"] = "CAVITATION"
        result["confidence"] = min(90, 70 + int((0.5 - npsh_margin) * 20) if npsh_margin < 0.5 else 70)
        result["severity"] = "High" if npsh_margin < 0.3 else "Medium"
        result["fault_type"] = "cavitation"
        return result
    
    if pattern == "UNDER_PERFORMANCE":
        result["diagnosis"] = "IMPELLER_WEAR"
        result["confidence"] = min(85, 60 + int(abs(deviations.get("head_dev", 0)) * 2))
        result["severity"] = "High" if deviations.get("head_dev", 0) < -15 else "Medium"
        result["fault_type"] = "wear"
        return result
    
    if pattern == "OVER_RESISTANCE":
        result["diagnosis"] = "SYSTEM_RESISTANCE_HIGH"
        result["confidence"] = min(90, 70 + int(abs(deviations.get("head_dev", 0))))
        result["severity"] = "High" if deviations.get("flow_dev", 0) < -30 else "Medium"
        result["fault_type"] = "system"
        return result
    
    if pattern == "EFFICIENCY_DROP":
        result["diagnosis"] = "EFFICIENCY_DROP"
        result["confidence"] = min(80, 65 + int(abs(deviations.get("eff_dev", 0))))
        result["severity"] = "High" if deviations.get("eff_dev", 0) < -20 else "Medium"
        result["fault_type"] = "efficiency"
        return result
    
    if pattern == "NORMAL":
        result["diagnosis"] = "NORMAL_OPERATION"
        result["confidence"] = 95
        result["severity"] = "Low"
        result["fault_type"] = "normal"
        return result
    
    result["diagnosis"] = "Tidak Terdiagnosa"
    result["confidence"] = 40
    result["severity"] = "Medium"
    result["fault_type"] = "unknown"
    return result

# ============================================================================
# 🔥 FAULT PROPAGATION MAP GENERATOR (FIXED KeyError)
# ============================================================================
def generate_fault_propagation_map(mech_result, hyd_result, elec_result,
                                  temp_data=None, pump_standard="ISO 10816-3/7"):
    propagation_data = []
    
    if pump_standard in PUMP_STANDARDS:
        temp_limits = PUMP_STANDARDS[pump_standard]["temp_limits"]
    else:
        temp_limits = BEARING_TEMP_LIMITS
    
    if elec_result.get("fault_type") == "voltage":
        if mech_result.get("diagnosis") in ["MISALIGNMENT", "LOOSENESS"]:
            propagation_data.append({
                "root_cause": "⚡ Electrical Supply Issue",
                "fault_chain": ["Voltage Unbalance", "Torque Pulsation", "Mechanical Stress", "Misalignment/Looseness"],
                "repair_actions": [
                    "✅ Balance 3-phase supply di MCC",
                    "✅ Check connection & terminal",
                    "✅ Verify transformer tap setting",
                    "✅ Laser alignment setelah electrical fix"
                ],
                "priority": "HIGH",
                "timeline": "1-3 hari"
            })
    
    if hyd_result.get("fault_type") == "cavitation":
        if mech_result.get("fault_type") == "wear" or mech_result.get("diagnosis") in ["BEARING_EARLY", "BEARING_DEVELOPED"]:
            propagation_data.append({
                "root_cause": "💧 Cavitation Damage",
                "fault_chain": ["Low NPSH Margin", "Bubble Collapse", "Impeller Erosion", "Unbalance", "Bearing Wear"],
                "repair_actions": [
                    "✅ Increase suction pressure",
                    "✅ Clean strainer/filter",
                    "✅ Check valve position",
                    "✅ Replace damaged impeller",
                    "✅ Replace bearing if worn"
                ],
                "priority": "CRITICAL",
                "timeline": "Immediate - 1 minggu"
            })
    
    if mech_result.get("fault_type") in ["low_freq", "high_freq"]:
        if hyd_result.get("fault_type") == "efficiency":
            propagation_data.append({
                "root_cause": "🔧 Mechanical Fault",
                "fault_chain": ["Unbalance/Misalignment/Bearing", "Increased Friction", "Efficiency Drop", "Motor Overload"],
                "repair_actions": [
                    "✅ Rotor balancing / Laser alignment",
                    "✅ Bearing replacement",
                    "✅ Check internal clearance",
                    "✅ Verify lubrication"
                ],
                "priority": "HIGH",
                "timeline": "1-2 minggu"
            })
    
    if temp_data:
        warning_threshold = temp_limits.get("warning_max", 80)
        critical_threshold = temp_limits.get("critical_min", 90)
        
        high_temps = [k for k, v in temp_data.items() if v and v > warning_threshold]
        if high_temps:
            propagation_data.append({
                "root_cause": "🌡️ Bearing Overheating",
                "fault_chain": ["Poor Lubrication", "Increased Friction", "Temperature Rise", "Bearing Damage"],
                "repair_actions": [
                    "✅ Check lubrication type & quantity",
                    "✅ Take oil sample analysis",
                    "✅ Verify bearing clearance",
                    "✅ Plan bearing replacement"
                ],
                "priority": "HIGH" if any(temp_data.get(k, 0) > critical_threshold for k in high_temps) else "MEDIUM",
                "timeline": "1-7 hari"
            })
    
    if temp_data:
        if temp_data.get("Pump_DE") and temp_data.get("Pump_NDE"):
            if abs(temp_data["Pump_DE"] - temp_data["Pump_NDE"]) > 15:
                propagation_data.append({
                    "root_cause": "🔍 Localized Bearing Fault",
                    "fault_chain": ["Uneven Load", "Localized Heating", "Bearing Damage"],
                    "repair_actions": [
                        "✅ Check bearing housing alignment",
                        "✅ Verify mounting procedure",
                        "✅ Inspect bearing raceway",
                        "✅ Replace bearing if damaged"
                    ],
                    "priority": "MEDIUM",
                    "timeline": "1-4 minggu"
                })
    
    if not propagation_data:
        mech_diag = mech_result.get("diagnosis", "Normal")
        hyd_diag = hyd_result.get("diagnosis", "Normal")
        elec_diag = elec_result.get("diagnosis", "Normal")
        
        if mech_diag != "Normal" or hyd_diag != "NORMAL_OPERATION" or elec_diag != "NORMAL_ELECTRICAL":
            propagation_data.append({
                "root_cause": "❓ Individual Domain Fault",
                "fault_chain": ["Single domain fault detected", "No strong cross-domain correlation"],
                "repair_actions": [
                    "✅ Address individual domain fault per recommendation",
                    "✅ Continue monitoring",
                    "✅ Collect more data for trend analysis"
                ],
                "priority": "MEDIUM",
                "timeline": "Routine maintenance"
            })
    
    return propagation_data

# ============================================================================
# CROSS-DOMAIN INTEGRATION LOGIC (UPDATED WITH STANDARD)
# ============================================================================
def aggregate_cross_domain_diagnosis(mech_result, hyd_result, elec_result,
                                    shared_context, temp_data=None, pump_standard="ISO 10816"):
    system_result = {
        "diagnosis": "Tidak Ada Korelasi Antar Domain Terdeteksi",
        "confidence": 0,
        "severity": "Low",
        "location": "N/A",
        "domain_breakdown": {},
        "correlation_notes": [],
        "temperature_notes": [],
        "affected_points": [],
        "pump_standard": pump_standard
    }
    
    system_result["domain_breakdown"] = {
        "mechanical": mech_result,
        "hydraulic": hyd_result,
        "electrical": elec_result
    }
    
    mech_fault = mech_result.get("fault_type")
    hyd_fault = hyd_result.get("fault_type")
    elec_fault = elec_result.get("fault_type")
    
    mech_sev = mech_result.get("severity", "Low")
    hyd_sev = hyd_result.get("severity", "Low")
    elec_sev = elec_result.get("severity", "Low")
    
    system_result["affected_points"] = mech_result.get("champion_points", [])
    
    correlation_bonus = 0
    correlated_faults = []
    
    if (elec_fault == "voltage" and 
        mech_result.get("diagnosis") in ["MISALIGNMENT", "LOOSENESS"] and
        hyd_result.get("details", {}).get("deviations", {}).get("head_dev", 0) < -5):
        correlation_bonus += 15
        correlated_faults.append("Voltage unbalance → torque pulsation → hydraulic instability")
        system_result["diagnosis"] = "Electrical-Mechanical-Hydraulic Coupled Fault"
    
    if (hyd_fault == "cavitation" and mech_fault == "wear" and
        elec_result.get("details", {}).get("current_unbalance", 0) > 5):
        correlation_bonus += 20
        correlated_faults.append("Cavitation → impeller erosion → unbalance → current fluctuation")
        system_result["diagnosis"] = "Cascading Failure: Cavitation Origin"
    
    if (elec_result.get("diagnosis") == "OVER_LOAD" and hyd_fault == "efficiency"):
        correlation_bonus += 10
        correlated_faults.append("High electrical input + low hydraulic output → internal mechanical/hydraulic loss")
        system_result["diagnosis"] = "Internal Loss Investigation Required"
    
    if temp_data:
        temp_adjustment, temp_notes = calculate_temperature_confidence_adjustment(
            temp_data,
            diagnosis_consistent=(mech_fault is not None and mech_fault != "normal"),
            pump_standard=pump_standard
        )
        correlation_bonus += temp_adjustment
        system_result["temperature_notes"] = temp_notes
        
        if temp_data.get("Pump_DE") and temp_data.get("Pump_NDE"):
            if abs(temp_data["Pump_DE"] - temp_data["Pump_NDE"]) > 15:
                correlated_faults.append(f"Pump DE-NDE ΔT >15°C → Localized fault on DE bearing")
        
        if temp_data.get("Motor_DE") and temp_data.get("Pump_DE"):
            if temp_data["Motor_DE"] > temp_data["Pump_DE"] + 10:
                correlated_faults.append("Motor DE > Pump DE → Possible electrical origin")
    
    severities = [mech_sev, hyd_sev, elec_sev]
    if "High" in severities:
        system_result["severity"] = "High"
    elif "Medium" in severities:
        system_result["severity"] = "Medium"
    else:
        system_result["severity"] = "Low"
    
    if temp_data:
        temp_limits = PUMP_STANDARDS.get(pump_standard, PUMP_STANDARDS["ISO 10816"])["temp_limits"]
        for temp in temp_data.values():
            if temp and temp > temp_limits["critical_min"]:
                system_result["severity"] = "High"
                correlated_faults.append("⚠️ Critical bearing temperature detected")
                break
    
    primary_fault = mech_result.get("diagnosis", "Normal")
    temp_adjustment_val = 0
    if temp_data:
        temp_adjustment_val, _ = calculate_temperature_confidence_adjustment(
            temp_data, diagnosis_consistent=(mech_fault is not None and mech_fault != "normal"),
            pump_standard=pump_standard
        )
    
    weighted_result = weighted_confidence_score(
        mech_result, hyd_result, elec_result,
        temp_data, temp_adjustment_val,
        primary_fault, pump_standard
    )
    
    system_result["confidence"] = weighted_result["final_confidence"]
    system_result["confidence_breakdown"] = weighted_result
    system_result["correlation_notes"] = correlated_faults if correlated_faults else ["Tidak ada korelasi kuat antar domain terdeteksi"]
    
    return system_result

# ============================================================================
# REPORT GENERATION - CSV (UPDATED WITH STANDARD)
# ============================================================================
def generate_unified_csv_report(machine_id, rpm, timestamp, mech_data, hyd_data,
                               elec_data, integrated_result, temp_data=None,
                               pump_standard="ISO 10816"):
    lines = []
    lines.append(f"MULTI-DOMAIN PUMP DIAGNOSTIC REPORT - {machine_id.upper()}")
    lines.append(f"Generated: {timestamp}")
    lines.append(f"Pump Standard: {pump_standard}")
    lines.append(f"RPM: {rpm} | 1x RPM: {rpm/60:.2f} Hz")
    lines.append(f"Standards: ISO 10816-3/7 (Mech) | API 610 (Hyd) | IEC 60034 (Elec) | {pump_standard} (Thresholds)")
    lines.append("")
    
    if temp_data:
        lines.append("=== BEARING TEMPERATURE ===")
        lines.append(f"Pump_DE: {temp_data.get('Pump_DE', 'N/A')}°C | Pump_NDE: {temp_data.get('Pump_NDE', 'N/A')}°C")
        lines.append(f"Motor_DE: {temp_data.get('Motor_DE', 'N/A')}°C | Motor_NDE: {temp_data.get('Motor_NDE', 'N/A')}°C")
        if temp_data.get('Pump_DE') and temp_data.get('Pump_NDE'):
            lines.append(f"Pump ΔT (DE-NDE): {abs(temp_data['Pump_DE'] - temp_data['Pump_NDE']):.1f}°C")
        if temp_data.get('Motor_DE') and temp_data.get('Motor_NDE'):
            lines.append(f"Motor ΔT (DE-NDE): {abs(temp_data['Motor_DE'] - temp_data['Motor_NDE']):.1f}°C")
        lines.append("")
    
    lines.append("=== MECHANICAL VIBRATION ===")
    if mech_data.get("points"):
        lines.append("POINT,Overall_Vel(mm/s),Band1(g),Band2(g),Band3(g),Status,Diagnosis")
        for point, data in mech_data["points"].items():
            vel = data.get('velocity', 0)
            bands = data.get('bands', {})
            b1 = bands.get('Band1', 0)
            b2 = bands.get('Band2', 0)
            b3 = bands.get('Band3', 0)
            point_diag = mech_data.get("point_diagnoses", {}).get(point, {})
            point_fault = point_diag.get("fault_type", "normal")
            
            velocity_limits = PUMP_STANDARDS.get(pump_standard, PUMP_STANDARDS["ISO 10816"])["velocity_limits"]
            if vel > velocity_limits["Zone D (Danger)"]:
                status = "Zone_D"
            elif vel > velocity_limits["Zone C (Unacceptable)"]:
                status = "Zone_C"
            elif vel > velocity_limits["Zone B (Acceptable)"]:
                status = "Zone_B"
            else:
                status = "Zone_A"
            
            lines.append(f"{point},{vel:.2f},{b1:.3f},{b2:.3f},{b3:.3f},{status},{point_fault}")
        
        lines.append(f"System Diagnosis: {mech_data.get('system_diagnosis', 'N/A')}")
        champion_points = mech_data.get('champion_points', [])
        if isinstance(champion_points, list):
            lines.append(f"Champion Points: {', '.join(champion_points)}")
        else:
            lines.append(f"Champion Point: {champion_points}")
        lines.append("")
    
    lines.append("=== HYDRAULIC PERFORMANCE ===")
    if hyd_data.get("measurements"):
        m = hyd_data["measurements"]
        lines.append(f"Fluid: {hyd_data.get('fluid_type', 'N/A')} | SG: {hyd_data.get('sg', 'N/A')}")
        lines.append(f"Suction: {m.get('suction_pressure', 0):.2f} bar | Discharge: {m.get('discharge_pressure', 0):.2f} bar")
        lines.append(f"Flow: {m.get('flow_rate', 0):.1f} m³/h | Power: {m.get('motor_power', 0):.1f} kW")
        lines.append(f"Calculated Head: {hyd_data.get('head_m', 0):.1f} m | Efficiency: {hyd_data.get('efficiency_percent', 0):.1f}%")
        lines.append(f"NPSH Margin: {hyd_data.get('npsh_margin_m', 0):.2f} m")
        lines.append(f"Diagnosis: {hyd_data.get('diagnosis', 'N/A')} | Confidence: {hyd_data.get('confidence', 0)}% | Severity: {hyd_data.get('severity', 'N/A')}")
        lines.append("")
    
    lines.append("=== ELECTRICAL CONDITION ===")
    if elec_data.get("measurements"):
        e = elec_data["measurements"]
        lines.append(f"Voltage L1-L2: {e.get('v_l1l2', 0):.1f}V | L2-L3: {e.get('v_l2l3', 0):.1f}V | L3-L1: {e.get('v_l3l1', 0):.1f}V")
        lines.append(f"Current L1: {e.get('i_l1', 0):.1f}A | L2: {e.get('i_l2', 0):.1f}A | L3: {e.get('i_l3', 0):.1f}A")
        lines.append(f"Voltage Unbalance: {elec_data.get('voltage_unbalance', 0):.2f}% | Current Unbalance: {elec_data.get('current_unbalance', 0):.2f}%")
        lines.append(f"Load Estimate: {elec_data.get('load_estimate', 0):.1f}%")
        lines.append(f"Diagnosis: {elec_data.get('diagnosis', 'N/A')} | Confidence: {elec_data.get('confidence', 0)}% | Severity: {elec_data.get('severity', 'N/A')}")
        lines.append("")
    
    lines.append("=== INTEGRATED DIAGNOSIS ===")
    lines.append(f"Overall Diagnosis: {integrated_result.get('diagnosis', 'N/A')}")
    lines.append(f"Overall Confidence: {integrated_result.get('confidence', 0)}%")
    lines.append(f"Overall Severity: {integrated_result.get('severity', 'N/A')}")
    lines.append(f"Pump Standard: {integrated_result.get('pump_standard', 'N/A')}")
    lines.append(f"Affected Points: {', '.join(integrated_result.get('affected_points', []))}")
    lines.append(f"Correlation Notes: {'; '.join(integrated_result.get('correlation_notes', []))}")
    if integrated_result.get("temperature_notes"):
        lines.append(f"Temperature Notes: {'; '.join(integrated_result['temperature_notes'])}")
    lines.append("")
    
    lines.append("=== FAULT PROPAGATION MAP FOR REPAIR ===")
    mech_result = integrated_result.get("domain_breakdown", {}).get("mechanical", {})
    hyd_result = integrated_result.get("domain_breakdown", {}).get("hydraulic", {})
    elec_result = integrated_result.get("domain_breakdown", {}).get("electrical", {})
    
    propagation_map = generate_fault_propagation_map(mech_result, hyd_result, elec_result, temp_data, pump_standard)
    for idx, prop in enumerate(propagation_map, 1):
        lines.append(f"Scenario {idx}: {prop['root_cause']}")
        lines.append(f"Priority: {prop['priority']} | Timeline: {prop['timeline']}")
        lines.append(f"Fault Chain: {' -> '.join(prop['fault_chain'])}")
        lines.append("Repair Actions:")
        for action in prop["repair_actions"]:
            lines.append(f"  - {action}")
        lines.append("")
    
    return "\n".join(lines)

# ============================================================================
# STREAMLIT UI - MAIN APPLICATION
# ============================================================================
def main():
    st.set_page_config(
        page_title="Pump Diagnostic Expert System",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    if "shared_context" not in st.session_state:
        st.session_state.shared_context = {
            "machine_id": "P-101",
            "rpm": 2950,
            "service_criticality": "Essential (Utility)",
            "pump_standard": "ISO 10816",
            "motor_power": 15.0,
            "motor_efficiency": 0.90,
            "fluid_type": "Diesel / Solar",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    st.markdown("""
    <div style="background-color:#1E3A5F; padding:15px; border-radius:8px; margin-bottom:20px; text-align:center;">
        <h2 style="color:white; margin:0">🔧💧⚡ Pump Diagnostic Expert System</h2>
        <p style="color:#E0E0E0; margin:5px 0 0 0">
            Integrated Mechanical • Hydraulic • Electrical Analysis | Pertamina Patra Niaga
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.subheader("📍 Shared Context")
        
        machine_id = st.text_input("Machine ID / Tag", value=st.session_state.shared_context["machine_id"])
        
        rpm = st.number_input("Operating RPM", min_value=600, max_value=3600,
                            value=st.session_state.shared_context["rpm"], step=10)
        
        motor_power_sidebar = st.number_input("⚡ Motor Power Nameplate (kW)",
                                            min_value=0.1, max_value=500.0,
                                            value=st.session_state.shared_context.get("motor_power", 15.0),
                                            step=0.5, key="sidebar_motor_power")
        
        motor_efficiency_sidebar = st.number_input("⚙️ Motor Efficiency (%)",
                                                  min_value=50, max_value=98,
                                                  value=int(st.session_state.shared_context.get("motor_efficiency", 0.90) * 100),
                                                  step=1, key="sidebar_motor_eff")
        
        pump_standard = st.selectbox("🏭 Pump Standard",
                                   list(PUMP_STANDARDS.keys()),
                                   index=list(PUMP_STANDARDS.keys()).index(
                                       st.session_state.shared_context["pump_standard"]),
                                   help="Pilih standar pump untuk menyesuaikan threshold diagnosa")
        
        standard_info = PUMP_STANDARDS[pump_standard]
        st.info(f"""
        **{pump_standard}**
        - {standard_info['description']}
        - Bearing Life: {standard_info['bearing_life_hours']:,} hours
        - Velocity Limits: Zone B = {standard_info['velocity_limits']['Zone B (Acceptable)']} mm/s
        - Temp Critical: ≥{standard_info['temp_limits']['critical_min']}°C
        """)
        
        service_type = st.selectbox("Service Criticality",
                                  ["Critical (Process)", "Essential (Utility)", "Standby"],
                                  index=["Critical (Process)", "Essential (Utility)", "Standby"].index(
                                      st.session_state.shared_context["service_criticality"]))
        
        fluid_type = st.selectbox("Fluid Type (BBM)",
                                list(FLUID_PROPERTIES.keys()),
                                index=list(FLUID_PROPERTIES.keys()).index(
                                    st.session_state.shared_context["fluid_type"]))
        
        st.session_state.shared_context.update({
            "machine_id": machine_id,
            "rpm": rpm,
            "service_criticality": service_type,
            "pump_standard": pump_standard,
            "motor_power": motor_power_sidebar,
            "motor_efficiency": motor_efficiency_sidebar / 100.0,
            "fluid_type": fluid_type
        })
        
        fluid_props = FLUID_PROPERTIES[fluid_type]
        st.info(f"""
        **Fluid Properties ({fluid_type}):**
        - SG: {fluid_props['sg']}
        - Vapor Pressure @38°C: {fluid_props['vapor_pressure_kpa_38C']} kPa
        - Risk Level: {fluid_props['risk_level']}
        """)
        
        st.divider()
        
        st.subheader("🧭 Navigasi")
        st.markdown("""
        - 🔧 **Mechanical**: Vibration analysis
        - 💧 **Hydraulic**: Performance troubleshooting
        - ⚡ **Electrical**: 3-phase condition monitoring
        - 🔗 **Integrated**: Cross-domain correlation
        """)
        
        st.divider()
        
        st.caption("📊 Status Analisis:")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            mech_done = "✅" if "mech_result" in st.session_state else "⏳"
            st.write(f"{mech_done} Mechanical")
        with col_s2:
            hyd_done = "✅" if "hyd_result" in st.session_state else "⏳"
            st.write(f"{hyd_done} Hydraulic")
        
        col_s3, col_s4 = st.columns(2)
        with col_s3:
            elec_done = "✅" if "elec_result" in st.session_state else "⏳"
            st.write(f"{elec_done} Electrical")
        with col_s4:
            int_done = "✅" if "integrated_result" in st.session_state else "⏳"
            st.write(f"{int_done} Integrated")
    
    tab_mech, tab_hyd, tab_elec, tab_integrated = st.tabs([
        "🔧 Mechanical", "💧 Hydraulic", "⚡ Electrical", "🔗 Integrated Summary"
    ])
    
    # TAB 1: MECHANICAL
    with tab_mech:
        st.header("🔧 Mechanical Vibration Analysis")
        st.caption(f"ISO 10816-3/7 | {pump_standard} Thresholds | Centrifugal Pump + Electric Motor")
        
        velocity_limits = PUMP_STANDARDS[pump_standard]["velocity_limits"]
        st.info(f"""
        **📊 {pump_standard} Velocity Thresholds:**
        - Zone A (Good): < {velocity_limits['Zone A (Good)']} mm/s
        - Zone B (Acceptable): {velocity_limits['Zone A (Good)']} - {velocity_limits['Zone B (Acceptable)']} mm/s
        - Zone C (Unacceptable): {velocity_limits['Zone B (Acceptable)']} - {velocity_limits['Zone C (Unacceptable)']} mm/s
        - Zone D (Danger): > {velocity_limits['Zone D (Danger)']} mm/s
        """)
        
        st.subheader("🌡️ Bearing Temperature (4 Points)")
        temp_cols = st.columns(4)
        temp_data = {}
        temp_limits = PUMP_STANDARDS[pump_standard]["temp_limits"]
        
        with temp_cols[0]:
            pump_de_temp = st.number_input("Pump DE (°C)", min_value=0, max_value=150,
                                         value=65, step=1, key="temp_pump_de")
            temp_data["Pump_DE"] = pump_de_temp
            status, color, _ = get_temperature_status(pump_de_temp, pump_standard)
            if status == "Critical":
                st.error(f"🔴 {pump_de_temp}°C - {status}")
            elif status == "Warning":
                st.warning(f"🟠 {pump_de_temp}°C - {status}")
            elif status == "Elevated":
                st.warning(f"🟡 {pump_de_temp}°C - {status}")
            else:
                st.success(f"🟢 {pump_de_temp}°C - {status}")
        
        with temp_cols[1]:
            pump_nde_temp = st.number_input("Pump NDE (°C)", min_value=0, max_value=150,
                                          value=63, step=1, key="temp_pump_nde")
            temp_data["Pump_NDE"] = pump_nde_temp
            status, color, _ = get_temperature_status(pump_nde_temp, pump_standard)
            if status == "Critical":
                st.error(f"🔴 {pump_nde_temp}°C - {status}")
            elif status == "Warning":
                st.warning(f"🟠 {pump_nde_temp}°C - {status}")
            elif status == "Elevated":
                st.warning(f"🟡 {pump_nde_temp}°C - {status}")
            else:
                st.success(f"🟢 {pump_nde_temp}°C - {status}")
        
        with temp_cols[2]:
            motor_de_temp = st.number_input("Motor DE (°C)", min_value=0, max_value=150,
                                          value=68, step=1, key="temp_motor_de")
            temp_data["Motor_DE"] = motor_de_temp
            status, color, _ = get_temperature_status(motor_de_temp, pump_standard)
            if status == "Critical":
                st.error(f"🔴 {motor_de_temp}°C - {status}")
            elif status == "Warning":
                st.warning(f"🟠 {motor_de_temp}°C - {status}")
            elif status == "Elevated":
                st.warning(f"🟡 {motor_de_temp}°C - {status}")
            else:
                st.success(f"🟢 {motor_de_temp}°C - {status}")
        
        with temp_cols[3]:
            motor_nde_temp = st.number_input("Motor NDE (°C)", min_value=0, max_value=150,
                                           value=66, step=1, key="temp_motor_nde")
            temp_data["Motor_NDE"] = motor_nde_temp
            status, color, _ = get_temperature_status(motor_nde_temp, pump_standard)
            if status == "Critical":
                st.error(f"🔴 {motor_nde_temp}°C - {status}")
            elif status == "Warning":
                st.warning(f"🟠 {motor_nde_temp}°C - {status}")
            elif status == "Elevated":
                st.warning(f"🟡 {motor_nde_temp}°C - {status}")
            else:
                st.success(f"🟢 {motor_nde_temp}°C - {status}")
        
        st.divider()
        
        st.subheader("📊 Input Data 12 Titik Pengukuran")
        points = [f"{machine} {end} {direction}"
                 for machine in ["Pump", "Motor"]
                 for end in ["DE", "NDE"]
                 for direction in ["Horizontal", "Vertical", "Axial"]]
        
        input_data = {}
        bands_inputs = {}
        cols = st.columns(3)
        
        for idx, point in enumerate(points):
            with cols[idx % 3]:
                with st.expander(f"📍 {point}", expanded=False):
                    overall = st.number_input("Overall Vel (mm/s)", min_value=0.0, max_value=30.0,
                                            value=1.0, step=0.1, key=f"mech_vel_{point}")
                    input_data[point] = overall
                    
                    st.caption("Freq Bands (g) - Bearing")
                    b1 = st.number_input("Band 1", min_value=0.0, value=0.2, step=0.05, key=f"m_b1_{point}")
                    b2 = st.number_input("Band 2", min_value=0.0, value=0.15, step=0.05, key=f"m_b2_{point}")
                    b3 = st.number_input("Band 3", min_value=0.0, value=0.1, step=0.05, key=f"m_b3_{point}")
                    bands_inputs[point] = {"Band1": b1, "Band2": b2, "Band3": b3}
                    
                    if overall > velocity_limits["Zone B (Acceptable)"]:
                        st.error(f"⚠️ {overall} mm/s (High for {pump_standard})")
        
        problematic_points = [p for p, v in input_data.items()
                            if v > velocity_limits["Zone B (Acceptable)"]]
        
        if problematic_points:
            st.markdown(f"""
            <div style="background-color:#ffeeba; padding:15px; border-radius:8px; border-left:5px solid #ffc107; margin-top:20px;">
                <h4 style="margin:0; color:#856404;">🎯 Multi-Point Detection: {len(problematic_points)} Titik Bermasalah</h4>
                <p style="margin:5px 0 0 0; color:#856404;">
                    Titik dengan vibrasi tinggi: <b>{', '.join(problematic_points)}</b><br>
                    Silakan masukkan data Spektrum FFT untuk <b>semua titik yang ditandai</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            fft_data_dict = {}
            for point in problematic_points:
                with st.expander(f"📈 Input FFT Spectrum untuk: {point}", expanded=True):
                    rpm_hz = rpm / 60
                    point_fft_peaks = []
                    for i in range(1, 4):
                        c1, c2 = st.columns(2)
                        with c1:
                            default_freq = rpm_hz * i
                            freq = st.number_input(f"Peak {i} Freq (Hz)", min_value=0.1,
                                                 value=default_freq, key=f"fft_f_{point}_{i}")
                        with c2:
                            amp = st.number_input(f"Peak {i} Amp (mm/s)", min_value=0.01,
                                                value=1.0, step=0.1, key=f"fft_a_{point}_{i}")
                        point_fft_peaks.append((freq, amp))
                    fft_data_dict[point] = point_fft_peaks
        else:
            rpm_hz = rpm / 60
            fft_data_dict = {p: [(rpm_hz, 0.1), (2*rpm_hz, 0.05)] for p in points}
            st.success("✅ Semua titik vibrasi dalam batas normal.")
        
        if st.button("🛡️ Validasi Data Input", key="validate_mech"):
            motor_power_val = st.session_state.shared_context.get("motor_power", 15.0)
            val = validate_input_data(
                vel_data=input_data,
                bands_data=bands_inputs,
                temp_data=temp_data,
                suction_pressure=0, discharge_pressure=1,
                flow_rate=1, motor_power=motor_power_val,
                v_l1l2=400, v_l2l3=400, v_l3l1=400,
                i_l1=10, i_l2=10, i_l3=10,
                rpm=rpm
            )
            
            if val["status"] == "REJECT":
                st.error(f"🚫 **DATA TIDAK VALID — {len(val['hard_errors'])} error kritis ditemukan.**")
                for e in val["hard_errors"]:
                    st.error(e)
            elif val["status"] == "WARNING":
                st.warning(f"⚠️ **{val['total_issues']} peringatan ditemukan.**")
                for w in val["soft_warnings"]:
                    st.warning(w)
                for c in val["consistency_warnings"]:
                    st.warning(c)
            else:
                st.success("✅ **Semua data input valid secara fisik.**")
        
        if st.button("🔍 Jalankan Mechanical Analysis", type="primary", key="run_mech"):
            motor_power_val = st.session_state.shared_context.get("motor_power", 15.0)
            val = validate_input_data(
                vel_data=input_data, bands_data=bands_inputs, temp_data=temp_data,
                suction_pressure=0, discharge_pressure=1, flow_rate=1, motor_power=motor_power_val,
                v_l1l2=400, v_l2l3=400, v_l3l1=400, i_l1=10, i_l2=10, i_l3=10, rpm=rpm
            )
            
            if val["status"] == "REJECT":
                st.error("🚫 Analisis dibatalkan — data input mengandung error fisik.")
                for e in val["hard_errors"]:
                    st.error(e)
            else:
                if val["status"] == "WARNING":
                    for w in val["soft_warnings"] + val["consistency_warnings"]:
                        st.warning(w)
                
                with st.spinner("Menganalisis pola getaran..."):
                    mech_system = diagnose_mechanical_system(
                        input_data, bands_inputs, fft_data_dict, rpm/60, temp_data, pump_standard
                    )
                    
                    st.session_state.mech_result = mech_system
                    st.session_state.mech_data = {
                        "points": {p: {"velocity": input_data[p], "bands": bands_inputs[p]} for p in points},
                        "point_diagnoses": mech_system["point_diagnoses"],
                        "system_diagnosis": mech_system["diagnosis"],
                        "champion_points": mech_system["champion_points"]
                    }
                    st.session_state.temp_data = temp_data
                    st.session_state.pump_standard = pump_standard
                    
                    st.success(f"✅ Analisis Selesai: {mech_system['diagnosis']}")
        
        if "mech_result" in st.session_state:
            result = st.session_state.mech_result
            champion_points = result.get("champion_points", [])
            if isinstance(champion_points, list):
                points_display = ", ".join(champion_points)
            else:
                points_display = champion_points
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Diagnosis Utama", result["diagnosis"])
            with col_b:
                st.metric("Titik Sumber", points_display)
            with col_c:
                st.metric("Severity", {"Low":"🟢","Medium":"🟠","High":"🔴"}.get(result["severity"],"⚪"))
            
            if result["diagnosis"] != "Normal":
                st.info(get_mechanical_recommendation(result["diagnosis"], points_display,
                                                    result["severity"], pump_standard))
            
            st.subheader("📋 Diagnosis Per Titik")
            point_df_data = []
            for point, diag in result.get("point_diagnoses", {}).items():
                point_df_data.append({
                    "Titik": point,
                    "Velocity (mm/s)": diag.get("velocity", 0),
                    "Fault Type": diag.get("fault_type", "normal"),
                    "Severity": diag.get("severity", "Low")
                })
            point_df = pd.DataFrame(point_df_data)
            st.dataframe(point_df, use_container_width=True)
    
    # TAB 2: HYDRAULIC (UPDATED WITH MOTOR EFFICIENCY)
    with tab_hyd:
        st.header("💧 Hydraulic Troubleshooting")
        st.caption("Single-Point Steady-State Measurement")
        
        def estimate_bep_efficiency(Q, H, P_motor, SG, motor_eff=0.90):
            P_hyd_design = (Q * H * SG * 9.81) / 3600
            P_shaft_est = P_motor * motor_eff
            if P_shaft_est > 0 and P_hyd_design > 0:
                eff = (P_hyd_design / P_shaft_est) * 100
                return min(90, max(50, eff))
            return 75
        
        def estimate_npshr_conservative(Q_m3h):
            if Q_m3h < 50:
                return 3.0
            elif Q_m3h < 200:
                return 4.0
            else:
                return 5.5
        
        st.subheader("📊 Data Primer Hidrolik")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            suction_pressure = st.number_input("Suction Pressure [bar]", min_value=-1.0,
                                             value=0.44, step=0.01, key="suction_p")
            discharge_pressure = st.number_input("Discharge Pressure [bar]", min_value=0.0,
                                               value=3.73, step=0.01, key="discharge_p")
            delta_p = discharge_pressure - suction_pressure
            st.metric("ΔP", f"{delta_p:.2f} bar")
        
        with col2:
            flow_rate = st.number_input("Flow Rate [m³/h]", min_value=0.0, value=100.0,
                                      step=1.0, key="flow_rate")
            motor_power = st.number_input("Motor Power [kW]", min_value=0.0,
                                        value=st.session_state.shared_context.get("motor_power", 15.0),
                                        step=0.5, key="motor_power")
        
        with col3:
            fluid_props = FLUID_PROPERTIES[fluid_type]
            sg = st.number_input("Specific Gravity", min_value=0.5, max_value=1.5,
                               value=fluid_props["sg"], step=0.01, key="sg_input")
            motor_efficiency = st.number_input("Motor Efficiency [%]", min_value=50, max_value=98,
                                             value=int(st.session_state.shared_context.get("motor_efficiency", 0.90) * 100),
                                             step=1, key="motor_eff_input")
            motor_efficiency = motor_efficiency / 100.0
        
        with st.expander("📋 Data Nameplate", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                rated_flow = st.number_input("Rated Flow Q [m³/h]", min_value=0.0,
                                           value=100.0, step=1.0, key="rated_flow")
                rated_head = st.number_input("Rated Head H [m]", min_value=0.0,
                                           value=59.73, step=0.1, key="rated_head")
            with col2:
                bep_efficiency = st.number_input("BEP Efficiency [%] (Optional)",
                                               min_value=0, max_value=100, value=0, step=1,
                                               key="bep_eff")
                npsh_required = st.number_input("NPSH Required [m] (Optional)",
                                              min_value=0.0, value=0.0, step=0.1,
                                              key="npshr")
        
        estimation_notes = []
        if bep_efficiency <= 0:
            bep_efficiency = estimate_bep_efficiency(rated_flow, rated_head, motor_power, sg, motor_efficiency)
            estimation_notes.append(f"BEP diestimasi: {bep_efficiency:.1f}%")
        
        if npsh_required <= 0:
            npsh_required = estimate_npshr_conservative(rated_flow)
            estimation_notes.append(f"NPSHr diestimasi: {npsh_required:.1f}m")
        
        if estimation_notes:
            st.info("🔧 **Auto-Estimation:** " + " | ".join(estimation_notes))
        
        analyze_hyd_disabled = suction_pressure >= discharge_pressure
        if analyze_hyd_disabled:
            st.error("❌ Discharge pressure harus lebih tinggi dari suction pressure.")
        
        if st.button("🛡️ Validasi Data Hydraulic", key="validate_hyd"):
            val = validate_input_data(
                vel_data={}, bands_data={}, temp_data={},
                suction_pressure=suction_pressure, discharge_pressure=discharge_pressure,
                flow_rate=flow_rate, motor_power=motor_power,
                v_l1l2=400, v_l2l3=400, v_l3l1=400, i_l1=10, i_l2=10, i_l3=10, rpm=rpm
            )
            
            if val["status"] == "REJECT":
                st.error(f"🚫 {len(val['hard_errors'])} error kritis:")
                for e in val["hard_errors"]:
                    st.error(e)
            elif val["status"] == "WARNING":
                for w in val["soft_warnings"] + val["consistency_warnings"]:
                    st.warning(w)
            else:
                st.success("✅ Data hydraulic valid. Lanjutkan analisis.")
        
        if st.button("💧 Generate Diagnosis", type="primary", key="run_hyd",
                    disabled=analyze_hyd_disabled):
            val = validate_input_data(
                vel_data={}, bands_data={}, temp_data={},
                suction_pressure=suction_pressure, discharge_pressure=discharge_pressure,
                flow_rate=flow_rate, motor_power=motor_power,
                v_l1l2=400, v_l2l3=400, v_l3l1=400, i_l1=10, i_l2=10, i_l3=10, rpm=rpm
            )
            
            if val["status"] == "REJECT":
                st.error("🚫 Analisis dibatalkan — error data fisik ditemukan.")
                for e in val["hard_errors"]:
                    st.error(e)
            else:
                if val["soft_warnings"] or val["consistency_warnings"]:
                    for w in val["soft_warnings"] + val["consistency_warnings"]:
                        st.warning(w)
                
                with st.spinner("Menganalisis performa hidrolik..."):
                    hyd_calc = calculate_hydraulic_parameters(
                        suction_pressure, discharge_pressure, flow_rate,
                        motor_power, sg, motor_efficiency
                    )
                    
                    design_params = {
                        "rated_flow_m3h": rated_flow,
                        "rated_head_m": rated_head,
                        "bep_efficiency": bep_efficiency,
                        "npsh_required_m": npsh_required
                    }
                    
                    context = {
                        "flow_aktual": flow_rate,
                        "suction_pressure_bar": suction_pressure
                    }
                    
                    hyd_result = diagnose_hydraulic_single_point(
                        hyd_calc, design_params, fluid_props, context
                    )
                    
                    st.session_state.hyd_result = hyd_result
                    st.session_state.hyd_data = {
                        "measurements": {
                            "suction_pressure": suction_pressure,
                            "discharge_pressure": discharge_pressure,
                            "flow_rate": flow_rate,
                            "motor_power": motor_power
                        },
                        "fluid_type": fluid_type,
                        "sg": sg,
                        "head_m": hyd_calc["head_m"],
                        "shaft_power_kw": hyd_calc.get("shaft_power_kw", 0),
                        "efficiency_percent": hyd_calc["efficiency_percent"],
                        "motor_efficiency": motor_efficiency,
                        "npsh_margin_m": hyd_result["details"].get("npsh_margin_m", 0),
                        "diagnosis": hyd_result["diagnosis"],
                        "confidence": hyd_result["confidence"],
                        "severity": hyd_result["severity"],
                        "estimation_note": " | ".join(estimation_notes) if estimation_notes else "Data OEM lengkap"
                    }
                    
                    st.success(f"✅ {hyd_result['diagnosis']} ({hyd_result['confidence']}%)")
        
        if "hyd_result" in st.session_state:
            result = st.session_state.hyd_result
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Diagnosis", result["diagnosis"])
            with col_b:
                st.metric("Severity", {"Low":"🟢","Medium":"🟠","High":"🔴"}.get(result["severity"],"⚪"))
            with col_c:
                st.metric("Domain", "Hydraulic")
            
            if result["diagnosis"] != "NORMAL_OPERATION":
                st.info(get_hydraulic_recommendation(result["diagnosis"], fluid_type,
                                                   result["severity"], pump_standard))
    
    # TAB 3: ELECTRICAL
    with tab_elec:
        st.header("⚡ Electrical Condition Analysis")
        st.caption("3-Phase Voltage/Current | Unbalance Detection")
        
        with st.expander("⚙️ Motor Nameplate", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                rated_voltage = st.number_input("Rated Voltage (V)", min_value=200, max_value=690,
                                              value=400, step=10, key="rated_v")
            with col2:
                fla = st.number_input("Full Load Amps - FLA (A)", min_value=10, max_value=500,
                                    value=85, step=5, key="rated_i")
        
        st.subheader("📊 Pengukuran 3-Phase")
        col1, col2 = st.columns(2)
        
        with col1:
            st.caption("Voltage (Line-to-Line)")
            v_l1l2 = st.number_input("L1-L2 (V)", min_value=0.0, value=400.0, step=1.0, key="v_l1l2")
            v_l2l3 = st.number_input("L2-L3 (V)", min_value=0.0, value=402.0, step=1.0, key="v_l2l3")
            v_l3l1 = st.number_input("L3-L1 (V)", min_value=0.0, value=398.0, step=1.0, key="v_l3l1")
        
        with col2:
            st.caption("Current (Per Phase)")
            i_l1 = st.number_input("L1 (A)", min_value=0.0, value=82.0, step=0.5, key="i_l1")
            i_l2 = st.number_input("L2 (A)", min_value=0.0, value=84.0, step=0.5, key="i_l2")
            i_l3 = st.number_input("L3 (A)", min_value=0.0, value=83.0, step=0.5, key="i_l3")
        
        if st.button("🛡️ Validasi Data Electrical", key="validate_elec"):
            motor_power_val = st.session_state.shared_context.get("motor_power", 15.0)
            val = validate_input_data(
                vel_data={}, bands_data={}, temp_data={},
                suction_pressure=0, discharge_pressure=1, flow_rate=1, motor_power=motor_power_val,
                v_l1l2=v_l1l2, v_l2l3=v_l2l3, v_l3l1=v_l3l1,
                i_l1=i_l1, i_l2=i_l2, i_l3=i_l3, rpm=rpm
            )
            
            if val["status"] == "REJECT":
                st.error(f"🚫 {len(val['hard_errors'])} error kritis:")
                for e in val["hard_errors"]:
                    st.error(e)
            elif val["status"] == "WARNING":
                for w in val["soft_warnings"] + val["consistency_warnings"]:
                    st.warning(w)
            else:
                st.success("✅ Data electrical valid. Lanjutkan analisis.")
        
        if st.button("⚡ Generate Electrical Diagnosis", type="primary", key="run_elec"):
            motor_power_val = st.session_state.shared_context.get("motor_power", 15.0)
            val = validate_input_data(
                vel_data={}, bands_data={}, temp_data={},
                suction_pressure=0, discharge_pressure=1, flow_rate=1, motor_power=motor_power_val,
                v_l1l2=v_l1l2, v_l2l3=v_l2l3, v_l3l1=v_l3l1,
                i_l1=i_l1, i_l2=i_l2, i_l3=i_l3, rpm=rpm
            )
            
            if val["status"] == "REJECT":
                st.error("🚫 Analisis dibatalkan — error data fisik ditemukan.")
                for e in val["hard_errors"]:
                    st.error(e)
            else:
                if val["soft_warnings"] or val["consistency_warnings"]:
                    for w in val["soft_warnings"] + val["consistency_warnings"]:
                        st.warning(w)
                
                with st.spinner("Menganalisis kondisi electrical..."):
                    elec_calc = calculate_electrical_parameters(
                        v_l1l2, v_l2l3, v_l3l1, i_l1, i_l2, i_l3,
                        rated_voltage, fla
                    )
                    
                    motor_specs = {
                        "rated_voltage": rated_voltage,
                        "fla": fla
                    }
                    
                    elec_result = diagnose_electrical_condition(elec_calc, motor_specs)
                    
                    st.session_state.elec_result = elec_result
                    st.session_state.elec_data = {
                        "measurements": {
                            "v_l1l2": v_l1l2, "v_l2l3": v_l2l3, "v_l3l1": v_l3l1,
                            "i_l1": i_l1, "i_l2": i_l2, "i_l3": i_l3
                        },
                        "voltage_unbalance": elec_calc["voltage_unbalance_percent"],
                        "current_unbalance": elec_calc["current_unbalance_percent"],
                        "load_estimate": elec_calc["load_estimate_percent"],
                        "diagnosis": elec_result["diagnosis"],
                        "confidence": elec_result["confidence"],
                        "severity": elec_result["severity"]
                    }
                    
                    st.success(f"✅ {elec_result['diagnosis']} ({elec_result['confidence']}%)")
        
        if "elec_result" in st.session_state:
            result = st.session_state.elec_result
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Diagnosis", result["diagnosis"])
            with col_b:
                st.metric("Severity", {"Low":"🟢","Medium":"🟠","High":"🔴"}.get(result["severity"],"⚪"))
            with col_c:
                st.metric("Domain", "Electrical")
            
            if result["diagnosis"] != "NORMAL_ELECTRICAL":
                st.info(get_electrical_recommendation(result["diagnosis"], result["severity"],
                                                    pump_standard))
    
    # TAB 4: INTEGRATED
    with tab_integrated:
        st.header("🔗 Integrated Diagnostic Summary")
        st.caption(f"Cross-Domain Correlation | {pump_standard} Thresholds | Temperature Analysis")
        
        analyses_complete = all([
            "mech_result" in st.session_state,
            "hyd_result" in st.session_state,
            "elec_result" in st.session_state
        ])
        
        if not analyses_complete:
            st.info("""
            💡 **Langkah Selanjutnya:**
            1. Jalankan analisis di tab **🔧 Mechanical**
            2. Jalankan analisis di tab **💧 Hydraulic**
            3. Jalankan analisis di tab **⚡ Electrical**
            4. Kembali ke tab ini untuk integrated diagnosis
            """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            status_mech = "✅" if "mech_result" in st.session_state else "⏳"
            st.metric("Mechanical", status_mech)
        with col2:
            status_hyd = "✅" if "hyd_result" in st.session_state else "⏳"
            st.metric("Hydraulic", status_hyd)
        with col3:
            status_elec = "✅" if "elec_result" in st.session_state else "⏳"
            st.metric("Electrical", status_elec)
        
        if analyses_complete:
            with st.spinner("Mengintegrasikan hasil tiga domain..."):
                temp_data = st.session_state.get("temp_data", None)
                pump_standard = st.session_state.get("pump_standard", "ISO 10816")
                
                integrated_result = aggregate_cross_domain_diagnosis(
                    st.session_state.mech_result,
                    st.session_state.hyd_result,
                    st.session_state.elec_result,
                    st.session_state.shared_context,
                    temp_data,
                    pump_standard
                )
                
                st.session_state.integrated_result = integrated_result
                
                st.subheader("📊 Overall Assessment")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    <div style="background-color:#f0f2f6; padding:15px; border-radius:8px; border-left:5px solid #1E3A5F; text-align:center;">
                        <h4 style="margin:0 0 10px 0; color:#1E3A5F">🔗 Integrated Diagnosis</h4>
                        <p style="margin:0; font-size:1.1em; font-weight:600; color:#2c3e50;">
                            {integrated_result["diagnosis"]}
                        </p>
                        <p style="margin:5px 0 0 0; font-size:0.9em; color:#666;">
                            Standard: <b>{pump_standard}</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    severity_config = {
                        "Low": ("🟢", "#27ae60"),
                        "Medium": ("🟠", "#f39c12"),
                        "High": ("🔴", "#c0392b")
                    }
                    sev_icon, sev_color = severity_config.get(integrated_result["severity"], ("⚪", "#95a5a6"))
                    st.markdown(f"""
                    <div style="background-color:#f0f2f6; padding:15px; border-radius:8px; border-left:5px solid {sev_color}; text-align:center;">
                        <h4 style="margin:0 0 10px 0; color:#1E3A5F">⚠️ Overall Severity</h4>
                        <p style="margin:0; font-size:1.5em; font-weight:700; color:{sev_color};">
                            {sev_icon} {integrated_result["severity"]}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                col3, col4, col5 = st.columns(3)
                with col3:
                    st.metric("Confidence", f"{integrated_result['confidence']}%")
                with col4:
                    correlation_text = "Detected" if integrated_result['correlation_notes'] and integrated_result['correlation_notes'][0] != "Tidak ada korelasi kuat antar domain terdeteksi" else "None"
                    st.metric("Cross-Domain Correlation", correlation_text)
                with col5:
                    temp_status = "Available" if temp_data else "N/A"
                    st.metric("Temperature Data", temp_status)
                
                affected_points = integrated_result.get("affected_points", [])
                if affected_points and affected_points != ["Tidak Ada (Normal)"]:
                    st.warning(f"📍 **Titik Terpengaruh:** {', '.join(affected_points)}")
                
                cb = integrated_result.get("confidence_breakdown", {})
                if cb:
                    st.divider()
                    st.subheader("🎯 Confidence Score Breakdown (Weighted Multi-Domain)")
                    st.caption("Bobot tiap domain disesuaikan berdasarkan jenis fault yang terdeteksi")
                    
                    bd = cb.get("breakdown", {})
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("🔧 Mechanical", f"{bd.get('mechanical', 0):.1f} poin")
                    with c2:
                        st.metric("💧 Hydraulic", f"{bd.get('hydraulic', 0):.1f} poin")
                    with c3:
                        st.metric("⚡ Electrical", f"{bd.get('electrical', 0):.1f} poin")
                    with c4:
                        st.metric("🌡️ Temperature", f"{bd.get('temperature', 0):.1f} poin")
                    
                    st.markdown(f"""
                    <div style='background:#f8f9fa; border:1px solid #dee2e6; border-radius:8px; padding:15px; margin-top:10px;'>
                        <b>📐 Perhitungan:</b><br>
                        Base confidence (weighted avg): <b>{cb.get('base_confidence', 0):.1f}%</b><br>
                        Cross-domain confirmation bonus: <b>+{cb.get('cross_bonus', 0)}%</b>
                        ({cb.get('confirming_domains', 0)} domain konfirmasi: {', '.join(cb.get('confirmation_detail', ['-']))})<br>
                        Pump standard adjustment ({pump_standard}): <b>{'+' if cb.get('severity_adj', 0) >= 0 else ''}{cb.get('severity_adj', 0)}%</b><br>
                        <hr style='margin:8px 0'>
                        <b>Final Confidence: {cb.get('final_confidence', 0)}%</b>
                    </div>
                    """, unsafe_allow_html=True)
                
                mech_res = st.session_state.mech_result
                hyd_res  = st.session_state.hyd_result
                elec_res = st.session_state.elec_result
                mech_data_stored = st.session_state.get("mech_data", {})
                vel_stored = {p: d.get("velocity", 0) for p, d in mech_data_stored.get("points", {}).items()}
                bands_stored = {p: d.get("bands", {}) for p, d in mech_data_stored.get("points", {}).items()}
                
                diff_result = perform_differential_diagnosis(
                    mech_res, hyd_res, elec_res,
                    temp_data, vel_stored, bands_stored,
                    {}, rpm / 60
                )
                
                if diff_result.get("applicable"):
                    st.divider()
                    st.subheader("🔬 Differential Diagnosis")
                    st.caption(f"Analisis pembeda antara dua fault yang memiliki gejala serupa")
                    
                    winner = diff_result["winner"]
                    loser  = diff_result["loser"]
                    w_score = diff_result["winner_score"]
                    l_score = diff_result["loser_score"]
                    
                    col_win, col_los = st.columns(2)
                    with col_win:
                        st.markdown(f"""
                        <div style='background:#e8f5e9; border-left:5px solid #27ae60; padding:15px; border-radius:8px;'>
                            <h4 style='color:#1b5e20; margin:0 0 8px 0;'>✅ DIPILIH: {winner}</h4>
                            <p style='color:#2e7d32; font-size:0.85em; margin:0;'>Skor evidence: <b>{w_score}</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                        if diff_result["winner_evidence"]:
                            for ev in diff_result["winner_evidence"]:
                                st.markdown(f"&nbsp;&nbsp;✔ {ev}")
                    
                    with col_los:
                        st.markdown(f"""
                        <div style='background:#fce4ec; border-left:5px solid #c0392b; padding:15px; border-radius:8px;'>
                            <h4 style='color:#b71c1c; margin:0 0 8px 0;'>❌ DISINGKIRKAN: {loser}</h4>
                            <p style='color:#c62828; font-size:0.85em; margin:0;'>Skor evidence: <b>{l_score}</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                        if diff_result["loser_evidence"]:
                            for ev in diff_result["loser_evidence"]:
                                st.markdown(f"&nbsp;&nbsp;✘ {ev}")
                    
                    st.info(f"💡 **Reasoning:** {diff_result['reasoning']}")
                
                st.divider()
                st.subheader("🗺️ Fault Propagation Map untuk Perbaikan")
                st.caption("Rantai fault dari root cause ke effect + action perbaikan yang diperlukan")
                
                propagation_map = generate_fault_propagation_map(
                    st.session_state.mech_result,
                    st.session_state.hyd_result,
                    st.session_state.elec_result,
                    temp_data,
                    pump_standard
                )
                
                if propagation_map:
                    for idx, prop in enumerate(propagation_map, 1):
                        priority = prop["priority"]
                        if priority == "CRITICAL":
                            priority_icon, border_color, bg_color = "🔴", "#c0392b", "#fff0f0"
                        elif priority == "HIGH":
                            priority_icon, border_color, bg_color = "🟠", "#e67e22", "#fff8f0"
                        elif priority == "MEDIUM":
                            priority_icon, border_color, bg_color = "🟡", "#f1c40f", "#fffdf0"
                        else:
                            priority_icon, border_color, bg_color = "🟢", "#27ae60", "#f0fff4"
                        
                        html_str = f"<div style='background-color: {bg_color}; border-left: 6px solid {border_color}; padding: 20px; border-radius: 8px; margin-bottom: 25px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); color: #2c3e50;'>"
                        html_str += f"<div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(0,0,0,0.1); padding-bottom: 12px; margin-bottom: 15px; flex-wrap: wrap; gap: 10px;'>"
                        html_str += f"<h4 style='margin: 0; color: #1E3A5F; font-size: 1.15em;'>{priority_icon} Scenario {idx}: {prop['root_cause']}</h4>"
                        html_str += f"<div style='font-size: 0.85em; font-weight: 600;'><span style='background-color: {border_color}; color: white; padding: 5px 12px; border-radius: 12px; margin-right: 8px;'>Priority: {priority}</span><span style='background-color: #1E3A5F; color: white; padding: 5px 12px; border-radius: 12px;'>Timeline: {prop['timeline']}</span></div></div>"
                        html_str += "<div style='font-weight: 600; color: #444; margin-bottom: 10px; font-size: 0.95em;'>🔗 Fault Chain:</div>"
                        html_str += "<div style='display: flex; align-items: center; flex-wrap: wrap; gap: 10px; background: rgba(255,255,255,0.6); padding: 15px; border-radius: 8px; border: 1px solid rgba(0,0,0,0.05); margin-bottom: 20px;'>"
                        n_nodes = len(prop["fault_chain"])
                        for i, fault in enumerate(prop["fault_chain"]):
                            html_str += f"<div style='background-color: #1E3A5F; color: white; padding: 8px 14px; border-radius: 6px; font-size: 0.85em; font-weight: 600; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.2);'>{fault}</div>"
                            if i < n_nodes - 1:
                                html_str += "<div style='color: #7f8c8d; font-weight: bold; font-size: 1.2em;'>→</div>"
                        html_str += "</div>"
                        html_str += "<div style='font-weight: 600; color: #444; margin-bottom: 10px; font-size: 0.95em;'>🔧 Repair Actions:</div>"
                        html_str += "<div style='background: rgba(255,255,255,0.6); padding: 10px 20px; border-radius: 8px; border: 1px solid rgba(0,0,0,0.05);'>"
                        for action in prop["repair_actions"]:
                            clean_action = action.replace("✅ ", "").strip()
                            html_str += f"<div style='padding: 8px 0; border-bottom: 1px dashed rgba(0,0,0,0.1); display: flex; align-items: flex-start; font-size: 0.9em; color: #2c3e50;'><span style='margin-right: 10px; font-size: 1.1em; line-height: 1.2;'>✅</span><span style='line-height: 1.4;'>{clean_action}</span></div>"
                        html_str += "</div></div>"
                        st.markdown(html_str, unsafe_allow_html=True)
                else:
                    st.info("ℹ️ Tidak ada fault propagation map yang dihasilkan. Semua domain dalam kondisi normal.")
                
                st.divider()
                st.subheader("📥 Export Report")
                
                if st.button("📊 Generate Unified CSV Report", type="primary"):
                    csv_report = generate_unified_csv_report(
                        machine_id,
                        rpm,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        st.session_state.get("mech_data", {}),
                        st.session_state.get("hyd_data", {}),
                        st.session_state.get("elec_data", {}),
                        integrated_result,
                        temp_data,
                        pump_standard
                    )
                    
                    st.download_button(
                        label="📥 Download CSV Report",
                        data=csv_report,
                        file_name=f"PUMP_DIAG_{machine_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    st.success("✅ Report generated successfully!")
                
                st.divider()
                st.caption(f"""
                **Standar Acuan**: ISO 10816-3/7 | ISO 13373-1 | API 610 | IEC 60034 | API 670
                **Pump Standard**: {pump_standard} | **Threshold Adjustment**: Active
                **Algoritma**: Hybrid rule-based dengan cross-domain correlation + confidence scoring
                ⚠️ Decision Support System - Verifikasi oleh personnel kompeten untuk keputusan kritis
                🏭 Pertamina Patra Niaga - Asset Integrity Management
                """)

if __name__ == "__main__":
    main()
