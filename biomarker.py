from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from dotenv import load_dotenv
import google.generativeai as genai
import os
import re
from typing import Dict, Any, Union, List


# ---------------- Initialize ----------------
app = FastAPI(title="LLM Model API", version="3.4")

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load environment variables
load_dotenv()

# ✅ Fetch Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️ Warning: GEMINI_API_KEY or GOOGLE_API_KEY not found. Biomarker analysis may fail.")

# ✅ Configure Gemini Client
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    
MODEL_ID = "gemini-2.5-flash"


# ---------------- Schema ----------------
class BiomarkerRequest(BaseModel):
    # ---------------- Patient Info ----------------
    age: int = Field(default=52, description="Patient age in years")
    gender: str = Field(default="female", description="Gender of the patient")
    height: float = Field(default=165, description="Height in cm")
    weight: float = Field(default=70, description="Weight in kg")

    # ---------------- Kidney Function ----------------
    urea: float = Field(default=30.0, description="Urea (S) in mg/dL")
    creatinine: float = Field(default=1.0, description="Creatinine (S) in mg/dL")
    uric_acid: float = Field(default=5.0, description="Uric Acid (S) in mg/dL")
    calcium: float = Field(default=9.5, description="Calcium (S) in mg/dL")
    phosphorus: float = Field(default=3.5, description="Phosphorus (S) in mg/dL")
    sodium: float = Field(default=140.0, description="Sodium (S) in mEq/L")
    potassium: float = Field(default=4.2, description="Potassium (S) in mEq/L")
    chloride: float = Field(default=102.0, description="Chloride (S) in mEq/L")
    amylase: float = Field(default=70.0, description="Amylase (S) in U/L")
    lipase: float = Field(default=35.0, description="Lipase (S) in U/L")
    bicarbonate: float = Field(default=24.0, description="Bicarbonate (S) in mEq/L")
    egfr: float = Field(default=100.0, description="Estimated GFR (S) in mL/min/1.73m²")
    serum_osmolality: float = Field(default=290.0, description="Serum Osmolality (S) in mOsm/kg")
    ionized_calcium: float = Field(default=1.25, description="Ionized Calcium (S) in mmol/L")
    
    # ---------------- Basic Check-up ----------------
    wbc: float = Field(default=6.0, description="White Blood Cell count (×10^3/μL)")
    hemoglobin: float = Field(default=14.0, description="Hemoglobin (g/dL)")
    mcv: float = Field(default=90.0, description="Mean Corpuscular Volume (fL)")
    rdw: float = Field(default=13.5, description="Red Cell Distribution Width (%)")
    lymphocytes: float = Field(default=30.0, description="Lymphocyte percentage (%)")
    
    # ---------------- Diabetic Profile ----------------
    fasting_blood_sugar: float = Field(default=85.0, description="Fasting Blood Sugar (mg/dL)")
    hb1ac: float = Field(default=5.4, description="HbA1c (%)")
    insulin: float = Field(default=10.0, description="Insulin (µIU/mL)")
    c_peptide: float = Field(default=1.2, description="C-Peptide (ng/mL)")
    homa_ir: float = Field(default=1.2, description="HOMA-IR")
    
    # ---------------- Lipid Profile ----------------
    total_cholesterol: float = Field(default=180.0, description="Total Cholesterol (mg/dL)")
    ldl: float = Field(default=90.0, description="LDL Cholesterol (mg/dL)")
    hdl: float = Field(default=50.0, description="HDL Direct (mg/dL)")
    cholesterol_hdl_ratio: float = Field(default=3.0, description="Cholesterol/HDL Ratio")
    triglycerides: float = Field(default=120.0, description="Triglycerides (mg/dL)")
    apo_a1: float = Field(default=140.0, description="Apo A-1 (mg/dL)")
    apo_b: float = Field(default=70.0, description="Apo B (mg/dL)")
    apo_ratio: float = Field(default=0.5, description="Apo B : Apo A-1 ratio")
    
    # ---------------- Liver Function ----------------
    albumin: float = Field(default=4.2, description="Albumin (g/dL)")
    total_protein: float = Field(default=7.0, description="Total Protein (g/dL)")
    alt: float = Field(default=25.0, description="ALT (U/L)")
    ast: float = Field(default=24.0, description="AST (U/L)")
    alp: float = Field(default=120.0, description="ALP (U/L)")
    ggt: float = Field(default=20.0, description="GGT (U/L)")
    ld: float = Field(default=180.0, description="LDH (U/L)")
    globulin: float = Field(default=3.0, description="Globulin (g/dL)")
    albumin_globulin_ratio: float = Field(default=1.4, description="Albumin/Globulin Ratio")
    magnesium: float = Field(default=2.0, description="Magnesium (mg/dL)")
    total_bilirubin: float = Field(default=0.7, description="Total Bilirubin (mg/dL)")
    direct_bilirubin: float = Field(default=0.3, description="Direct Bilirubin (mg/dL)")
    indirect_bilirubin: float = Field(default=0.4, description="Indirect Bilirubin (mg/dL)")
    ammonia: float = Field(default=35.0, description="Ammonia (NH3) (µmol/L)")
    
    # ---------------- Cardiac Profile ----------------
    hs_crp: float = Field(default=1.0, description="High-Sensitivity CRP (mg/L)")
    ck: float = Field(default=150.0, description="Creatine Kinase (U/L)")
    ck_mb: float = Field(default=20.0, description="CK-MB (U/L)")
    homocysteine: float = Field(default=10.0, description="Homocysteine (µmol/L)")
    
    # ---------------- Mineral & Heavy Metal ----------------
    zinc: float = Field(default=90.0, description="Zinc (µg/dL)")
    copper: float = Field(default=100.0, description="Copper (µg/dL)")
    selenium: float = Field(default=120.0, description="Selenium (µg/L)")
    
    # ---------------- Iron Profile ----------------
    iron: float = Field(default=100.0, description="Iron (µg/dL)")
    tibc: float = Field(default=300.0, description="TIBC (µg/dL)")
    transferrin: float = Field(default=250.0, description="Transferrin (mg/dL)")
    
    # ---------------- Vitamins ----------------
    vitamin_d: float = Field(default=35.0, description="Vitamin D (ng/mL)")
    vitamin_b12: float = Field(default=500.0, description="Vitamin B12 (pg/mL)")
    
    # ---------------- Hormone Profile ----------------
    total_testosterone: float = Field(default=450.0, description="Total Testosterone (ng/dL)")
    free_testosterone: float = Field(default=15.0, description="Free Testosterone (pg/mL)")
    estrogen: float = Field(default=60.0, description="Estrogen / Estradiol (pg/mL)")
    progesterone: float = Field(default=1.0, description="Progesterone (ng/mL)")
    dhea_s: float = Field(default=250.0, description="DHEA-S (µg/dL)")
    shbg: float = Field(default=40.0, description="SHBG (nmol/L)")
    lh: float = Field(default=5.0, description="LH (IU/L)")
    fsh: float = Field(default=6.0, description="FSH (IU/L)")
    
    # ---------------- Thyroid Profile ----------------
    tsh: float = Field(default=2.0, description="TSH (µIU/mL)")
    free_t3: float = Field(default=3.2, description="Free T3 (pg/mL)")
    free_t4: float = Field(default=1.2, description="Free T4 (ng/dL)")
    total_t3: float = Field(default=120.0, description="Total T3 (ng/dL)")
    total_t4: float = Field(default=8.0, description="Total T4 (µg/dL)")
    reverse_t3: float = Field(default=15.0, description="Reverse T3 (ng/dL)")
    tpo_ab: float = Field(default=5.0, description="Thyroid Antibodies – TPO Ab (IU/mL)")
    tg_ab: float = Field(default=3.0, description="Thyroid Antibodies – TG Ab (IU/mL)")
    
    # ---------------- Adrenal / Stress / Other Hormones ----------------
    cortisol: float = Field(default=12.0, description="Cortisol (µg/dL)")
    acth: float = Field(default=25.0, description="ACTH (pg/mL)")
    igf1: float = Field(default=200.0, description="IGF-1 (ng/mL)")
    leptin: float = Field(default=10.0, description="Leptin (ng/mL)")
    adiponectin: float = Field(default=10.0, description="Adiponectin (µg/mL)")
    
    # ---------------- Blood Marker Cancer Profile ----------------
    ca125: float = Field(default=20.0, description="CA125 (U/mL)")
    ca15_3: float = Field(default=25.0, description="CA15-3 (U/mL)")
    ca19_9: float = Field(default=30.0, description="CA19-9 (U/mL)")
    psa: float = Field(default=1.0, description="PSA (ng/mL)")
    cea: float = Field(default=2.0, description="CEA (ng/mL)")
    calcitonin: float = Field(default=5.0, description="Calcitonin (pg/mL)")
    afp: float = Field(default=5.0, description="AFP (ng/mL)")
    tnf: float = Field(default=2.0, description="Tumor Necrosis Factor (pg/mL)")
    
    # ---------------- Immune Profile ----------------
    ana: float = Field(default=0.5, description="ANA (IU/mL)")
    ige: float = Field(default=100.0, description="IgE (IU/mL)")
    igg: float = Field(default=1200.0, description="IgG (mg/dL)")
    anti_ccp: float = Field(default=10.0, description="Anti-CCP (U/mL)")
    dsdna: float = Field(default=0.5, description="dsDNA (IU/mL)")
    ssa_ssb: float = Field(default=5.0, description="SSA/SSB (IU/mL)")
    rnp: float = Field(default=1.0, description="RNP (IU/mL)")
    sm_antibodies: float = Field(default=0.5, description="Sm Antibodies (IU/mL)")
    anca: float = Field(default=0.5, description="ANCA (IU/mL)")
    anti_ena: float = Field(default=0.5, description="Anti-ENA (IU/mL)")
    il6: float = Field(default=3.0, description="IL-6 (pg/mL)")
    allergy_panel: float = Field(default=10.0, description="Comprehensive Allergy Profile (IgE & Food Sensitivity IgG)")




# ---------------- Cleaning Utility ----------------
def clean_json(data: Union[Dict, List, str]) -> Union[Dict, List, str]:
    """Recursively removes separators, extra whitespace, and artifacts from all string values."""
    if isinstance(data, str):
        text = re.sub(r"-{3,}", "", data)
        text = re.sub(r"\s+", " ", text)
        text = text.strip(" -\n\t\r")
        return text
    elif isinstance(data, list):
        return [clean_json(i) for i in data if i and clean_json(i)]
    elif isinstance(data, dict):
        return {k.strip(): clean_json(v) for k, v in data.items()}
    return data


# ---------------- Parser ----------------
def parse_medical_report(text: str):
    """
    Parses Gemini markdown response → structured JSON.
    Detects section headers, **bold keys**, and table entries.
    """
    def clean_line(line: str) -> str:
        return re.sub(r"[\-\*\u2022]+\s*", "", line.strip())

    def parse_bold_entities(block: str) -> Dict[str, str]:
        """Extracts **bold** entities and maps text until next bold or section."""
        entities = {}
        pattern = re.compile(r"\*\*(.*?)\*\*(.*?)(?=\*\*|###|$)", re.S)
        for match in pattern.finditer(block):
            key = match.group(1).strip().strip(":")
            val = match.group(2).strip().replace("\n", " ")
            val = re.sub(r"\s+", " ", val)
            if key:
                entities[key] = val
        return entities

    data = {
        "executive_summary": {"top_priorities": [], "key_strengths": []},
        "system_analysis": {},
        "personalized_action_plan": {},
        "interaction_alerts": [],
        "normal_ranges": {},
        "biomarker_table": []
    }

    # --- Executive Summary ---
    exec_match = re.search(r"###\s*Executive Summary(.*?)(?=###|$)", text, re.S | re.I)
    if exec_match:
        block = exec_match.group(1)
        priorities = re.findall(r"\d+\.\s*(.*?)\n", block)
        if priorities:
            data["executive_summary"]["top_priorities"] = [clean_line(p) for p in priorities]
        strengths_match = re.search(r"\*\*Key Strengths:\*\*(.*)", block, re.S)
        if strengths_match:
            strengths_text = strengths_match.group(1)
            strengths = [clean_line(s) for s in strengths_text.splitlines() if clean_line(s)]
            data["executive_summary"]["key_strengths"] = strengths

    # --- System Analysis ---
    sys_match = re.search(r"###\s*System[- ]Specific Analysis(.*?)(?=###|$)", text, re.S | re.I)
    if sys_match:
        sys_block = sys_match.group(1)
        data["system_analysis"] = parse_bold_entities(sys_block)

    # --- Personalized Action Plan ---
    plan_match = re.search(r"###\s*Personalized Action Plan(.*?)(?=###|$)", text, re.S | re.I)
    if plan_match:
        plan_block = plan_match.group(1)
        data["personalized_action_plan"] = parse_bold_entities(plan_block)

    # --- Interaction Alerts ---
    alerts_match = re.search(r"###\s*Interaction Alerts(.*?)(?=###|$)", text, re.S | re.I)
    if alerts_match:
        alerts_block = alerts_match.group(1)
        alerts = [clean_line(a) for a in alerts_block.splitlines() if clean_line(a)]
        data["interaction_alerts"] = alerts

    # --- Normal Ranges ---
    normal_match = re.search(r"###\s*Normal Ranges(.*?)(?=###|$)", text, re.S | re.I)
    if normal_match:
        normal_block = normal_match.group(1)
        for match in re.findall(r"-\s*([^:]+):\s*([^\n]+)", normal_block):
            biomarker, rng = match
            data["normal_ranges"][biomarker.strip()] = rng.strip()

    # --- Tabular Mapping ---
    table_match = re.search(r"###\s*Tabular Mapping(.*)", text, re.S | re.I)
    if table_match:
        table_block = table_match.group(1)
        # robust row matcher: capture any table rows with 5 pipe-separated columns
        table_pattern = r"\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|"
        for biomarker, value, status, insight, ref in re.findall(table_pattern, table_block):
            # normalize
            biomarker_s = biomarker.strip()
            value_s = value.strip()
            status_s = status.strip()
            insight_s = insight.strip()
            ref_s = ref.strip()

            # ---------- ONLY SKIP rows where ALL five fields are empty ----------
            if not any([biomarker_s, value_s, status_s, insight_s, ref_s]):
                # This is the empty-row you showed: skip it and continue
                continue

            # ---------- ALSO SKIP rows that are pure separator artifacts ----------
            # e.g., ":-----------" or "--------" in biomarker column (common AI artifacts)
            def is_separator_cell(s: str) -> bool:
                # treat as separator if contains no alphanumeric chars
                return not bool(re.search(r"[A-Za-z0-9]", s))

            if all(is_separator_cell(c) for c in [biomarker_s, value_s, status_s, insight_s, ref_s]):
                continue

            # ---------- Append the cleaned/valid row ----------
            data["biomarker_table"].append({
                "biomarker": biomarker_s,
                "value": value_s,
                "status": status_s,
                "insight": insight_s,
                "reference_range": ref_s,
            })

    return data


# ---------------- Endpoint ----------------
@app.post("/predict")
def predict(data: BiomarkerRequest):
    """Accepts biomarker input and returns structured and complete detailed medical insights."""
    try:
        # --- Prompt Template ---
        prompt = """
You are an advanced **Medical Insight Generation AI** trained to analyze **biomarkers and lab results**.

⚠️ IMPORTANT — OUTPUT FORMAT INSTRUCTIONS:
Return your report in this strict markdown structure.

------------------------------
### Executive Summary
**Top 3 Health Priorities:**
1. ...
2. ...
3. ...
make it more detailed 

**Key Strengths:**
- ...
- ...
make it detailed
------------------------------
### System-Specific Analysis

**Cardiovascular System**  
Status: Normal. Explanation: Lipid profile including Total Cholesterol, LDL, HDL, Triglycerides, Apo A-1, Apo B, Apo Ratio, and Cholesterol/HDL Ratio are within reference ranges, indicating low risk of atherosclerosis, coronary artery disease, and other cardiovascular disorders. hs-CRP, CK, CK-MB, and Homocysteine levels are normal, reflecting minimal systemic inflammation and proper myocardial health.

**Metabolic & Glycemic Control**  
Status: Normal. Explanation: Fasting Blood Sugar, HbA1c, Insulin, C-Peptide, and HOMA-IR are within healthy ranges, suggesting effective glucose metabolism, insulin sensitivity, and low risk of prediabetes or diabetes.

**Liver Function**  
Status: Normal. Explanation: ALT, AST, ALP, GGT, LDH, Total Bilirubin, Direct and Indirect Bilirubin, Albumin, Globulin, Albumin/Globulin Ratio, Total Protein, Ammonia, and Magnesium are within reference ranges, reflecting normal hepatocellular integrity, protein synthesis, and biliary excretion. Abnormalities could indicate hepatic injury, cholestasis, or metabolic liver disorders.

**Renal Function**  
Status: Normal. Explanation: Urea, Creatinine, eGFR, Uric Acid, Sodium, Potassium, Chloride, Phosphorus, Calcium, Ionized Calcium, Bicarbonate, Serum Osmolality, Amylase, and Lipase are within expected ranges, suggesting proper kidney filtration, electrolyte balance, and pancreatic enzyme activity. Deviations may indicate renal impairment, electrolyte disorders, or pancreatitis risk.

**Thyroid Function**  
Status: Normal. Explanation: TSH, Free T3, Free T4, Total T3, Total T4, Reverse T3, TPO Ab, and TG Ab are within reference limits, showing normal thyroid hormone production, peripheral conversion, and autoimmune status. Abnormal levels may indicate hypothyroidism, hyperthyroidism, or thyroid autoimmunity.

**Adrenal & Stress Hormones**  
Status: Normal. Explanation: Cortisol, ACTH, DHEA-S, IGF-1, Leptin, and Adiponectin are within normal ranges, reflecting healthy adrenal function, stress response, metabolic regulation, and energy homeostasis. Abnormalities could indicate adrenal insufficiency, Cushing’s syndrome, metabolic disorders, or leptin/adiponectin imbalance.

**Sex Hormones & Reproductive Health**  
Status: Normal. Explanation: Total Testosterone, Free Testosterone, SHBG, Estrogen, Progesterone, LH, and FSH are within expected ranges based on gender and menstrual cycle, indicating balanced gonadal function, fertility potential, and hormonal homeostasis. Deviations may impact reproductive function, libido, or secondary sexual characteristics.

**Vitamins & Minerals**  
Status: Normal. Explanation: Vitamin D, Vitamin B12, Iron, TIBC, Transferrin, Zinc, Copper, Selenium, and Magnesium are within reference ranges, supporting optimal hematologic function, enzymatic reactions, immune defense, and bone health. Deficiencies may lead to anemia, metabolic disturbances, or immune dysfunction.

**Hematology & Immune Function**  
Status: Normal. Explanation: Hemoglobin, MCV, RDW, WBC, Lymphocytes, Albumin, Globulin, ANA, IgE, IgG, Anti-CCP, dsDNA, SSA/SSB, RNP, Sm Antibodies, ANCA, Anti-ENA, IL-6, and Allergy Panel are within normal limits, indicating proper oxygen transport, red blood cell morphology, and immune competence. Deviations could indicate anemia, infection, inflammation, or autoimmune conditions.

**Cancer Markers**  
Status: Normal. Explanation: CA125, CA15-3, CA19-9, PSA, CEA, AFP, Calcitonin, and TNF are within reference ranges, suggesting low risk for malignancy or tumor activity. Elevated values may require further imaging or diagnostic evaluation.

**Inflammatory Markers**  
Status: Normal. Explanation: hs-CRP, IL-6, and Homocysteine are within recommended ranges, reflecting low systemic inflammation and minimal cardiovascular or metabolic risk. Elevations may indicate chronic inflammation, autoimmune activity, or thrombotic risk.

------------------------------
### Personalized Action Plan
**Nutrition:** ...
make it detailed
**Lifestyle:** ...
make it detailed
**Testing:** ...
make it detailed
**Medical Consultation:** ...
make it detailed
------------------------------
### Interaction Alerts
- ...
- ...
make it detailed
------------------------------
### Normal Ranges
# Kidney Function
- Urea (S): 17–43 mg/dL
- Creatinine (Men): 0.74–1.35 mg/dL
- Creatinine (Women): 0.59–1.04 mg/dL
- Uric Acid (Men): 3.4–7.0 mg/dL
- Uric Acid (Women): 2.4–6.0 mg/dL
- Calcium (S): 8.5–10.5 mg/dL
- Phosphorus (S): 2.5–4.5 mg/dL
- Sodium (S): 135–145 mEq/L
- Potassium (S): 3.5–5.1 mEq/L
- Chloride (S): 98–107 mEq/L
- Bicarbonate (S): 22–28 mEq/L
- eGFR: ≥90 mL/min/1.73m²
- Serum Osmolality: 275–295 mOsm/kg
- Ionized Calcium: 1.12–1.32 mmol/L
- Amylase (S): 23–85 U/L
- Lipase (S): 0–160 U/L

# Basic Check-up
- WBC: 4–10 ×10^3/μL
- Hemoglobin: 13–17 g/dL
- MCV: 80–100 fL
- RDW: 11.5–14.5 %
- Lymphocytes: 20–40 %

# Diabetic Profile
- Fasting Blood Sugar: 70–99 mg/dL
- HbA1c: <5.7 %
- Insulin: 2–20 µIU/mL
- C-Peptide: 0.5–2.0 ng/mL
- HOMA-IR: <1 Optimal, 1–2 Normal, >2 Insulin Resistance

# Lipid Profile
- Total Cholesterol: <200 mg/dL
- LDL: <100 mg/dL
- HDL (Men): ≥40 mg/dL
- HDL (Women): ≥50 mg/dL
- Triglycerides: <150 mg/dL
- Apo A-1: 120–160 mg/dL
- Apo B: <90 mg/dL
- Apo B/A1 ratio: 0.3–0.7
- Cholesterol/HDL Ratio: <3.5 Optimal

# Liver Function
- Albumin: 3.5–5.0 g/dL
- Total Protein: 6.0–8.3 g/dL
- ALT: 10–40 U/L
- AST: 10–40 U/L
- ALP: 44–147 U/L
- GGT: 8–61 U/L
- LDH: 140–280 U/L
- Globulin: 2.0–3.5 g/dL
- Albumin/Globulin Ratio: 1.1–2.5
- Magnesium: 1.7–2.2 mg/dL
- Total Bilirubin: 0.1–1.2 mg/dL
- Direct Bilirubin: 0.0–0.3 mg/dL
- Indirect Bilirubin: 0.2–0.9 mg/dL
- Ammonia: 15–45 µmol/L

# Cardiac Profile
- hs-CRP: 1–3 mg/L
- CK: 40–200 U/L
- CK-MB: 0–25 U/L
- Homocysteine: 5–15 µmol/L

# Minerals & Heavy Metals
- Zinc: 70–120 µg/dL
- Copper: 70–140 µg/dL
- Selenium: 70–150 µg/L

# Iron Profile
- Iron (Men): 60–170 µg/dL
- Iron (Women): 50–170 µg/dL
- TIBC: 250–450 µg/dL
- Transferrin: 200–360 mg/dL

# Vitamins
- Vitamin D: 30–60 ng/mL
- Vitamin B12: 200–900 pg/mL

# Hormones
- Total Testosterone (Men): 300–1000 ng/dL
- Total Testosterone (Women): 15–70 ng/dL
- Free Testosterone (Men): 5–21 pg/mL
- Free Testosterone (Women): 0.5–4.2 pg/mL
- Estrogen (Men): 10–40 pg/mL
- Estrogen (Women Follicular): 30–120 pg/mL
- Estrogen (Women Ovulation): 130–370 pg/mL
- Estrogen (Women Luteal): 70–250 pg/mL
- Estrogen (Women Postmenopause): <20–30 pg/mL
- Progesterone: 0.2–1.4 ng/mL
- SHBG (Men): 10–57 nmol/L
- SHBG (Women): 18–144 nmol/L
- LH: 1.7–8.6 IU/L
- FSH: 1.5–12.4 IU/L
- DHEA-S (Men): 280–640 µg/dL
- DHEA-S (Women): 65–380 µg/dL
- Cortisol (AM): 6–23 µg/dL
- Cortisol (PM): 2–14 µg/dL
- IGF-1: 100–300 ng/mL
- Leptin (Men): 0.5–8 ng/mL
- Leptin (Women): 5–25 ng/mL
- Adiponectin: 5–30 µg/mL

# Thyroid
- TSH: 0.4–4.0 µIU/mL
- Free T3: 2.0–4.4 pg/mL
- Free T4: 0.8–1.8 ng/dL
- Total T3: 80–180 ng/dL
- Total T4: 4.5–12 µg/dL
- Reverse T3: 9–24 ng/dL
- TPO Ab: <35 IU/mL
- TG Ab: <40 IU/mL

# Cancer Markers
- CA125: <35 U/mL
- CA15-3: <30 U/mL
- CA19-9: <37 U/mL
- PSA: <4 ng/mL
- CEA: <5 ng/mL
- Calcitonin: <10 pg/mL
- AFP: <10 ng/mL
- TNF: <8 pg/m



------------------------------
### Tabular Mapping
| Biomarker | Value | Status | Insight | Reference Range |
| Albumin | X | Normal | ... | 3.5–5.0 g/dL |
| Creatinine | X | High | ... | 0.7–1.3 mg/dL |
| Glucose | X | ... | ... | 70–100 mg/dL |
------------------------------
"""

        # --- Format User Data ---
        user_message = f"""
**Patient Info**
- Age: {data.age}
- Gender: {data.gender}
- Height: {data.height} cm
- Weight: {data.weight} kg

**Metabolic & Glycemic Control**
- Fasting Blood Sugar: {data.fasting_blood_sugar} mg/dL
- HbA1c: {data.hb1ac} %
- Insulin: {data.insulin} µIU/mL
- C-Peptide: {data.c_peptide} ng/mL
- HOMA-IR: {data.homa_ir}
- Leptin: {data.leptin} ng/mL

**Cardiovascular System**
- Total Cholesterol: {data.total_cholesterol} mg/dL
- LDL: {data.ldl} mg/dL
- HDL: {data.hdl} mg/dL
- Triglycerides: {data.triglycerides} mg/dL
- ApoB: {data.apo_b} mg/dL
- Cholesterol/HDL Ratio: {data.cholesterol_hdl_ratio}
- hs-CRP: {data.hs_crp} mg/L
- Homocysteine: {data.homocysteine} µmol/L

**Liver Function**
- ALT: {data.alt} U/L
- AST: {data.ast} U/L
- GGT: {data.ggt} U/L
- Total Bilirubin: {data.total_bilirubin} mg/dL
- Total Protein: {data.total_protein} g/dL

**Renal Function**
- Creatinine: {data.creatinine} mg/dL
- eGFR: {data.egfr} mL/min/1.73m2
- Uric Acid: {data.uric_acid} mg/dL

**Vitamins & Minerals**
- Vitamin D: {data.vitamin_d} ng/mL
- Vitamin B12: {data.vitamin_b12} pg/mL
- Iron: {data.iron} µg/dL
- Zinc: {data.zinc} µg/dL

**Thyroid Function**
- TSH: {data.tsh} µIU/mL
- Free T3: {data.free_t3} pg/mL
- Free T4: {data.free_t4} ng/dL

**Sex Hormones & Reproductive Health**
- Total Testosterone: {data.total_testosterone} ng/dL
- Free Testosterone: {data.free_testosterone} pg/mL
- Estrogen (Estradiol): {data.estrogen} pg/mL
- SHBG: {data.shbg} nmol/L

**Adrenal & Stress Hormones**
- Cortisol: {data.cortisol} µg/dL
- DHEA-S: {data.dhea_s} µg/dL

**Autoimmune / Inflammatory Markers**
- Anti-CCP: {data.anti_ccp} U/mL
"""

        # --- Gemini Call ---
        model = genai.GenerativeModel(MODEL_ID)
        response = model.generate_content(f"{prompt}\n\n{user_message}")

        if not response or not getattr(response, "text", None):
            raise ValueError("Empty response from Gemini model.")

        report_text = response.text.strip()

        # --- Parse + Clean ---
        parsed_output = parse_medical_report(report_text)
        cleaned_output = clean_json(parsed_output)

        return cleaned_output

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")