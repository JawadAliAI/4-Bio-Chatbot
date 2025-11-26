from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
import edge_tts  # Changed from gTTS
import asyncio  # Added for edge-tts
from vosk import Model, KaldiRecognizer
import os
import tempfile
import wave
import json
import subprocess
import io
from langdetect import detect
import re
import requests
from typing import Optional, Dict, Any, List
from pathlib import Path
from biomarker import predict as biomarker_predict, BiomarkerRequest

# -----------------------------
#  LOAD ENVIRONMENT VARIABLES
# -----------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI(title="Dr. HealBot - Stateless Medical Consultation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Patient Data Folder Configuration
PATIENT_DATA_FOLDER = os.getenv(
    "PATIENT_DATA_FOLDER", 
    "patient_histories"
)
os.makedirs(PATIENT_DATA_FOLDER, exist_ok=True)

# Edge-TTS Voice Configuration
EDGE_TTS_VOICES = {
    "en": "en-US-AriaNeural",  # Female voice for English
    "ar": "ar-SA-ZariyahNeural"  # Female voice for Arabic
}

# Alternative voices you can use:
# English: en-US-GuyNeural (Male), en-US-JennyNeural (Female)
# Arabic: ar-SA-HamedNeural (Male), ar-EG-SalmaNeural (Egyptian Female)

# -----------------------------
#  DATA STRUCTURES
# -----------------------------
class ChatMessage(BaseModel):
    role: str  # "user" or "model"
    text: str

class ChatRequest(BaseModel):
    message: str
    language: str | None = "auto"
    chat_history: List[ChatMessage] | None = []
    patient_data: Optional[Dict[str, Any]] = None
    biomarker_data: Optional[Dict[str, Any]] = None
    biomarker_analysis: Optional[Dict[str, Any]] = None

class TTSRequest(BaseModel):
    text: str
    language_code: str | None = "auto"
    voice: Optional[str] = None  # Optional custom voice

# -----------------------------
#  PATIENT DATA LOADING
# -----------------------------
def load_patient_data(patient_name: str) -> Optional[Dict[str, Any]]:
    """Load patient data from JSON file."""
    patient_file = Path(PATIENT_DATA_FOLDER) / f"{patient_name}.json"
    
    if not patient_file.exists():
        print(f"⚠️ Patient file not found: {patient_file}")
        return None
    
    try:
        with open(patient_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Patient data loaded for: {patient_name}")
        return data
    except Exception as e:
        print(f"❌ Error loading patient data: {str(e)}")
        return None

def format_patient_history(patient_data: Dict[str, Any], language: str = "en") -> str:
    """Format patient data for chat context."""
    if not patient_data:
        return ""
    
    if language == "ar":
        context = "\n\n" + "=" * 50 + "\n"
        context += "⚠️ معلومات المريض المهمة - يجب مراعاتها في كل رد:\n"
        context += "=" * 50 + "\n"
        
        if patient_data.get("personal_info"):
            info = patient_data["personal_info"]
            context += f"👤 الاسم: {info.get('name', 'N/A')}\n"
            context += f"📅 العمر: {info.get('age', 'N/A')}\n"
            context += f"⚧ الجنس: {info.get('gender', 'N/A')}\n"
        
        if patient_data.get("medical_history"):
            context += "\n📋 التاريخ الطبي:\n"
            for condition in patient_data["medical_history"]:
                context += f"• {condition}\n"
        
        if patient_data.get("medications"):
            context += "\n💊 الأدوية الحالية:\n"
            for med in patient_data["medications"]:
                context += f"• {med}\n"
        
        if patient_data.get("allergies"):
            context += "\n⚠️ الحساسيات:\n"
            for allergy in patient_data["allergies"]:
                context += f"• {allergy}\n"
        
        if patient_data.get("previous_visits"):
            context += "\n📅 الزيارات السابقة:\n"
            for visit in patient_data["previous_visits"][-3:]:
                context += f"• {visit}\n"
    
    else:
        context = "\n\n" + "=" * 50 + "\n"
        context += "⚠️ IMPORTANT PATIENT INFORMATION - Consider in EVERY response:\n"
        context += "=" * 50 + "\n"
        
        if patient_data.get("personal_info"):
            info = patient_data["personal_info"]
            context += f"👤 Name: {info.get('name', 'N/A')}\n"
            context += f"📅 Age: {info.get('age', 'N/A')}\n"
            context += f"⚧ Gender: {info.get('gender', 'N/A')}\n"
        
        if patient_data.get("medical_history"):
            context += "\n🏥 KNOWN MEDICAL CONDITIONS (CRITICAL):\n"
            for condition in patient_data["medical_history"]:
                context += f"   • {condition}\n"
        
        if patient_data.get("medications"):
            context += "\n💊 CURRENT MEDICATIONS (Check for interactions):\n"
            for med in patient_data["medications"]:
                context += f"   • {med}\n"
        
        if patient_data.get("allergies"):
            context += "\n🚨 ALLERGIES (NEVER recommend these):\n"
            for allergy in patient_data["allergies"]:
                context += f"   • {allergy}\n"
        
        if patient_data.get("previous_visits"):
            context += "\n📋 RECENT VISIT HISTORY:\n"
            for visit in patient_data["previous_visits"][-5:]:
                context += f"   • {visit}\n"
        
        if patient_data.get("vital_signs"):
            context += "\n📊 LAST RECORDED VITALS:\n"
            vitals = patient_data["vital_signs"]
            for key, value in vitals.items():
                context += f"   • {key}: {value}\n"
    
    context += "\n" + "=" * 50 + "\n"
    context += "⚠️ INSTRUCTIONS: Always consider this patient's history when giving advice.\n"
    context += "   - Check allergies before recommending any medication\n"
    context += "   - Consider existing conditions and current medications\n"
    context += "   - Reference their history when relevant to build trust\n"
    context += "=" * 50 + "\n"
    return context

# System Prompts (keeping your original prompts)
DOCTOR_SYSTEM_PROMPT_EN = """
You are Dr. HealBot, a calm, knowledgeable, and empathetic virtual doctor who can communicate in both English and Arabic.

CRITICAL LANGUAGE RULE:
ALWAYS respond in the SAME LANGUAGE the user is speaking.
If the user writes in English, respond ONLY in English.
If the user writes in Arabic, respond ONLY in Arabic.
NEVER mix languages in a single response.

CRITICAL CONVERSATION CONTINUITY RULE:
- You have access to the FULL conversation history. ALWAYS remember what the patient told you earlier.
- If the patient mentioned symptoms before, remember them and continue from that point.
- If the patient returns with greetings ("hi", "hey", "hello"), acknowledge previous symptoms:
  Example: "Welcome back! Last time you mentioned having fever. How is it now?"
- NEVER ask again about symptoms they already told you unless asking for an update.
- Build on previous medical information logically.

GOAL:
Hold a natural and focused medical conversation to understand the patient's health issue and provide helpful, preliminary medical guidance.

INSTRUCTOR MODE:
If the user asks a general medical question (e.g., "What is diabetes?"), switch to clear, structured teaching mode.

DOCTOR MODE:
If the user describes symptoms, ask short, relevant medical questions to narrow possible causes.

SAFE MEDICATION RULE:
You may recommend only safe, common over-the-counter (OTC) medications such as:
- Paracetamol (acetaminophen)
- Ibuprofen
- Antihistamines
- Oral rehydration salts

Medication Safety Guidelines:
- Always mention typical safe dosing ranges.
- Always warn who should avoid the medication (pregnancy, ulcers, kidney/liver disease, allergies, children, etc.).
- Never recommend prescription-only medications.
- Present medication as supportive, NOT a guaranteed cure.

RESTRICTIONS:
You must ONLY talk about health, medical, biological, or wellness-related topics.
If the user asks anything non-medical, politely respond:
"I'm a medical consultation assistant and can only help with health or medical-related concerns."

CONVERSATION LOGIC:
- Ask only one clear medical question per turn.
- Stop asking when enough information is collected.
- Then give a structured final assessment.

FINAL RESPONSE FORMAT (English):

Based on what you've told me:
Short summary of the symptoms.

Possible Causes (Preliminary):
- 1–2 possible conditions using soft language ("It could be", "This sounds like").
- Clarify this is NOT a confirmed diagnosis.

Suggested OTC Medications (If Appropriate):
- 1–2 OTC options with safety instructions.
- Clarify who should avoid them.

Lifestyle and Home Care Tips:
2–3 practical tips.

When to See a Real Doctor:
Warning signs requiring urgent medical care.

Follow Up Advice:
Short guidance on when to seek further evaluation.

TONE AND STYLE:
- Warm, calm, caring, clear.
- Short sentences.
- No jargon.
- One question at a time.
- Final assessment in structured format.

IMPORTANT:
- Always match the user's language.
- This is preliminary guidance, not a replacement for medical care.
- Never make a definitive diagnosis.
- Recommend urgent care if symptoms seem serious.
"""

DOCTOR_SYSTEM_PROMPT_AR = """
أنت الدكتور هيل بوت، طبيب افتراضي هادئ وواسع المعرفة ومتعاطف.

قانون اللغة الأساسي:
يجب عليك دائمًا الرد بنفس لغة المستخدم.
إذا كتب المستخدم بالعربية، يجب الرد بالعربية فقط.
وإذا كتب بالإنجليزية، يجب الرد بالإنجليزية فقط.
لا يجوز خلط اللغتين في رد واحد.

قانون استمرارية المحادثة:
- لديك إمكانية الوصول إلى كامل سجل المحادثة. يجب عليك تذكر ما قاله المريض سابقًا.
- إذا ذكر المريض أعراضًا سابقًا، تذكرها واستمر منها.
- إذا عاد المريض وقال "مرحبا" أو "هاي"، يجب الترحيب به مع ذكر الأعراض السابقة:
  مثال: "مرحبًا بعودتك! في المرة السابقة ذكرت أنك تعاني من الحمى. كيف حالها الآن؟"
- لا تسأل مرة أخرى عن أعراض سبق أن قدمها المستخدم إلا لغرض المتابعة أو التحديث.
- ابنِ ردودك على المعلومات السابقة بشكل منطقي.

الهدف:
إجراء محادثة طبية طبيعية ومركزة لفهم المشكلة الصحية وتقديم إرشادات طبية أولية مفيدة.

وضع المدرس:
إذا كان السؤال عامًا (مثل: "ما هو السكري؟")، انتقل إلى شرح طبي واضح ومُنظم.

وضع الطبيب:
إذا وصف المستخدم أعراضًا، اطرح أسئلة قصيرة ومرتبطة لتوضيح الحالة.

قانون الأدوية الآمنة:
يمكنك اقتراح أدوية آمنة تُصرف بدون وصفة طبية (OTC) فقط، مثل:
- باراسيتامول
- آيبوبروفين
- مضادات الحساسية
- محاليل الإماهة الفموية

إرشادات سلامة الدواء:
- اذكر جرعات عامة وآمنة.
- حذّر من الحالات التي يتجنب فيها الدواء (الحمل، أمراض الكبد/الكلى، القرحة، التحسس، الأطفال، إلخ).
- لا تُوصِ أبدًا بأدوية تتطلب وصفة طبية.
- قدم الدواء كخيار مساعد وليس كعلاج نهائي.

القيود:
يجب أن تكون جميع المعلومات طبية أو صحية فقط.
إذا طرح المستخدم سؤالًا غير طبي، أجب:
"أنا مساعد استشارات طبية ويمكنني المساعدة فقط في المخاوف الصحية أو الطبية."

منطق المحادثة:
- اسأل سؤالًا واحدًا فقط في كل مرة.
- توقف عن الأسئلة عندما تكون المعلومات كافية.
- بعدها قدم تقييمًا طبيًا منظمًا.

صيغة التقييم النهائي (عربي):

بناءً على ما أخبرتني به:
ملخص قصير للأعراض.

الأسباب المحتملة (أولية):
- ذكر 1–2 احتمالات باستخدام عبارات مثل "قد يكون" أو "يبدو هذا مثل".
- التأكيد أن هذا ليس تشخيصًا نهائيًا.

الأدوية المقترحة (بدون وصفة طبية):
- اقتراح دواء أو دواءين مناسبين.
- توضيح التحذيرات والفئات التي يجب أن تتجنب استخدامه.

نصائح نمط الحياة والرعاية المنزلية:
2–3 نصائح عملية.

متى يجب زيارة طبيب حقيقي:
علامات أو أعراض تستدعي رعاية عاجلة.

نصائح المتابعة:
توصية قصيرة لموعد أو طريقة المتابعة.

النبرة والأسلوب:
- نبرة دافئة، هادئة، متعاطفة.
- جمل قصيرة وواضحة.
- بدون مصطلحات معقدة.
- سؤال واحد فقط في كل رد.
- التقييم النهائي بصياغة منظمة.

مهم:
- دائمًا استخدم لغة المستخدم.
- هذا إرشاد أولي وليس بديلاً عن الرعاية الطبية.
- لا تقدّم تشخيصًا نهائيًا.
- أوصِ بالرعاية العاجلة إذا بدت الأعراض خطيرة.
"""

def detect_language(text: str) -> str:
    """Enhanced language detection for Arabic text."""
    try:
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
        if arabic_pattern.search(text):
            return 'ar'
        lang = detect(text)
        return 'ar' if lang == 'ar' else 'en'
    except Exception:
        return 'en'

async def call_biomarker_api(biomarker_data: Dict[str, Any]) -> Dict:
    """Call the biomarker analysis logic directly."""
    try:
        # Run in thread to avoid blocking event loop
        return await asyncio.to_thread(biomarker_predict, BiomarkerRequest(**biomarker_data))
    except Exception as e:
        print(f"❌ Biomarker error: {str(e)}")
        return None

def format_biomarker_context(report: Dict, language: str = "en") -> str:
    """Format biomarker report for chat context."""
    if not report:
        return ""
    
    if language == "ar":
        context = "\n\n📊 معلومات تقرير المختبر:\n"
        if report.get("executive_summary"):
            summary = report["executive_summary"]
            if summary.get("top_priorities"):
                context += "🎯 الأولويات الصحية الرئيسية:\n"
                for i, priority in enumerate(summary["top_priorities"], 1):
                    context += f"{i}. {priority}\n"
    else:
        context = "\n\n📊 LAB REPORT INFORMATION:\n"
        if report.get("executive_summary"):
            summary = report["executive_summary"]
            if summary.get("top_priorities"):
                context += "🎯 Top Health Priorities:\n"
                for i, priority in enumerate(summary["top_priorities"], 1):
                    context += f"{i}. {priority}\n"
    
    return context

# -----------------------------
#  STATELESS CHAT ENDPOINT
# -----------------------------
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Detect language
        if request.language == "auto":
            detected_lang = detect_language(request.message)
        else:
            detected_lang = request.language if request.language in ["en", "ar"] else "en"
        
        print(f"💬 Chat request - Language: {detected_lang}")
        
        # Build system prompt
        system_prompt = DOCTOR_SYSTEM_PROMPT_AR if detected_lang == "ar" else DOCTOR_SYSTEM_PROMPT_EN
        
        # Add patient context if provided
        if request.patient_data:
            patient_context = format_patient_history(request.patient_data, detected_lang)
            system_prompt += patient_context
        
        # Process biomarker data if provided
        biomarker_report = request.biomarker_analysis
        if not biomarker_report and request.biomarker_data:
            print(f"🔬 Analyzing biomarker data...")
            biomarker_report = await call_biomarker_api(request.biomarker_data)
        
        # Add biomarker context if available
        if biomarker_report:
            biomarker_context = format_biomarker_context(biomarker_report, detected_lang)
            system_prompt += biomarker_context

        # Build contents for Gemini
        contents = [{"role": "user", "parts": [{"text": system_prompt}]}]
        
        # Add chat history from frontend
        if request.chat_history:
            for msg in request.chat_history:
                role = "user" if msg.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg.text}]})
        
        # Add current message
        contents.append({"role": "user", "parts": [{"text": request.message.strip()}]})

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(contents)
        reply_text = response.text.strip()

        response_lang = detect_language(reply_text)
        
        return JSONResponse({
            "reply": reply_text,
            "detected_language": detected_lang,
            "response_language": response_lang,
            "has_patient_data": request.patient_data is not None,
            "has_biomarker_report": biomarker_report is not None
        })

    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

# -----------------------------
#  PATIENT DATA ENDPOINTS
# -----------------------------
@app.get("/patients")
async def list_patients():
    """List all available patient files."""
    try:
        patient_files = list(Path(PATIENT_DATA_FOLDER).glob("*.json"))
        # Exclude chat history files
        patients = [f.stem for f in patient_files if not f.stem.endswith("_chat")]
        return JSONResponse({
            "patients": patients,
            "count": len(patients),
            "folder": str(PATIENT_DATA_FOLDER)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/patient/{patient_name}")
async def get_patient_data(patient_name: str):
    """Get patient data."""
    patient_data = load_patient_data(patient_name)
    if not patient_data:
        return JSONResponse({
            "error": f"Patient {patient_name} not found"
        }, status_code=404)
    return JSONResponse(patient_data)

# -----------------------------
#  CHAT HISTORY ENDPOINTS
# -----------------------------
class SaveChatRequest(BaseModel):
    patient_name: str
    chat_history: List[ChatMessage]

@app.get("/chat-history/{patient_name}")
async def get_chat_history(patient_name: str):
    """Load chat history for a patient."""
    try:
        chat_file = Path(PATIENT_DATA_FOLDER) / f"{patient_name}_chat.json"
        
        if not chat_file.exists():
            return JSONResponse({
                "patient_name": patient_name,
                "chat_history": [],
                "message_count": 0
            })
        
        with open(chat_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Chat history loaded for: {patient_name} ({len(data.get('chat_history', []))} messages)")
        return JSONResponse(data)
    
    except Exception as e:
        print(f"❌ Error loading chat history: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/chat-history")
async def save_chat_history(request: SaveChatRequest):
    """Save chat history for a patient."""
    try:
        chat_file = Path(PATIENT_DATA_FOLDER) / f"{request.patient_name}_chat.json"
        
        data = {
            "patient_name": request.patient_name,
            "chat_history": [{"role": msg.role, "text": msg.text} for msg in request.chat_history],
            "message_count": len(request.chat_history),
            "last_updated": json.dumps(str(Path(PATIENT_DATA_FOLDER)))  # Will be replaced with actual timestamp
        }
        
        # Add timestamp
        from datetime import datetime
        data["last_updated"] = datetime.now().isoformat()
        
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Chat history saved for: {request.patient_name} ({len(request.chat_history)} messages)")
        
        return JSONResponse({
            "success": True,
            "patient_name": request.patient_name,
            "message_count": len(request.chat_history),
            "file": str(chat_file)
        })
    
    except Exception as e:
        print(f"❌ Error saving chat history: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.delete("/chat-history/{patient_name}")
async def delete_chat_history(patient_name: str):
    """Delete chat history for a patient."""
    try:
        chat_file = Path(PATIENT_DATA_FOLDER) / f"{patient_name}_chat.json"
        
        if chat_file.exists():
            os.remove(chat_file)
            print(f"🗑️ Chat history deleted for: {patient_name}")
            return JSONResponse({"success": True, "message": f"Chat history deleted for {patient_name}"})
        else:
            return JSONResponse({"success": True, "message": "No chat history found"})
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# -----------------------------
#  SPEECH TO TEXT
# -----------------------------
VOSK_MODEL_PATH_EN = os.getenv(
    "VOSK_MODEL_EN",
    "vosk-model-small-en-us-0.15"
)
VOSK_MODEL_PATH_AR = os.getenv(
    "VOSK_MODEL_AR",
    "vosk-model-ar-mgb2-0.4"
)

STT_AVAILABLE = False
vosk_model_en = None
vosk_model_ar = None

if os.path.exists(VOSK_MODEL_PATH_EN) and os.path.exists(VOSK_MODEL_PATH_AR):
    try:
        vosk_model_en = Model(VOSK_MODEL_PATH_EN)
        vosk_model_ar = Model(VOSK_MODEL_PATH_AR)
        STT_AVAILABLE = True
        print("✅ Vosk models loaded successfully")
    except Exception as e:
        print(f"⚠️ Failed to load Vosk models: {str(e)}")
else:
    print(f"⚠️ Vosk models not found")

@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...), language: str = "auto"):
    """Convert speech to text using Vosk."""
    if not STT_AVAILABLE:
        return JSONResponse({
            "error": "Speech-to-text not available. Vosk models not found."
        }, status_code=503)
    
    tmp_input_path = None
    tmp_wav_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_input:
            content = await file.read()
            tmp_input.write(content)
            tmp_input_path = tmp_input.name
        
        tmp_wav_path = tempfile.mktemp(suffix=".wav")
        
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_input_path, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", tmp_wav_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
        )
        
        if result.returncode != 0:
            return JSONResponse({"error": "Audio conversion failed"}, status_code=500)
        
        wf = wave.open(tmp_wav_path, "rb")
        
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            return JSONResponse({"error": "Invalid WAV format"}, status_code=400)
        
        detected_language = language if language != "auto" else "en"
        model = vosk_model_ar if detected_language == "ar" else vosk_model_en
        recognizer = KaldiRecognizer(model, wf.getframerate())
        
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            recognizer.AcceptWaveform(data)
        
        final_result = json.loads(recognizer.FinalResult())
        wf.close()
        
        transcript = final_result.get("text", "").strip()
        
        return JSONResponse({
            "transcript": transcript,
            "detected_language": detected_language,
            "status": "success" if transcript else "no_speech"
        })
    
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "Audio processing timeout"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        for path in [tmp_input_path, tmp_wav_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

# -----------------------------
#  TEXT TO SPEECH (EDGE-TTS)
# -----------------------------
@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """Convert text to speech using edge-tts (Render-compatible)."""
    try:
        # Detect language if auto
        if req.language_code == "auto":
            detected_lang = detect_language(req.text)
        else:
            detected_lang = req.language_code if req.language_code in ["en", "ar"] else "en"
        
        # Double-check for Arabic characters
        if re.search(r'[\u0600-\u06FF]', req.text):
            detected_lang = "ar"
        
        # Select voice (use custom voice if provided, otherwise use default)
        voice = req.voice if req.voice else EDGE_TTS_VOICES.get(detected_lang, EDGE_TTS_VOICES["en"])
        
        print(f"🔊 TTS Request - Language: {detected_lang}, Voice: {voice}")
        
        # Use BytesIO instead of temporary files (more reliable on Render)
        audio_buffer = io.BytesIO()
        
        try:
            # Generate speech directly to BytesIO
            communicate = edge_tts.Communicate(req.text.strip(), voice)
            
            # Write to BytesIO buffer
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])
            
            # Reset buffer position for reading
            audio_buffer.seek(0)
            
            # Check if we got any audio data
            if audio_buffer.getbuffer().nbytes == 0:
                raise Exception("No audio data generated")
            
            print(f"✅ TTS generated {audio_buffer.getbuffer().nbytes} bytes")
            
            return StreamingResponse(
                audio_buffer,
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": "inline; filename=speech.mp3",
                    "X-Language": detected_lang,
                    "X-Voice": voice,
                    "Cache-Control": "no-cache"
                }
            )
        
        except Exception as inner_e:
            print(f"❌ TTS generation error: {str(inner_e)}")
            raise
    
    except Exception as e:
        print(f"❌ TTS error: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "error": str(e),
            "type": "tts_error",
            "details": "Text-to-speech generation failed"
        }, status_code=500)
# -----------------------------
#  GET AVAILABLE VOICES
# -----------------------------
@app.get("/tts/voices")
async def get_available_voices():
    """Get list of available edge-tts voices."""
    try:
        voices = await edge_tts.list_voices()
        
        # Filter for medical-relevant voices (English and Arabic)
        medical_voices = {
            "english": [],
            "arabic": []
        }
        
        for voice in voices:
            locale = voice.get("Locale", "")
            name = voice.get("ShortName", "")
            gender = voice.get("Gender", "")
            
            if locale.startswith("en-"):
                medical_voices["english"].append({
                    "name": name,
                    "locale": locale,
                    "gender": gender,
                    "display_name": f"{voice.get('LocalName', name)} ({gender})"
                })
            elif locale.startswith("ar-"):
                medical_voices["arabic"].append({
                    "name": name,
                    "locale": locale,
                    "gender": gender,
                    "display_name": f"{voice.get('LocalName', name)} ({gender})"
                })
        
        return JSONResponse({
            "default_voices": EDGE_TTS_VOICES,
            "available_voices": medical_voices,
            "total_en": len(medical_voices["english"]),
            "total_ar": len(medical_voices["arabic"])
        })
    
    except Exception as e:
        print(f"❌ Error fetching voices: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

# -----------------------------
#  ROOT ENDPOINT
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("chatbot.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api-info")
def api_info():
    return {
        "message": "Dr. HealBot Stateless API with Edge-TTS",
        "version": "4.1",
        "features": [
            "Stateless chat (frontend manages history)",
            "Patient data loading from files",
            "Bilingual support (English/Arabic)",
            "Biomarker analysis integration",
            "Speech-to-text (Vosk)",
            "Text-to-speech (Edge-TTS with natural voices)"
        ],
        "endpoints": {
            "chat": "POST /chat - Send message with history",
            "patients": "GET /patients - List all patients",
            "patient": "GET /patient/{name} - Get patient data",
            "chat_history_get": "GET /chat-history/{name} - Get chat history",
            "chat_history_save": "POST /chat-history - Save chat history",
            "chat_history_delete": "DELETE /chat-history/{name} - Delete chat history",
            "stt": "POST /stt - Speech to text",
            "tts": "POST /tts - Text to speech (Edge-TTS)",
            "tts_voices": "GET /tts/voices - Get available voices"
        },
        "tts_info": {
            "provider": "Microsoft Edge-TTS",
            "default_voices": EDGE_TTS_VOICES,
            "features": [
                "High-quality neural voices",
                "Natural speech synthesis",
                "Multiple voice options",
                "Free to use"
            ]
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🏥 Dr. HealBot API with Edge-TTS Starting...")
    print("="*60)
    print(f"📁 Patient Data Folder: {PATIENT_DATA_FOLDER}")
    print(f"🔊 TTS Provider: Microsoft Edge-TTS")
    print(f"🗣️ Default English Voice: {EDGE_TTS_VOICES['en']}")
    print(f"🗣️ Default Arabic Voice: {EDGE_TTS_VOICES['ar']}")
    print(f"🎤 STT Available: {STT_AVAILABLE}")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8002)
