import os
import json
import re
from ollama import chat # modif 2
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import joblib
from super_tools import sensor_aliases

MEMORY_DIR = "user_memory"

class PromptProcessedMemory:
    def __init__(
        self,
        user_id: str,
        user_messages: List[Dict],
        bert_model_path: str,
        label_encoder_path: str,
    ):
        self.user_id = user_id
        self.user_messages = user_messages
        self.prompt_history: List[str] = []
        self.response_history: List[str] = []
        self.latest_prompt: Optional[str] = None

        self.last_logger: Optional[str] = None
        self.last_logger_list: List[str] = []
        self.last_logger_id: Optional[str] = None
        self.last_date: Optional[str] = None
        self.intent: Optional[str] = None
        self.analysis_result: Optional[str] = None
        self.last_data: Optional[str] = None  # ✅ Untuk analisa tanpa perlu fetch ulang

        self.prev_intent: Optional[str] = None
        self.prev_target: Optional[str] = None
        self.prev_date: Optional[str] = None

        # self.logger_list = logger_list  # ✅ Perubahan: Simpan logger_list sebagai atribut

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.bert_model = BertForSequenceClassification.from_pretrained(bert_model_path).to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
            self.label_encoder = joblib.load(label_encoder_path)
            self.bert_model.eval()
        except Exception as e:
            print(f"[MODEL LOAD ERROR] {e}")
            raise RuntimeError("Gagal memuat model BERT atau label encoder.")

        self._load_user_memory()
        self._split_prompt_response()
        self._extract_context_memory()

    def _get_memory_path(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        return os.path.join(MEMORY_DIR, f"{self.user_id}.json")

    def _load_user_memory(self):
        try:
            path = self._get_memory_path()
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.last_logger = data.get("last_logger")
                    self.last_logger_id = data.get("last_logger_id")
                    self.last_date = data.get("last_date")
        except Exception as e:
            print(f"[MEMORY LOAD ERROR] {e}")

    def _save_user_memory(self):
        try:
            data = {
                "last_logger": self.last_logger,
                "last_logger_id": self.last_logger_id,
                "last_date": self.last_date
            }
            with open(self._get_memory_path(), "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[MEMORY SAVE ERROR] {e}")

    def _split_prompt_response(self):
        for msg in self.user_messages:
            if msg.get("role") == "user":
                self.prompt_history.append(msg["content"])
            elif msg.get("role") == "assistant":
                self.response_history.append(msg["content"])
        if self.prompt_history:
            self.latest_prompt = self.prompt_history[-1]

    def get_context_window(self, window_size: int = 6) -> List[Dict[str, str]]:
        prompts = self.prompt_history[-window_size:]
        responses = self.response_history[-window_size:]
        context = []
        for u, a in zip(prompts, responses):
            context.append({"role": "user", "content": u})
            context.append({"role": "assistant", "content": a})
        return context

    def _predict_intent_bert(self, text: str) -> str:
        print("text dari _predict_intent_bert :", text)
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1)
        return self.label_encoder.inverse_transform(pred.cpu().numpy())[0]
    def _extract_context_memory(self, text: Optional[str] = None):

        def normalize_logger_name(name: str) -> str:
            """Normalisasi nama logger: lowercase, strip, dan pastikan prefix 'pos ' ada"""
            name = " ".join(name.strip().lower().split())
            if not name.startswith("pos "):
                name = "pos " + name
            return name

        # === Gabungkan teks prompt dan response
        combined_text = text.lower() if text else " ".join(self.prompt_history + self.response_history).lower()
        print("📥 combined_text:", combined_text)

        # === Ambil daftar logger valid dari fetch_list_logger
        try:
            logger_data = fetch_list_logger()
            logger_names_from_db = [normalize_logger_name(lg["nama_lokasi"]) for lg in logger_data if "nama_lokasi" in lg]
            normalized_valid_loggers = set(logger_names_from_db)
            print(f"✅ {len(normalized_valid_loggers)} logger valid dimuat.")
        except Exception as e:
            print("❌ [ERROR] fetch_list_logger gagal:", e)
            normalized_valid_loggers = set()

        # === Regex deteksi kasar logger
        logger_pattern = r"\b(?:pos|afmr|awlr|awr|arr|adr|awqr|avwr|awgc)\s+(?:[a-z]{3,}(?:\s+[a-z]{3,}){0,3})"
        raw_matches = re.findall(logger_pattern, combined_text)
        print("🔍 logger_match (raw):", raw_matches)

        # === Pisahkan gabungan seperti 'dan', lalu validasi
        expanded_matches = self._clean_logger_list(raw_matches)
        print("🧩 expanded_matches:", expanded_matches)

        # Daftar kata yang bukan bagian dari nama logger
        stop_words = {"kemarin", "selama","hari", "ini", "lalu", "terakhir", "minggu", "bulan", "tahun", "tanggal"}

        cleaned_matches = set()
        for match in expanded_matches:
            # Hapus kata tidak relevan dari match
            filtered = " ".join(word for word in match.split() if word.lower() not in stop_words)
            norm = normalize_logger_name(filtered)
            if norm in normalized_valid_loggers:
                cleaned_matches.add(norm)
            else:
                print(f"⚠️ Tidak valid: {norm}")

        if cleaned_matches:
            self.last_logger_list = list(cleaned_matches)
            self.last_logger = self.last_logger_list[-1]
            print("📌 last_logger_list:", self.last_logger_list)
            print("📌 last_logger (terakhir):", self.last_logger)
        else:
            print("🚫 Tidak ada logger valid ditemukan.")

        # === Ekstraksi tanggal
        date_keywords = [
            "hari ini", "kemarin", "kemaren", "minggu ini", "minggu lalu", "bulan lalu",
            "awal bulan", "akhir bulan", "tahun lalu", "minggu terakhir", "bulan terakhir", "hari terakhir"
        ]
        for phrase in date_keywords:
            if phrase in combined_text:
                self.last_date = phrase
                print("🗓️ Deteksi tanggal (keyword):", phrase)
                break

        relative_date_patterns = [
            r"\d+\s+hari\s+(lalu|terakhir)",
            r"\d+\s+minggu\s+(lalu|terakhir)",
            r"\d+\s+bulan\s+(lalu|terakhir)"
        ]
        for pattern in relative_date_patterns:
            match = re.search(pattern, combined_text)
            if match:
                self.last_date = match.group(0)
                print("🗓️ Deteksi tanggal (relatif):", self.last_date)
                break

    # def _extract_context_memory(self, text: Optional[str] = None):
    #     def normalize_logger_name(name: str) -> str:
    #         """Normalisasi nama logger: lowercase, strip, dan pastikan prefix 'pos ' ada"""
    #         name = " ".join(name.strip().lower().split())
    #         if not name.startswith("pos "):
    #             name = "pos " + name
    #         return name

    #     # === Gabungkan teks history atau pakai yang diberikan
    #     combined_text = text.lower() if text else " ".join(self.prompt_history + self.response_history).lower()
    #     print("combined_text adalah :", combined_text)

    #     # === Ambil daftar logger dari fetch_list_logger
    #     try:
    #         logger_data = fetch_list_logger()
    #         logger_names_from_db = [normalize_logger_name(lg["nama_lokasi"]) for lg in logger_data if "nama_lokasi" in lg]
    #         normalized_valid_loggers = set(logger_names_from_db)
    #     except Exception as e:
    #         print("[ERROR] Gagal mengambil daftar logger dari fetch_list_logger()", e)
    #         normalized_valid_loggers = set()

    #     # === Regex kasar deteksi logger
    #     logger_pattern = r"\b(?:pos|afmr|awlr|awr|arr|adr|awqr|avwr|awgc)\s+(?:[a-z]{3,}(?:\s+[a-z]{3,}){0,3})"
    #     raw_matches = re.findall(logger_pattern, combined_text)
    #     print("logger_match (raw):", raw_matches)

    #     # === Pisahkan gabungan 'dan', lalu validasi terhadap database
    #     expanded_matches = self._clean_logger_list(raw_matches)
    #     print("expanded_matches:", expanded_matches)

    #     cleaned_matches = set()
    #     for match in expanded_matches:
    #         norm = normalize_logger_name(match)
    #         if norm in normalized_valid_loggers:
    #             cleaned_matches.add(norm)

    #     if cleaned_matches:
    #         self.last_logger_list = list(cleaned_matches)
    #         self.last_logger = self.last_logger_list[-1]
    #         print("last_logger_list setelah extract:", self.last_logger_list)
    #         print("last_logger_list:", self.last_logger_list)
    #         print("last_logger:", self.last_logger)

    #     # === Ekstraksi tanggal eksplisit atau relatif
    #     date_keywords = [
    #         "hari ini", "kemarin", "kemaren", "minggu ini", "minggu lalu", "bulan lalu",
    #         "awal bulan", "akhir bulan", "tahun lalu", "minggu terakhir", "bulan terakhir", "hari terakhir"
    #     ]
    #     for phrase in date_keywords:
    #         if phrase in combined_text:
    #             self.last_date = phrase
    #             break

    #     relative_date_patterns = [
    #         r"\d+\s+hari\s+(lalu|terakhir)",
    #         r"\d+\s+minggu\s+(lalu|terakhir)",
    #         r"\d+\s+bulan\s+(lalu|terakhir)"
    #     ]
    #     for pattern in relative_date_patterns:
    #         match = re.search(pattern, combined_text)
    #         if match:
    #             self.last_date = match.group(0)
    #             break


    # def _extract_context_memory(self, text: Optional[str] = None):

    #     def normalize_logger_name(name: str) -> str:
    #         return " ".join(name.strip().lower().split())
        
    #     combined_text = text.lower() if text else " ".join(self.prompt_history + self.response_history).lower()
    #     print("combined_text adalah :", combined_text)

    #     # === Ambil daftar nama lokasi dari fetch_list_logger
    #     try:
    #         logger_data = fetch_list_logger()
    #         logger_names_from_db = [normalize_logger_name(lg["nama_lokasi"]) for lg in logger_data if "nama_lokasi" in lg]
    #         normalized_valid_loggers = set(logger_names_from_db)
    #     except Exception as e:
    #         print("[ERROR] Gagal mengambil daftar logger dari fetch_list_logger()", e)
    #         normalized_valid_loggers = set()

    #     # === Regex deteksi logger
    #     logger_pattern = r"\b(?:pos|afmr|awlr|awr|arr|adr|awqr|avwr|awgc)\s+(?:[a-z]{3,}\s*){1,4}"
    #     raw_matches = re.findall(logger_pattern, combined_text)
    #     print("logger_match (raw)", raw_matches)

    #     # === Bersihkan dan validasi
    #     cleaned_matches = set()
    #     for match in raw_matches:
    #         norm = normalize_logger_name(match)
    #         if norm in normalized_valid_loggers:
    #             cleaned_matches.add(norm)

    #     if cleaned_matches:
    #         self.last_logger_list = list(cleaned_matches)
    #         self.last_logger = self.last_logger_list[-1]

    #     # === Ekstraksi tanggal
    #     date_keywords = [
    #         "hari ini", "kemarin", "kemaren", "minggu ini", "minggu lalu", "bulan lalu",
    #         "awal bulan", "akhir bulan", "tahun lalu", "minggu terakhir", "bulan terakhir", "hari terakhir"
    #     ]
    #     for phrase in date_keywords:
    #         if phrase in combined_text:
    #             self.last_date = phrase
    #             break

    #     relative_date_patterns = [
    #         r"\d+\s+hari\s+(lalu|terakhir)",
    #         r"\d+\s+minggu\s+(lalu|terakhir)",
    #         r"\d+\s+bulan\s+(lalu|terakhir)"
    #     ]
    #     for pattern in relative_date_patterns:
    #         match = re.search(pattern, combined_text)
    #         if match:
    #             self.last_date = match.group(0)
    #             break

    def _should_use_memory(self, prompt: str) -> bool:
        prompt = prompt.lower()
        ambiguous_terms = [
            "data di atas", "kesimpulan", "bagaimana tadi", "tadi", "lanjutkan",
            "jelaskan", "makna", "apa artinya", "pos tersebut", "lokasi tersebut"
        ]
        return any(term in prompt for term in ambiguous_terms)

    def _clean_logger_list(self, raw_logger_list: List[str]) -> List[str]:
        cleaned = []
        for entry in raw_logger_list:
            if " dan " in entry:
                parts = entry.split(" dan ")
                cleaned.extend([part.strip() for part in parts])
            else:
                cleaned.append(entry.strip())
        return cleaned

    def update_last_logger(self, logger_name: str, logger_id: str):
        self.last_logger = logger_name
        self.last_logger_id = logger_id
        self._save_user_memory()

    def update_last_date(self, date_text: str):
        self.last_date = date_text
        self._save_user_memory()

    def process_new_prompt(self, new_prompt: str) -> Dict:
        print(f"new_prompt : {new_prompt}")

        is_same_prompt = self.latest_prompt == new_prompt
        self.latest_prompt = new_prompt
        self.prompt_history.append(new_prompt)

        # ✅ Reset date dulu setiap prompt baru
        self.last_date = None

        # ✅ Prediksi intent baru
        try:
            self.intent = self._predict_intent_bert(new_prompt)
        except Exception as e:
            print(f"[INTENT PREDICTION ERROR] {e}")
            self.intent = "unknown_intent"

        # if self._should_use_memory(new_prompt): # change from this 
        #     self._extract_context_memory()
        # else:
        #     self._extract_context_memory(text=new_prompt)

        # ✅ Deteksi apakah harus pakai context memory sebelumnya to this line of code
        print("new_prompt di function process_new_prompt", new_prompt) 
        if self._should_use_memory(new_prompt) and not self._contains_explicit_logger_or_date(new_prompt):
            context_window = self.get_context_window(window_size=6)
            self._extract_context_memory(text=" ".join([m["content"] for m in context_window if m["role"] == "user"]))
        else:
            self._extract_context_memory(text=new_prompt)

        # ✅ Ambil target
        if self.intent in ["fetch_logger_by_date", "show_logger_data", "analyze_logger_by_date", "show_selected_parameter", "compare_parameter_across_loggers", "compare_logger_data", "compare_logger_by_date"]:
            print(f"self.intent == {self.intent}")
            raw_targets = self.last_logger_list if self.last_logger_list else ([self.last_logger] if self.last_logger else [])
            
            print("raw_targets :", raw_targets)

            target = self._clean_logger_list(raw_targets)
        else:
            target = self.last_logger

        # if self.intent == "show_logger_data":
        #     raw_targets = self.last_logger_list if self.last_logger_list else ([self.last_logger] if self.last_logger else [])
        #     target = self._clean_logger_list(raw_targets)
        # else:
        #     target = self.last_logger
        # ✅ Return dan cache
        print("\n")
        print("Dari function process_new_prompt untuk deteksi intent") 
        print("=================")
        print(f"intent: {self.intent}, target: {target},date: {self.last_date}") # intent: compare_logger_by_date, target: None,date: kemarin
        print("\n")
        return {
            "intent": self.intent,
            "target": target,
            "date": self.last_date,
            "latest_prompt": new_prompt
        }

    def _contains_explicit_logger_or_date(self, text: str) -> bool:
        text = text.lower()
        logger_found = bool(re.search(r"\b(pos|awlr|awr|arr|adr)\s+[a-z]+", text))
        date_keywords = ["hari ini", "kemarin", "tanggal", "minggu lalu", "bulan lalu"]
        return logger_found or any(keyword in text for keyword in date_keywords)

    # def process_new_prompt(self, new_prompt: str) -> Dict:
    #     print(f"new_prompt : {new_prompt}")

    #     # === Simpan sebelumnya untuk perbandingan ===
    #     prev_intent = self.intent
    #     prev_target = self.last_logger_list or self.last_logger
    #     prev_date = self.last_date

    #     self.last_date = None  # Reset date setiap prompt baru

    #     try:
    #         self.intent = self._predict_intent_bert(new_prompt)
    #     except Exception as e:
    #         print(f"[INTENT PREDICTION ERROR] {e}")
    #         self.intent = "unknown_intent"

    #     self.latest_prompt = new_prompt
    #     self.prompt_history.append(new_prompt)

    #     if self._should_use_memory(new_prompt):
    #         context_window = self.get_context_window(window_size=6)
    #         self._extract_context_memory(text=" ".join([m["content"] for m in context_window if m["role"] == "user"]))
    #     else:
    #         self._extract_context_memory(text=new_prompt)

    #     if self.intent in ["fetch_logger_by_date", "analyze_logger_by_date", "compare_logger_by_date"]:
    #         self.last_date = self.last_date  # akan ditangkap dari _extract_context_memory
    #     else:
    #         self.last_date = None

    #     if self.intent == "show_logger_data":
    #         print("self.last_logger_list : ",self.last_logger_list)
            
    #         raw_targets = self.last_logger_list if self.last_logger_list else ([self.last_logger] if self.last_logger else [])
    #         print("Get raw_targets : ",raw_targets)
    #         target = self._clean_logger_list(raw_targets)
            
    #     else:
    #         target = self.last_logger

    #     # === Bandingkan hasil intent sebelumnya ===
    #     if (
    #         prev_intent == self.intent and
    #         prev_target == (self.last_logger_list or self.last_logger) and
    #         prev_date == self.last_date
    #     ):
    #         print("[INFO] Intent sama dengan sebelumnya, kembalikan hasil dari cache.")
    #         return {
    #             "intent": self.intent,
    #             "target": self.last_logger_list or self.last_logger,
    #             "date": self.last_date,
    #             "latest_prompt": new_prompt
    #         }

    #     # === Simpan hasil intent untuk perbandingan selanjutnya ===
    #     self.prev_intent = self.intent
    #     self.prev_target = self.last_logger_list or self.last_logger
    #     self.prev_date = self.last_date

    #     return {
    #         "intent": self.intent,
    #         "target": target,
    #         "date": self.last_date,
    #         "latest_prompt": new_prompt
    #     }

import warnings
warnings.filterwarnings('ignore')
import torch
import re
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
from super_tools import original_fetch_data_range, find_and_fetch_latest_data, fetch_list_logger_from_prompt_flexibleV1, original_fetch_status_logger, fetch_list_logger, original_compare_by_date, extract_date_structured, general_stesy, summarize_logger_data, extreme_keywords, is_extreme_only

class PromptValidator:
    INTENT_REQUIREMENTS = {
        "show_logger_data": {"target": True, "date": False},
        "fetch_logger_by_date": {"target": True, "date": True},
        "compare_logger_by_date": {"target": True, "date": True},
        "compare_logger_data": {"target": True, "date": False},
        "analyze_logger_by_date": {"target": True, "date": True},
        "fetch_status_rain": {"target": False, "date": False},
        "show_list_logger": {"target": False, "date": False}, 
        "ai_limitation": {"target": False, "date": False}
    }

    def __init__(self, prompt: str, predicted_intent: str, target: list = None, date: str = None):
        self.prompt = prompt.lower()
        self.predicted_intent = predicted_intent
        self.target = target or []
        self.date = date

    def is_ambiguous_prompt(self) -> bool:
        ambiguous_phrases = [
            "lanjutkan", "iya", "ya", "bagaimana tadi", "data di atas",
            "tolong", "oke", "lihat semua", "teruskan", "apa itu", "jelaskan"
        ]

        print("[DEBUG] Checking ambiguous prompt:", self.prompt)
        for phrase in ambiguous_phrases:
            if re.search(rf"\\b{re.escape(phrase)}\\b", self.prompt):
                print(f"[DEBUG] Found ambiguous phrase: '{phrase}'")
                return True

        return len(self.prompt.strip().split()) < 3


    def is_intent_mismatch(self) -> bool:
        if ("tampilkan" in self.prompt or "lihat" in self.prompt) and "data" in self.prompt:
            if any(keyword in self.prompt for keyword in ["awr", "arr", "logger", "pos"]):
                if self.predicted_intent not in ["show_logger_data"]:
                    return True
        if "tidak hujan" in self.prompt or "sedang hujan" in self.prompt:
            if self.predicted_intent != "fetch_status_rain":
                return True
        return False

    def is_missing_required_fields(self) -> bool:
        reqs = self.INTENT_REQUIREMENTS.get(self.predicted_intent, {})
        if reqs.get("target") and not self.target:
            return True
        if reqs.get("date") and self.date is None:
            return True
        return False

    def should_fallback_to_ai(self) -> bool:
        return self.is_ambiguous_prompt() or self.is_intent_mismatch() or self.is_missing_required_fields()

# === Contoh integrasi dalam PromptProcessedMemory ===
# validator = PromptValidator(new_prompt, self.intent)
# if validator.should_fallback_to_ai():
#     self.intent = "ai_limitation"  # biar dijawab oleh smart_respond()

class IntentHandler:
    def __init__(self, model_path, tokenizer_path, label_encoder_path, max_length: int = 128):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.max_length = max_length
        self.model.eval()

    def predict_intent(self, prompt: str) -> str:
        print("prompt dari predict_intent",prompt)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        prediction = outputs.logits.argmax(dim=1).item()
        return self.label_encoder.inverse_transform([prediction])[0]

class IntentManager:
    def __init__(self, memory):
        self.memory = memory
        self.intent_map = {
            "show_logger_data": self.fetch_latest_data,
            "fetch_logger_by_date": self.fetch_data_range,
            "fetch_status_rain": self.fetch_status_logger,
            "show_list_logger": self.show_list_logger,
            "compare_logger_by_date": self.compare_by_date,
            "compare_logger_data": self.compare_latest_data,
            "compare_parameter_across_loggers": self.compare_across_loggers,
            "show_selected_parameter": self.show_selected_parameter,
            "get_logger_photo_path": self.get_photo_path,
            "how_it_works": self.explain_system,
            "analyze_logger_by_date": self.analyze_by_date,
            "ai_limitation": self.ai_limitation,
            "show_online_logger" : self.connection_status
        }

    # def handle_intent(self):
    #     prompt = self.memory.latest_prompt
    #     intent = self.memory.intent
    #     target = self.memory.last_logger_list or [self.memory.last_logger]
    #     date = self.memory.last_date

    #     # ✅ Kirim parameter lengkap ke validator
    #     print(f"Dari Prompt {prompt} Intent adalah : {intent}, target logger adalah : {target}, tanggal yang dicari adalah : {target}")
        
    #     validator = PromptValidator(prompt, intent, target, date)
    #     print("validator :",validator.should_fallback_to_ai())
    #     if validator.should_fallback_to_ai():
    #         print("[INFO] Prompt ambigu atau intent tidak cocok, gunakan smart_respond()")
    #         return self.smart_respond()

    #     func = self.intent_map.get(intent, self.fallback_response)
    #     return func()
    
    def handle_intent(self):
        prompt = self.memory.latest_prompt
        intent = self.memory.intent
        target = self.memory.last_logger_list or [self.memory.last_logger]
        date = self.memory.last_date

        print(f"Dari Prompt {prompt} Intent adalah : {intent}, target logger adalah : {target}, tanggal yang dicari adalah : {date}")
        func = self.intent_map.get(intent, self.fallback_response)
        return func()
    
    def fetch_latest_data(self):
        print("show_logger_data ini telah berjalan")
        prompt = self.memory.latest_prompt.lower()
        data_last = self.memory.last_data

        print(f"Data Terakhir adalah : {data_last}")
        print("self.memory.last_logger_list ", self.memory.last_logger_list)
        print(f"Type data dari self.memory.last_logger_list adalah {type(self.memory.last_logger_list)}")
        print(f"Panjang data dari self.memory.last_logger_list adalah {len(self.memory.last_logger_list)}")
        print("self.memory.last_logger ", self.memory.last_logger)

        target_loggers = self.memory.last_logger_list or [self.memory.last_logger]
        logger_list = fetch_list_logger()

        if not target_loggers or not logger_list:
            return "Target logger atau daftar logger tidak tersedia."

        # === DETEKSI PARAMETER YANG DIMINTA DI DALAM PROMPT ===
        matched_parameters = []
        for param, aliases in sensor_aliases.items():
            for alias in aliases:
                if alias in prompt:
                    matched_parameters.append(param)
                    break

        print("Matched Parameters:", matched_parameters)
        print("Matched Parameters Type:", type(matched_parameters))
        print("Matched Parameters Length:", len(matched_parameters))

        # === FETCH DATA TERBARU ===
        fetched = find_and_fetch_latest_data(target_loggers, matched_parameters, logger_list)
        print("fetched data adalah :", fetched)

        summaries = []
        for item in fetched:
            nama_lokasi = item['logger_name']
            data = item['data']
            summary = summarize_logger_data(nama_lokasi, data)
            summaries.append(summary)

        return "\n\n---\n\n".join(summaries)

    # def fetch_latest_data(self):
    #     print("show_logger_data ini telah berjalan")
    #     prompt = self.memory.latest_prompt
    #     data_last = self.memory.last_data

    #     print(f"Data Terakhir adalah : {data_last}")
    #     print("self.memory.last_logger_list ",self.memory.last_logger_list)
    #     print(f"Type data dari self.memory.last_logger_list adalah {type(self.memory.last_logger_list)}")
    #     print(f"Panjang data dari self.memory.last_logger_list adalah {len(self.memory.last_logger_list)}")
    #     print("self.memory.last_logger ",self.memory.last_logger)

    #     target_loggers = self.memory.last_logger_list or [self.memory.last_logger] # Error no
    #     logger_list = fetch_list_logger()

    #     if not target_loggers or not logger_list:
    #         return "Target logger atau daftar logger tidak tersedia."

    #     fetched = find_and_fetch_latest_data(target_loggers, logger_list)
    #     print("fetched data adalah :",fetched)

    #     summaries = []
    #     for item in fetched:
    #         nama_lokasi = item['logger_name']
    #         data = item['data']
    #         summary = summarize_logger_data(nama_lokasi, data)
    #         summaries.append(summary)

    #     return "\n\n---\n\n".join(summaries)

        # if not fetched:
        #     return "Tidak ditemukan data untuk logger yang disebutkan."

        # for item in fetched:
        #     print(f"\n📍 {item['logger_name']}")
        #     for key, value in item['data'].items():
        #         print(f"{key}: {value}")

        # return f"Berhasil mengambil data terbaru dari {len(fetched)} logger."


    def fetch_data_range(self):
        print("fetch_logger_by_date ini telah berjalan")
        prompt = self.memory.latest_prompt
        print("prompt", prompt)

        # === Ambil target logger
        target_loggers = self.memory.last_logger_list or [self.memory.last_logger]
        print("target_loggers", target_loggers)
        logger_list = fetch_list_logger()

        # === Ekstraksi tanggal dari prompt
        date_info = extract_date_structured(prompt)
        print("date_info", date_info)

        # === Validasi: Tidak boleh lebih dari 2 minggu lalu
        # if date_info["awal_tanggal"]:
        #     start_date = datetime.strptime(date_info["awal_tanggal"], "%Y-%m-%d")
        #     today = datetime.today()
        #     max_allowed_start = today - timedelta(days=14)

        #     if start_date < max_allowed_start:
        #         return (
        #             "⚠️ Permintaan data terlalu lama. "
        #             "Sistem hanya mengizinkan pengambilan data maksimal 2 minggu ke belakang dari hari ini.\n"
        #             f"Silakan ubah rentang tanggal menjadi setelah {max_allowed_start.strftime('%Y-%m-%d')}."
        #         )

         # === Deteksi parameter dari prompt (sensor yang ingin ditampilkan)
        matched_parameters = []
        for param, aliases in sensor_aliases.items():
            for alias in aliases:
                if alias in prompt:
                    matched_parameters.append(param)
                    break
        print("matched_parameters:", matched_parameters)

        # === Teruskan jika valid
        return original_fetch_data_range(
            prompt=prompt,
            target_loggers=target_loggers,
            matched_parameters=matched_parameters,  # <-- tambahkan ini ke original_fetch_data_range
            logger_list=logger_list
        )

    def fetch_status_logger(self):
        data_last = self.memory.last_data

        print(f"Data Terakhir adalah : {data_last}")
        return original_fetch_status_logger(prompt=self.memory.latest_prompt)

    def show_list_logger(self):
        return fetch_list_logger_from_prompt_flexibleV1(self.memory.latest_prompt)

    def compare_by_date(self):
        print("compare_logger_by_date ini telah berjalan")
        prompt = self.memory.latest_prompt
        target_loggers = self.memory.last_logger_list or [self.memory.last_logger]
        logger_list = fetch_list_logger()
         # === Deteksi parameter dari prompt (sensor yang ingin ditampilkan)
        matched_parameters = []
        for param, aliases in sensor_aliases.items():
            for alias in aliases:
                if alias in prompt:
                    matched_parameters.append(param)
                    break
        print("matched_parameters:", matched_parameters)

        return original_compare_by_date(
            prompt=prompt,
            target_loggers=target_loggers,
            matched_parameters = matched_parameters,
            logger_list=logger_list
        )

    def compare_latest_data(self):
        prompt = self.memory.latest_prompt
        target_loggers = self.memory.last_logger_list or [self.memory.last_logger]
        logger_list = fetch_list_logger()
        matched_parameters = []
        fetched = find_and_fetch_latest_data(target_loggers, matched_parameters, logger_list)
        # 🔥 Gunakan summary sebagai tampilan perbandingan data
        summaries = []
        for item in fetched:
            nama_lokasi = item['logger_name']
            data = item['data']
            summary = summarize_logger_data(nama_lokasi, data)
            summaries.append(summary)

        return "\n\n---\n\n".join(summaries)
        # for item in fetched:
        #     print(f"\n📍 {item['logger_name']}")
        #     for key, value in item['data'].items():
        #         print(f"{key}: {value}")

        # return f"Berhasil mengambil data dari {len(fetched)} logger."

    def compare_across_loggers(self):
        print("compare_across_loggers ini telah berjalan")
        prompt = self.memory.latest_prompt.lower()
        logger_list = fetch_list_logger()

        if not logger_list:
            return "Daftar logger tidak tersedia."

        # Cek apakah hanya ada nilai ekstrem tanpa sensor alias
        if is_extreme_only(prompt, sensor_aliases):
            system_prompt = (
                "Anda adalah asisten yang membantu pengguna menjelaskan parameter sensor yang mereka maksud "
                "ketika mereka menyebutkan istilah ekstrem seperti 'terpanas', 'terdingin', 'paling lembap', dsb.\n\n"
                "Tugas Anda adalah menebak parameter yang paling relevan berdasarkan kata ekstrem tersebut, "
                "lalu meminta konfirmasi dari pengguna. Misalnya:\n"
                "- Jika pengguna berkata 'pos terpanas', Anda bisa bertanya 'Apakah maksud Anda suhu udara?'\n"
                "- Jika pengguna berkata 'paling lembap', Anda bisa bertanya 'Apakah maksud Anda kelembaban udara?'\n"
                "- Jika tidak bisa dipastikan, minta pengguna memilih salah satu dari daftar sensor yang tersedia."
            )
            response = chat(
                model='llama3.1:8b',
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['message']['content']

        # Cek parameter berdasarkan sensor_aliases
        selected_param = None
        for param_key, aliases in sensor_aliases.items():
            for alias in aliases:
                if alias.lower() in prompt:
                    selected_param = param_key
                    break
            if selected_param:
                break

        if not selected_param:
            return (
                "Parameter yang ingin dibandingkan tidak dikenali. "
                "Silakan sebutkan seperti suhu udara, kelembaban udara, tekanan udara, atau curah hujan."
            )

        # Ambil data logger dengan filtered matched_parameters [selected_param]
        name_fragments = [logger["nama_lokasi"] for logger in logger_list]
        matched_parameters = [selected_param]  # harus dalam list untuk filter
        fetched = find_and_fetch_latest_data(name_fragments, matched_parameters, logger_list)

        if not fetched:
            return "Tidak ditemukan data terbaru untuk logger yang disebutkan." 

        # Siapkan data untuk konteks
        comparison_data = []
        for item in fetched:
            logger_name = item["logger_name"]
            value = item["data"].get(selected_param, "Data tidak tersedia")
            comparison_data.append(f"{logger_name}: {value}")

        # Prompt sistem untuk kasus normal dan ekstrem + sensor
        system_prompt = (
            "Kamu adalah asisten cerdas yang menjawab pertanyaan berdasarkan data logger cuaca. "
            "Jika pengguna menanyakan yang berkaitan dengan kondisi ekstrem seperti 'terdingin', 'paling panas', 'terbasah', 'paling', 'tertinggi', atau 'terendah', "
            "berikan jawaban hanya untuk satu pos dengan nilai ekstrem tersebut beserta nama pos dan nilai parameternya. "
            "Untuk permintaan lain, berikan jawaban yang jelas dan sesuai dengan semua data pos yang relevan. "
            "Catatan penting: Sistem hanya dapat menampilkan data untuk hari ini, data terbaru, atau dalam rentang 24 jam terakhir. "
            "Data untuk waktu tertentu seperti 'kemarin', 'minggu lalu', atau rentang waktu spesifik tidak tersedia dan tidak dapat ditampilkan."
        )

        context_data = "\n".join(comparison_data)

        response = chat(
            model='llama3.1:8b',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{self.memory.latest_prompt}\n\nBerikut data yang tersedia:\n{context_data}"}
            ]
        )

        return response['message']['content']

    def show_selected_parameter(self):
        print("show_selected_parameter ini telah berjalan")
        prompt = self.memory.latest_prompt.lower()
        logger_list = fetch_list_logger()

        if not logger_list:
            return "Daftar logger tidak tersedia."

        # Deteksi parameter dari sensor_aliases → matched_parameters (bisa lebih dari satu jika perlu)
        matched_parameters = []
        for param_key, aliases in sensor_aliases.items():
            for alias in aliases:
                if alias.lower() in prompt:
                    matched_parameters.append(param_key)
                    break  # Hanya ambil satu alias per param_key

        if not matched_parameters:
            return "Parameter tidak dikenali. Silakan sebutkan suhu udara, kelembaban udara, curah hujan, atau tekanan udara."

        # Deteksi rentang tanggal
        date_info = extract_date_structured(prompt)
        print("Extracted date_info:", date_info)

        # Deteksi lokasi (logger)
        target_loggers = self.memory.last_logger_list or [self.memory.last_logger]
        print("Target logger:", target_loggers)

        # Jika tidak ada tanggal, ambil data terkini dari logger
        if not date_info.get("awal_tanggal") or not date_info.get("akhir_tanggal"):
            fetched = find_and_fetch_latest_data(
                name_list=target_loggers,
                matched_parameters=matched_parameters,
                logger_list=logger_list
            )

            summaries = []
            for item in fetched:
                logger_name = item['logger_name']
                for param in matched_parameters:
                    value = item['data'].get(param, "Data tidak tersedia")
                    summaries.append(f"{logger_name}: {param} = {value}")

            # Gabungkan jadi konteks untuk LLaMA
            context_data = "\n".join(summaries)

            system_prompt = (
                "Kamu adalah asisten cerdas yang menjawab pertanyaan berdasarkan data logger. "
                "Berikan jawaban yang jelas dan sesuai dengan permintaan pengguna seperti nama pos dan parameter yang diminta."
            )

            response = chat(
                model='llama3.1:8b',
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{self.memory.latest_prompt}\n\nBerikut data yang tersedia:\n{context_data}"}
                ]
            )
            return response['message']['content']

        # Jika ada rentang tanggal, gunakan original_fetch_data_range
        summaries = original_fetch_data_range(
            prompt=self.memory.latest_prompt,
            target_loggers=target_loggers,
            logger_list=logger_list
        )

        system_prompt = (
            "Kamu adalah asisten cerdas yang menjawab pertanyaan berdasarkan data logger. "
            "Berikan jawaban yang jelas dan sesuai dengan permintaan pengguna seperti nama pos dan parameter yang diminta."
        )
        response = chat(
            model='llama3.1:8b',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{self.memory.latest_prompt}\n\nBerikut data yang tersedia:\n{summaries}"}
            ]
        )
        return response['message']['content']


    def get_photo_path(self):
        return "Menjalankan get_photo_path"

    def explain_system(self):
        return "Menjalankan explain_system"

    def analyze_by_date(self):
        print("analyze_logger_by_date ini telah berjalan")

        prompt = self.memory.latest_prompt
        target_loggers = self.memory.last_logger_list or [self.memory.last_logger]
        logger_list = fetch_list_logger()

        print("prompt :", prompt)
        print("target loggers HEHE:", target_loggers)
        print("self.memory.last_data :", self.memory.last_data)

        if self.memory.analysis_result:
            print("[INFO] Menggunakan hasil analisis dari memory")
            return self.memory.analysis_result

        if self.memory.last_data:
            print("[INFO] Menggunakan last_data dari memory")

            fetched_data = self.memory.last_data
        else:
            print("[INFO] Tidak ditemukan last_data, mencoba fetch ulang")
            fetched_data = self.fetch_data_range()
            print("fetched_data adalah 1 : ",fetched_data)
            if isinstance(fetched_data, str):
                print("[ERROR] Gagal fetch ulang, response:", fetched_data)
                return fetched_data  # e.g. "Tanggal tidak dikenali..."

            self.memory.last_data = fetched_data
            self.memory._save_user_memory()  # ✅ Tambahan di sini
            print("[INFO] Data berhasil di-fetch dan disimpan ke memory")

        # summary_prompt = {
        #     "role": "user",
        #     "content": f"Berikan analisis data logger berikut:\n\n{fetched_data}"
        # }
        print("fetched_data untuk summary_prompt: ",fetched_data)

        summary_prompt = {
            "role": "user",
            "content": (
                "Berikan kesimpulan dalam satu paragraf dari data logger berikut. "
                "Analisis tren parameter yang tersedia (jika terlihat), sebutkan nilai tertinggi dan terendah, dan "
                "jelaskan apakah nilainya termasuk kategori rendah, sedang, atau tinggi berdasarkan konteks umum. "
                "Gunakan gaya bahasa informatif dan ringkas.\n\n"
                f"{fetched_data}"
            )
        }
        print("Summary Adalah:", summary_prompt)
        result = general_stesy(messages=[summary_prompt])
        self.memory.analysis_result = result
        self.memory._save_user_memory()

        return result

    def connection_status(self):
        print("self.memory.latest_prompt",self.memory.latest_prompt)
        prompt = self.memory.latest_prompt
        print("show_online_logger telah berjalan")
        list_connection = fetch_list_logger()
        
        koneksi_aliases = sensor_aliases.get("Koneksi", [])
        prompt_lower = prompt.lower()

        # Deteksi apakah prompt mengandung kata terkait koneksi
        if any(alias in prompt_lower for alias in koneksi_aliases):
            # Kelompokkan kata kunci untuk status
            keywords_aktif = ["terhubung", "aktif", "online", "connect", "tersambung", "hidup"]
            keywords_nonaktif = ["terputus", "tidak terhubung", "offline", "disconnect", "mati"]

            # Cek maksud pengguna
            if any(k in prompt_lower for k in keywords_nonaktif):
                target_status = "Terputus"
            elif any(k in prompt_lower for k in keywords_aktif):
                target_status = "Terhubung"
            else:
                return (
                    "Prompt Anda menyebutkan status koneksi, namun tidak jelas apakah ingin menampilkan logger yang terhubung atau terputus.\n"
                    "Contoh:\n- 'Berikan pos yang tidak aktif'\n- 'Tampilkan logger yang online'"
                )

            # Filter logger berdasarkan status koneksi
            filtered = [l for l in list_connection if l["koneksi"].lower() == target_status.lower()]
            if not filtered:
                return f"Tidak ditemukan logger dengan status koneksi: *{target_status}*"

            hasil = f"### Daftar Pos dengan koneksi *{target_status}*:\n"
            for logger in filtered:
                hasil += f"- *{logger['nama_lokasi']}* di {logger['alamat']} ({logger['kabupaten']})\n"
            return hasil
        else:
            return "Prompt tidak mengandung kata kunci koneksi seperti aktif, offline, signal, dll."

    def ai_limitation(self):
        prompt = self.memory.latest_prompt
        model_name = "llama3.1:8b"
        print("Intent deteksi limitation berjalan")

        messages_llm = [
            {
                "role": "system",
                "content": (
                    "Anda adalah AI virtual assistant Smart Telemetry Systems (STESY) yang hanya menjawab pertanyaan dalam konteks khusus berikut:.\n"
                    "- Telemetri\n"
                    "- Hidrologi\n"
                    "- Sungai\n"
                    "- Cuaca\n"
                    "- Klimatologi\n"
                    "- Analisis data logger\n\n"
                    "Jika pertanyaan user sesuai konteks di atas, berikan jawaban secara informatif, jelas, dan dalam format markdown jika relevan.\n\n"
                    "Namun jika pertanyaan user berada di luar topik (misalnya tentang sejarah, teknologi umum, hiburan, atau tidak ada hubungannya dengan sistem telemetri), "
                    "**tolak dengan sopan boleh tambahkan emoticon**:\n"
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = chat(model=model_name, messages=messages_llm)
        final_content = response["message"]["content"]

        return final_content

    def fallback_response(self):
        print("Fallback intent dijalankan")
        return "Intent tidak dikenali"
