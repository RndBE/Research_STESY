import os
import json
import re
from ollama import chat # modif 2
from typing import List, Dict, Optional
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

        self.prev_intent: Optional[str] = None
        self.prev_target: Optional[str] = None
        self.prev_date: Optional[str] = None

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
        combined_text = text.lower() if text else " ".join(self.prompt_history + self.response_history).lower()
        print("combined_text adalah :", combined_text)
        logger_pattern = r"\b(?:pos|afmr|awlr|awr|arr|adr|awqr|avwr|awgc)\s+(?:[a-zA-Z]+\s*){1,4}"
        logger_match = re.findall(logger_pattern, combined_text)

        print("logger_match", logger_match)    

        if logger_match:
            self.last_logger = logger_match[-1].strip()
            self.last_logger_list = [match.strip() for match in logger_match]

        date_keywords = [
            "hari ini", "kemarin", "kemaren", "minggu ini", "minggu lalu", "bulan lalu",
            "awal bulan", "akhir bulan", "tahun lalu",
            "minggu terakhir", "bulan terakhir", "hari terakhir"
        ]
        for phrase in date_keywords:
            if phrase in combined_text:
                self.last_date = phrase
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
                break

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
        print("new_prompt", new_prompt) 
        if self._should_use_memory(new_prompt) and not self._contains_explicit_logger_or_date(new_prompt):
            context_window = self.get_context_window(window_size=6)
            self._extract_context_memory(text=" ".join([m["content"] for m in context_window if m["role"] == "user"]))
        else:
            self._extract_context_memory(text=new_prompt)

        # ✅ Ambil target
        if self.intent == "show_logger_data":
            raw_targets = self.last_logger_list if self.last_logger_list else ([self.last_logger] if self.last_logger else [])
            target = self._clean_logger_list(raw_targets)
        else:
            target = self.last_logger

        # ✅ Return dan cache
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
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
from super_tools import original_fetch_data_range, find_and_fetch_latest_data, fetch_list_logger_from_prompt_flexibleV1, original_fetch_status_logger, fetch_list_logger, original_compare_by_date, extract_date_structured
from super_tools import general_stesy, summarize_logger_data

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
        return any(phrase in self.prompt for phrase in ambiguous_phrases) or len(self.prompt.strip().split()) < 3

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
            "ai_limitation": self.ai_limitation
        }

    def handle_intent(self):
        prompt = self.memory.latest_prompt
        intent = self.memory.intent
        target = self.memory.last_logger_list or [self.memory.last_logger]
        date = self.memory.last_date

        # ✅ Kirim parameter lengkap ke validator
        validator = PromptValidator(prompt, intent, target, date)
        if validator.should_fallback_to_ai():
            print("[INFO] Prompt ambigu atau intent tidak cocok, gunakan smart_respond()")
            return self.smart_respond()

        func = self.intent_map.get(intent, self.fallback_response)
        return func()
    
    # def handle_intent(self):
    #     intent = self.memory.intent
    #     print(intent)
    #     func = self.intent_map.get(intent, self.fallback_response)
    #     return func()

    def fetch_latest_data(self):
        print("show_logger_data ini telah berjalan")
        prompt = self.memory.latest_prompt

        print("self.memory.last_logger_list ",self.memory.last_logger_list)
        print(f"Type data dari self.memory.last_logger_list adalah {type(self.memory.last_logger_list)}")
        print(f"Panjang data dari self.memory.last_logger_list adalah {len(self.memory.last_logger_list)}")
        print("self.memory.last_logger ",self.memory.last_logger)

        target_loggers = self.memory.last_logger_list or [self.memory.last_logger] # Error no
        logger_list = fetch_list_logger()

        if not target_loggers or not logger_list:
            return "Target logger atau daftar logger tidak tersedia."

        fetched = find_and_fetch_latest_data(target_loggers, logger_list)
        print("fetched data adalah :",fetched)

        summaries = []
        for item in fetched:
            nama_lokasi = item['logger_name']
            data = item['data']
            summary = summarize_logger_data(nama_lokasi, data)
            summaries.append(summary)

        return "\n\n---\n\n".join(summaries)

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

        target_loggers = self.memory.last_logger_list or [self.memory.last_logger]
        print("target_loggers", target_loggers)
        logger_list = fetch_list_logger()

        return original_fetch_data_range(
            prompt=prompt,
            target_loggers=target_loggers,
            logger_list=logger_list
        )

    def fetch_status_logger(self):
        return original_fetch_status_logger(prompt=self.memory.latest_prompt)

    def show_list_logger(self):
        return fetch_list_logger_from_prompt_flexibleV1(self.memory.latest_prompt)

    def compare_by_date(self):
        print("compare_logger_by_date ini telah berjalan")
        prompt = self.memory.latest_prompt
        target_loggers = self.memory.last_logger_list or [self.memory.last_logger]
        logger_list = fetch_list_logger()

        return original_compare_by_date(
            prompt=prompt,
            target_loggers=target_loggers,
            logger_list=logger_list
        )

    def compare_latest_data(self):
        prompt = self.memory.latest_prompt
        target_loggers = self.memory.last_logger_list or [self.memory.last_logger]
        logger_list = fetch_list_logger()

        fetched = find_and_fetch_latest_data(target_loggers, logger_list)
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

        # Ambil data logger
        name_fragments = [logger["nama_lokasi"] for logger in logger_list]
        fetched = find_and_fetch_latest_data(name_fragments, logger_list)

        if not fetched:
            return "Tidak ditemukan data terbaru untuk logger yang disebutkan."

        # Siapkan data untuk konteks
        comparison_data = []
        for item in fetched:
            logger_name = item["logger_name"]
            value = item["data"].get(selected_param, "Data tidak tersedia")
            comparison_data.append(f"{logger_name}: {value}")

        # Prompt sistem
        system_prompt = (
            "Kamu adalah asisten cerdas yang menjawab pertanyaan berdasarkan data logger cuaca. "
            "Berikan jawaban yang singkat, jelas, dan sesuai dengan permintaan pengguna."
        )

        # Pertanyaan asli dan data perbandingan
        user_question = self.memory.latest_prompt
        context_data = "\n".join(comparison_data)

        # Panggil Ollama
        response = chat(
            model='llama3.1',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_question}\n\nBerikut datanya:\n{context_data}"}
            ]
        )

        return response['message']['content']

    def show_selected_parameter(self):
        print("show_selected_parameter ini telah berjalan")
        prompt = self.memory.latest_prompt.lower()
        logger_list = fetch_list_logger()

        if not logger_list:
            return "Daftar logger tidak tersedia."

        # Deteksi parameter
        selected_param = None
        for param_key, aliases in sensor_aliases.items():
            for alias in aliases:
                if alias.lower() in prompt:
                    selected_param = param_key
                    break
            if selected_param:
                break

        if not selected_param:
            return "Parameter tidak dikenali. Silakan sebutkan suhu udara, kelembaban udara, curah hujan, atau tekanan udara."

        # Deteksi rentang tanggal
        date_info = extract_date_structured(prompt)
        print("Extracted date_info:", date_info)

        # Deteksi lokasi (logger)
        target_loggers = self.memory.last_logger_list or [self.memory.last_logger]
        print("Target logger:", target_loggers)

        # Jika tidak ada tanggal, anggap permintaan data saat ini
        if not date_info.get("awal_tanggal") or not date_info.get("akhir_tanggal"):
            fetched = find_and_fetch_latest_data(target_loggers, logger_list)
            summaries = []
            for item in fetched:
                logger_name = item['logger_name']
                value = item['data'].get(selected_param, "Data tidak tersedia")
                summaries.append(f"{logger_name}: {selected_param} = {value}")
            
            # Gabungkan data jadi konteks untuk LLaMA
            context_data = "\n".join(summaries)

            system_prompt = (
                "Kamu adalah asisten cerdas yang menjawab pertanyaan berdasarkan data logger cuaca. "
                "Berikan jawaban yang singkat, jelas, dan sesuai dengan permintaan pengguna."
            )

            # Panggil LLaMA chat dengan konteks data dan pertanyaan asli
            response = chat(
                model='llama3.1',
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{self.memory.latest_prompt}\n\nBerikut data yang tersedia:\n{context_data}"}
                ]
            )
            return response['message']['content']

        # Jika ada rentang tanggal, ambil data range
        summaries = original_fetch_data_range(
            prompt=self.memory.latest_prompt,
            target_loggers=target_loggers,
            logger_list=logger_list
        )

        # Bisa juga kirim data ringkas ini ke LLaMA agar jawabannya lebih natural
        system_prompt = (
            "Kamu adalah asisten cerdas yang menjawab pertanyaan berdasarkan data logger cuaca. "
            "Berikan jawaban yang singkat, jelas, dan sesuai dengan permintaan pengguna."
        )
        response = chat(
            model='llama3.1',
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

        if self.memory.analysis_result:
            print("[INFO] Menggunakan hasil analisis dari memory")
            return self.memory.analysis_result

        fetched_data = self.fetch_data_range()

        if isinstance(fetched_data, str):
            return fetched_data

        self.memory.last_data = fetched_data

        summary_prompt = {
            "role": "user",
            "content": f"Berikan analisis data logger berikut:\n\n{fetched_data}"
        }

        result = general_stesy(messages=[summary_prompt])
        self.memory.analysis_result = result
        return result
    
    # === Fungsi smart_respond untuk IntentManager ===
    def smart_respond(self):
        prompt = self.memory.latest_prompt
        print("+++++ Function smart_respond Berjalan +++++")
        print("prompt:", prompt)

        unavailable_features = [
            "get_logger_photo_path", "compare_across_loggers", "show_selected_parameter", "explain_system"
        ]
        if self.memory.intent in unavailable_features:
            return f"Fitur untuk `{self.memory.intent}` belum tersedia saat ini. Silakan gunakan perintah lain seperti menampilkan data logger, status hujan, atau analisis data."

        validator = PromptValidator(prompt, self.memory.intent, self.memory.last_logger_list or [self.memory.last_logger], self.memory.last_date)

        if validator.is_ambiguous_prompt():
            return (
                "Saya mendeteksi bahwa pertanyaan Anda masih kurang jelas. "
                "Coba berikan informasi lebih lengkap, seperti:\n\n"
                "- 'Tampilkan data terbaru dari ARR Gemawang'\n"
                "- 'Bagaimana status hujan di AWR Beji?'\n"
                "- 'Berikan perbandingan data kelembaban minggu ini'\n\n"
                "Saya siap membantu jika pertanyaannya lebih spesifik."
            )

        if validator.is_intent_mismatch():
            return (
                f"Sepertinya sistem memprediksi intent Anda sebagai `{self.memory.intent}`, "
                "tetapi isi pertanyaannya tidak sepenuhnya cocok.\n\n"
                "Silakan cek kembali prompt Anda. Apakah maksud Anda ingin menampilkan data terbaru atau menganalisis tren?\n"
                "Contoh:\n- 'Tampilkan data hari ini dari ARR Hargorejo'\n- 'Bandingkan kelembaban antara dua logger'"
            )

        if validator.is_missing_required_fields():
            return (
                "Saya tidak menemukan lokasi logger atau tanggal yang disebutkan dalam pertanyaan Anda sesuai dengan kebutuhan intent.\n"
                "Silakan lengkapi prompt Anda, misalnya:\n- 'Tampilkan data dari ARR Gemawang'\n- 'Ambil data tanggal 1 Mei dari AWR Kaliurang'"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "Anda adalah asisten telemetri STESY. Jawablah pertanyaan ringan dengan ramah dan informatif.\n"
                    "Jika pertanyaan terlalu umum atau tidak spesifik, minta klarifikasi dengan sopan.\n"
                    "Jika pengguna menyebut fitur yang belum tersedia, beri tahu dengan jujur.\n"
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = chat(model="llama3.1:8b", messages=messages)
        return response["message"]["content"]

    def ai_limitation(self):
        # sebuah function yang dapat merespon, merekomendasikan dan menyarankan output.
        # sebuah function yang dapat memberikan respon jika prompt memiliki struktur yang kurang
        # sebuah function yang dapat memberikan struktur yang benar sesuai intent nya
        """
        Menangani intent 'ai_limitation' atau percakapan ringan/chat umum.
        Memberikan jawaban ramah atau netral menggunakan smart_respond().
        """
        prompt = self.memory.latest_prompt
        print(f"[AI Limitation] Prompt: {prompt}")
        
        return self.smart_respond()

    def fallback_response(self):
        print("Fallback intent dijalankan")
        return "Intent tidak dikenali"
