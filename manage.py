import os
import json
import re
import difflib
from ollama import chat # modif 2
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import joblib
from super_tools import sensor_aliases,get_logger_info, search_logger_info,original_fetch_data_range_async
from difflib import get_close_matches
from flask import Flask, request, jsonify
from typing import List, Dict, Tuple
from nltk.tokenize import word_tokenize
from difflib import SequenceMatcher

MEMORY_DIR = "user_memory"

# === Stopword Loader ===
def get_stopwords_list(stop_file_path):
    """Load stopwords dari file"""
    try:
        with open(stop_file_path, 'r', encoding="utf-8") as f:
            stopwords = f.readlines()
            stop_set = set(m.strip() for m in stopwords)
            return list(frozenset(stop_set))
    except FileNotFoundError:
        print(f"[ERROR] File not found: {stop_file_path}")
        return []

# Load stopwords lokal
STOPWORD_PATH = "id.beacon.stopwords.txt"
# stopwords = get_stopwords_list(STOPWORD_PATH)


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

        self.last_logger_clarification = None  # ✅ Diperlukan
        self.last_logger: Optional[str] = None
        self.last_logger_list: List[str] = []
        self.last_logger_id: Optional[str] = None
        self.last_date: Optional[str] = None
        self.intent: Optional[str] = None
        self.analysis_result: Optional[str] = None
        self.last_data: Optional[str] = None  # ✅ Untuk analisa tanpa perlu fetch ulang
        self.last_logger_source: Optional[str] = None 

        self.last_logger_data: Dict[str, Dict] = {}
        self.last_logger_ids: List[str] = []

        self.prev_intent: Optional[str] = None
        self.prev_target: Optional[str] = None
        self.prev_date: Optional[str] = None
        
        # ✅ Tambahkan ini untuk memperbaiki error
        self.logger_suggestions: List[str] = []

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

    CONFIRM_YES_SYNONYMS = {
            # Karakter ekspresif/random
            "?", "??", "???", "????", "!", "!!", "!!!", "!!!!", "!!!!!", "...", "....", "......","ya",
            # Variasi dasar & pertanyaan
            "ya", "iya", "iya?", "iya!", "iya?!", "betul", "betul?", "benar", "benar?", "bener", "bener?",
            "betool", "betool?", "betool!!", "betool!", "benerrr", "benerrrr!","boleh","yaa","yaaa","yaaaa",
            # Variasi informal dan slang
            "yoi", "yap", "y", "yo", "sip", "ok", "oke", "okey", "yes", "yess", "yesss", "you bet", "yosh", "yoa", "yoi",
            # Tambahan penekanan/emosi
            "ya dong", "iya deh", "iya banget", "iya lah", "iyalah", "iyalah sayang", "iyaa", "iyaa dong", "iyaa!", "iyap", 
            "yes dong", "yes lah", "yes banget", "yes yes!",
            # Kata penegasan atau afirmatif
            "udah pasti", "jelas", "jelas banget", "pastinya", "tentu", "tentu saja", "pasti", "pastilah", "bener banget",
            # Chat slang/kasual + ekspresi acak
            "gas", "cus", "gass", "gasss", "gaskeun", "mantap", "mantul", "sip deh", "sippp", "sippp!", "sipppp", "siap", "siap!", 
            "go!", "ayoo!", "ayok", "hayuk", "lanjut", "langsung aja", "okedeh", "ok sip", "ok gas", "langsung gas",
            # Emoji & kombinasi
            "ya 👍", "oke 👍", "sip 👍", "yoi 💪", "yes ✅", "betul ✅", "iyes ✅", "mantap 🔥", "sippp 🔥", "cus 💨", "gaspol 🔥",
            # Tambahan karakter random dan ekspresi lebih bebas
            "iyaa~", "iyaaa", "iyaaa!!", "iyaa bgt", "iyes!", "okeee", "oke deh~", "yes!", "yeees", "yeeees!", "yaaaa",
            "gaskeun!", "gasskan!", "gasskeun dong", "mantab!", "mantabb!", "mantabb banget!", "langsungkeun!", "yappp", "yaaa gpp",
            # Tambahan dari permintaan
            "ya dua-duanya", "dua-duanya", "keduanya", "dua duanya", "ya keduanya", "ya dua duanya", "kedua duanya",
        }
    def _get_memory_path(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        return os.path.join(MEMORY_DIR, f"{self.user_id}.json")

    def _load_user_memory(self):
        try:
            path = self._get_memory_path()
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.memory_history = data.get("history", [])

                    if self.memory_history:
                        last_entry = self.memory_history[-1]
                        # 🔁 Atur ulang state memory dari history
                        self.intent = last_entry.get("intent")
                        self.prev_intent = last_entry.get("prev_intent", self.intent)  # fallback
                        self.latest_prompt = last_entry.get("prompt")
                        self.last_logger = last_entry.get("logger")
                        self.last_logger_id = last_entry.get("id_logger")
                        self.last_date = last_entry.get("last_date")
                        self.last_logger_list = last_entry.get("last_logger_list", [])
                        self.prev_logger = last_entry.get("prev_logger", self.intent)  # fallback last_entry.get("prev_logger", self.last_logger) digunakan seperti prev_intent tapi untuk logger
                        self.prev_date = last_entry.get("prev_date", self.last_date)  # fallback last_entry.get("prev_logger", self.last_date) digunakan seperti prev_intent tapi untuk tanggal
                        self.last_data = last_entry.get("response")  # untuk cache respons
        except Exception as e:
            print(f"[MEMORY LOAD ERROR] {e}")

    def _save_user_memory(self):
        print("_save_user_memory telah berjalan")
        try:
            now = datetime.now()
            response = self.response_history[-1] if self.response_history else ""
            prompt = self.latest_prompt or ""
            print("response di _save_user_memory :",response)
            data_entry = {
                "user_id": self.user_id,
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "prompt": prompt,
                "response": response,
                "intent": self.intent,
                "prev_intent": self.prev_intent,  # ✅ Simpan ini
                "logger": self.last_logger,
                "id_logger": self.last_logger_id,
                "last_date": self.last_date,
                "last_logger_list": self.last_logger_list,
                "prev_logger" : self.prev_target, # ✅ Simpan Logger yang telah disebutkan
                "prev_date" : self.prev_date # ✅ Simpan Tanggal yang telah disebutkan
            }

            path = self._get_memory_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)

            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    history = data.get("history", [])
            else:
                history = []

            history.append(data_entry)
            if len(history) > 100:
                history = history[-100:]

            with open(path, "w", encoding="utf-8") as f:
                json.dump({"history": history}, f, ensure_ascii=False, indent=2)
            print(f"[MEMORY SAVED] {data_entry}")
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

    def get_context_window(self, window_size: int = 4) -> List[Dict[str, str]]:
        prompts = self.prompt_history[-window_size:]
        responses = self.response_history[-window_size:]
        context = []
        for u, a in zip(prompts, responses):
            context.append({"role": "user", "content": u})
            context.append({"role": "assistant", "content": a})
        return context

    def _predict_intent_bert(self, text: str) -> str:
        # print("text dari _predict_intent_bert :", text)
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1)
        return self.label_encoder.inverse_transform(pred.cpu().numpy())[0]
    
    # def handle_confirmation_prompt_if_needed(self, user_reply: str) -> Optional[dict]:
    #     print("func handle_confirmation_prompt_if_needed sedang berjalan....")
    #     print(f"Original prompt input : {user_reply}")

    #     # 🔁 Jika user_reply berupa list → gabungkan jadi string
    #     if isinstance(user_reply, list):
    #         user_reply = " ".join(user_reply)
    #         print(f"📦 Diubah dari list ke string: {user_reply}")

        
        
    #     normalized_reply = user_reply.strip().lower()
    #     print(f"🧾 normalized prompt: {normalized_reply}")
    #     print(f"⚠️ last_logger_clarification : {self.last_logger_clarification}")
    #     clarification = self.last_logger_clarification

    #     if not clarification:
    #         print("❌ Tidak ada klarifikasi sebelumnya.")
    #         return None

    #     candidates = clarification.get("candidates", [])
    #     ambiguous_input = clarification.get("ambiguous_input", "")

    #     if len(candidates) != 2:
    #         print("❌ Jumlah kandidat tidak valid.")
    #         return None

    #     # ✅ Jika memenuhi syarat lokasi ambigu dan tipe logger khas
    #     if (
    #         any(keyword in self.latest_prompt.lower() for keyword in ["sapon", "kali bawang"]) and
    #         all(any(kw in logger.lower() for kw in ["arr", "awlr"]) for logger in candidates)
    #     ):
    #         print("🔁 Memanggil handle_dual_logger_selection() karena cocok kondisi.")
    #         result = self.handle_dual_logger_selection(user_reply, candidates)
    #         if result:
    #             selected = result["logger"]
    #             self.last_logger = selected[0] if isinstance(selected, list) else selected
    #             self.last_logger_list = selected if isinstance(selected, list) else [selected]
    #             self.last_logger_clarification = None

    #             return {
    #                 "prompt": self.latest_prompt or user_reply,
    #                 "ambiguous_input": ambiguous_input,
    #                 "confirmed": True,
    #                 "logger": selected
    #             }

    #     # ✅ fallback default: jika hanya afirmatif umum → ambil semua kandidat
    #     if normalized_reply in self.CONFIRM_YES_SYNONYMS:
    #         print("✅ Klarifikasi umum afirmatif, ambil semua kandidat.")
    #         self.last_logger = candidates[0]
    #         self.last_logger_list = candidates
    #         self.last_logger_clarification = None
    #         return {
    #             "prompt": self.latest_prompt or user_reply,
    #             "ambiguous_input": ambiguous_input,
    #             "confirmed": True,
    #             "logger": candidates
    #         }

    #     print("⚠️ Tidak dikenali sebagai klarifikasi.")
    #     return None
    #     # normalized_reply = user_reply.strip().lower()

    #     # if normalized_reply in CONFIRM_YES_SYNONYMS and getattr(self, "last_logger_clarification", None):
    #     #     clarification = self.last_logger_clarification
    #     #     print(f"Klarifikasi Kandidat adalah  : {clarification}") # 
    #     #     candidates = clarification.get("candidates", [])
    #     #     # GANTI KE CODE INI 
    #     #     # if (
    #     #     #     len(self.last_logger_list) == 2
    #     #     #     and any(k in self.latest_prompt.lower() for k in ["sapon", "kali bawang"])
    #     #     #     and all(
    #     #     #         any(kw in logger.lower() for kw in ["arr", "awlr"])
    #     #     #         for logger in self.last_logger_list
    #     #     #     )
    #     #     # ):
    #     #     #     print("if jalan") #
    #     #     #     result = self.handle_dual_logger_selection()
    #     #     #     if result:
    #     #     #         print(f"🎯 Logger yang dipilih: {result}")
    #     #     #         selected = result["logger"]
    #     #     #         if isinstance(selected, str):
    #     #     #             self.last_logger_list = [selected]
    #     #     #         elif isinstance(selected, list):
    #     #     #             print("selected :", selected)
    #     #     #             target = selected
    #     #     #             self.last_logger_list = selected # bagaimana mengoper variable ini langsung ke return yang ada dibawah sana, agar melewati if else lain
    #     #     #         # self.last_logger = self.last_logger_list[0]

    #     #     if len(candidates) == 2:
    #     #         selected_logger = candidates  # ✅ default ke kandidat pertama # KETEMU bntr
    #     #         print("🧪 Type of selected_logger:", type(selected_logger))
    #     #         print("🧪 selected_logger content:", selected_logger)
                
    #     #         self.last_logger = selected_logger
    #     #         self.last_logger_list = [selected_logger]
    #     #         self.last_logger_clarification = None  # 🔁 Reset setelah dipakai

    #     #         print("📌 Klarifikasi dipilih otomatis:", selected_logger) # 
    #     #         return {"confirmed": True, "logger": selected_logger}  # [ERROR] string indices must be integers, not 'str'
    #     #         #  return {"prompt":"self.latest_prompt","confirmed": True, "logger": selected_logger} ubah ke return seperti ini
 
    #     # return None
    def validate_logger_clarification(self, clarification):
        if not isinstance(clarification, list) or len(clarification) < 2:
            return None

        ignore_tokens = {"pos", "arr", "awlr", "awr", "adr", "awqr", "avwr", "awgc"}

        def get_location_key(name):
            tokens = [t for t in name.lower().split() if t not in ignore_tokens]
            return " ".join(tokens[-2:]) if tokens else ""

        base_key = get_location_key(clarification[0])
        if not base_key:
            return None

        if all(get_location_key(name) == base_key for name in clarification[1:]):
            return clarification
        return None
    
    def normalize_logger_list_from_history(self, list_random):
        print("📌 func normalize_logger_list_from_history sedang berjalan....")
        print(f"📥 list_random adalah: {list_random}")

        if list_random is None:
            print("❌ list_random bernilai None — langsung return None")
            return None

        if not isinstance(list_random, list) or len(list_random) == 0:
            print("⚠️ list_random kosong atau bukan list — return None")
            return None

        # ✅ Kandidat benar (hardcoded)
        list_candidates = [
            ['Pos ARR Sapon', 'Pos AWLR Sapon'],
            ['Pos ARR Kali Bawang', 'Pos AWLR Kali Bawang']
        ]

        from difflib import SequenceMatcher

        def flatten(text):
            return text.lower().replace("pos ", "").replace("  ", " ").strip()

        def similarity_score(a, b):
            return SequenceMatcher(None, flatten(a), flatten(b)).ratio()

        for candidate in list_candidates:
            scores = []
            for cand_logger in candidate:
                best_score = max([similarity_score(cand_logger, r) for r in list_random])
                scores.append(best_score)
            avg_score = sum(scores) / len(scores)
            print(f"🔎 Kandidat: {candidate} — Skor rata-rata: {avg_score:.2f}")
            if avg_score > 0.75:
                print(f"✅ Klarifikasi cocok ditemukan: {candidate}")
                return candidate

        print("❌ Tidak ada klarifikasi cocok.")
        return None

    
    def handle_confirmation_prompt_if_needed(self, user_reply: str) -> Optional[dict]:
        print("func handle_confirmation_prompt_if_needed sedang berjalan....")
        print(f"Original prompt input : {user_reply}")

        # 🔁 Jika user_reply berupa list → gabungkan jadi string
        if isinstance(user_reply, list):
            user_reply = " ".join(user_reply)
            print(f"📦 Diubah dari list ke string: {user_reply}")

        normalized_reply = user_reply.strip().lower()
        print(f"🧾 normalized prompt: {normalized_reply}")

        # Coba ambil klarifikasi langsung
        clarification = self.last_logger_clarification

        # ✅ Pasang disini
        clarification = self.normalize_logger_list_from_history(clarification)
        # print(f"clarification yang baru : {clarification}")

        if clarification:
            print(f"✅ Klarifikasi valid ditemukan: {clarification}")
            self.last_logger_clarification = clarification

        # Jika tidak ada klarifikasi tersimpan, coba ekstrak dari response_history
        if not clarification:
            print("⚠️ last_logger_clarification kosong — mencoba ekstraksi dari response_history...")

            # Pola untuk mendeteksi 'pos {tipe} {lokasi}'
            pattern = r"(pos\s+[a-z]+(?:\s+[a-z]+){0,2})"
            logger_types = ['arr', 'awlr', 'awr', 'afmr', 'awqr', 'avwr', 'awgc']
            history_text = " ".join(self.response_history).lower()
            matches = re.findall(pattern, history_text)
            candidates = [m for m in matches if any(t in m for t in logger_types)]
            print("matches :", matches)
            print("candidates :", candidates,"Panjang data adalah", len(candidates))

            if len(candidates) == 2:
                clarification = {
                    "ambiguous_input": "logger sebelumnya",
                    "candidates": candidates
                }
                self.last_logger_clarification = clarification
                print("✅ Klarifikasi berhasil diambil dari response_history:", clarification)

            elif len(candidates) > 2:
                print(f"🔁 Panjang dari kandidat adalah {len(candidates)}")
                candidates = self.normalize_logger_list_from_history(candidates)

                if candidates is None or len(candidates) < 2:
                    print("❌ Tidak ditemukan kandidat logger yang valid setelah normalisasi.")
                    return None

                clarification = {
                    "ambiguous_input": "logger sebelumnya",
                    "candidates": candidates
                }
                self.last_logger_clarification = clarification
                print("✅ Klarifikasi berhasil diambil dari response_history setelah normalisasi:", clarification)

            else:
                print("❌ Tidak cukup kandidat untuk klarifikasi.")
                return None

        print(f"⚠️ last_logger_clarification : {clarification}")
        candidates = clarification.get("candidates", [])
        ambiguous_input = clarification.get("ambiguous_input", "")

        if len(candidates) != 2:
            print("❌ Jumlah kandidat tidak valid.")
            return None

        # ✅ Jika memenuhi syarat lokasi ambigu dan tipe logger khas
        if (
            any(keyword in self.latest_prompt.lower() for keyword in ["sapon", "kali bawang"]) and
            all(any(kw in logger.lower() for kw in ["arr", "awlr"]) for logger in candidates)
        ):
            print("🔁 Memanggil handle_dual_logger_selection() karena cocok kondisi.")
            result = self.handle_dual_logger_selection(user_reply, candidates)
            if result:
                selected = result["logger"]
                self.last_logger = selected[0] if isinstance(selected, list) else selected
                self.last_logger_list = selected if isinstance(selected, list) else [selected]
                self.last_logger_clarification = None

                return {
                    "prompt": self.latest_prompt or user_reply,
                    "ambiguous_input": ambiguous_input,
                    "confirmed": True,
                    "logger": selected
                }

        # ✅ fallback default: jika hanya afirmatif umum → ambil semua kandidat
        if normalized_reply in self.CONFIRM_YES_SYNONYMS:
            print("✅ Klarifikasi umum afirmatif, ambil semua kandidat.")
            self.last_logger = candidates[0]
            self.last_logger_list = candidates
            self.last_logger_clarification = None
            return {
                "prompt": self.latest_prompt or user_reply,
                "ambiguous_input": ambiguous_input,
                "confirmed": True,
                "logger": candidates
            }

        print("⚠️ Tidak dikenali sebagai klarifikasi.")
        return None

    def confirm_logger_from_previous_suggestion(self, previous_assistant_message: str, user_reply: str) -> Optional[str]:
        """
        Jika user menjawab 'ya' dan assistant sebelumnya memberi saran logger,
        maka ambil logger yang disarankan pertama.t
        """
        CONFIRM_YES_SYNONYMS = {
            # Karakter ekspresif/random
            "?", "??", "???", "????", "!", "!!", "!!!", "!!!!", "!!!!!", "...", "....", "......","ya","ya kaliurang",
            # Variasi dasar & pertanyaan
            "ya", "iya", "iya?", "iya!", "iya?!", "betul", "betul?", "benar", "benar?", "bener", "bener?",
            "betool", "betool?", "betool!!", "betool!", "benerrr", "benerrrr!","boleh",
            # Variasi informal dan slang
            "yoi", "yap", "y", "yo", "sip", "ok", "oke", "okey", "yes", "yess", "yesss", "you bet", "yosh", "yoa", "yo’i",
            # Tambahan penekanan/emosi
            "ya dong", "iya deh", "iya banget", "iya lah", "iyalah", "iyalah sayang", "iyaa", "iyaa dong", "iyaa!", "iyap", 
            "yes dong", "yes lah", "yes banget", "yes yes!",
            # Kata penegasan atau afirmatif
            "udah pasti", "jelas", "jelas banget", "pastinya", "tentu", "tentu saja", "pasti", "pastilah", "bener banget",
            # Chat slang/kasual + ekspresi acak
            "gas", "cus", "gass", "gasss", "gaskeun", "mantap", "mantul", "sip deh", "sippp", "sippp!", "sipppp", "siap", "siap!", 
            "go!", "ayoo!", "ayok", "hayuk", "lanjut", "langsung aja", "okedeh", "ok sip", "ok gas", "langsung gas",
            # Emoji & kombinasi
            "ya 👍", "oke 👍", "sip 👍", "yoi 💪", "yes ✅", "betul ✅", "iyes ✅", "mantap 🔥", "sippp 🔥", "cus 💨", "gaspol 🔥",
            # Tambahan karakter random dan ekspresi lebih bebas
            "iyaa~", "iyaaa", "iyaaa!!", "iyaa bgt", "iyes!", "okeee", "oke deh~", "yes!", "yeees", "yeeees!", "yaaaa",
            "gaskeun!", "gasskan!", "gasskeun dong", "mantab!", "mantabb!", "mantabb banget!", "langsungkeun!", "yappp", "yaaa gpp",
            # Tambahkan kombinasi konfirmasi dan nama lokasi logger
            "ya kaliurang", "boleh kaliurang", "iya kaliurang", "ok kaliurang", "sip kaliurang",
            "ya bronggang", "boleh bronggang", "iya bronggang", "ok bronggang", "sip bronggang",
            "ya kemput", "boleh kemput", "iya kemput", "ok kemput", "sip kemput",
            "ya seturan", "boleh seturan", "iya seturan", "ok seturan", "sip seturan",
            "ya papringan", "boleh papringan", "iya papringan", "ok papringan", "sip papringan",
            "ya pogung", "boleh pogung", "iya pogung", "ok pogung", "sip pogung",
            "ya sinduadi", "boleh sinduadi", "iya sinduadi", "ok sinduadi", "sip sinduadi",
            "ya bendungan", "boleh bendungan", "iya bendungan", "ok bendungan", "sip bendungan",
            "ya opak pulo", "boleh opak pulo", "iya opak pulo", "ok opak pulo", "sip opak pulo",
            "ya bantar", "boleh bantar", "iya bantar", "ok bantar", "sip bantar",
            "ya kedungmiri", "boleh kedungmiri", "iya kedungmiri", "ok kedungmiri", "sip kedungmiri",
            "ya bunder", "boleh bunder", "iya bunder", "ok bunder", "sip bunder",
            "ya gemawang", "boleh gemawang", "iya gemawang", "ok gemawang", "sip gemawang",
            "ya singkung", "boleh singkung", "iya singkung", "ok singkung", "sip singkung",
            "ya sapon", "boleh sapon", "iya sapon", "ok sapon", "sip sapon",
            "ya hargorejo", "boleh hargorejo", "iya hargorejo", "ok hargorejo", "sip hargorejo",
            "ya sanden", "boleh sanden", "iya sanden", "ok sanden", "sip sanden",
            "ya ngawen", "boleh ngawen", "iya ngawen", "ok ngawen", "sip ngawen",
            "ya angin-angin", "boleh angin-angin", "iya angin-angin", "ok angin-angin", "sip angin-angin",
            "ya seyegan", "boleh seyegan", "iya seyegan", "ok seyegan", "sip seyegan",
            "ya godean", "boleh godean", "iya godean", "ok godean", "sip godean",
            "ya prumpung", "boleh prumpung", "iya prumpung", "ok prumpung", "sip prumpung",
            "ya giriwungu", "boleh giriwungu", "iya giriwungu", "ok giriwungu", "sip giriwungu",
            "ya borrowarea", "boleh borrowarea", "iya borrowarea", "ok borrowarea", "sip borrowarea",
            "ya terong", "boleh terong", "iya terong", "ok terong", "sip terong",
            "ya kedungkeris", "boleh kedungkeris", "iya kedungkeris", "ok kedungkeris", "sip kedungkeris",
            "ya barongan", "boleh barongan", "iya barongan", "ok barongan", "sip barongan",
            "ya pengasih", "boleh pengasih", "iya pengasih", "ok pengasih", "sip pengasih",
            "ya kali bawang", "boleh kali bawang", "iya kali bawang", "ok kali bawang", "sip kali bawang",
            "ya wonokromo", "boleh wonokromo", "iya wonokromo", "ok wonokromo", "sip wonokromo",
            "ya tegal", "boleh tegal", "iya tegal", "ok tegal", "sip tegal",
            "ya beran", "boleh beran", "iya beran", "ok beran", "sip beran",
            "ya gembongan", "boleh gembongan", "iya gembongan", "ok gembongan", "sip gembongan",
            "ya santan", "boleh santan", "iya santan", "ok santan", "sip santan",
            "ya plataran", "boleh plataran", "iya plataran", "ok plataran", "sip plataran",
            "ya nyemengan", "boleh nyemengan", "iya nyemengan", "ok nyemengan", "sip nyemengan",
            "ya gedhangan", "boleh gedhangan", "iya gedhangan", "ok gedhangan", "sip gedhangan",
            "ya beji", "boleh beji", "iya beji", "ok beji", "sip beji",
            "ya gumuk", "boleh gumuk", "iya gumuk", "ok gumuk", "sip gumuk"
        }
        print("confirm_logger_from_previous_suggestion telah berjalan")

        user_reply = user_reply.strip().lower()
        prev_msg = previous_assistant_message.strip().lower()
        print(f"latest User Messages is : {user_reply}")
        print(f"Previous Messages is : {prev_msg}")

        if user_reply in CONFIRM_YES_SYNONYMS and "manakah yang Anda maksud" in prev_msg:
            # Ambil logger dari kalimat seperti: "Apakah maksud Anda: 'pos arr kemput'?"
            match = re.findall(r"'(pos [^']+)'", previous_assistant_message)
            print("match :", match)
            if match:
                confirmed_logger = match[0]
                print(f"🤖 Logger dikonfirmasi oleh user: {confirmed_logger}")
                return confirmed_logger
            else :
                print("Gagal Mendapatkan konfirmasi logger")
        return None
    
    # def _extract_context_memory(self, text: Optional[str] = None):
    #     print("Function _extract_context_memory sedang berjalan")

    #     def normalize_logger_name(name: str) -> str:
    #         name = " ".join(name.strip().lower().split())
    #         return name if name.startswith("pos ") else "pos " + name

    #     combined_text = text.lower() if text else " ".join(self.prompt_history + self.response_history).lower()
    #     print("📥 combined_text:", combined_text)
    #     print(f"Latest Prompt : {self.latest_prompt}")

    #     try:
    #         print("Try code condition is working .....")
    #         logger_data = fetch_list_logger()
    #         logger_names_from_db = [normalize_logger_name(lg["nama_lokasi"]) for lg in logger_data if "nama_lokasi" in lg]
    #         normalized_valid_loggers = set(logger_names_from_db)
    #         print(f"✅ {len(normalized_valid_loggers)} logger valid dimuat.")
    #         from collections import Counter
    #         stripped_names = [name.replace("pos ", "").strip() for name in logger_names_from_db]
    #         normalized_counter = Counter(stripped_names)
    #     except Exception as e:
    #         print("❌ [ERROR] fetch_list_logger gagal:", e)
    #         normalized_valid_loggers = set()
    #         normalized_counter = {}

    #     logger_pattern = r"\b(?:logger|pos|afmr|awlr|awr|arr|adr|awqr|avwr|awgc)\s+(?:[a-z]{3,}(?:\s+[a-z]{3,}){0,3})"
    #     raw_matches = re.findall(logger_pattern, combined_text)
    #     print(f"🔍 logger_match (raw): {raw_matches}")

    #     expanded_matches = self._clean_logger_list(raw_matches)

    #     print("🧩 expanded_matches (awal):", expanded_matches)

    #     if any(keyword in expanded_matches for keyword in ["kali bawang", "sapon"]):
    #         print("⚠️ Deteksi kata eksplisit 'kali bawang' atau 'sapon' — token overlap dilewati")
    #     else:
    #         print("🔄 Mencoba pencocokan dengan token overlap...")
    #         technical_tokens = {"pos", "logger", "data", "afmr", "awlr", "awr", "arr", "adr", "awqr", "avwr", "awgc"}
    #         tokens = {word for word in self.latest_prompt.lower().split() if word not in technical_tokens and len(word) > 2}
    #         print("🧠 Token bersih dari prompt:", tokens)

    #         for logger in logger_names_from_db:
    #             lokasi_tokens = {
    #                 word for word in logger.replace("pos ", "").lower().split()
    #                 if word not in technical_tokens and len(word) > 2
    #             }
    #             if lokasi_tokens & tokens:
    #                 if logger not in expanded_matches:
    #                     expanded_matches.append(logger)
    #                 print(f"✅ Token overlap: {lokasi_tokens & tokens} cocok dengan '{logger}'")

    #     # ✅ Bersihkan hanya sekali
    #     stopwords = {
    #         "kemarin", "kemaren", "hari", "ini", "lalu", "terakhir",
    #         "minggu", "bulan", "tahun", "tanggal", "besok", "selama", "depan"
    #     }
    #     cleaned_matches = set()
    #     self.logger_suggestions = {}
    #     self.last_logger_clarification = None

    #     for match in expanded_matches:
    #         filtered = " ".join(word for word in match.split() if word.lower() not in stopwords)
    #         norm = normalize_logger_name(filtered)
    #         print(f"🧹 filtered: {filtered} → norm: {norm}")

    #         if norm in normalized_valid_loggers:
    #             cleaned_matches.add(norm)
    #         else:
    #             stripped_norm = norm.replace("pos ", "").strip()
    #             if stripped_norm in {"sapon", "kali bawang"}:
    #                 suggestions = [l for l in normalized_valid_loggers if stripped_norm in l]
    #                 print(f"⚠️ Logger ambigu '{stripped_norm}' — Saran eksplisit: {suggestions}")
    #             else:
    #                 n_suggestions = min(normalized_counter.get(stripped_norm, 2), 3)
    #                 suggestions = get_close_matches(norm, normalized_valid_loggers, n=n_suggestions, cutoff=0.7)
    #                 print(f"📌 Saran fuzzy match: {suggestions}")

    #             if suggestions:
    #                 self.logger_suggestions[norm] = suggestions

    #                 if len(suggestions) == 2:
    #                     self.last_logger_clarification = {
    #                         "ambiguous_input": norm,
    #                         "candidates": suggestions
    #                     }
    #                     print("📌 last_logger_clarification disimpan:", self.last_logger_clarification)

    #             print(f"⚠️ Tidak valid: {norm} — Saran: {suggestions}")

    #     if cleaned_matches:
    #         self.last_logger_list = list(cleaned_matches)
    #         self.last_logger = self.last_logger_list[-1]
    #         print("📌 last_logger_list:", self.last_logger_list)
    #         print("📌 last_logger (terakhir):", self.last_logger)
    #     else:
    #         print("🚫 Tidak ada logger valid — mempertahankan logger sebelumnya.")
    #         self.last_logger_list = self.last_logger_list or []

    #     # 📅 Deteksi tanggal eksplisit
    #     date_keywords = [
    #         "hari ini", "kemarin", "kemaren", "minggu ini", "minggu lalu", "bulan lalu",
    #         "awal bulan", "akhir bulan", "tahun lalu", "minggu terakhir", "bulan terakhir", "hari terakhir"
    #     ]
    #     for phrase in date_keywords:
    #         if phrase in combined_text:
    #             self.last_date = phrase
    #             print("🗓️ Deteksi tanggal (keyword):", phrase)
    #             break

    #     # 📅 Deteksi tanggal relatif
    #     relative_date_patterns = [
    #         r"\d+\s+hari\s+(lalu|terakhir)",
    #         r"\d+\s+minggu\s+(lalu|terakhir)",
    #         r"\d+\s+bulan\s+(lalu|terakhir)"
    #     ]
    #     for pattern in relative_date_patterns:
    #         match = re.search(pattern, combined_text)
    #         if match:
    #             self.last_date = match.group(0)
    #             print("🗓️ Deteksi tanggal (relatif):", self.last_date)
    #             break


    def _extract_context_memory(self, text: Optional[str] = None):
        print("Function _extract_context_memory sedang berjalan")
        print(f"Latest Prompt : {self.latest_prompt}")

        # if isinstance(self.latest_prompt, str) and self.latest_prompt.strip().lower() in self.CONFIRM_YES_SYNONYMS:
        #     if self.last_logger_clarification:
        #         print("⛔ Prompt hanya berupa konfirmasi umum & sudah ada klarifikasi logger — keluar dari context extraction.")
        #         return

        if self.latest_prompt.strip().lower() in self.CONFIRM_YES_SYNONYMS:
            print(f"Klarifikasi Logger {self.last_logger_clarification}")
            
        def normalize_logger_name(name: str) -> str:
            name = " ".join(name.strip().lower().split())
            return name if name.startswith("pos ") else "pos " + name

        combined_text = text.lower() if text else " ".join(self.prompt_history + self.response_history).lower()
        print("📥 combined_text:", combined_text)
        
        try:
            print("Try code condition is working .....")
            logger_data = fetch_list_logger()
            logger_names_from_db = [normalize_logger_name(lg["nama_lokasi"]) for lg in logger_data if "nama_lokasi" in lg]
            normalized_valid_loggers = set(logger_names_from_db)
            print(f"✅ {len(normalized_valid_loggers)} logger valid dimuat.")
            from collections import Counter
            stripped_names = [name.replace("pos ", "").strip() for name in logger_names_from_db]
            normalized_counter = Counter(stripped_names)
            print(f"normalized_counter adalah {normalized_counter}")
        except Exception as e:
            print("❌ [ERROR] fetch_list_logger gagal:", e)
            normalized_valid_loggers = set()
            normalized_counter = {}
        logger_pattern = r"\b(?:data|logger|pos|afmr|awlr|awr|arr|adr|awqr|avwr|awgc)\s+(?:[a-z]{3,}(?:\s+[a-z]{3,}){0,3})"
        raw_matches = re.findall(logger_pattern, combined_text)
        print(f"🔍 logger_match (raw): {raw_matches}")

        expanded_matches = self._clean_logger_list(raw_matches)

        print("🧩 expanded_matches (awal):", expanded_matches) 

        if any(keyword in expanded_matches for keyword in ["kali bawang", "sapon"]): 
            print("⚠️ Deteksi kata eksplisit 'kali bawang' atau 'sapon' — token overlap dilewati")
        else:
            print("🔄 Mencoba pencocokan dengan token overlap...")
            technical_tokens = {"pos", "logger", "data", "afmr", "awlr", "awr", "arr", "adr", "awqr", "avwr", "awgc"}
            tokens = {word for word in self.latest_prompt.lower().split() if word not in technical_tokens and len(word) > 2}
            print("🧠 Token bersih dari prompt:", tokens)

            already_matched = set(normalize_logger_name(m) for m in raw_matches)

            for logger in logger_names_from_db:
                lokasi_tokens = {
                    word for word in logger.replace("pos ", "").lower().split()
                    if word not in technical_tokens and len(word) > 2
                }
                if lokasi_tokens & tokens:
                    normalized_logger = normalize_logger_name(logger)
                    if normalized_logger not in expanded_matches and normalized_logger not in already_matched:
                        expanded_matches.append(normalized_logger)
                        print(f"✅ Token overlap: {lokasi_tokens & tokens} cocok dengan '{logger}'")

        stopwords = {
            "kemarin", "kemaren", "hari", "ini", "lalu", "terakhir",
            "minggu", "bulan", "tahun", "tanggal", "besok", "selama", "depan"
        }
        cleaned_matches = set()
        self.logger_suggestions = {}
        self.last_logger_clarification = None
        
        ambiguous_keywords = {"sapon", "kali bawang"}
        for match in expanded_matches:
            filtered = " ".join(word for word in match.split() if word.lower() not in stopwords)
            norm = normalize_logger_name(filtered)
            print(f"🧹 filtered: {filtered} → norm: {norm}")

            if norm in normalized_valid_loggers:
                cleaned_matches.add(norm)
            else:
                stripped_norm = norm.replace("pos ", "").strip().lower()

                # Deteksi apakah nama logger mengandung kata ambigu eksplisit
                ambiguous_detected = False
                for keyword in ambiguous_keywords:
                    if keyword in stripped_norm:
                        suggestions = [l for l in normalized_valid_loggers if keyword in l]
                        ambiguous_detected = True
                        print(f"⚠️ Logger ambigu mengandung '{keyword}' — Saran eksplisit: {suggestions}")
                        break

                if not ambiguous_detected:
                    n_suggestions = min(normalized_counter.get(stripped_norm, 1), 3)
                    suggestions = get_close_matches(norm, normalized_valid_loggers, n=n_suggestions, cutoff=0.7)
                    print(f"📌 Saran fuzzy match: {suggestions}")

                if suggestions:
                    self.logger_suggestions[norm] = suggestions

                    if len(suggestions) == 2:
                        self.last_logger_clarification = {
                            "ambiguous_input": norm,
                            "candidates": suggestions
                        }
                        print("📌 last_logger_clarification disimpan:", self.last_logger_clarification)

                print(f"⚠️ Tidak valid: {norm} — Saran: {suggestions}")

        # for match in expanded_matches: # ['pos sapon'] masuk ke logger ambigu, ['data sapon'] tidak masuk ke logger ambigu
        #     filtered = " ".join(word for word in match.split() if word.lower() not in stopwords)
        #     norm = normalize_logger_name(filtered)
        #     print(f"🧹 filtered: {filtered} → norm: {norm}") # 🧹 filtered: pos sapon → norm: pos sapon, 🧹 filtered: data sapon → norm: pos data sapon

        #     if norm in normalized_valid_loggers:
        #         cleaned_matches.add(norm)
        #     else:
        #         stripped_norm = norm.replace("pos ", "").strip().lower()
        #         if stripped_norm in {"sapon", "kali bawang"}:
        #             suggestions = [l for l in normalized_valid_loggers if stripped_norm in l]
        #             print(f"⚠️ Logger ambigu '{stripped_norm}' — Saran eksplisit: {suggestions}") # ⚠️ Logger ambigu 'sapon' — Saran eksplisit: ['pos arr sapon', 'pos awlr sapon']
        #         else:
        #             n_suggestions = min(normalized_counter.get(stripped_norm, 1), 3)
        #             suggestions = get_close_matches(norm, normalized_valid_loggers, n=n_suggestions, cutoff=0.7)
        #             print(f"📌 Saran fuzzy match: {suggestions}")

        #         if suggestions:
        #             self.logger_suggestions[norm] = suggestions

        #             if len(suggestions) == 2:
        #                 self.last_logger_clarification = {
        #                     "ambiguous_input": norm,
        #                     "candidates": suggestions
        #                 }
        #                 print("📌 last_logger_clarification disimpan:", self.last_logger_clarification)

        #         print(f"⚠️ Tidak valid: {norm} — Saran: {suggestions}")

        if self.last_logger_clarification:
            self.last_logger_list = []
            self.last_logger = None
            print("🛑 Logger dikosongkan karena menunggu klarifikasi dari user")
        # elif cleaned_matches:
        #     print("cleaned_matches :", cleaned_matches)
        #     print("panjang cleaned_matches", len(cleaned_matches))

        #     if self.last_logger_clarification or len(cleaned_matches) > 1:
        #         self.last_logger_list = list(cleaned_matches)
        #         self.last_logger = None
        #         print("⚠️ Logger valid ditemukan lebih dari satu — menunggu klarifikasi")
        #         self.last_logger_clarification = {
        #             "ambiguous_input": self.latest_prompt,
        #             "candidates": self.last_logger_list
        #         }
        #     else:
        #         self.last_logger_list = list(cleaned_matches)
        #         self.last_logger = self.last_logger_list[0]
        #         print("📌 last_logger_list:", self.last_logger_list)
        #         print("📌 last_logger (terakhir):", self.last_logger)

        elif cleaned_matches:
            print("cleaned_matches :",cleaned_matches)
            print("panjang cleaned_matches", len(cleaned_matches))
            # ✅ Jika sebelumnya ada saran klarifikasi, jangan pilih dua sekaligus
            if self.last_logger_clarification:
                self.last_logger_list = []
                self.last_logger = None
                print("🛑 Logger valid ditemukan, tapi sedang menunggu klarifikasi — tidak disimpan")
            else:
                self.last_logger_list = list(cleaned_matches)
                self.last_logger = self.last_logger_list[0]
                print("📌 last_logger_list:", self.last_logger_list)
                print("📌 last_logger (terakhir):", self.last_logger)
        else:
            print("🚫 Tidak ada logger valid — mempertahankan logger sebelumnya.")
            self.last_logger_list = self.last_logger_list or []

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

    def _should_use_memory(self, prompt: str) -> bool:
        print("_should_use_memory telah berjalan HEHE")
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

    def update_last_logger_list(self, logger_list: List[str], source: str):
        self.last_logger_list = logger_list
        self.last_logger_source = source
        print(f"[MEMORY] last_logger_list diupdate dari source '{source}': {logger_list}")

    def update_last_logger_list_from_response(self, response_text: str):
        import re
        matches = re.findall(r"- (Pos [A-Z]{3} [\w\s]+)", response_text)
        if matches:
            self.last_logger_list = [m.lower() for m in matches]
            print(f"[MEMORY] Updated last_logger_list dari response: {self.last_logger_list}")

    def resolve_ambiguous_prompt_with_llm(self, user_messages: List[Dict], model_name: str = "llama3.1:8b") -> Tuple[str, bool]:
        print("\n🧠 resolve_ambiguous_prompt_with_llm berjalan...")

        # ✅ Jika tidak ada konteks percakapan sebelumnya (user_messages terlalu sedikit), kembalikan langsung
        if len(user_messages) <= 2:
            print("⚠️ Tidak ada konteks sebelumnya. Gunakan prompt terakhir langsung.")
            return user_messages[-1]["content"], False

        # Prompt untuk reasoning
        reasoning_prompt = (
            "Anda adalah asisten AI untuk sistem monitoring telemetri. "
            "Tugas Anda adalah menjawab pertanyaan terakhir dari pengguna berdasarkan **informasi yang telah disebutkan secara eksplisit** dalam riwayat percakapan sebelumnya. "
            "**Wajib periksa riwayat secara teliti dan jangan pernah mengarang, menyimpulkan, atau mengisi kekosongan dari asumsi.** "
            "Jika jawaban dapat ditemukan secara jelas dari data sebelumnya, berikan jawaban langsung (maksimal satu kalimat). "
            "Jika pertanyaan terakhir terlalu ambigu (misalnya 'ya', 'lanjutkan', 'jelaskan', 'ok', 'sip', 'oke', 'baiklah', 'iya?', 'betul?'), balas dengan: [AMBIGUOUS]. "
            "Jika informasi untuk menjawab tidak tersedia secara eksplisit di chat sebelumnya, balas dengan: [NO ANSWER]. "
            "Jawaban hanya boleh salah satu dari:\n"
            "- jawaban langsung (jika tersedia di chat sebelumnya),\n"
            "- [AMBIGUOUS], atau\n"
            "- [NO ANSWER]."
            "\n\nContoh:\n"
            "User: tampilkan data pos kali bawang\n"
            "Assistant: Pos tidak ditemukan. Apakah maksud Anda 'pos arr kali bawang'?\n"
            "User: iya?\n"
            "→ Maka Anda harus membalas: [AMBIGUOUS]\n\n"
            "User: tampilkan suhu udara tertinggi\n"
            "→ Jika suhu udara belum disebutkan di percakapan sebelumnya, maka Anda harus membalas: [NO ANSWER]"
            "**DILARANG KERAS MEMBERIKAN JAWABAN DILUAR [AMBIGUOUS] dan [NO ANSWER], JIKA JAWABANG BERADA DI LUAR ITU MAKA PILIH [NO ANSWER]** "
        )

        messages_for_llm = [{"role": "system", "content": reasoning_prompt}] + user_messages

        try:
            response = chat(messages=messages_for_llm, model=model_name)
            content = response['message']['content'] if isinstance(response, dict) else response.message.content
            content = content.strip()
            print("🧠 Jawaban awal dari LLM:", content)

            if content == "[AMBIGUOUS]":
                # Susun ulang pertanyaan eksplisit
                clarification_prompt = (
                    "Anda adalah asisten AI untuk sistem monitoring telemetri. "
                    "Gunakan konteks berdasarkan **informasi yang telah disebutkan secara eksplisit** dalam riwayat percakapan sebelumnya untuk menyusun ulang maksud pengguna secara eksplisit. "
                    "Tulis ulang dalam satu kalimat perintah eksplisit singkat, seperti 'Tampilkan suhu udara pos yang terpanas'. "
                    "Jawaban hanya boleh satu kalimat perintah yang jelas."
                )
                clarification_messages = [{"role": "system", "content": clarification_prompt}] + user_messages

                clarification_response = chat(messages=clarification_messages, model=model_name)
                clarified_text = clarification_response['message']['content'] if isinstance(clarification_response, dict) else clarification_response.message.content
                return clarified_text.strip(), False

            elif content == "[NO ANSWER]":
                return user_messages[-1]["content"], False

            else:
                return content, True

        except Exception as e:
            print("❌ Gagal memproses LLM:", e)
            return user_messages[-1]["content"], False
    
    def handle_dual_logger_selection(self, user_reply: str, candidates: List[str]) -> Optional[dict]:
        print("handle_dual_logger_selection sedang berjalan....")
        prompt = user_reply.strip().lower()
        print(f"Candidates : {candidates}")

        ambiguous_keywords = ["sapon", "kali bawang"]

        if len(candidates) != 2:
            print("❌ Jumlah kandidat bukan 2.")
            return None

        if not any(keyword in prompt or any(keyword in l.lower() for l in candidates) for keyword in ambiguous_keywords):
            print("⚠️ Lokasi bukan 'sapon' atau 'kali bawang'.")
            return None

        norm_cand_1 = candidates[0].lower()
        norm_cand_2 = candidates[1].lower()

        if any(
            phrase in prompt
            for phrase in ["ya", "yaa", "yaaa", "ya keduanya", "keduanya", "ya dua-duanya", "dua duanya"]
        ) and not (norm_cand_1 in prompt or norm_cand_2 in prompt):
            print("✅ User memilih kedua logger.")
            return {"confirmed": True, "logger": [candidates[0], candidates[1]]}

        if norm_cand_1 in prompt:
            print(f"✅ User memilih logger: {candidates[0]}")
            return {"confirmed": True, "logger": candidates[0]}

        if norm_cand_2 in prompt:
            print(f"✅ User memilih logger: {candidates[1]}")
            return {"confirmed": True, "logger": candidates[1]}

        print("⚠️ Klarifikasi tidak dikenali.")
        return None

    def extract_date_keywords_or_relative(self, text: str) -> str:
        """
        Deteksi frasa waktu eksplisit dan relatif dari teks.
        Jika ditemukan, simpan ke self.last_date dan return string hasilnya.
        """
        combined_text = text.lower().strip()

        # 1. Pola waktu relatif
        relative_date_patterns = [
            r"\d+\s+hari\s+(lalu|terakhir)",
            r"\d+\s+minggu\s+(lalu|terakhir)",
            r"\d+\s+bulan\s+(lalu|terakhir)",
            r"\d+\s+tahun\s+(lalu|terakhir)"
        ]
        for pattern in relative_date_patterns:
            match = re.search(pattern, combined_text)
            if match:
                self.last_date = match.group(0)
                print("🗓️ Deteksi tanggal (relatif):", self.last_date)
                return self.last_date

        # 2. Kata kunci eksplisit
        date_keywords = [
            "hari ini", "kemarin", "kemaren", "minggu ini", "minggu lalu",
            "bulan lalu", "awal bulan", "akhir bulan", "tahun lalu",
            "minggu terakhir", "bulan terakhir", "hari terakhir"
        ]
        for phrase in date_keywords:
            if phrase in combined_text:
                self.last_date = phrase
                print("🗓️ Deteksi tanggal (keyword):", phrase)
                return phrase

        print("❌ Tidak ada kata kunci tanggal terdeteksi.")
        return None
    # def handle_dual_logger_selection(self):
    #     print("handle_dual_logger_selection sedang berjalan....")
    #     """
    #     Tangani klarifikasi pengguna terhadap 2 logger ambigu khusus ['sapon', 'kali bawang']:
    #     - Jika user bilang 'ya keduanya' → ambil dua-duanya.
    #     - Jika menyebut satu → ambil yang disebut.
    #     - Jika tidak dikenali atau bukan lokasi ambigu → return None.
    #     """
    #     prompt = self.latest_prompt.strip().lower()
    #     candidates = self.last_logger_list
    #     print(f"Candidates : {candidates}")
    #     ambiguous_keywords = ["sapon", "kali bawang"]

    #     if len(candidates) != 2:
    #         print("❌ Jumlah kandidat bukan 2.")
    #         return None

    #     # Periksa apakah keyword ambigu ada di prompt atau logger
    #     if not any(keyword in prompt or any(keyword in l.lower() for l in candidates) for keyword in ambiguous_keywords):
    #         print("⚠️ Lokasi bukan 'sapon' atau 'kali bawang'.")
    #         return None

    #     norm_cand_1 = candidates[0].lower()
    #     norm_cand_2 = candidates[1].lower()
        
    #     # Case 1: User pilih keduanya
    #     if any(
    #         phrase in prompt
    #         for phrase in ["ya", "yaa", "yaaa", "ya keduanya", "keduanya", "ya dua-duanya", "dua duanya"]
    #     ) and not (norm_cand_1 in prompt or norm_cand_2 in prompt):
    #         print("✅ User memilih kedua logger.")
    #         return {"confirmed": True, "logger": [candidates[0], candidates[1]]}

    #     # Case 2: User menyebut salah satu logger
    #     if norm_cand_1 in prompt:
    #         print(f"✅ User memilih logger: {candidates[0]}")
    #         return {"confirmed": True, "logger": candidates[0]}

    #     if norm_cand_2 in prompt:
    #         print(f"✅ User memilih logger: {candidates[1]}")
    #         return {"confirmed": True, "logger": candidates[1]}

    #     print("⚠️ Klarifikasi tidak dikenali.")
    #     return None
    
    def extract_previous_prompt_for_reclassification(self, history: list) -> str:
        """
        Ambil prompt user sebelum klarifikasi logger jika formatnya:
        user → assistant → user (misal: 'keduanya')
        """
        if not history or len(history) < 3:
            print("⚠️ Chat history terlalu pendek.")
            return None

        # Ambil 3 pesan terakhir
        last_user = history[-1]
        assistant_before = history[-2]
        user_before_that = history[-3]

        if (
            last_user.get("role") == "user"
            and assistant_before.get("role") == "assistant"
            and user_before_that.get("role") == "user"
        ):
            print(f"🧠 Klasifikasi ulang menggunakan prompt sebelumnya: {user_before_that['content']}")
            return user_before_that["content"]

        print("⚠️ Format 3 pesan terakhir tidak sesuai untuk klasifikasi ulang.")
        return None


    def process_new_prompt(self, new_prompt: str) -> Dict:
        print("\nprocess_new_prompt sedang berjalan\n")

        payload = request.get_json(force=True)
        model_name = payload.get("model", "llama3.1:8b")
        user_messages = payload.get("messages", [])

        print("USER MESSAGES :\n", user_messages)
        print("\nLAST PROMPT", new_prompt)

        self.latest_prompt = new_prompt
        self.prompt_history.append(new_prompt)
        self.last_date = None  # reset date saat proses baru

        print("\nnew_prompt di function process_new_prompt", new_prompt)
        print(f"\nIntent di function procces_new_prompt adalah : {self.intent}")
        print(f"\nPos Logger terakhir {self.last_logger}")
        print(f"\nIntent terakhir adalah : {self.prev_intent}")

        # ✅ Jika new_prompt adalah hasil konfirmasi
        if isinstance(new_prompt, dict) and new_prompt.get("confirmed") is True:
            confirmed_logger = new_prompt["logger"]
            self.latest_prompt = new_prompt.get("prompt", "")
            self.last_logger_list = confirmed_logger if isinstance(confirmed_logger, list) else [confirmed_logger]
            self.last_logger = self.last_logger_list[0]
            self.last_logger_clarification = None

            # 🔁 Ambil prompt sebelumnya untuk referensi tanggal
            prev_prompt = self.extract_previous_prompt_for_reclassification(user_messages)

            # 🧠 Jika intent belum ada → klasifikasi intent
            if self.intent is None:
                if prev_prompt:
                    predicted_intent = self._predict_intent_bert(prev_prompt)
                    self.intent = predicted_intent
                    self.prev_intent = predicted_intent
                    print(f"🔁 Intent diklasifikasi ulang berdasarkan prompt sebelumnya: {predicted_intent}")
                else:
                    print("⚠️ Tidak ditemukan prompt sebelumnya untuk klasifikasi ulang.")

            # 📅 Isi tanggal jika intent mengandung _by_date dan belum ada tanggal
            if (self.intent and self.intent.endswith("_by_date") and self.last_date is None) or (
                self.intent in ["fetch_logger_by_date", "analyze_logger_by_date", "compare_logger_by_date"]
                and self.last_date is None
            ):
                if prev_prompt:
                    extracted_date = self.extract_date_keywords_or_relative(prev_prompt)
                    if extracted_date:
                        print(f"📅 Tanggal berhasil diekstrak dari prompt sebelumnya: {extracted_date}")
                        self.last_date = extracted_date
                    else:
                        print("⚠️ Tidak berhasil ekstrak tanggal dari prompt sebelumnya.")
                else:
                    print("⚠️ Tidak ada prev_prompt untuk ekstraksi tanggal.")

            target = self.last_logger_list

            print("🎯 Prompt hasil klarifikasi — langsung return.")
            print(f"[FINAL] Intent yang digunakan: {self.intent}, Target: {target}, Date: {self.last_date}")
            return {
                "intent": self.intent,
                "target": target,
                "date": self.last_date,
                "latest_prompt": self.latest_prompt,
                "logger_suggestions": self.logger_suggestions
            }

        # ✅ Jika tidak ada klarifikasi, coba deteksi prompt ambigu
        clarification_response = self.handle_confirmation_prompt_if_needed(self.latest_prompt)
        print("clarification_response adalah", clarification_response)
        if clarification_response:
            return clarification_response

        # ✅ Lanjutkan dengan proses normal (intent langsung terdeteksi)
        print("new_prompt adalah :", new_prompt)
        self._extract_context_memory(text=new_prompt)
        print("func _extract_context_memory telah selesai berjalan !")
        print(f"Prompt terbaru setelah _extract_context_memory adalah {self.latest_prompt}")
        print(f"self.last_logger adalah ini : {self.last_logger}")
        print(f"LIST dari self.last_logger adalah ini : {self.last_logger_list}")

        print(f"\n Intent baru {self.intent} dan intent lama adalah {self.prev_intent}")
        print(f"Intent Sebelumnya {self.prev_intent}, target sebelumnya {self.prev_target}, waktu sebelumnya {self.prev_date}")

        if self.intent not in ["ai_limitation", "unknown_intent"]:
            self.prev_intent = self.intent

        if self.last_logger:
            self.prev_logger = self.last_logger
            print(f"Nama pos sebelumnya adalah : {self.prev_logger}")

        if self.last_date:
            self.prev_date = self.last_date
            print(f"Tanggal sebelumnya adalah : {self.prev_date}")

        try:
            self.intent = self._predict_intent_bert(new_prompt)
        except Exception as e:
            print(f"[INTENT PREDICTION ERROR] {e}")
            self.intent = "unknown_intent"

        ambiguous_intents = ["ai_limitation", "unknown_intent"]
        effective_intent = self.intent
        # menaruh kondisi untuk handler disini atau

        if self.intent in ambiguous_intents and self.prev_intent in [
            "compare_parameter_across_loggers", "analyze_logger_by_date", 
            "compare_logger_by_date", "compare_logger_data", "show_logger_data"
        ]:
            print(f"\U0001f9e0 Menggunakan prev_intent karena intent saat ini ambigu: {self.intent}")
            effective_intent = self.prev_intent
            raw_targets = self.last_logger_list if self.last_logger_list else ([self.last_logger] if self.last_logger else [])
            print(f"IN AMBIGOUS CONDITION RAW TARGET IS : {raw_targets}")
            target = self._clean_logger_list(raw_targets)

        else: # kenapa harus pake kode  ini
            print("else atas berjalan....")
            print(f"last_logger_clarification : {self.last_logger_clarification}")
            # DEBUG Here
            print(f"Daftar logger yang teridentifikasi {self.last_logger_list}")
            if self.last_logger_clarification and not self.last_logger:
                print("\u26a0\ufe0f Sedang menunggu klarifikasi logger — target dikosongkan")
                self.last_logger_list = []  # ✅ Bersihkan agar tidak dipakai di handle_intent
                target = []

            elif self.intent in [
                "show_logger_data", "analyze_logger_by_date", "fetch_logger_by_date",
                "show_selected_parameter", "compare_parameter_across_loggers",
                "compare_logger_data", "compare_logger_by_date"
            ]:
                if self.last_logger_clarification and self.last_logger:
                    print(" Klarifikasi selesai — gunakan hanya satu logger hasil klarifikasi")
                    target = [self.last_logger]
                    self.last_logger_list = [self.last_logger]  # ✅ Sinkronisasi agar handle_intent aman
                    self.last_logger_clarification = None
                else:
                    raw_targets = self.last_logger_list if self.last_logger_list else ([self.last_logger] if self.last_logger else [])
                    print("PRINT RAW TARGET :",raw_targets) # PRINT RAW TARGET : ['pos arr hargorejo']
                    target = self._clean_logger_list(raw_targets)
                    print("PRINT _clean_logger_list TARGET :",target)
                    if len(target) > 1:
                        print(f" Ditemukan beberapa target: {target}, mengambil yang pertama saja.")
            else:
                print("Else berjalan ....")
                target = self.last_logger
        
        # tetap dibawah sini
        print(f"Nilai last_logger_list {self.last_logger_list} dan panjang data : {len(self.last_logger_list)}")
        # print(f"Klarifikasi Logger adalah : {self.last_logger_clarification['candidates']}") #IGNORE
        # get specific logger condition and prompt
        print(f"Prompt nya adalah : {self.latest_prompt}")
        print(f"Intentnya adalah {effective_intent}")
        # if (
        #     len(self.last_logger_list) == 2
        #     and any(k in self.latest_prompt.lower() for k in ["sapon", "kali bawang"])
        #     and all(
        #         any(kw in logger.lower() for kw in ["arr", "awlr"])
        #         for logger in self.last_logger_list
        #     )
        # ):
        #     print("if jalan") #
        #     result = self.handle_dual_logger_selection()
        #     if result:
        #         print(f"🎯 Logger yang dipilih: {result}")
        #         selected = result["logger"]
        #         if isinstance(selected, str):
        #             self.last_logger_list = [selected]
        #         elif isinstance(selected, list):
        #             print("selected :", selected)
        #             target = selected
        #             self.last_logger_list = selected # bagaimana mengoper variable ini langsung ke return yang ada dibawah sana, agar melewati if else lain
        #         # self.last_logger = self.last_logger_list[0]

        print(f"self.logger_suggestions adalah, {self.logger_suggestions}")
        self.intent = effective_intent
        print(f"[FINAL] Intent yang digunakan: {self.intent}, Target: {target}, Date: {self.last_date}")

        return {
            "intent": self.intent,
            "target": target,
            "date": self.last_date,
            "latest_prompt": self.latest_prompt,
            "logger_suggestions": self.logger_suggestions
        }

        # # Langkah 1: Daftar intent ambigu
        # ambiguous_intents = ["ai_limitation", "unknown_intent"]

        # # Langkah 2: Tentukan intent yang akan digunakan
        # effective_intent = self.intent
        # if self.intent in ambiguous_intents and self.prev_intent in [
        #     "compare_parameter_across_loggers", "analyze_logger_by_date", 
        #     "compare_logger_by_date", "compare_logger_data", "show_logger_data"
        # ]:
        #     print(f"🧠 Menggunakan prev_intent karena intent saat ini ambigu: {self.intent}")
        #     effective_intent = self.prev_intent
        #     # Target juga ambil dari sebelumnya
        #     raw_targets = self.last_logger_list if self.last_logger_list else ([self.last_logger] if self.last_logger else [])
        #     target = self._clean_logger_list(raw_targets)
        # else:
        #     # Jika intent valid, pakai target dari intent sekarang
        #     if self.intent in [
        #         "show_logger_data", "analyze_logger_by_date", "fetch_logger_by_date",
        #         "show_selected_parameter", "compare_parameter_across_loggers",
        #         "compare_logger_data", "compare_logger_by_date"
        #     ]:
        #         raw_targets = self.last_logger_list if self.last_logger_list else ([self.last_logger] if self.last_logger else [])
        #         print("raw_targets :", raw_targets)
        #         target = self._clean_logger_list(raw_targets)
        #     else:
        #         target = self.last_logger
        
        # self.intent = effective_intent  # ✅ force override agar digunakan juga oleh handle_intent()

        
        # if self.prev_intent in ["compare_parameter_across_loggers"]: # kode ini digunakan untuk mengambil konteks sebelumnya melalui intent sebelumnya, 
        #     print(f"self.prev_intent == {self.prev_intent} HEHE Previous Intent is Finally saved")
        #     raw_targets = self.last_logger_list if self.last_logger_list else ([self.last_logger] if self.last_logger else [])
        #     print("raw_targets :", raw_targets)
        #     target = self._clean_logger_list(raw_targets)

        # if self.intent in [
        #     "show_logger_data", "analyze_logger_by_date", "fetch_logger_by_date",
        #     "show_selected_parameter", "compare_parameter_across_loggers",
        #     "compare_logger_data", "compare_logger_by_date"
        # ]:
        #     print(f"self.intent == {self.intent} HEHE it works at least")
        #     raw_targets = self.last_logger_list if self.last_logger_list else ([self.last_logger] if self.last_logger else [])
        #     print("raw_targets :", raw_targets)
        #     target = self._clean_logger_list(raw_targets)
        # else:
        #     target = self.last_logger

        # print("\nDari function process_new_prompt untuk deteksi intent")
        # print("=================")
        # print(f" ✅ intent: {self.intent}, target: {target}, date: {self.last_date}")
        # print("\n")

        # return {
        #     "intent": self.intent,
        #     "target": target,
        #     "date": self.last_date,
        #     "latest_prompt": self.latest_prompt,
        #     "logger_suggestions": self.logger_suggestions if not target else {}
        # }

    def handle_direct_answer(self, answer_text: str) -> Dict:
        print("📤 Menangani jawaban langsung dari LLM")
        self.intent = "direct_answer"  # ✅ Simpan intent untuk digunakan oleh IntentManager
        self.analysis_result = answer_text
        return {
            "intent": "direct_answer",
            "target": None,
            "date": None,
            "latest_prompt": self.latest_prompt,
            "logger_suggestions": {},
            "response": answer_text
        }

    # def process_new_prompt(self, new_prompt: str) -> Dict:
    #     print("process_new_prompt sedang berjalan")
    #     # print(f"new_prompt : {new_prompt}")

    #     is_same_prompt = self.latest_prompt == new_prompt
    #     self.latest_prompt = new_prompt
    #     self.prompt_history.append(new_prompt)

    #     self.last_date = None

    #     try:
    #         predicted_intent = self._predict_intent_bert(new_prompt)
    #         print("predicted_intent :", predicted_intent)
    #         # Simpan intent sebelumnya hanya jika intent valid dan bukan "ai_limitation"
    #         if predicted_intent != "ai_limitation" and self.intent is not None:
    #             self.prev_intent = self.intent

    #         self.intent = predicted_intent

    #         print(f"self.intent {self.intent}")
    #         print(f"self.prev_intent {self.prev_intent}")

    #     except Exception as e:
    #         print(f"[INTENT PREDICTION ERROR] {e}")
    #         self.intent = "unknown_intent"

    #     print("new_prompt di function process_new_prompt", new_prompt)
    #     print(f"Intent di function procces_new_prompt adalah : {self.intent}")
    #     print(f"Pos Logger terakhir {self.last_logger}")

    #     if self._should_use_memory(new_prompt) and not self._contains_explicit_logger_or_date(new_prompt):
    #         print(f"DATA TERAKHIR adalah : {self.last_data}")
    #         print("Menggunakan Memory")
    #         context_window = self.get_context_window(window_size=4)
    #         self._extract_context_memory(text=" ".join([m["content"] for m in context_window if m["role"] == "user"]))
    #     else:
    #         self._extract_context_memory(text=new_prompt)

    #     if self.intent in [
    #         "show_logger_data", "analyze_logger_by_date", "fetch_logger_by_date",
    #         "show_selected_parameter", "compare_parameter_across_loggers", 
    #         "compare_logger_data", "compare_logger_by_date"
    #     ]:
    #         print(f"self.intent == {self.intent} HEHE it works at least ")
    #         raw_targets = self.last_logger_list if self.last_logger_list else ([self.last_logger] if self.last_logger else [])
    #         print("raw_targets :", raw_targets)
    #         target = self._clean_logger_list(raw_targets)
    #     # if self.intent in ['compare_parameter_across_loggers']:
    #     #     print("HEHE it works at least")
    #     #     print(f"Pos Logger terakhir {self.last_logger}")
    #     else:
    #         target = self.last_logger

    #     # # ✅ Tambahkan di sini
    #     # if (not target or target == []) and hasattr(self, "logger_suggestions"):
    #     #     logger_fallbacks = list(self.logger_suggestions.values())[0]
    #     #     print(f"[FALLBACK] Menggunakan saran logger: {logger_fallbacks}")
    #     #     target = logger_fallbacks

    #     # ✅ Return intent info + logger_suggestions
    #     print("\nDari function process_new_prompt untuk deteksi intent")
    #     print("=================")
    #     print(f" ✅ intent: {self.intent}, target: {target}, date: {self.last_date}")
    #     print("\n")

    #     return {
    #         "intent": self.intent,
    #         "target": target,
    #         "date": self.last_date,
    #         "latest_prompt": new_prompt,
    #         "logger_suggestions": self.logger_suggestions if not target else {}
    #     }
    
    def reset_memory_if_chat_too_long(self, max_user_messages: int = 4):
        user_msg_count = sum(1 for m in self.user_messages if m["role"] == "user")
        print(f"[INFO] Jumlah chat user: {user_msg_count}")

        if user_msg_count >= max_user_messages:
            print(f"🧹 Mereset memori karena user telah mengirim {user_msg_count} chat.")
            self.last_logger = None
            self.last_logger_id = None
            self.last_logger_list = []
            self.last_data = None
            self.last_date = None
            self.intent = None
            self.analysis_result = None
            self._save_user_memory()
            self._memory_was_reset = True
        else:
            self._memory_was_reset = False

    def memory_was_reset(self) -> bool:
        return getattr(self, "_memory_was_reset", False)

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
import asyncio
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
            "show_logger_data": self.fetch_latest_data, # safe id_logger, date_info, fetched data(before llm)
            "fetch_logger_by_date": self.fetch_data_range, # safe id_logger, date_info, fetched data(before llm)
            "fetch_status_rain": self.fetch_status_logger, # safe id_logger, fetched data(before llm)
            "show_list_logger": self.show_list_logger, # safe id_logger, date_info, fetched data(before llm)
            "compare_logger_by_date": self.compare_by_date, # safe id_logger, date_info, fetched data(before llm)
            "compare_logger_data": self.compare_latest_data, # safe id_logger, date_info, fetched data(before llm)
            "compare_parameter_across_loggers": self.compare_across_loggers, # safe id_logger, date_info, fetched data(before llm)
            "show_selected_parameter": self.show_selected_parameter, # safe id_logger, date_info, fetched data(before llm)
            "get_logger_photo_path": self.get_photo_path, # safe id_logger, date_info, fetched data(before llm)
            "how_it_works": self.explain_system, # safe id_logger, date_info, fetched data(before llm)
            "analyze_logger_by_date": self.analyze_by_date, # safe id_logger, date_info, fetched data(before llm)
            "ai_limitation": self.ai_limitation, # safe id_logger, date_info, fetched data(before llm) # abaikan 
            "show_online_logger" : self.connection_status, # safe id_logger, date_info, fetched data(before llm)
            "show_logger_info": self.show_logger_info,
            "direct_answer": self.direct_answer # abaikan
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
        target = self.memory.last_logger_list or [self.memory.last_logger] # [self.memory.last_logger] # 
        date = self.memory.last_date

        print("func handle_intent telah berjalan.....")
        print(f"Dari Prompt {prompt} Intent adalah : {intent}, target logger adalah : {target}, tanggal yang dicari adalah : {date}")
        func = self.intent_map.get(intent, self.fallback_response)
        return func()
    

    def fetch_latest_data(self):
        print("intent show_logger_data ini telah berjalan")
        model_name = "llama3.1:8b"
        print("Intent Sebelumnya :", self.memory.prev_intent)
        prompt = self.memory.latest_prompt.lower()
        intent = self.memory.intent
        # target_loggers = [self.memory.last_logger] 
        target_loggers = self.memory.last_logger_list or [self.memory.last_logger]
        logger_list = fetch_list_logger()

        print(f"Dari Prompt {prompt} Intent adalah : {intent}, target logger adalah : {target_loggers}")
        print("Data Sebelumnya :", self.memory.last_data)

        if target_loggers == [None]:
            return "Maaf nama lokasi tidak diketahui, Silahkan memberikan nama lokasi logger"

        if not target_loggers or not logger_list:
            return "Target logger atau daftar logger tidak tersedia."
        
        # 🔍 Cek apakah ada summary sebelumnya dari LLM di history
        def _find_last_logger_response(messages):
            for msg in reversed(messages):
                if msg["role"] == "assistant" and "Data Monitoring Telemetri" in msg["content"]:
                    return msg["content"]
            return None

        # if self.memory.last_data != None :
        #     print("Data yang disimpan", self.memory.last_data)
        # if self.memory.last_data == None :
        #     print("Tidak Menyimpan data")
        # Ambil user_messages dari request
        payload = request.get_json(force=True)
        user_messages = payload.get("messages", [])

        # === Deteksi parameter yang diminta dari prompt
        matched_parameters = []
        for param, aliases in sensor_aliases.items():
            for alias in aliases:
                if alias in prompt:
                    matched_parameters.append(param)
                    break
        print("Matched Parameters:", matched_parameters)
        # 🔁 Cek apakah prompt saat ini hanya ingin parameter tertentu dari summary sebelumnya
        latest_summary = _find_last_logger_response(user_messages)
        if latest_summary and matched_parameters:
            print("✅ Menggunakan summary sebelumnya untuk menjawab permintaan parameter tertentu.")

            extracted_lines = []
            for param in matched_parameters:
                pattern = rf"\|\s*{param}\s*\|\s*([^|]+)\s*\|"
                import re
                match = re.search(pattern, latest_summary, re.IGNORECASE)
                if match:
                    extracted_lines.append(f"**{param}:** {match.group(1).strip()}")
                else:
                    extracted_lines.append(f"**{param}:** Tidak ditemukan dalam ringkasan sebelumnya.")

            extracted_text = "\n".join(extracted_lines)

            user_prompt = (
                "Berikut ini adalah nilai hasil ekstraksi parameter dari ringkasan data logger sebelumnya:\n\n"
                f"{extracted_text}\n\n"
                "Tolong berikan interpretasi atau penjelasan singkat untuk masing-masing parameter. "
                "Gunakan format seperti: *Nilai [parameter] [angka dan satuan] menunjukkan bahwa ...*. "
                "Jelaskan dalam konteks umum, apakah nilainya rendah, sedang, atau tinggi, dan bagaimana kondisi tersebut dapat dipahami oleh orang awam. "
                "Jawaban Anda cukup dalam **satu kalimat saja per parameter**."
            )
            messages = [
                {"role": "system", "content": "Anda adalah asisten telemetri pintar. "
                "Jawaban Anda **harus hanya berdasarkan informasi eksplisit yang diberikan**. "
                "Jika data tidak tersedia, katakan dengan sopan — jangan buat asumsi atau menebak-nebak."
            },
                {"role": "user", "content": user_prompt}
            ]
            response = chat(model=model_name, messages=messages)
            return response["message"]["content"]
        # === Fetch data terbaru
        fetched = find_and_fetch_latest_data(target_loggers, matched_parameters, logger_list)

        summaries = []
        # last_data_buffer = []

        for item in fetched:
            nama_lokasi = item['logger_name']
            data = item['data']

            # Cek apakah data mengandung parameter yang diminta
            data_keys = set()
            if isinstance(data, dict):
                data_keys = set(data.keys())
            elif isinstance(data, list) and len(data) > 0:
                data_keys = set(data[0].keys())
            
            # Apakah semua matched_parameters ada di data_keys? Kalau tidak, buat pesan khusus
            missing_params = [p for p in matched_parameters if p not in data_keys]

            if missing_params:
                missing_param_str = ', '.join(missing_params)
                user_prompt = (
                    f"Tolong jawab permintaan data logger dari pengguna sesuai konteks prompt{prompt}.\n\n"
                    f"Namun, parameter berikut tidak ditemukan dalam data logger dari lokasi **{nama_lokasi}**:\n"
                    f"- {missing_param_str}\n\n"
                    "Mohon sampaikan bahwa data tersebut tidak tersedia saat ini, dan tidak perlu menampilkan data lain jika tidak relevan.\n"
                    "Gunakan format markdown dan awali dengan nama lokasi sebagai judul."
                )

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Anda adalah asisten telemetri pintar. Tugas Anda adalah memberi tahu pengguna secara sopan jika parameter "
                            "yang mereka minta tidak tersedia, tanpa membuat ringkasan yang tidak relevan."
                        )
                    },
                    {"role": "user", "content": user_prompt}
                ]

                response = chat(model=model_name, messages=messages)
                summaries.append(response["message"]["content"])
            else:
                user_prompt = (
                    f"Tampilkan data lengkap dari logger **{nama_lokasi}** dalam format markdown.\n\n"
                    f"Berikut adalah semua parameter yang tersedia:\n\n"
                    f"{data}\n\n"
                    "Berikan kesimpulan dalam satu paragraf pendek dari data logger berikut. \n"
                    "Analisis tren parameter yang tersedia (jika terlihat), sebutkan nilai tertinggi dan terendah, dan jelaskan apakah nilainya termasuk kategori rendah, sedang, atau tinggi berdasarkan konteks umum. \n"
                    "Cukup tampilkan semua data dalam urutan seperti yang diberikan.\n"
                    f"Awali dengan judul: **Data Monitoring Telemetri {nama_lokasi}**\n"
                    "Tambahkan garis pemisah '=====' di bawah judul."
                )

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Anda adalah asisten telemetri pintar.\n"
                            "Tugas Anda adalah memberikan satu ringkasan berdasarkan data logger yang tersedia.\n"
                            "Ringkasan harus objektif, dan relevan — hindari penjelasan berlebihan."
                        )
                    },
                    {"role": "user", "content": user_prompt}
                ]
                response = chat(model=model_name, messages=messages, options={"num_predict": 1024})
                summaries.append(response["message"]["content"])
            

                # Ada data parameter, buat ringkasan normal
                # summary = summarize_logger_data(nama_lokasi, data)
                # summaries.append(summary)

            #  2 code dibawah ini sudah saya tambahkan ke _init_(), 
            # bagaimana menyesuaikan 2 baris dibawah ini ke dalam function ini untuk menyimpan nama logger
            # self.last_logger_data: Dict[str, Dict] = {}
            # self.last_logger_ids: List[str] = []

            # ✅ Simpan hanya 1 logger (yang pertama) ke memory
            if item.get("logger_id"): # not self.memory.last_logger_id and 
                self.memory.last_logger_id = item["logger_id"]
                self.memory.last_logger = nama_lokasi
                self.memory.last_data = data  # Simpan hanya data mentah untuk analisa lanjutan
                self.memory.analysis_result = None  # Reset analisis jika sebelumnya ada

                print(f"✅ Disimpan ke memory: {self.memory.intent} / {self.memory.last_logger_id} / {self.memory.last_logger}")
                print("DATA YANG TERSIMPAN ADALAH : ",self.memory.last_data)
                
        self.memory._save_user_memory()

        return "\n\n---\n\n".join(summaries)

    # def fetch_latest_data(self):
    #     print("show_logger_data ini telah berjalan")
    #     prompt = self.memory.latest_prompt.lower()
    #     data_last = self.memory.last_data

    #     print(f"Data Terakhir adalah : {data_last}")
    #     print("self.memory.last_logger_list ", self.memory.last_logger_list)
    #     print(f"Type data dari self.memory.last_logger_list adalah {type(self.memory.last_logger_list)}")
    #     print(f"Panjang data dari self.memory.last_logger_list adalah {len(self.memory.last_logger_list)}")
    #     print("self.memory.last_logger ", self.memory.last_logger)

    #     target_loggers = self.memory.last_logger_list or [self.memory.last_logger]
    #     logger_list = fetch_list_logger()

    #     if not target_loggers or not logger_list:
    #         return "Target logger atau daftar logger tidak tersedia."

    #     # === DETEKSI PARAMETER YANG DIMINTA DI DALAM PROMPT ===
    #     matched_parameters = []
    #     for param, aliases in sensor_aliases.items():
    #         for alias in aliases:
    #             if alias in prompt:
    #                 matched_parameters.append(param)
    #                 break

    #     print("Matched Parameters:", matched_parameters)
    #     print("Matched Parameters Type:", type(matched_parameters))
    #     print("Matched Parameters Length:", len(matched_parameters))

    #     # === FETCH DATA TERBARU ===
    #     fetched = find_and_fetch_latest_data(target_loggers, matched_parameters, logger_list)
    #     print("fetched data adalah :", fetched)

    #     summaries = []
    #     for item in fetched:
    #         nama_lokasi = item['logger_name']
    #         data = item['data']
    #         summary = summarize_logger_data(nama_lokasi, data)
    #         summaries.append(summary)

    #     return "\n\n---\n\n".join(summaries)

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

    #     if not fetched:
    #         return "Tidak ditemukan data untuk logger yang disebutkan."

    #     for item in fetched:
    #         print(f"\n📍 {item['logger_name']}")
    #         for key, value in item['data'].items():
    #             print(f"{key}: {value}")

    #     return f"Berhasil mengambil data terbaru dari {len(fetched)} logger."
    
    def fetch_data_range(self):
        print("fetch_logger_by_date ini telah berjalan") # fetch_logger_by_date ini telah berjalan
        model_name = "llama3.1:8b"
        prompt = self.memory.latest_prompt
        date = self.memory.last_date
        print("prompt", prompt) # prompt ya
        target_loggers = self.memory.last_logger_list or [self.memory.last_logger]
        print("target_loggers", target_loggers) # target_loggers ['Pos ARR Sapon', 'Pos AWLR Sapon']
        print("Tanggal adalah : ",date) # Tanggal adalah :  2 hari terakhir

        def generate_return_response(user_prompt: str, fallback_message: str) -> str:
            # Ambil struktur: judul, bullet list, dan format markdown dari user_prompt
            # Gantikan kontennya dengan fallback_message namun tetap menjaga format

            lines = user_prompt.strip().splitlines()
            header_line = next((line for line in lines if "**" in line), "**Informasi Telemetri**")
            separator_line = "=====" if any("=====" in line for line in lines) else ""

            # Gabungkan dalam struktur yang sama
            response = f"{header_line}\n{separator_line}\n\n{fallback_message}"
            return response


        # ✅ Filter None dari list
        target_loggers = [t for t in target_loggers if t is not None]
        return_response = (
                "Maaf sistem tidak menemukan pos logger aktif dalam memori.\n\n"
                "Hal ini bisa terjadi karena:\n"
                "1. Anda belum menyebutkan nama pos secara eksplisit\n"
                "2. Memori sistem telah direset karena sesi terlalu panjang\n\n"
                "Silakan ketik ulang permintaan Anda dengan menyebutkan nama pos.\n"
                "Contoh:\n"
                "- `berikan data pos arr ngawen kemarin`\n"
                "- `tampilkan data pos awlr bunder 2 hari terakhir`"
            )
        # ❗Jika tidak ada target valid, berikan pesan penjelasan
        if not target_loggers:
            # Gunakan user_prompt sebagai template format
            formatted_response = generate_return_response(user_prompt, return_response)
            return formatted_response

        logger_list = fetch_list_logger()

        # === Ekstraksi tanggal dari prompt
        date_info = extract_date_structured(date)
        print("date_info", date_info)
        if date_info.get("awal_tanggal") == [None]:
            print("Mohon maaf, tanggal yang Anda masukkan belum dapat dikenali. Silakan berikan tanggal secara lengkap dengan format tahun-bulan-tanggal")
        
        # Simpan tanggal ke memory
        if date_info.get("awal_tanggal") or date_info.get("akhir_tanggal"):
            self.memory.last_date = f"{date_info.get('awal_tanggal')} s/d {date_info.get('akhir_tanggal')}"
            self.memory._save_user_memory()

        # === Deteksi parameter dari prompt
        matched_parameters = []
        for param, aliases in sensor_aliases.items():
            for alias in aliases:
                if alias in prompt:
                    matched_parameters.append(param)
                    break

        print(f"Type Data dari : {type(date_info)}")
        print("matched_parameters:", matched_parameters)

        # === Ambil data dari original fetch
        fetched_data = original_fetch_data_range(
            date_info = date_info,
            target_loggers=target_loggers,
            matched_parameters=matched_parameters,
            logger_list=logger_list
        )
        # fetched_data = asyncio.run( original_fetch_data_range_async(
        #         date_info=date_info,
        #         target_loggers=target_loggers,
        #         matched_parameters=matched_parameters,
        #         logger_list=logger_list
        #     )
        # )
        # print(f"target_loggers {target_loggers.values}")
        #  Oper ke LLM
        summaries = []
        user_prompt = (
           f"Tampilkan data lengkap dari logger secara rapi dan tersusun.\n\n"
            f"Berikut adalah semua parameter yang tersedia:\n\n"
            f"{fetched_data}\n\n"
            "Berikan kesimpulan dalam satu paragraf pendek dari data logger berikut. \n"
            "Analisis tren parameter yang tersedia (jika terlihat), sebutkan nilai tertinggi dan terendah, dan jelaskan apakah nilainya termasuk kategori rendah, sedang, atau tinggi berdasarkan konteks umum. \n"
            "Cukup tampilkan semua data dalam urutan seperti yang diberikan.\n"
            f"Awali dengan judul: **Data Monitoring Telemetri **\n"
            "Tambahkan garis pemisah '=====' di bawah judul."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Anda adalah asisten telemetri pintar.\n"
                    "Tugas Anda adalah memberikan satu ringkasan berdasarkan data logger yang tersedia.\n"
                    "Ringkasan harus objektif, dan relevan — hindari penjelasan berlebihan."
                )
            },
            {"role": "user", "content": user_prompt}
        ]
        response = chat(model=model_name, messages=messages, options={"num_predict": 1024})
        summaries.append(response["message"]["content"])

        # === Simpan ke memory jika berhasil (list of summaries)
        if isinstance(fetched_data, str) and "Tidak ditemukan" in fetched_data:
            print("[INFO] Data tidak ditemukan, tidak menyimpan ke memory")
            return fetched_data

        if isinstance(fetched_data, list):
            # Kalau original_fetch_data_range kamu return list (data mentah)
            first_data = fetched_data[0] if fetched_data else None
            if first_data and isinstance(first_data, dict):
                self.memory.last_logger = first_data.get("logger_name")
                self.memory.last_logger_id = first_data.get("logger_id")
                self.memory.last_data = first_data.get("data")
                self.memory.analysis_result = None  # reset analisa
                self.memory._save_user_memory()
                print(f"✅ Disimpan ke memory: fetch_logger_by_date / {self.memory.last_logger_id} / {self.memory.last_logger}")

        elif isinstance(fetched_data, str):
            # Jika kamu return string (ringkasan markdown), tetap bisa tampilkan
            print("Ringkasan string (markdown):", fetched_data)

        return "\n\n---\n\n".join(summaries)


    # def fetch_data_range(self):
    #     print("fetch_logger_by_date ini telah berjalan")
    #     prompt = self.memory.latest_prompt
    #     print("prompt", prompt)

    #     # === Ambil target logger
    #     target_loggers = self.memory.last_logger_list or [self.memory.last_logger]
    #     print("target_loggers", target_loggers)
    #     logger_list = fetch_list_logger()

    #     # === Ekstraksi tanggal dari prompt
    #     date_info = extract_date_structured(prompt)
    #     print("date_info", date_info)

    #     # === Validasi: Tidak boleh lebih dari 2 minggu lalu
    #     # if date_info["awal_tanggal"]:
    #     #     start_date = datetime.strptime(date_info["awal_tanggal"], "%Y-%m-%d")
    #     #     today = datetime.today()
    #     #     max_allowed_start = today - timedelta(days=14)

    #     #     if start_date < max_allowed_start:
    #     #         return (
    #     #             "⚠️ Permintaan data terlalu lama. "
    #     #             "Sistem hanya mengizinkan pengambilan data maksimal 2 minggu ke belakang dari hari ini.\n"
    #     #             f"Silakan ubah rentang tanggal menjadi setelah {max_allowed_start.strftime('%Y-%m-%d')}."
    #     #         )

    #      # === Deteksi parameter dari prompt (sensor yang ingin ditampilkan)
    #     matched_parameters = []
    #     for param, aliases in sensor_aliases.items():
    #         for alias in aliases:
    #             if alias in prompt:
    #                 matched_parameters.append(param)
    #                 break
    #     print("matched_parameters:", matched_parameters)

    #     # === Teruskan jika valid
    #     return original_fetch_data_range(
    #         prompt=prompt,
    #         target_loggers=target_loggers,
    #         matched_parameters=matched_parameters,  # <-- tambahkan ini ke original_fetch_data_range
    #         logger_list=logger_list
    #     )

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
        print("compare_latest_data ini telah berjalan")
        prompt = self.memory.latest_prompt
        target_loggers = self.memory.last_logger_list or [self.memory.last_logger]
        logger_list = fetch_list_logger()
        matched_parameters = []
        fetched = find_and_fetch_latest_data(target_loggers, matched_parameters, logger_list)

        print(f"Fetched Data adalah : {fetched}")

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
        print(f"compare_across_loggers prompt {prompt}")
        logger_list = fetch_list_logger()
        print(f"Type data logger list adalah : {type(logger_list)}")
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

        # Tentukan nama lokasi logger
        name_fragments = [logger["nama_lokasi"] for logger in logger_list]
        matched_parameters = [selected_param]  # list parameter untuk filter

        # Deteksi rentang tanggal
        date_info = extract_date_structured(prompt)
        print("Extracted date_info:", date_info)

        # Jika tidak ada tanggal, ambil data terkini dari logger
        if not date_info.get("awal_tanggal") or not date_info.get("akhir_tanggal"):
            fetched = find_and_fetch_latest_data(
                name_list=name_fragments,
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

        # Gunakan original_fetch_data_range untuk fetch data dengan deteksi waktu di prompt
        fetched_data = original_fetch_data_range(
            prompt=prompt,
            target_loggers=name_fragments,
            matched_parameters=matched_parameters,
            logger_list=logger_list
        )

        # Karena original_fetch_data_range mengembalikan string summary,
        # kita langsung gunakan untuk konteks chat
        system_prompt = (
            "Kamu adalah asisten cerdas yang menjawab pertanyaan berdasarkan data logger cuaca. "
            "Jika pengguna menanyakan yang berkaitan dengan kondisi ekstrem seperti 'terdingin', 'paling panas', 'terbasah', 'paling', 'tertinggi', atau 'terendah', "
            "berikan jawaban hanya untuk satu pos dengan nilai ekstrem tersebut beserta nama pos dan nilai parameternya. "
            "Untuk permintaan lain, berikan jawaban yang jelas dan sesuai dengan semua data pos yang relevan. "
        )

        response = chat(
            model='llama3.1:8b',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{self.memory.latest_prompt}\n\nBerikut data yang tersedia:\n{fetched_data}"}
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
            matched_parameters=matched_parameters,
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
        """
        Fungsi untuk membatasi jawaban model agar hanya dalam konteks STESY.
        Menggunakan prompt terakhir dan riwayat chat dari self.memory.
        """
        print("🔒 func ai_limitation dijalankan untuk membatasi respons ke konteks telemetri")
        model_name = "llama3.1:8b"
        prompt = self.memory.latest_prompt or ""
        print(f":: Prompt: {prompt}")

        # 1. Prompt sistem yang diperkuat
        messages_llm = [
            {
                "role": "system",
                "content": (
                    "Anda adalah STESY, AI virtual assistant untuk Smart Telemetry Systems (STESY).\n\n"
                    "Anda **hanya** menjawab pertanyaan yang terkait dengan topik-topik berikut:\n"
                    "- Telemetri\n- Hidrologi\n- Sungai\n- Cuaca\n- Klimatologi\n- Analisis data logger STESY\n\n"
                    "Jika pertanyaan **di luar konteks** (contohnya tentang teori fisika, sejarah, AI, matematika, atau lainnya), "
                    "tolak dengan sopan dan jelas. Jangan jawab isinya, tapi sampaikan penolakan secara singkat, walaupun Anda tahu jawabannya.\n\n"
                    "Topik yang harus **DITOLAK** meliputi:\n"
                    "- Fisika, Kimia, Matematika, Biologi, Sejarah, Teknologi Umum, Kesehatan, Agama, dll\n\n"
                    "Jawab HANYA jika pertanyaan berkaitan langsung dengan:\n"
                    "- Lokasi pos logger STESY\n- Data suhu, curah hujan, kelembaban, tekanan, angin\n"
                    "- Perbandingan dan evaluasi parameter logger\n- Cuaca lokal dan kondisi sungai berdasarkan logger\n\n"
                    "Contoh jawaban penolakan:\n"
                    "🙏 Maaf, saya hanya bisa menjawab topik terkait data telemetri dan lingkungan. Silakan ajukan pertanyaan lain dalam konteks tersebut.\n"
                )
            }
        ]

        # 2. Ambil history terakhir dari memori
        chat_history = self.memory.get_context_window(window_size=4) or []
        print(f"📜 Chat history: {chat_history}")
        print(f"Panjang data dari chat_history adalah : {len(chat_history)}")
        messages_llm.extend(chat_history)

        # Tambahkan prompt user
        messages_llm.append({"role": "user", "content": prompt})

        print(f"Type data dari messages_llm adalah : {type(messages_llm)}")
        print(f"Panjang data dari messages_llm adalah : {len(messages_llm)}")

        # 3. Jalankan model dengan seluruh konteks
        try:
            response = chat(model=model_name, messages=messages_llm)
            content = response.get("message", {}).get("content", "")
            print(f"jenis response adalah : {response}")
            print(f"📜 response Chat: {content}")
            content = (content or "").strip()
            print(f"Content adalah : {content}")

            # 🔐 Fallback jika content kosong
            if not content:
                print("⚠️   Model tidak memberikan respons. Fallback digunakan.")
                fallback_text = (
                    "📌 *Sistem STESY hanya menjawab dalam konteks berikut:*\n"
                    "- Telemetri\n- Hidrologi\n- Sungai\n- Cuaca\n- Klimatologi\n- Analisis data logger\n\n"
                    "🙏 *Pertanyaan Anda tampaknya di luar topik tersebut.*\n"
                    "Silakan ajukan pertanyaan yang relevan dengan sistem telemetri atau lingkungan. 🌿"
                )
                cont = f"{fallback_text}\n\n{prompt}"
                formatted_response = general_stesy(
                    [{"role": "user", "content": cont}], 
                    model_name=model_name
                )
                return formatted_response

            return content

        except Exception as e:
            print(f"❌ Terjadi error saat menjalankan model: {e}")
            return (
                "⚠️ Sistem STESY mengalami gangguan saat menjawab pertanyaan Anda. "
                "Pastikan pertanyaan Anda sesuai topik seperti data logger atau cuaca."
            )
    # def ai_limitation(self):
    #     """
    #     Fungsi untuk membatasi jawaban model agar hanya dalam konteks STESY.
    #     Menggunakan prompt terakhir dan riwayat chat dari self.memory.
    #     """
    #     print("🔒 func ai_limitation dijalankan untuk membatasi respons ke konteks telemetri")
    #     model_name = "llama3.1:8b"
    #     prompt = self.memory.latest_prompt
    #     print(f":: Prompt: {prompt}")
        
    #     # 1. Prompt sistem yang diperkuat
    #     messages_llm = [
    #         {
    #             "role": "system",
    #             "content": (
    #                 "Anda adalah STESY, AI virtual assistant untuk Smart Telemetry Systems (STESY).\n\n"
    #                 "Anda **hanya** menjawab pertanyaan yang terkait dengan topik-topik berikut:\n"
    #                 "- Telemetri\n"
    #                 "- Hidrologi\n"
    #                 "- Sungai\n"
    #                 "- Cuaca\n"
    #                 "- Klimatologi\n"
    #                 "- Analisis data logger STESY\n\n"
    #                 "Jika pertanyaan **di luar konteks**(contohnya tentang teori fisika, sejarah, AI, matematika, atau lainnya), tolak dengan sopan dan jelas. Jangan jawab isinya, tapi sampaikan penolakan secara singkat, walaupun Anda tahu jawabannya.\n\n"
    #                 "Topik yang harus **DITOLAK** meliputi (namun tidak terbatas pada):\n"
    #                 "- Fisika (teori Archimedes, hukum Newton, dll)\n"
    #                 "- Kimia (reaksi, unsur, senyawa, tabel periodik)\n"
    #                 "- Matematika (aljabar, kalkulus, statistik, geometri)\n"
    #                 "- Biologi (anatomi, sistem organ, klasifikasi makhluk hidup)\n"
    #                 "- Geografi umum (benua, geologi, pegunungan, iklim dunia)\n"
    #                 "- Sejarah & tokoh sejarah\n"
    #                 "- Teknologi umum & komputer\n"
    #                 "- Kesehatan & medis\n"
    #                 "- Agama, etika, filsafat\n"
    #                 "- Ekonomi, politik, hukum\n"
    #                 "- Hiburan (film, musik, novel, anime)\n"
    #                 "- Pendidikan, tips belajar, psikologi\n\n"
    #                 "Jawab HANYA jika pertanyaan berkaitan langsung dengan:\n"
    #                 "- Lokasi pos logger STESY\n"
    #                 "- Data suhu, curah hujan, kelembaban, tekanan, angin\n"
    #                 "- Perbandingan dan evaluasi parameter logger\n"
    #                 "- Cuaca lokal dan kondisi sungai berdasarkan logger\n\n"
    #                 "Contoh pertanyaan yang valid:\n"
    #                 "- \"Berikan suhu udara di pos ARR Sapon kemarin\"\n"
    #                 "- \"Bandingkan curah hujan di pos Beji dan pos Tegal\"\n"
    #                 "- \"Bagaimana analisis tekanan udara minggu lalu di ARR Kaliurang\"\n\n"
    #                 "Contoh pertanyaan yang harus ditolak:\n"
    #                 "- \"Apa itu teori Archimedes?\"\n"
    #                 "- \"Siapa presiden Indonesia pertama?\"\n"
    #                 "- \"Berapa 7 dikali 8?\"\n\n"
    #                 "Contoh jawaban penolakan:\n"
    #                 "\"🙏 Maaf, saya hanya bisa menjawab topik terkait data telemetri dan lingkungan. Silakan ajukan pertanyaan lain dalam konteks tersebut.\"\n"
    #             )
    #         }
    #     ]

    #     # 2. Ambil history terakhir dari memori
    #     chat_history = self.memory.get_context_window(window_size=4)
    #     print(f"📜 Chat history: {chat_history}")
    #     print(f"Panjang data dari chat_history adalah : {len(chat_history)}")
    #     messages_llm.extend(chat_history)
        
    #     print(f"Type data dari messages_llm adalah : {type(messages_llm)}")
    #     print(f"Panjang data dari messages_llm adalah : {len(messages_llm)}")

    #     # 3. Jalankan model dengan seluruh konteks
    #     try:
    #         response = chat(model=model_name, messages=messages_llm)
            
    #         content = response.get("message", {}).get("content", "").strip()
    #         print(f"jenis response adalah : {response}")
    #         print(f"📜 response Chat: {response.get('message', {}).get('content', '')}")
    #         print(f"Content adalah : {content}")


    #         # 🔐 Fallback jika content kosong
    #         if not content:
    #             print("⚠️   Model tidak memberikan respons. Fallback digunakan.")

    #             fallback_text = (
    #                 "📌 *Sistem STESY hanya menjawab dalam konteks berikut:*\n"
    #                 "- Telemetri\n- Hidrologi\n- Sungai\n- Cuaca\n- Klimatologi\n- Analisis data logger\n\n"
    #                 "🙏 *Pertanyaan Anda tampaknya di luar topik tersebut.*\n"
    #                 "Silakan ajukan pertanyaan yang relevan dengan sistem telemetri atau lingkungan. 🌿"
    #             )
    #             cont = fallback_text + "\n\n" + prompt 
    #             # 🔁 Kirim fallback_text ke LLM agar diformat lebih alami
    #             formatted_response = general_stesy(
    #                 [{"role": "user", "content": cont}], 
    #                 model_name=model_name
    #             )
    #             return formatted_response

    #     except Exception as e:
    #         print(f"❌ Terjadi error saat menjalankan model: {e}")
    #         return (
    #             "⚠️ Sistem STESY mengalami gangguan saat menjawab pertanyaan Anda. "
    #             "Pastikan pertanyaan Anda sesuai topik seperti data logger atau cuaca."
    #         )



    # def ai_limitation(self):
    #     print("func ai_limitation telah berjalan untuk merespon di luar cakupan")
    #     model_name = "llama3.1:8b"

    #     # Ambil prompt terakhir dari user
    #     prompt = self.memory.latest_prompt
    #     print(f"Latest prompt: {prompt}")

    #     # 🔍 Jika tidak ada target maupun tanggal → arahkan user dengan bantuan LLM
    #     if not self.memory.last_logger and not self.memory.last_date:
    #         print("⚠️ Tidak ditemukan nama logger maupun tanggal — arahkan user dengan contoh pertanyaan.")

    #         messages_llm = [
    #             {
    #                 "role": "system",
    #                 "content": (
    #                     f"Anda adalah STESY, AI assistant untuk Smart Telemetry Systems.\n"
    #                     f"Jika pengguna tidak menyebutkan nama pos maupun tanggal, bantu arahkan dengan sopan.\n"
    #                     f"Berikan contoh pertanyaan, gunakan markdown dan emoji jika perlu."
    #                 )
    #             },
    #             {
    #                 "role": "user",
    #                 "content": (
    #                     f"Pengguna hanya memberikan permintaan umum seperti 'tampilkan datanya', 'saya ingin lihat info', tanpa nama pos atau tanggal.\n\n"
    #                     f"Berikut adalah contoh pertanyaan yang sesuai:\n"
    #                     f"1. Tampilkan suhu udara di Pos ARR Gemawang kemarin.\n"
    #                     f"2. Bagaimana kondisi kelembaban di AWLR Kaliurang minggu lalu?\n"
    #                     f"3. Di mana pos dengan curah hujan tertinggi hari ini?"
    #                 )
    #             }
    #         ]

    #         response = chat(model=model_name, messages=messages_llm)
    #         return response["message"]["content"]

    #     # Jika target atau tanggal ada, tetap gunakan konteks dari chat history
    #     full_history = self.memory.get_context_window(window_size=4)
    #     print(f"📜 Chat history untuk ai_limitation: {full_history}")

    #     messages_llm = [{
    #         "role": "system",
    #         "content": (
    #             "Anda adalah STESY, AI virtual assistant untuk Smart Telemetry Systems.\n"
    #             "Anda hanya menjawab pertanyaan tentang:\n"
    #             "- Telemetri\n- Hidrologi\n- Sungai\n- Cuaca\n- Klimatologi\n- Analisis data logger\n\n"
    #             "Jika pertanyaan tidak relevan atau ambigu, jawab dengan sopan dan arahkan pengguna.\n"
    #             "Jika pengguna hanya mengucapkan 'ok', 'terima kasih', atau 'iya', cukup balas ringkas dan ramah."
    #         )
    #     }]

    #     # Tambahkan history user + asisten (jika ada)
    #     for msg in full_history:
    #         messages_llm.append({
    #             "role": msg["role"],
    #             "content": msg["content"]
    #         })
    #     print(f"HEHE ayo masuk {prompt}")
    #     # Tambahkan prompt terbaru jika belum ada
    #     if not any(m["content"] == prompt for m in messages_llm if m["role"] == "user"):
    #         messages_llm.append({"role": "user", "content": prompt})

    #     # Kirim ke LLaMA
    #     response = chat(model=model_name, messages=messages_llm)
    #     return response["message"]["content"]

    def show_logger_info(self):
        print("Function show_logger_info langsung dijalankan")
        prompt  = self.memory.latest_prompt
        model_name = "llama3.1:8b"
        print("prompt", prompt)
        get_info = search_logger_info(prompt)
        print("get_info :", get_info)

        messages_llm = [
            {
                "role": "system",
                "content": (
                    "Anda adalah AI virtual assistant untuk Smart Telemetry Systems (STESY).\n"
                    "Tugas Anda adalah menjawab pertanyaan pengguna secara langsung berdasarkan informasi yang tersedia dari sistem.\n"
                    "Jika data ditemukan, berikan jawaban yang tepat dan singkat.\n"
                    "Jika data **tidak tersedia**, berikan jawaban maaf yang sopan tanpa mengarang atau menebak.\n"
                    "JANGAN mengatakan 'saya tidak tahu' jika data tersedia, dan JANGAN mengarang jika data tidak tersedia.\n\n"
                    "Keterangan singkatan umum:\n"
                    "- WS: Weather Station\n"
                    "- TB: Tipping Bucket (curah hujan)\n"
                    "- US: Ultrasonic Sensor (tinggi muka air)"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Pertanyaan pengguna: {prompt}\n\n"
                    f"Berikut informasi yang tersedia dari sistem:\n\n{get_info}\n\n"
                    "Jika informasi yang diminta tidak ditemukan dalam data di atas, jawab dengan:\n"
                    f"\"Maaf, saya tidak menemukan informasi mengenai '{prompt}' dalam database.\"\n\n"
                    "Jika informasi tersedia, berikan jawaban langsung berdasarkan data tersebut."
                )
            }
        ]

        response = chat(model=model_name, messages=messages_llm)
        print("response adalah ",response)
        final_content = response["message"]["content"]
        print("final_content adalah ",final_content)
        # 
        return final_content # "Menjalankan function show_logger_info"

    def direct_answer(self):
        print("✅ Handler untuk jawaban langsung dijalankan")
        return self.memory.analysis_result or "✅ Jawaban telah diberikan langsung oleh sistem."

    def fallback_response(self):
        print("⚠️ fallback_response dijalankan karena intent tidak dikenali.")
        prompt = self.memory.latest_prompt
        model_name = "llama3.1:8b"

        only_logger = self.memory.last_logger and not self.memory.last_date # .memory.
        only_date = self.memory.last_date and not self.memory.last_logger

        rekomendasi_khusus = ""
        clean_logger = self.memory.last_logger.strip().title() if self.memory.last_logger else "pos yang Anda maksud"

        if only_logger:
            rekomendasi_khusus = (
                f"🛰️ Sistem hanya mengenali lokasi logger: **{clean_logger}**.\n"
                "Coba tambahkan periode waktu atau parameter yang ingin ditampilkan, seperti:\n"
                f"- `Tampilkan suhu udara di {clean_logger} kemarin`\n"
                f"- `Berikan ringkasan kelembaban di {clean_logger} selama seminggu terakhir`\n\n"
            )
        elif only_date:
            rekomendasi_khusus = (
                f"🕒 Sistem hanya mengenali tanggal: **{self.memory.last_date}**.\n"
                "Coba tambahkan nama pos logger untuk memperjelas, seperti:\n"
                f"- `Tampilkan data logger ARR Sapon tanggal {self.memory.last_date}`\n"
                f"- `Bagaimana tekanan udara di AWLR Wates pada {self.memory.last_date}`\n\n"
            )

        contoh_prompt_umum = (
            "Berikut contoh pertanyaan yang dapat dikenali sistem:\n"
            "1. Bagaimana ringkasan kelembaban dari ARR Gemawang selama minggu lalu?\n"
            "2. Tampilkan data terbaru dari ARR Gemawang.\n"
            "3. Di mana tingkat cahaya paling rendah tercatat?\n"
        )

        intro = "🙏 Maaf, sistem belum dapat memahami maksud Anda sepenuhnya.\n\n"

        messages_llm = [
            {
                "role": "system",
                "content": (
                    "Anda adalah STESY, AI virtual assistant untuk Smart Telemetry System.\n"
                    "Jika pengguna tidak memberikan input lengkap, bantu berikan arahan dengan sopan dan jelas.\n"
                    "Formatkan respons secara ramah dan mudah dimengerti."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = chat(model=model_name, messages=messages_llm)
        final_reply = intro + rekomendasi_khusus + contoh_prompt_umum + "\n\n" + response["message"]["content"]
        return final_reply
