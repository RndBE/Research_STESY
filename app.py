# flask code 
# flask chat_endpoint()

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timezone
from manage import PromptProcessedMemory, IntentManager
from super_tools import fetch_list_logger  # Ensure this exists or update with correct path

app = Flask(__name__)
CORS(app) # testing GPU

# === Konfigurasi model ===
BERT_MODEL_PATH = "_new_model/intent_model"
LABEL_ENCODER_PATH = "_new_model/label_encoder.pkl"

@app.route("/chat", methods=["POST"])
# def chat_endpoint():
#     payload = request.get_json(force=True)
#     model_name = payload.get("model", "llama3.1:8b")
#     user_messages = payload.get("messages", [])
#     user_id = payload.get("uuid")
#     print("🔑 user_id:", user_id)

#     try:
#         latest_user_msg = next((m["content"] for m in reversed(user_messages) if m["role"] == "user"), "")
#         prev_assistant_msg = next((m["content"] for m in reversed(user_messages) if m["role"] == "assistant"), "")

#         memory = PromptProcessedMemory(
#             user_id=user_id,
#             user_messages=user_messages,
#             bert_model_path=BERT_MODEL_PATH,
#             label_encoder_path=LABEL_ENCODER_PATH
#         )

#         result = memory.process_new_prompt(latest_user_msg)
#         print("Flask API code")
#         print(f"result : {result}") # {'intent': 'show_logger_data', 'target': [], 'date': None, 'latest_prompt': 'berikan data di pos sapon', 'logger_suggestions': {'pos sapon': ['pos arr sapon', 'pos awlr sapon']}}
#         if isinstance(result, dict) and result.get("message"):
#             return jsonify({
#                 "model": model_name,
#                 "created_at": datetime.now(timezone.utc).isoformat() + "Z",
#                 "message": result["message"]
#             })

#         return jsonify({
#             "model": model_name,
#             "created_at": datetime.now(timezone.utc).isoformat() + "Z",
#             "message": {"role": "assistant", "content": result}
#         })

#     except Exception as e:
#         print(f"[ERROR] {e}")
#         return jsonify({
#             "model": model_name,
#             "created_at": datetime.now(timezone.utc).isoformat() + "Z",
#             "message": {"role": "assistant", "content": "Maaf, terjadi kesalahan saat memproses permintaan Anda."}
#         }), 500
    
def chat_endpoint():
    payload = request.get_json(force=True)
    model_name = payload.get("model", "llama3.1:8b")
    user_messages = payload.get("messages", [])
    user_id = payload.get("uuid") # or request.remote_addr  # 🆕 Pakai UUID dari frontend
    print("🔑 user_id:", user_id)

    try:
        latest_user_msg = next((m["content"] for m in reversed(user_messages) if m["role"] == "user"), "")
        last_msg = user_messages[-1]["content"].strip().lower() if user_messages else ""
        prev_assistant_msg = user_messages[-2]["content"].strip().lower() if len(user_messages) >= 2 and user_messages[-2]["role"] == "assistant" else ""

        # 🧠 Inisialisasi memory & intent manager
        memory = PromptProcessedMemory(
            user_id=user_id,
            user_messages=user_messages,
            bert_model_path=BERT_MODEL_PATH, # dont change this
            label_encoder_path=LABEL_ENCODER_PATH # dont change this
        )

        # 🧹 Reset memori jika chat terlalu banyak
        memory.reset_memory_if_chat_too_long()

        intent_manager = IntentManager(memory)

        # ✅ Deteksi jika user mengkonfirmasi saran logger sebelumnya
        print("Flask API CODE is running ⚡")
        print("prev_assistant_msg", prev_assistant_msg)
        # confirmed_logger = memory.confirm_logger_from_previous_suggestion(prev_assistant_msg, last_msg)
        confirm_result = memory.handle_confirmation_prompt_if_needed(last_msg)
        print(f"Sugesti logger : {memory.last_logger_clarification}")
        print(f"🔁 confirm_result:", confirm_result)
        
        # Cek apakah handle_confirmation_prompt_if_needed mengembalikan dict = siap lanjut fetch
        if isinstance(confirm_result, dict) and confirm_result.get("confirmed"):
            # Gunakan logger terkonfirmasi dan langsung fetch intent
            confirmed_logger = confirm_result["logger"]
            print(f"🔔 Hasil yang dikonfirmasi: {confirm_result}")
            print(f"✅ Logger dikonfirmasi: {confirmed_logger}")
            intent_info = memory.process_new_prompt(confirm_result)
            result = intent_manager.handle_intent()

            return jsonify({
                "model": model_name,
                "created_at": datetime.now(timezone.utc).isoformat() + "Z",
                "message": {"role": "assistant", "content": result}
            })

        # Jika hanya string klarifikasi (bukan dict), langsung tampilkan ke user
        elif isinstance(confirm_result, str):
            return jsonify({
                "model": model_name,
                "created_at": datetime.now(timezone.utc).isoformat() + "Z",
                "message": {"role": "assistant", "content": confirm_result}
            })

        # ✅ Jalankan intent normal
        intent_info = memory.process_new_prompt(latest_user_msg)
        intent = intent_info["intent"]
        target = intent_info.get("target")
        date_text = intent_info.get("date")
        logger_suggestions = intent_info.get("logger_suggestions", {})

        print(f"Intent adalah : {intent}, target logger adalah : {target}, tanggal yang dicari adalah : {date_text}")
        print("1. latest_user_msg", latest_user_msg)
        print("2. intent_info", intent_info)

        # ✅ Step 3: Simpan ke history memory
        print("[DEBUG] Akan memanggil _save_user_memory...")
        memory._save_user_memory() #  update GIT

        # ❓ Tawarkan konfirmasi jika logger tidak dikenali
        if logger_suggestions and any(len(v) == 2 for v in logger_suggestions.values()):
            suggestions_text = []
            for invalid_logger, candidates in logger_suggestions.items():
                if candidates and len(candidates) == 2:
                    # Format kandidat: kapitalisasi awal setiap kata + AWLR/ARR kapital penuh
                    def format_logger_name(name):
                        words = name.lower().split()
                        return ' '.join(w.upper() if w in {"awlr", "arr", "awr", "avwr"} else w.capitalize() for w in words)

                    c1 = format_logger_name(candidates[0])
                    c2 = format_logger_name(candidates[1])
                    # invalid = format_logger_name(invalid_logger)

                    suggestions_text.append(
                        f"Kami mendeteksi ada 2 logger dengan nama mirip {invalid_logger}. Pos manakah yang Anda maksud: {c1} atau {c2}?"
                    )
            return jsonify({
                "model": model_name,
                "created_at": datetime.now(timezone.utc).isoformat() + "Z",
                "message": {"role": "assistant", "content": "\n".join(suggestions_text)}
            })

        # ✅ Jalankan handler intent
        result = intent_manager.handle_intent()

        # 🔔 Tambahkan konfirmasi jika memori direset
        if memory.memory_was_reset():
            confirmation_msg = (
                ""
            )
            result = confirmation_msg + result

        return jsonify({
            "model": model_name,
            "created_at": datetime.now(timezone.utc).isoformat() + "Z",
            "message": {"role": "assistant", "content": result}
        })

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({
            "model": model_name,
            "created_at": datetime.now(timezone.utc).isoformat() + "Z",
            "message": {"role": "assistant", "content": "Maaf, terjadi kesalahan saat memproses permintaan Anda."}
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)