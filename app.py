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
            bert_model_path=BERT_MODEL_PATH,
            label_encoder_path=LABEL_ENCODER_PATH
        )

        # 🧹 Reset memori jika chat terlalu banyak
        memory.reset_memory_if_chat_too_long()

        intent_manager = IntentManager(memory)

        # ✅ Deteksi jika user mengkonfirmasi saran logger sebelumnya
        print("prev_assistant_msg", prev_assistant_msg)
        confirmed_logger = memory.confirm_logger_from_previous_suggestion(prev_assistant_msg, last_msg)
        print(f"confirmed_logger adalah : {confirmed_logger}" )
        if confirmed_logger:
            intent_info = memory.process_new_prompt(confirmed_logger)
            result = intent_manager.handle_intent()
            return jsonify({
                "model": model_name,
                "created_at": datetime.now(timezone.utc).isoformat() + "Z",
                "message": {"role": "assistant", "content": result}
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
        memory._save_user_memory()

        # ❓ Tawarkan konfirmasi jika logger tidak dikenali
        if (not target or target == []) and logger_suggestions:
            suggestions_text = []
            for invalid_logger, candidates in logger_suggestions.items():
                if candidates:
                    # suggestions_text.append(
                    #     f"Nama logger '{invalid_logger}' tidak dikenali. Apakah maksud Anda: " +
                    #     ", ".join(f"'{c}'" for c in candidates) + "?"
                    # )
                    suggestions_text.append( # kami mendeteksi ada 
                        f"Kami mendeteksi ada '{len(candidates)}' logger dengan nama yang sama. Apakah pos yang anda maksud adalah " +
                        ", ".join(f"'{c}'" for c in candidates) + "?"
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