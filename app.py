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
    user_id = request.remote_addr

    try: 
        latest_user_msg = next((m["content"] for m in reversed(user_messages) if m["role"] == "user"), "")

        # ✅ Inisialisasi memory dan intent manager
        memory = PromptProcessedMemory(
            user_id=user_id,
            user_messages=user_messages,
            bert_model_path=BERT_MODEL_PATH,
            label_encoder_path=LABEL_ENCODER_PATH
        )
        # print(memory)
        intent_manager = IntentManager(memory)

        # print("intent_manager", intent_manager)

        # ✅ Proses dan jalankan intent
        intent_info = memory.process_new_prompt(latest_user_msg)

        intent = intent_info["intent"]
        target = intent_info.get("target")
        date_text = intent_info.get("date")
        print(f"Intent adalah : {intent}, target logger adalah : {target}, tanggal yang dicari adalah : {date_text}")
        
        print("1. latest_user_msg", latest_user_msg)
        print("2. intent_info", intent_info)
        # intent_info

        result = intent_manager.handle_intent()

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
            "message": {"role": "assistant", "content": "❌ Maaf, terjadi kesalahan saat memproses permintaan Anda."}
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

