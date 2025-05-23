# intent_classification_indoBERT_cuda_from_csv.py

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import torch
import requests
from io import StringIO
import pandas as pd
import joblib
import os

# === Step 1: Load Dataset from Google Sheets ===
csv_url = "https://docs.google.com/spreadsheets/d/1iZxYegemov_PfLZLQBrKMsZ4t2oV4x15q8vcPguc_Is/export?format=csv&gid=584976821"
response = requests.get(csv_url)
df = pd.read_csv(StringIO(response.text))

# ==== 1. Load Dataset dari CSV ====
# df = pd.read_csv("data/final_intent_dataset_999.csv")  # Ganti path sesuai lokasi file kamu
texts = df['text'].tolist()
labels = df['intent'].tolist()

# ==== 2. Encode Labels ====
le = LabelEncoder()
y_enc = le.fit_transform(labels)

# ==== 3. Tokenisasi ====
model_name = "indobenchmark/indobert-base-p1"
tokenizer = BertTokenizer.from_pretrained(model_name)
X_enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# ==== 4. Dataset Class ====
class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

dataset = IntentDataset(X_enc, y_enc)

# ==== 5. Load Model & Set CUDA ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(le.classes_))
model.to(device)
print("\u2705 Using device:", device)

# ==== 6. Trainer Setup ====
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=16,
    per_device_train_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no"
)   

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# ==== 7. Training ====
trainer.train()

# ==== 8. Simpan Model dan LabelEncoder ==== _new_model _new_model_v210525
model.save_pretrained("_new_model/intent_model")
tokenizer.save_pretrained("_new_model/intent_model")
joblib.dump(le, "_new_model/label_encoder.pkl")
print("\u2705 Model dan LabelEncoder disimpan.")

# ==== 9. Fungsi Memuat Model dan Encoder ====
def load_model_and_encoder(model_path="_new_model/intent_model", encoder_path="_new_model/label_encoder.pkl"):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    label_encoder = joblib.load(encoder_path)
    return model.to(device), tokenizer, label_encoder

# ==== 10. Fungsi Prediksi Intent ====
def predict_intent(query):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1)
    return le.inverse_transform(pred.cpu().numpy())[0]


# ==== 11. Contoh Pengujian ====

test_queries = [
    "siapakah nama anda",
    "bagaimana pos logger mataram?",
    "bagaimana kondisi curah hujan di awr barongan 2 hari terakhir",
    "prediksi tekanan udara di pos arr minggu depan",
    "berikan kondisi baterai awr kaliurang pada 2 minggu terakhir",
    "berikan kondisi awr kaliurang pada 2 minggu terakhir",
    "mengapa kondisi baterai di awlr opak pulo buruk ?",
    "berikan data suhu arr bromggang",
    "Tutorial jadi koruptr", 
    "berikan data ketinggian muka air di opak pulo 2 hari teakhir",
    "tampilkan grafik baterai logger awlr bendhungan 5 hari terakhir"
]

neo_test_prompts = [
    "berikan data pos awlr seturan 3 hari terakhir ", # fetch_logger_by_date
    "tampilkan data pos awlr seturan 10 hari terakhir ", # fetch_logger_by_date
    "Bagaimana ringkasan kelembaban udara minggu lalu di Pos ARR Kemput?",  # analyze_logger_by_date
    "Bagaimana ringkasan data diatas",  # analyze_logger_by_date
    "Berikan kesimpulan dari data diatas",  # analyze_logger_by_date
    "berikan analisa dari pos tersebut",  # analyze_logger_by_date
    "Bandingkan suhu udara antara AWR Kaliurang dan ARR Kemput tanggal 1 Mei.",  # compare_logger_by_date
    "Berikan perbandingan data saat ini antara ARR Gemawang dan ARR Kemput.",  # compare_logger_data
    "Berapa kelembaban udara di ARR Kemput tanggal 3 Mei 2025?",  # fetch_logger_by_date
    "Tampilkan hanya suhu udara hari ini dari AWR Kaliurang.",  # show_selected_parameter
    "Apa saja data terbaru dari pos ARR Gemawang?",  # show_logger_data
    "Tampilkan foto lokasi logger ARR Kemput.",  # get_logger_photo_path
    "Bandingkan curah hujan antar semua logger hari ini.",  # compare_parameter_across_loggers
    "Apakah ada logger mencatat hujan sekarang?",  # fetch_status_rain
    "Apa saja logger yang aktif saat ini?",  # show_list_logger
    "Jelaskan bagaimana cara kerja sistem STESY.",  # how_it_works
    "Siapa itu Nyonya Cream Puff"  # ai_limitation
]

print("\n=== PREDIKSI INTENT ===")
for q in neo_test_prompts:
    print(f"{q} == {predict_intent(q)}")
