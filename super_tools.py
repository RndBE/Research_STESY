import pandas as pd
import csv

# Declaration Function yang dibutuhkan
import json, csv, re
from rapidfuzz import process, fuzz
from ollama import chat
from datetime import datetime, timezone
import requests


extreme_keywords = [
    "tertinggi", "terendah", "terdingin", "terpanas",
    "paling panas", "paling dingin", "terbanyak", "tersedikit",
    "terkecil", "terbesar", "paling kering", "paling basah",
    "terbasah", "paling lembab", "paling kering"
]

sensor_aliases = {
    "Tinggi_Muka_Air": ["tinggi muka air", "water level", "level air", "ketinggian air"],
    "Max_Baterai_Logger": ["max baterai logger", "baterai maksimum", "baterai max", "maximum battery"],
    "Rerata_Baterai_Logger": ["rerata baterai logger", "rata-rata baterai", "average battery", "baterai rata-rata"],
    "Min_Baterai_Logger": ["min baterai logger", "baterai minimum", "baterai min", "minimum battery"],
    "Temperatur_Logger": ["temperatur logger", "suhu logger", "temperature logger", "logger temperature"],
    "Kecepatan_Angin": ["kecepatan angin", "wind speed", "speed angin", "angin cepat"],
    "Arah_Angin": ["arah angin", "wind direction", "direction angin", "mata angin"],
    "Kecerahan": ["kecerahan", "brightness", "intensitas cahaya", "tingkat cahaya"],
    "Arah_Cahaya": ["arah cahaya", "light direction", "direction cahaya", "sudut cahaya"],
    "Curah_Hujan_1": ["curah hujan", "rainfall", "hujan"],
    "Temperatur_Udara": ["temperatur udara", "suhu udara", "air temperature", "udara panas", "temperatur", "udara temperatur", "tingkat panas udara", "derajat suhu", "tingkat suhu", "temperature", "cuaca panas", "suhu lingkungan", "panas udara", "suhu"],
    "Kelembaban_Udara": ["kelembaban udara", "humidity air", "kadar air udara", "udara lembab", "kelembaban", "udara kelembaban", "kelembapan", "kelembapan udara", "tingkat kelembaban", "humidity", "kadar kelembaban", "kelembaban lingkungan", "kelembaban relatif", "lembab", "lembap"],
    "Tekanan_Udara": ["tekanan udara", "pressure", "air pressure", "barometer"],
    "Kelembaban_Logger": ["kelembaban logger", "humidity logger", "logger kelembaban", "logger humidity"],
    "Koneksi": ["koneksi", "terputus", "terhubung", "aktif","status koneksi", "tersambung", "tidak terhubung","offline", "online", "disconnect", "connect", "penghubung", "sinyal", "signal", "mati", "hidup", "jaringan", "network"]
}

def general_stesy(messages, model_name="llama3.1:8b"):
    print("+++++ Function General Stesy Berjalan +++++")

    system_prompt = {
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
            "Hapus Koneksi logger"
            "Namun jika pertanyaan user berada di luar topik (misalnya tentang sejarah, teknologi umum, hiburan, atau tidak ada hubungannya dengan sistem telemetri), "
            "**tolak dengan sopan boleh tambahkan emoticon**:\n"
        )
    }

    response = chat(model=model_name, messages=[system_prompt] + messages)
    print(f"respone adalah {response['message']['content']}")
    return response["message"]["content"]

def original_fetch_status_logger(prompt: str):
    print("fetch_status_rain ini telah berjalan")

    prompt = prompt.lower()

    kabupaten_map = {
        "sleman": "Sleman",
        "bantul": "Bantul",
        "kulonprogo": "Kulon Progo",
        "kulon progo": "Kulon Progo",
        "gunungkidul": "Gunung Kidul",
        "gunung kidul": "Gunung Kidul",
    }

    kabupaten_in_prompt = None
    for k_raw, k_clean in kabupaten_map.items():
        if k_raw in prompt:
            kabupaten_in_prompt = k_clean
            break

    try:
        logger_resp = requests.get("https://dpupesdm.monitoring4system.com/api/list_logger", timeout=10)
        logger_resp.raise_for_status()
        logger_list = logger_resp.json()
    except requests.RequestException as e:
        return f"Terjadi kesalahan saat mengakses API: {str(e)}"

    lokasi_to_kab = {
        row.get("nama_lokasi"): row.get("kabupaten")
        for row in logger_list if row.get("nama_lokasi") and row.get("kabupaten")
    }

    try:
        response = requests.get("https://dpupesdm.monitoring4system.com/api/get_rain", timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return f"Terjadi kesalahan saat mengakses API: {str(e)}"

    regex_map = {
        "tidak_hujan": r"(tidak\s+hujan|belum\s+(turun\s+)?hujan|masih\s+terang)",
        "hujan_lebat": r"(hujan\s+(lebat|deras|sangat\s+lebat))",
        "tidak_hujan_lebat": r"(tidak\s+hujan\s+lebat|bukan\s+hujan\s+lebat)",
        "sering_hujan": r"sering\s+hujan",
        "tidak_cerah": r"(tidak\s+cerah|berawan|mendung)",
        "cerah": r"\b(cerah|cuaca\s+baik|terang)\b"
    }

    pos_tidak_hujan = []
    pos_hujan = []
    pos_lebat = []
    pos_tidak_lebat = []

    for entry in data:
        nama = entry.get("nama_lokasi", "")
        ch = entry.get("curah_hujan", "")
        status = entry.get("status", "").strip().lower()
        kab = lokasi_to_kab.get(nama)

        if kabupaten_in_prompt and kab != kabupaten_in_prompt:
            continue

        line = f"- {nama} ({ch}, {status})"

        if status == "tidak hujan":
            pos_tidak_hujan.append(line)
        else:
            pos_hujan.append(line)
            if "lebat" in status:
                pos_lebat.append(line)
            else:
                pos_tidak_lebat.append(line)

    if re.search(regex_map["tidak_hujan_lebat"], prompt):
        combined = pos_tidak_lebat + pos_tidak_hujan
        return f"Berikut logger di {kabupaten_in_prompt or 'semua kabupaten'} yang tidak hujan lebat:\n" + "\n".join(combined) if combined else "Semua pos mencatat hujan lebat saat ini."

    if re.search(regex_map["tidak_hujan"], prompt):
        return f"Berikut logger di {kabupaten_in_prompt or 'semua kabupaten'} yang tidak hujan:\n" + "\n".join(pos_tidak_hujan) if pos_tidak_hujan else "Semua pos mencatat hujan saat ini."

    if re.search(regex_map["hujan_lebat"], prompt):
        return f"Berikut logger di {kabupaten_in_prompt or 'semua kabupaten'} dengan hujan lebat:\n" + "\n".join(pos_lebat) if pos_lebat else "Tidak ada pos dengan hujan lebat saat ini."

    if re.search(regex_map["tidak_cerah"], prompt):
        return f"Berikut logger di {kabupaten_in_prompt or 'semua kabupaten'} yang sedang hujan:\n" + "\n".join(pos_hujan) if pos_hujan else "Semua pos tampak cerah saat ini."

    if re.search(regex_map["cerah"], prompt):
        if pos_tidak_hujan:
            return f"Ya, terdapat pos dengan cuaca cerah (tidak hujan) di {kabupaten_in_prompt or 'semua kabupaten'}:\n" + "\n".join(pos_tidak_hujan)
        else:
            return "Saat ini tidak ada pos yang mencatat cuaca cerah."

    if re.search(regex_map["sering_hujan"], prompt):
        return "Maaf, data intensitas hujan historis tidak tersedia saat ini."

    return f"Berikut logger di {kabupaten_in_prompt or 'semua kabupaten'} yang sedang hujan:\n" + "\n".join(pos_hujan) if pos_hujan else "Tidak ada logger yang mencatat hujan saat ini."


import aiohttp
import asyncio
from rapidfuzz import process, fuzz

import aiohttp
import asyncio
from rapidfuzz import process, fuzz

async def original_fetch_data_range_async(date_info, target_loggers, matched_parameters, logger_list):
    interval = date_info.get("interval")
    start_date = date_info.get("awal_tanggal")
    end_date = date_info.get("akhir_tanggal")
    print("🗓️ Date Info:", date_info)

    if not interval or not start_date or not end_date:
        return "Tanggal tidak dikenali dalam prompt."

    def normalize(text):
        return text.lower().replace("pos", "").replace("logger", "").strip()

    normalized_choices = {
        normalize(logger['nama_lokasi']): logger for logger in logger_list
    }
    all_logger_names = list(normalized_choices.keys())
    tasks = []

    async def fetch_logger_data(session, logger):
        logger_id = logger['id_logger']
        logger_name = logger['nama_lokasi']
        url = (
            f"https://dpupesdm.monitoring4system.com/api/data_range"
            f"?id_logger={logger_id}&interval={interval}"
            f"&awal={start_date}&akhir={end_date}"
        )
        print(f"[FETCH RANGE] {logger_name} → {url}")
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                content_type = resp.headers.get("Content-Type", "")
                if "application/json" not in content_type:
                    raise ValueError(f"Invalid Content-Type: {content_type}")

                data = await resp.json()

                if isinstance(data, list) and data:
                    if matched_parameters:
                        filtered_data = []
                        for entry in data:
                            filtered_entry = {
                                k: v for k, v in entry.items()
                                if k in matched_parameters or k.lower() in ["id_logger", "waktu"]
                            }
                            filtered_data.append(filtered_entry)
                        return logger_name, filtered_data
                    return logger_name, data
        except Exception as e:
            print(f"[ERROR] Gagal fetch data untuk {logger_name}: {e}")
        return logger_name, []

    async with aiohttp.ClientSession() as session:
        for name_fragment in target_loggers:
            query = normalize(name_fragment)
            fuzzy_results = process.extract(query, all_logger_names, scorer=fuzz.token_set_ratio, limit=1)

            print(f"\n[DEBUG] Fuzzy match for: '{name_fragment}'")
            for r in fuzzy_results:
                print(f"Match: {r[0]} | Score: {r[1]}")
            print("==============================")

            if not fuzzy_results or fuzzy_results[0][1] < 40:
                continue

            best_match = fuzzy_results[0][0]
            matched_logger = normalized_choices.get(best_match)

            if matched_logger:
                tasks.append(fetch_logger_data(session, matched_logger))

        results = await asyncio.gather(*tasks)

    combined_data = {name: data for name, data in results if data}
    if combined_data:
        print("✅ Data berhasil diambil:", list(combined_data.keys()))
        return combined_data
    else:
        print("[INFO] Data tidak ditemukan, tidak menyimpan ke memory")
        return "Tidak ditemukan data yang cocok untuk logger yang dimaksud."


def original_fetch_data_range(date_info, target_loggers: list, matched_parameters: list, logger_list: list):
    
    # ⛑️ Cek dan konversi jika date_info bukan dict
    # if isinstance(date_info, str):
    #     try:
    #         import json
    #         date_info = json.loads(date_info)  # jika string adalah JSON
    #     except Exception as e:
    #         print("❌ Gagal parsing date_info string:", e)
    #         return "Format date_info tidak valid. Harus dictionary."

    # if not isinstance(date_info, dict):
    #     print("❌ Tipe date_info tidak didukung:", type(date_info))
    #     return "Format date_info tidak valid."
    
    print(f"Type Data dari {type(date_info)}")
    interval = date_info.get("interval")
    start_date = date_info.get("awal_tanggal")
    end_date = date_info.get("akhir_tanggal")
    print("🗓️ Date Info:", date_info)

    if not interval or not start_date or not end_date:
        return "Tanggal tidak dikenali dalam prompt."

    def normalize(text):
        return text.lower().replace("pos", "").replace("logger", "").strip()

    normalized_choices = {
        normalize(logger['nama_lokasi']): logger
        for logger in logger_list
    }
    all_logger_names = list(normalized_choices.keys())
    # summaries = [] 
    all_data = []

    for name_fragment in target_loggers:
        query = normalize(name_fragment)
        fuzzy_results = process.extract(query, all_logger_names, scorer=fuzz.token_set_ratio, limit=3)

        print(f"\n[DEBUG] Fuzzy match for: '{name_fragment}'")
        for r in fuzzy_results:
            print(f"Match: {r[0]} | Score: {r[1]}")
        print("==============================")

        if not fuzzy_results or fuzzy_results[0][1] < 40:
            continue

        best_match = fuzzy_results[0][0]
        matched_logger = normalized_choices.get(best_match)

        if matched_logger:
            logger_id = matched_logger['id_logger']
            logger_name = matched_logger['nama_lokasi']
            try:
                url = (
                    f"https://dpupesdm.monitoring4system.com/api/data_range"
                    f"?id_logger={logger_id}&interval={interval}"
                    f"&awal={start_date}&akhir={end_date}"
                )
                print(f"[FETCH RANGE] {logger_name} → {url}")
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                logger_data = resp.json()

                if isinstance(logger_data, list) and logger_data:
                    if matched_parameters:
                        filtered_data = []
                        for entry in logger_data:
                            filtered_entry = {
                                k: v for k, v in entry.items()
                                if k in matched_parameters or k.lower() in ["id_logger", "waktu"]
                            }
                            filtered_data.append(filtered_entry)
                        logger_data = filtered_data

                    # ✅ Simpan per-logger
                    all_data.append({
                        "logger": logger_name,
                        "data": logger_data
                    })

            except Exception as e:
                print(f"[ERROR] Gagal fetch data untuk {logger_name}: {e}")
    # 🔚 Return hasil
    if all_data:
        print("fetched data adalah :", all_data)
        return all_data
    else:
        return "Tidak ditemukan data yang cocok untuk logger yang dimaksud."



# def original_fetch_data_range(prompt: str, target_loggers: list, matched_parameters: list, logger_list: list ):
#     print("prompt :", prompt)
#     date_info = extract_date_structured(prompt)
#     interval = date_info.get("interval")
#     start_date = date_info.get("awal_tanggal")
#     end_date = date_info.get("akhir_tanggal")
#     print("date_info", date_info)

#     if not interval or not start_date or not end_date:
#         return "Tanggal tidak dikenali dalam prompt."

#     def normalize(text):
#         return text.lower().replace("pos", "").replace("logger", "").strip()

#     normalized_choices = {
#         normalize(logger['nama_lokasi']): logger
#         for logger in logger_list
#     }
#     all_logger_names = list(normalized_choices.keys())
#     summaries = []

#     for name_fragment in target_loggers:
#         query = normalize(name_fragment)
#         fuzzy_results = process.extract(query, all_logger_names, scorer=fuzz.token_set_ratio, limit=3)

#         print(f"\n[DEBUG] Fuzzy match for: '{name_fragment}'")
#         for r in fuzzy_results:
#             print(f"Match: {r[0]} | Score: {r[1]}")
#         print("==============================")

#         if not fuzzy_results or fuzzy_results[0][1] < 40:
#             continue

#         best_match = fuzzy_results[0][0]
#         matched_logger = normalized_choices.get(best_match)

#         if matched_logger:
#             logger_id = matched_logger['id_logger']
#             logger_name = matched_logger['nama_lokasi']
#             try:
#                 url = (
#                     f"https://dpupesdm.monitoring4system.com/api/data_range"
#                     f"?id_logger={logger_id}&interval={interval}"
#                     f"&awal={start_date}&akhir={end_date}"
#                 )
#                 print(f"[FETCH RANGE] {logger_name} → {url}")
#                 resp = requests.get(url, timeout=20)
#                 resp.raise_for_status()
#                 data = resp.json()

#                 if isinstance(data, list) and data:
#                     # === Filter parameter jika matched_parameters tersedia
#                     if matched_parameters:
#                         filtered_data = []
#                         for entry in data:
#                             filtered_entry = {
#                                 k: v for k, v in entry.items()
#                                 if k in matched_parameters or k.lower() in ["id_logger", "waktu"]
#                             }
#                             filtered_data.append(filtered_entry)
#                         data = filtered_data

#                     summary = summarize_logger_data(logger_name, data)
#                     summaries.append(summary)

#             except Exception as e:
#                 print(f"[ERROR] Gagal fetch data untuk {logger_name}: {e}")

#     if summaries:
#         print("fetched data adalah :", summaries)
#         return "\n\n---\n\n".join(summaries)
#     else:
#         return "Tidak ditemukan data yang cocok untuk logger yang dimaksud."



def original_compare_by_date(prompt: str, target_loggers: list, matched_parameters: list, logger_list: list ):
    return original_fetch_data_range(prompt, target_loggers, matched_parameters, logger_list)
    # return original_fetch_data_range(prompt, target_loggers, logger_list)


def find_closest_logger(name_fragment, logger_list, threshold=40, max_candidates=3):
    name_fragment = name_fragment.strip()

    if len(name_fragment) < 4:
        return None

    normalized_choices = {
        normalize_text(logger['nama_lokasi']): logger
        for logger in logger_list
    }
    names = list(normalized_choices.keys())

    query = normalize_text(name_fragment)
    results = process.extract(query, names, scorer=fuzz.token_set_ratio, limit=max_candidates)

    print(f"\n[DEBUG] Fuzzy match for: '{name_fragment}'")
    print("=== Fuzzy Matching Results ===")
    for r in results:
        print(f"Match: {r[0]} | Score: {r[1]}")
    print("==============================")

    if not results:
        return None

    top_match = results[0]
    top_score = top_match[1]

    if len(results) > 1 and top_score < 60 and top_score - results[1][1] < 10:
        return None

    if top_score < threshold:
        return None

    return normalized_choices[top_match[0]]

import requests
from rapidfuzz import process, fuzz
from datetime import datetime, timedelta
import re, calendar

def normalize_text(text: str) -> str:
    return text.lower().replace("pos", "").replace("logger", "").strip()

def find_and_fetch_old_data(name_list, logger_list, prompt_text, threshold=40, max_candidates=3):
    results = []

    # === Ekstrak rentang tanggal dari prompt ===
    date_info = extract_date_structured(prompt_text)
    interval = date_info.get("interval")
    start_date = date_info.get("awal_tanggal")
    end_date = date_info.get("akhir_tanggal")

    if not interval or not start_date or not end_date:
        print("[WARNING] Tanggal tidak dikenali.")
        return []

    # === Persiapan fuzzy matching ===
    normalized_choices = {
        normalize_text(logger['nama_lokasi']): logger
        for logger in logger_list
    }
    all_logger_names = list(normalized_choices.keys())

    for name_fragment in name_list:
        name_fragment = name_fragment.strip()
        if len(name_fragment) < 4:
            continue

        query = normalize_text(name_fragment)
        fuzzy_results = process.extract(query, all_logger_names, scorer=fuzz.token_set_ratio, limit=max_candidates)

        print(f"\n[DEBUG] Fuzzy match for: '{name_fragment}'")
        print("=== Fuzzy Matching Results ===")
        for r in fuzzy_results:
            print(f"Match: {r[0]} | Score: {r[1]}")
        print("==============================")

        if not fuzzy_results or fuzzy_results[0][1] < threshold:
            continue

        best_match_name = fuzzy_results[0][0]
        matched_logger = normalized_choices.get(best_match_name)

        if matched_logger:
            logger_id = matched_logger.get("id_logger")
            logger_name = matched_logger.get("nama_lokasi")
            try:
                url = (
                    f"https://dpupesdm.monitoring4system.com/api/data_range"
                    f"?id_logger={logger_id}&interval={interval}"
                    f"&awal={start_date}&akhir={end_date}"
                )
                print(f"[FETCH RANGE] {logger_name} → {url}")
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    results.append({
                        "logger_name": logger_name,
                        "logger_id": logger_id,
                        "data": data
                    })
            except Exception as e:
                print(f"[ERROR] Gagal fetch data range untuk {logger_name}: {e}")

    return results

def extract_date_structured(text):  # v23.05.25_fx
    from datetime import datetime, timedelta
    import re, calendar

    MONTHS_ID = {
        "januari": "01", "februari": "02", "maret": "03", "april": "04",
        "mei": "05", "juni": "06", "juli": "07", "agustus": "08",
        "september": "09", "oktober": "10", "november": "11", "desember": "12"
    }

    print("Recall Function : extract_date_structured ")
    text = text.lower().strip()
    today = datetime.today()

    def last_day(year, month):
        return calendar.monthrange(year, month)[1]

    def output(start, end, interval=None):
        if not interval:
            delta_days = (end - start).days
            interval = "hari" if delta_days == 0 else "hari"
        return {
            "interval": interval,
            "awal_tanggal": start.strftime("%Y-%m-%d"),
            "akhir_tanggal": end.strftime("%Y-%m-%d")
        }

    # === Format: 1 mei hingga 13 mei 2025 ===
    match = re.search(r'(\d{1,2})\s+([a-z]+)\s+(dan|hingga|sampai|ke|s\.d\.|menuju)\s+(\d{1,2})\s+([a-z]+)\s+(\d{4})', text)
    if match:
        d1 = int(match.group(1))
        m1 = match.group(2)
        d2 = int(match.group(4))
        m2 = match.group(5)
        y = int(match.group(6))
        if m1 in MONTHS_ID and m2 in MONTHS_ID:
            start = datetime(y, int(MONTHS_ID[m1]), d1)
            end = datetime(y, int(MONTHS_ID[m2]), d2)
            return output(start, end, interval="hari")

    # === Format: 1 mei hingga 13 mei (tanpa tahun, default tahun ini) ===
    match = re.search(r'(\d{1,2})\s+([a-z]+)\s+(dan|hingga|sampai|ke|s\.d\.|menuju)\s+(\d{1,2})\s+([a-z]+)', text)
    if match:
        d1 = int(match.group(1))
        m1 = match.group(2)
        d2 = int(match.group(4))
        m2 = match.group(5)
        y = today.year
        if m1 in MONTHS_ID and m2 in MONTHS_ID:
            start = datetime(y, int(MONTHS_ID[m1]), d1)
            end = datetime(y, int(MONTHS_ID[m2]), d2)
            return output(start, end, interval="hari")

    # === Format: 1 mei 2024 hingga 13 mei 2025 ===
    match = re.search(r'(\d{1,2})\s+([a-z]+)\s+(\d{4})\s+(hingga|sampai|s\.d\.|ke|menuju)\s+(\d{1,2})\s+([a-z]+)\s+(\d{4})', text)
    if match:
        d1, m1, y1 = int(match.group(1)), match.group(2).lower(), int(match.group(3))
        d2, m2, y2 = int(match.group(5)), match.group(6).lower(), int(match.group(7))
        if m1 in MONTHS_ID and m2 in MONTHS_ID:
            start = datetime(y1, int(MONTHS_ID[m1]), d1)
            end = datetime(y2, int(MONTHS_ID[m2]), d2)
            return output(start, end, interval="hari")

    # === Format: 1 mei (tanpa tahun) ===
    match = re.search(r'(\d{1,2})\s+([a-z]+)\b', text)
    if match:
        d1 = int(match.group(1))
        m1 = match.group(2)
        if m1 in MONTHS_ID:
            y = today.year
            date_obj = datetime(y, int(MONTHS_ID[m1]), d1)
            return output(date_obj, date_obj)

    # === Format tanggal eksplisit (YYYY-MM-DD atau DD-MM-YYYY) ===
    match = re.search(r'\b(\d{4}-\d{2}-\d{2})\b|\b(\d{2}[-/]\d{2}[-/]\d{4})\b', text)
    if match:
        date_str = match.group(0)
        for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"]:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                return output(parsed_date, parsed_date)
            except ValueError:
                continue
    match = re.search(r"(\d+)\s+minggu\s+(lalu|terakhir)", text)
    if match:
        weeks = int(match.group(1))
        start = today - timedelta(weeks=weeks)
        end = today
        return output(start, end)

    if "hari ini" in text:
        return output(today, today)

    if re.search(r"\bkema(?:rin|ren)\b", text):
        day = today - timedelta(days=1)
        return output(day, day, interval="hari")

    if "minggu ini" in text:
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6)
        return output(start, end)

    if "minggu lalu" in text:
        ref = today - timedelta(weeks=1)
        start = ref - timedelta(days=ref.weekday())
        end = start + timedelta(days=6)
        return output(start, end)

    # match = re.search(r"(\d+)\s+minggu\s+(lalu|terakhir)", text)
    # if match:
    #     weeks = int(match.group(1))
    #     start = today - timedelta(weeks=weeks)
    #     end = today
    #     return output(start, end)

    if "awal bulan" in text:
        start = today.replace(day=1)
        return output(start, start)

    if "akhir bulan" in text:
        end = today.replace(day=last_day(today.year, today.month))
        return output(end, end)

    if "bulan lalu" in text:
        prev_month = today.replace(day=1) - timedelta(days=1)
        start = prev_month.replace(day=1)
        end = prev_month.replace(day=last_day(prev_month.year, prev_month.month))
        return output(start, end)

    match = re.search(r"(\d+)\s+hari\s+(lalu|terakhir)", text)
    if match:
        days = int(match.group(1))
        start = today - timedelta(days=days)
        end = today
        return output(start, end)

    match = re.search(r"(\d+)\s+bulan\s+(lalu|terakhir)", text)
    if match:
        months = int(match.group(1))
        year = today.year
        month = today.month - months
        while month <= 0:
            month += 12
            year -= 1
        start = datetime(year, month, 1)
        end = today
        return output(start, end)

    match = re.search(r"dari\s+(" + "|".join(MONTHS_ID.keys()) + r")\s+(\d{4})\s+hingga\s+(" + "|".join(MONTHS_ID.keys()) + r")\s+(\d{4})", text)
    if match:
        smonth = int(MONTHS_ID[match.group(1)])
        syear = int(match.group(2))
        emonth = int(MONTHS_ID[match.group(3)])
        eyear = int(match.group(4))
        start = datetime(syear, smonth, 1)
        end = datetime(eyear, emonth, last_day(eyear, emonth))
        return output(start, end)

    match = re.search(r"dari\s+(" + "|".join(MONTHS_ID.keys()) + r")\s+ke\s+(" + "|".join(MONTHS_ID.keys()) + r")\s+(\d{4})", text)
    if match:
        smonth = int(MONTHS_ID[match.group(1)])
        emonth = int(MONTHS_ID[match.group(2)])
        year = int(match.group(3))
        start = datetime(year, smonth, 1)
        end = datetime(year, emonth, last_day(year, emonth))
        return output(start, end)

    match = re.search(r"data dari\s+(" + "|".join(MONTHS_ID.keys()) + r")", text)
    if match:
        month_num = int(MONTHS_ID[match.group(1)])
        year = today.year
        start = datetime(year, month_num, 1)
        end = datetime(year, month_num, last_day(year, month_num))
        return output(start, end)

    match = re.search(r'\b(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})\b', text)
    if match:
        day = int(match.group(1))
        month_text = match.group(2)
        year = int(match.group(3))
        month_num = MONTHS_ID.get(month_text.lower())
        if month_num:
            date_obj = datetime(year, int(month_num), day)
            return output(date_obj, date_obj)

    match = re.search(r"(\d{4})\s+(" + "|".join(MONTHS_ID.keys()) + r")|(" + "|".join(MONTHS_ID.keys()) + r")\s+(\d{4})", text)
    if match:
        year = int(match.group(1) or match.group(4))
        month = int(MONTHS_ID.get(match.group(2) or match.group(3)))
        start = datetime(year, month, 1)
        end = datetime(year, month, last_day(year, month))
        return output(start, end)

    if "tahun lalu" in text:
        year = today.year - 1
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31)
        return output(start, end)

    return {
        "interval": None,
        "awal_tanggal": None,
        "akhir_tanggal": None
    }

# add new summarize function
def summarize_logger_data(nama_lokasi, latest_data, model_name="llama3.1:8b"):
    def estimate_tokens(text: str) -> int:
        return int(len(text) / 4)

    def format_day_parameters(data: dict) -> str:
        return "\n".join(
            [f"* **{key.replace('_',' ').title()}**: {value}" 
             for key, value in data.items() if key.lower() != "id_logger"]
        )

    # === Tangani data satu hari vs multi hari ===
    if isinstance(latest_data, list):
        parameter_text = ""
        for day in latest_data:
            tanggal = day.get("Waktu", "Tanggal tidak diketahui")
            parameter_text += f"\n### {tanggal}\n"
            parameter_text += format_day_parameters(day) + "\n"

        waktu = f"{latest_data[0].get('Waktu', '?')} hingga {latest_data[-1].get('Waktu', '?')}"
        koneksi = latest_data[-1].get("Koneksi", "Koneksi Terputus")

    elif isinstance(latest_data, dict):
        parameter_text = format_day_parameters(latest_data)
        waktu = latest_data.get("Waktu", "waktu tidak diketahui")
        koneksi = latest_data.get("Koneksi", "Koneksi Terputus")

    else:
        return "⚠️ Format data logger tidak dikenali (harus dict atau list of dict)."

    print("parameternya adalah:\n", parameter_text)

    # === Prompt LLM untuk Kesimpulan Naratif ===
    # "Berikan kesimpulan dalam satu paragraf dari data logger berikut. "
    # "Analisis tren parameter yang tersedia (jika terlihat), sebutkan nilai tertinggi dan terendah, dan "
    # "jelaskan apakah nilainya termasuk kategori rendah, sedang, atau tinggi berdasarkan konteks umum. "
    # "Gunakan gaya bahasa informatif dan ringkas.\n\n"
    # Analisis tren parameter yang tersedia (jika terlihat), sebutkan nilai tertinggi dan terendah, dan jelaskan apakah nilainya termasuk kategori rendah, sedang, atau tinggi berdasarkan konteks umum. 

    user_prompt = (
        f"Tampilkan data lengkap dari logger **{nama_lokasi}** dalam format markdown.\n\n"
        f"Data diambil pada waktu: {waktu}.\n"
        # f"Status koneksi logger: {koneksi}.\n\n"
        f"Berikut adalah semua parameter yang tersedia:\n\n"
        f"{parameter_text}\n\n"
        "Berikan kesimpulan dalam satu paragraf dari data logger berikut. \n"
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

    full_prompt = messages[0]["content"] + "\n\n" + user_prompt
    estimated_tokens = estimate_tokens(full_prompt)
    print(f"[DEBUG] Estimasi jumlah token prompt: {estimated_tokens}")

    response = chat(model=model_name, messages=messages, options={"num_predict": 1024})
    return response["message"]["content"]


def preprocess_name_list(name_list):
    new_list = []
    for name in name_list:
        # Replace " dan " with comma, then split
        name = name.lower().replace(" dan ", ",").replace(" & ", ",")
        parts = [part.strip() for part in name.split(",") if part.strip()]
        new_list.extend(parts)
    return new_list


def normalize_text(text):
    return text.lower().replace("pos", "").replace("logger", "").replace("dan", "").replace("hingga", "").strip()

def find_and_fetch_latest_data(name_list, matched_parameters, logger_list, threshold=70, max_candidates=3):
    results = []

    # Preprocessing nama lokasi
    name_list = preprocess_name_list(name_list)

    normalized_choices = {
        normalize_text(logger['nama_lokasi']): logger
        for logger in logger_list
    }
    all_logger_names = list(normalized_choices.keys())

    for name_fragment in name_list:
        name_fragment = name_fragment.strip()
        if len(name_fragment) < 4:
            continue

        query = normalize_text(name_fragment)
        print("query :", query)

        fuzzy_results = process.extract(query, all_logger_names, scorer=fuzz.token_set_ratio, limit=max_candidates)
        
        print(f"\n[DEBUG] Fuzzy match for: '{name_fragment}'")
        for r in fuzzy_results:
            print(f"Match: {r[0]} | Score: {r[1]}")
        print("==============================")

        if not fuzzy_results or fuzzy_results[0][1] < 40:
            continue

        best_match = fuzzy_results[0][0]
        matched_logger = normalized_choices.get(best_match)

        # print(f"\n[DEBUG] Fuzzy match for: '{name_fragment}'")
        # print("=== Fuzzy Matching Results ===")
        # for r in fuzzy_results:
        #     print(f"Match: {r[0]} | Score: {r[1]}")
        # print("==============================")

        # if not fuzzy_results:
        #     continue

        # match_name, match_score, _ = fuzzy_results[0]
        # print(f"match_name : {match_name}, match_score : {match_score}, _ {_}")
        # # match_name : awr barongan, match_score : 100.0, _ 27
        # if match_score < threshold:
        #     continue

        # matched_logger = normalized_choices.get(match_name)
        print("matched_logger :", matched_logger)
        # matched_logger = normalized_choices.get(top_match[0])
        if matched_logger:
            logger_id = matched_logger["id_logger"]
            try:
                url = f"https://dpupesdm.monitoring4system.com/api/data_new?id_logger={logger_id}"
                print(f"[FETCH] {matched_logger['nama_lokasi']} → {url}")
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                data = resp.json()
                print("1. Langkah 1")
                if isinstance(data, list) and len(data) > 0:
                    latest_entry = data[0]
                    print("2. Langkah 2")
                    print("Langkah 3 :", matched_parameters)

                    if matched_parameters:
                        print("Langkah 4 :", len(matched_parameters))
                        filtered_entry = {}
                        for k, v in latest_entry.items():
                            if k in matched_parameters or k.lower() in ["id_logger", "waktu", "nama_lokasi"]:
                                # Ambil nilai pertama kalau berupa list
                                if isinstance(v, list) and len(v) == 1:
                                    filtered_entry[k] = v[0]
                                else:
                                    filtered_entry[k] = v
                        latest_entry = filtered_entry

                    results.append({
                        "logger_name": matched_logger["nama_lokasi"],
                        "logger_id": logger_id,
                        "data": latest_entry
                    })
            except Exception as e:
                print(f"[ERROR] Gagal fetch data untuk {matched_logger['nama_lokasi']}: {e}")

    return results

# def find_and_fetch_latest_data(name_list, logger_list, threshold=80, max_candidates=3):
#     results = []
    
#      # Tambahkan pre-processing
#     name_list = preprocess_name_list(name_list)

#     normalized_choices = {  
#         normalize_text(logger['nama_lokasi']): logger
#         for logger in logger_list
#     }
#     all_logger_names = list(normalized_choices.keys())

#     for name_fragment in name_list:
#         name_fragment = name_fragment.strip()
#         if len(name_fragment) < 4:
#             continue

#         query = normalize_text(name_fragment)
#         print("query :", query)

#         fuzzy_results = process.extract(query, all_logger_names, scorer=fuzz.token_set_ratio, limit=max_candidates)

#         print(f"\n[DEBUG] Fuzzy match for: '{name_fragment}'")
#         print("=== Fuzzy Matching Results ===")
#         for r in fuzzy_results:
#             print(f"Match: {r[0]} | Score: {r[1]}")
#         print("==============================")

#         if not fuzzy_results:
#             continue

#         top_match = fuzzy_results[0]
#         if top_match[1] < threshold:
#             continue

#         matched_logger = normalized_choices.get(top_match[0])
#         if matched_logger:
#             logger_id = matched_logger["id_logger"]
#             try:
#                 url = f"https://dpupesdm.monitoring4system.com/api/data_new?id_logger={logger_id}"
#                 print(f"[FETCH] {matched_logger['nama_lokasi']} → {url}")
#                 resp = requests.get(url, timeout=20)
#                 resp.raise_for_status()
#                 data = resp.json()
#                 if isinstance(data, list) and len(data) > 0:
#                     results.append({
#                         "logger_name": matched_logger["nama_lokasi"],
#                         "logger_id": logger_id,
#                         "data": data[0]
#                     })
#             except Exception as e:
#                 print(f"[ERROR] Gagal fetch data untuk {matched_logger['nama_lokasi']}: {e}")

#     return results

def find_closest_logger(name_fragment, logger_list, threshold=40, max_candidates=3):
    name_fragment = name_fragment.strip()

    if len(name_fragment) < 4:
        return None

    normalized_choices = {
        normalize_text(logger['nama_lokasi']): logger
        for logger in logger_list
    }
    names = list(normalized_choices.keys())
    query = normalize_text(name_fragment)
    results = process.extract(query, names, scorer=fuzz.token_set_ratio, limit=max_candidates)

    print(f"\n[DEBUG] Fuzzy match for: '{name_fragment}'")
    print("=== Fuzzy Matching Results ===")
    for r in results:
        print(f"Match: {r[0]} | Score: {r[1]}")
    print("==============================")

    if not results:
        return None

    top_match = results[0]
    top_score = top_match[1]

    if len(results) > 1 and top_score < 60 and top_score - results[1][1] < 10:
        return None

    if top_score < threshold:
        return None


def get_logger_info(): 
    url = "https://dpupesdm.monitoring4system.com/api/list_logger"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data)
    except requests.RequestException as e:
        print(f"❌ Gagal mengambil data dari API: {e}")
        return None
import difflib
def search_logger_info(prompt: str):
    df = get_logger_info()
    if df is None:
        return "❌ Tidak bisa mengambil data logger saat ini."

    prompt = prompt.lower()

    # Tentukan jenis informasi yang diminta
    if "sensor" in prompt:
        info_type = "sensor"
    elif "hp" in prompt or "nomor" in prompt or "no hp" in prompt:
        info_type = "hp"
    elif "penjaga" in prompt or "siapa" in prompt or "jaga" in prompt:
        info_type = "penjaga"
    else:
        info_type = "full"

    # Ekstrak kemungkinan nama lokasi dari prompt
    logger_pattern = r"\b(?:logger|pos|afmr|awlr|awr|arr|adr|awqr|avwr|awgc)\s+(?:[a-z]{3,}(?:\s+[a-z]{3,}){0,3})"
    match = re.search(logger_pattern, prompt)
    location_text = match.group(0) if match else prompt  # fallback ke seluruh prompt

    # Fuzzy match ke nama_lokasi
    pos_list = df['nama_lokasi'].dropna().str.lower().tolist()
    best_match = difflib.get_close_matches(location_text, pos_list, n=1, cutoff=0.3)

    if not best_match:
        return "🕵️‍♂️ Maaf, saya tidak menemukan pos yang relevan dengan permintaan Anda."

    pos_data = df[df['nama_lokasi'].str.lower() == best_match[0]].iloc[0]
    lokasi = pos_data['nama_lokasi']

    # Ambil masing-masing data dan validasi isinya
    nama_penjaga = str(pos_data.get('nama_penjaga', '')).strip()
    no_hp = str(pos_data.get('no_penjaga', '')).strip()
    sensor = str(pos_data.get('sensor', '')).strip()

    # Buat respon berdasarkan jenis info
    if info_type == "sensor":
        return f"Jenis sensor di {lokasi}: {sensor if sensor else 'Data sensor belum tersedia.'}"
    elif info_type == "hp":
        return f"Nomor HP penjaga di {lokasi}: {no_hp if no_hp else 'Nomor HP belum tercatat dalam sistem.'}"
    elif info_type == "penjaga":
        return f"Nama penjaga di {lokasi}: {nama_penjaga if nama_penjaga else 'Nama penjaga belum tersedia.'}"
    else:
        return (
            f"{lokasi}\n"
            f"Penjaga: {nama_penjaga if nama_penjaga else 'Belum tersedia'}\n"
            f"Sensor: {sensor if sensor else 'Belum tersedia'}\n"
            f"No HP: {no_hp if no_hp else 'Belum tersedia'}"
        )


def fetch_list_logger_from_prompt_flexibleV1(user_prompt: str):
    print("fetch_list_logger_from_prompt_flexibleV1 telah berjalan")

    # Daftar kabupaten yang didukung
    kabupaten_list = ['Sleman', 'Bantul', 'Kulon Progo', 'Gunung Kidul']
    # gunungkidul, kulonprogo
    def extract_kabupaten(prompt: str) -> str:
        prompt = prompt.lower().replace(" ", "")
        
        # Kamus mapping ejaan longgar ke format baku
        kabupaten_map = {
            "sleman": "Sleman",
            "bantul": "Bantul",
            "gunungkidul": "Gunung Kidul",
            "gunungkidul": "Gunung Kidul",  # tambahan jaga-jaga
            "kulonprogo": "Kulon Progo",
        }

        for raw_kab, standard_kab in kabupaten_map.items():
            if raw_kab in prompt:
                return standard_kab
        return None


    kabupaten = extract_kabupaten(user_prompt)

    # Fetch data dari API
    url = "https://dpupesdm.monitoring4system.com/api/list_logger"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        loggers = response.json()
    except requests.RequestException as e:
        print(f"Error fetching logger data: {e}")
        return "⚠️ Gagal mengambil data logger dari server."

    # Filter field penting
    filtered_loggers = []
    for row in loggers:
        filtered_loggers.append({
            "nama_lokasi": row.get("nama_lokasi"),
            "alamat": row.get("alamat"),
            "kabupaten": row.get("kabupaten"), # nama lokasi harus dalam bentuk Kulon Progo dan Gunung Kidul
        })

    df = pd.DataFrame(filtered_loggers)
    print(f"Panjang DataFrame di Kabupaten {kabupaten} adalah {len(df)}")
    if kabupaten:
        df = df[df['kabupaten'].str.lower() == kabupaten.lower()]
        print(f"Filter berdasarkan kabupaten: {kabupaten}")
    else:
        print("Tidak ada kabupaten disebutkan, tampilkan semua logger.")
    print(f"DataFrame {kabupaten} : {df}")
    result_list = df.to_dict(orient='records')

    # === Ringkasan otomatis ===
    def summarize_list_logger_internal():
        print("summarize_list_logger_internal telah berjalan")

        if not result_list:
            return "⚠️ Tidak ditemukan logger untuk permintaan Anda."
        
        markdown_list = ""
        for i, log in enumerate(result_list, start=1):
            markdown_list += (
                f"\n### {i}. {log['nama_lokasi']}\n"
                f"* **Alamat**: {log['alamat']}\n"
                f"* **Kabupaten**: {log['kabupaten']}\n"
            )

        user_msg = (
            f"Tampilkan daftar logger berdasarkan prompt berikut:\n\n"
            f"`{user_prompt}`\n\n"
            f"Jumlah logger ditemukan: {len(result_list)} pos.\n\n"
            f"{markdown_list}\n\n"
            "JANGAN Potong daftar logger. "
            "Buat ringkasan pendek di awal (maksimal 2 kalimat), lalu tampilkan semua logger. "
            "Awali dengan judul: **Daftar Pos Telemetri Berdasarkan Permintaan**\n"
            "Tambahkan garis pemisah '=====' di bawah judul."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Anda adalah asisten telemetri yang menyusun ringkasan dan daftar pos logger.\n"
                    "Buat ringkasan pendek yang relevan, lalu tampilkan semua logger dalam format markdown."
                )
            },
            {"role": "user", "content": user_msg}
        ]

        response = chat(model="llama3.1:8b", messages=messages, options={"num_predict": 1024})
        return response["message"]["content"]

    return summarize_list_logger_internal()


def fetch_data_range(id_logger, start_date, end_date, interval="hari"):
    url = f"https://dpupesdm.monitoring4system.com/api/data_range?id_logger={id_logger}&interval={interval}&awal={start_date}&akhir={end_date}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()
    
import pandas as pd

def fetch_list_logger():
    url = "https://dpupesdm.monitoring4system.com/api/list_logger"
    
    try:
        response = requests.get(url, timeout=10)  # tambahkan timeout untuk mencegah hanging
        response.raise_for_status()  # akan melempar error kalau status code bukan 2xx
        loggers = response.json()

        # Pastikan hanya field yang kita butuhkan
        cleaned_loggers = []
        for row in loggers:
            cleaned_loggers.append({
                "id_logger": row.get("id_logger"),
                "nama_lokasi": row.get("nama_lokasi"),
                "latitude": row.get("latitude"),
                "longitude": row.get("longitude"),
                "alamat": row.get("alamat"),
                "kabupaten": row.get("kabupaten"),
                "koneksi": row.get("koneksi")
            })

        return cleaned_loggers

    except requests.RequestException as e:
        print(f"Error fetching logger data: {e}")
        return []

def is_extreme_only(prompt, sensor_aliases):
        prompt_lower = prompt.lower()
        has_extreme = any(k in prompt_lower for k in extreme_keywords)
        has_sensor = any(
            any(alias.lower() in prompt_lower for alias in aliases)
            for aliases in sensor_aliases.values()
        )
        return has_extreme and not has_sensor

def fetch_latest_data(id_logger):
    url = f"https://dpupesdm.monitoring4system.com/api/data_new?id_logger={id_logger}"
    print(url)
    resp = requests.get(url, timeout=20) # 
    resp.raise_for_status() # 
    data = resp.json() # 
    return data[0] if isinstance(data, list) and len(data) > 0 else {}

neo_test_prompts = [
    "Bagaimana ringkasan kelembaban udara minggu lalu di Pos ARR Kemput?",  # analyze_logger_by_date
    "Bandingkan suhu udara antara AWR Kaliurang dan ARR Kemput tanggal 1 Mei.",  # compare_logger_by_date
    "Berikan perbandingan data saat ini antara ARR Gemawang dan ARR Kemput.",  # compare_logger_data
    "Berapa kelembaban udara di ARR Kemput tanggal 3 Mei 2025?",  # fetch_logger_by_date
    "Tampilkan hanya suhu udara hari ini dari AWR Kaliurang.",  # show_selected_parameter
    "Apa saja data terbaru dari pos ARR Gemawang?",  # show_logger_data
    "Tampilkan foto lokasi logger ARR Kemput.",  # get_logger_photo_path
    "Bandingkan curah hujan antar semua logger hari ini.",  # compare_parameter_across_loggers
    "Apakah ada logger mencatat hujan sekarang?",  # fetch_status_rain
    "Apa saja logger yang aktif saat ini?",  # show_list_logger
    "Jelaskan bagaimana cara kerja sistem STESY."  # how_it_works
]

