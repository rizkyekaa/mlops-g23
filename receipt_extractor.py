import json
import os
from pathlib import Path

import lmstudio as lms


# =========================
# Konfigurasi
# =========================

IMAGE = Path("data/raw/nota-sample.jpeg")
MODEL = os.environ["LM_STUDIO_MODEL"]


# =========================
# Validasi file gambar
# =========================

if not IMAGE.exists():
    raise FileNotFoundError(
        f"File gambar tidak ditemukan: {IMAGE}"
    )


# =========================
# Siapkan gambar
# =========================

print(f"Membaca gambar: {IMAGE}")

image = lms.prepare_image(str(IMAGE))


# =========================
# Load model
# =========================

print(f"Menggunakan model: {MODEL}")

model = lms.llm(MODEL)


# =========================
# Buat chat
# =========================

chat = lms.Chat()

chat.add_user_message(
    """
Baca nota pada gambar dengan teliti dan ekstrak informasi berikut:

- merchant
- tanggal
- item
- subtotal
- pajak
- total

Gunakan struktur JSON berikut:

{
  "merchant": "",
  "tanggal": "",
  "item": [
    {
      "nama": "",
      "jumlah": 0,
      "harga": 0
    }
  ],
  "subtotal": 0,
  "pajak": 0,
  "total": 0
}

Aturan:
- Keluarkan hanya JSON valid.
- Jangan gunakan markdown.
- Jangan tambahkan penjelasan di luar JSON.
- Jika pajak tidak terlihat, isi dengan 0.
- Jangan mengarang informasi yang tidak terlihat pada nota.
- Jika informasi tidak dapat dibaca, gunakan null.
- Nilai uang harus berupa angka, bukan string.
""",
    images=[image],
)


# =========================
# Jalankan model
# =========================

print("Sedang memproses nota...")

prediction = model.respond(chat)


# =========================
# Ambil hasil response
# =========================

content = prediction.content.strip()


print("\n=== RAW RESPONSE ===")
print(content)
print("====================")


# =========================
# Bersihkan markdown
# =========================

if content.startswith("```json"):
    content = content[len("```json"):].strip()

elif content.startswith("```"):
    content = content[len("```"):].strip()


if content.endswith("```"):
    content = content[:-3].strip()


# =========================
# Konversi response menjadi JSON
# =========================

try:
    result = json.loads(content)

except json.JSONDecodeError as e:
    print("\nERROR: Response dari model bukan JSON yang valid.")
    print("\nResponse setelah dibersihkan:")
    print(content)

    raise ValueError(
        f"Gagal parsing JSON: {e}"
    ) from e


# =========================
# Pastikan folder reports tersedia
# =========================

Path("reports").mkdir(
    parents=True,
    exist_ok=True
)


# =========================
# Simpan hasil
# =========================

OUTPUT = Path("reports/receipt.json")

OUTPUT.write_text(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# =========================
# Tampilkan hasil
# =========================

print("\n=== HASIL EKSTRAKSI ===")

print(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    )
)

print(f"\nJSON berhasil disimpan ke: {OUTPUT}")
