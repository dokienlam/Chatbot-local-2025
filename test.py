import json
from gtts import gTTS

path = 'C:/Users/ADMIN/Documents/GitHub/Chatbot-local-2025/responses.json'

# Mở file JSON và đọc dữ liệu
with open(path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Giả sử dữ liệu là một chuỗi văn bản
text = data['responese']  # Thay đổi theo cấu trúc file JSON của bạn

# Chuyển văn bản thành giọng nói
tts = gTTS(text, lang='vi')

# Lưu file âm thanh
tts.save('output.mp3')

print("Đã chuyển văn bản thành giọng nói và lưu vào 'output.mp3'")
