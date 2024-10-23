from flask import Flask, request, render_template, send_from_directory, redirect, url_for, flash, jsonify, send_file
# from werkzeug.utils import secure_filename
import os
from datetime import datetime
import speech_recognition as sr
# import soundfile
import time
import warnings
warnings.filterwarnings("ignore")


from pydub import AudioSegment
from io import BytesIO
# from main import MainApp 

app = Flask(__name__)
app.secret_key = "Lamii"

DATA_DIR = "data/"
CHROMA_PATH = "chroma/"
llm_model = 'llama3.1'
embeddings_model = 'nomic-embed-text'

# main_app = MainApp(DATA_DIR, CHROMA_PATH, llm_model, embeddings_model)

INPUT_FOLDER = './input'
OUTPUT_FOLDER = './outputs'
app.config['UPLOAD_FOLDER'] = OUTPUT_FOLDER

os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def upload_form():
    return render_template('index.html')


@app.route('/get', methods=['POST'])
def process_message():
    # Lấy dữ liệu từ yêu cầu POST
    data = request.get_json()
    user_message = data.get('message')

    # Xử lý tin nhắn (ví dụ: trả lời người dùng)
    response = f"Bạn đã nói: {user_message}"

    # Trả về phản hồi JSON cho client
    print(user_message)
    return jsonify({'response': response}), user_message

@app.route('/outputs', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded.'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    # Đọc file audio trực tiếp từ request mà không lưu vào thư mục
    try:
        # Chuyển đổi file audio sang WAV trong bộ nhớ
        audio = AudioSegment.from_file(BytesIO(audio_file.read()))
        wav_io = BytesIO()
        audio.export(wav_io, format='wav')  # Xuất file dưới định dạng WAV vào bộ nhớ
        wav_io.seek(0)  # Đặt con trỏ về vị trí đầu của file

        # Chuyển đổi file WAV sang văn bản
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)  # Đọc toàn bộ nội dung file
            try:
                text = recognizer.recognize_google(audio_data, language='vi-VN')  # Chuyển đổi giọng nói thành văn bản
                print(f'Nhận diện được văn bản: {text}')  # In ra console để kiểm tra
                return jsonify({'text': text})  # Trả về văn bản đã nhận diện
            except sr.UnknownValueError:
                return jsonify({'error': 'Could not understand audio'}), 400
            except sr.RequestError as e:
                return jsonify({'error': f'Request error: {e}'}), 500

    except Exception as e:
        return jsonify({'error': f'Failed to process audio: {e}'}), 500

# @app.route("/get", methods=['POST'])
# def get_bot_response():
#     data = request.get_json()
#     user_text = data.get('msg', '')
#     if user_text:
#         # Giả sử có hàm xử lý logic chatbot ở đây
#         response = f"Phản hồi cho: {user_text}"
#         return jsonify({'response': response})
#     return jsonify({'response': "Không nhận được câu hỏi!"})


# @app.route("/get")
# def get_bot_response():
#     userText = request.args.get('msg')
#     print(userText)
    # if userText:
    #     ans = main_app.ask_text(userText)
    #     return jsonify(ans)  
    # return jsonify("Không nhận được câu hỏi!") 

# @app.route('/outputs', methods=['GET'])
# def upload_audio():
#     filename = "input/user_recording.webm"
#     flash('File uploaded successfully!')

#     return jsonify({'file': filename})


# @app.route('/outputs', methods=['POST'])
# def upload_audio():
#     if 'audio' not in request.files:
#         return jsonify({'error': 'No audio file uploaded.'}), 400

#     audio_file = request.files['audio']
#     if audio_file.filename == '':
#         return jsonify({'error': 'No selected file.'}), 400

#     # Lưu tệp vào thư mục OUTPUT_FOLDER
#     audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
#     audio_file.save(audio_file_path)

#     flash('File uploaded successfully!')
#     return jsonify({'file': audio_file.filename})


# @app.route("/bot_speech_to_text")
# def bot_speech_to_text():
#     file_wav = "input/user_recording.wav"
#     ans = speech_to_text(file_wav)
#     return jsonify(ans) 


# def speech_to_text(file_name):
#     r = sr.Recognizer()
#     try:
#         print("đang convert")
#         print(file_name)
#         with sr.AudioFile(file_name) as source:
#             audio_data = r.record(source)  
#             text = r.recognize_google(audio_data, language='vi-VN')
#             print("text:", text)
#             ans = main_app.ask_wav(text)
#             print(ans)
#             return ans
        
    # except sr.UnknownValueError:
    #     print("Không nhận diện được giọng nói.")
    # except sr.RequestError as e:
    #     print(f"Lỗi trong việc yêu cầu; {e}")
    # except FileNotFoundError:
    #     print(f"Không tìm thấy file")
    # except KeyboardInterrupt:
    #     print("Dừng nhận diện.")
    # return None

@app.route('/download/<filename>')
def download_file(filename):
    # Kiểm tra nếu file tồn tại
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        # Trả file về trình duyệt để tải xuống
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    else:
        return "File không tồn tại!", 404

# @app.route('/downloads/outputs/<filename>')
# def download_output_file(filename):  # Đổi tên hàm này
#     file_path = os.path.join("outputs", filename)
#     if not os.path.exists(file_path):
#         return "File not found", 404
#     return send_from_directory("outputs", filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
