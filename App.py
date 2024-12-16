from flask import Flask, request, render_template, send_from_directory, redirect, url_for, flash, jsonify, send_file
import os
from datetime import datetime
import speech_recognition as sr
import time
from Deployment_Graph import main, main2
from Deployment_RAG import *
import warnings
warnings.filterwarnings("ignore")
import json

from pydub import AudioSegment
from io import BytesIO

app = Flask(__name__)
app.secret_key = "Lamii"

INPUT_FOLDER = './input'
OUTPUT_FOLDER = './outputs'
app.config['UPLOAD_FOLDER'] = OUTPUT_FOLDER

os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
main2()


@app.route('/')
def upload_form():
    return render_template('index.html')


@app.route('/get', methods=['POST'])
def process_message():
    data = request.get_json()
    user_message = data.get('message')
    response = main(user_message)

    response_json = jsonify({'response': response})
    response_json.headers['Content-Type'] = 'application/json; charset=utf-8'
    
    return response_json

@app.route('/outputs', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files or request.files['audio'].filename == '':
        return jsonify({'error': 'No audio file uploaded or no selected file.'}), 400

    audio_file = request.files['audio']
    audio = AudioSegment.from_file(BytesIO(audio_file.read()))
    wav_io = BytesIO()
    audio.export(wav_io, format='wav')
    wav_io.seek(0)

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_io) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language='vi-VN')
        print(f'Nhận diện được văn bản: {text}') 

    response = main(text)
    response_json = jsonify({
        'recognized_text': text,
        'response': response
    })
    response_json.headers['Content-Type'] = 'application/json; charset=utf-8'
    
    return response_json



@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    else:
        return "File không tồn tại!", 404

if __name__ == '__main__':
    app.run(debug=True)

