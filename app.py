from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from pymongo import MongoClient
from bson import ObjectId
import jwt
from google.cloud import speech
from google.cloud import storage
import gridfs
import subprocess
import librosa

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# MongoDB 연결 설정
mongo_url = "mongodb+srv://ckdgml1302:admin@cluster0.cw4wxud.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_url)
db = client.get_database('test')
recordings_collection = db['recordings']  # recordings 컬렉션
fs = gridfs.GridFS(db, collection='recordings')  # recordings GridFS

# 환경 변수 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\infra-falcon-388803-1d495943b174.json"
GCS_BUCKET_NAME = 'sttproject3-3'

# JWT 비밀 키
SECRET_KEY = 'your_secret_key'

# 토큰 검증 미들웨어
def verify_token(token):
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return decoded['userId']
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError) as e:
        print("Token verification failed:", str(e))
        return None

def convert_audio(input_data, input_format='m4a', output_format='wav', sample_rate=16000):
    input_file = "input." + input_format
    output_file = "output." + output_format

    with open(input_file, 'wb') as f:
        f.write(input_data)

    # ffmpeg를 사용하여 m4a를 wav로 변환하고 샘플 레이트를 16000Hz, 모노로 설정
    subprocess.run(['ffmpeg', '-i', input_file, '-ar', str(sample_rate), '-ac', '1', output_file], check=True)

    with open(output_file, 'rb') as f:
        output_data = f.read()

    os.remove(input_file)
    os.remove(output_file)

    return output_data

def upload_to_gcs(file_path, bucket_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    return f"gs://{bucket_name}/{destination_blob_name}"

def analyze_speech_rate(audio_file_path):
    y, sr = librosa.load(audio_file_path)
    duration = librosa.get_duration(y=y, sr=sr)
    words = librosa.effects.split(y, top_db=20)
    word_count = len(words)
    speech_rate = word_count / duration  # words per second
    
    # 점수화: 이상적인 속도를 150 ~ 170 wpm으로 가정하고 점수를 계산
    ideal_rate = 160 / 60  # words per second
    if speech_rate < ideal_rate * 0.8:
        score = "느린 속도"
    elif speech_rate > ideal_rate * 1.2:
        score = "빠른 속도"
    else:
        score = "적당한 속도"
    
    return speech_rate, score

@app.route('/recordings/<file_id>/transcript', methods=['GET'])
def get_transcript(file_id):
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return jsonify({'error': 'Authorization header missing'}), 403

    token = auth_header.split(' ')[1]
    user_id = verify_token(token)
    if not user_id:
        return jsonify({'error': 'Invalid token'}), 401

    print(f"file_id: {file_id}, user_id: {user_id}")

    try:
        # recordings.files에서 메타데이터 조회
        recording_file = fs.find_one({'_id': ObjectId(file_id)})
        if not recording_file:
            return jsonify({'error': 'Recording file not found'}), 404

        # recordings 컬렉션에서 메타데이터 조회
        recording_metadata = recordings_collection.find_one({
            'fileId': ObjectId(file_id),
            'userId': ObjectId(user_id)
        })
        if not recording_metadata:
            return jsonify({'error': 'Recording metadata not found'}), 404

        audio_data = recording_file.read()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    try:
        # 오디오 파일 변환 (m4a to wav) 및 샘플 레이트 16000Hz, 모노로 변환
        converted_audio = convert_audio(audio_data, input_format='m4a', output_format='wav', sample_rate=16000)

        # 변환된 오디오 파일을 임시로 저장하여 분석에 사용
        with open("temp.wav", "wb") as f:
            f.write(converted_audio)

        # GCS에 파일 업로드
        gcs_uri = upload_to_gcs("temp.wav", GCS_BUCKET_NAME, f"{file_id}.wav")

        # 비동기식 STT 변환
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(uri=gcs_uri)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,  # 16000Hz로 변경
            language_code="ko-KR"
        )

        operation = client.long_running_recognize(config=config, audio=audio)
        response = operation.result(timeout=300)

        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript

        # 말하기 속도 분석
        speech_rate, speed_score = analyze_speech_rate("temp.wav")

        # MongoDB에 STT 결과 업데이트
        recordings_collection.update_one(
            {'_id': ObjectId(recording_metadata['_id'])},
            {'$set': {'transcript': transcript}}
        )

        # 임시 파일 삭제
        os.remove("temp.wav")

        # 결과를 클라이언트에게 반환
        return jsonify({
            'transcript': transcript,
            'speech_rate': speech_rate,
            'speed_score': speed_score
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
