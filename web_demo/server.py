"""
웹 데모 서버 실행 스크립트
"""
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
import threading

class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def run_server(port=8000):
    """웹 서버 실행"""
    # web_demo 디렉토리로 이동
    web_demo_dir = os.path.join(os.path.dirname(__file__), 'web_demo')
    if os.path.exists(web_demo_dir):
        os.chdir(web_demo_dir)
    else:
        print("❌ web_demo 디렉토리를 찾을 수 없습니다.")
        return
    
    # 필요한 파일 확인
    required_files = ['index.html', 'classifier.js']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ 다음 파일들이 없습니다: {missing_files}")
        return
    
    # 모델 파일 확인
    model_files = ['model.json']
    model_missing = [f for f in model_files if not os.path.exists(f)]
    
    if model_missing:
        print("⚠️ TensorFlow.js 모델 파일이 없습니다.")
        print("다음 명령어로 모델을 변환하세요:")
        print("python src/convert_to_tfjs.py --model_path models/best_model.h5 --output_dir web_demo")
        print("\n모델 없이 웹 데모를 실행합니다. (모델 로딩 오류 발생 예상)")
    
    try:
        # HTTP 서버 시작
        server = HTTPServer(('localhost', port), CORSHTTPRequestHandler)
        url = f"http://localhost:{port}"
        
        print(f"🚀 웹 데모 서버 시작됨: {url}")
        print("브라우저가 자동으로 열립니다...")
        print("종료하려면 Ctrl+C를 누르세요")
        
        # 브라우저 자동 열기 (1초 후)
        def open_browser():
            import time
            time.sleep(1)
            webbrowser.open(url)
        
        threading.Thread(target=open_browser, daemon=True).start()
        
        # 서버 실행
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n✅ 서버가 정상적으로 종료되었습니다.")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ 포트 {port}가 이미 사용 중입니다.")
            print(f"다른 포트로 실행하세요: python web_demo/server.py --port {port+1}")
        else:
            print(f"❌ 서버 시작 오류: {e}")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='웹 데모 서버 실행')
    parser.add_argument('--port', type=int, default=8000, help='서버 포트 (기본값: 8000)')
    
    args = parser.parse_args()
    
    print("=== 외국인등록증 뒷면 분류기 웹 데모 ===")
    run_server(args.port)
