class ForeignerCardClassifier {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.webcamStream = null;
        
        this.initializeElements();
        this.setupEventListeners();
        this.loadModel();
    }
    
    initializeElements() {
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.imagePreview = document.getElementById('imagePreview');
        this.result = document.getElementById('result');
        this.resultText = document.getElementById('resultText');
        this.progress = document.getElementById('progress');
        this.progressBar = document.getElementById('progressBar');
        this.modelStatus = document.getElementById('modelStatus');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.startWebcamBtn = document.getElementById('startWebcam');
        this.stopWebcamBtn = document.getElementById('stopWebcam');
        this.captureBtn = document.getElementById('capturePhoto');
        this.webcam = document.getElementById('webcam');
        this.canvas = document.getElementById('canvas');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusText = document.getElementById('statusText');
        this.confidenceBar = document.getElementById('confidenceBar');
        this.confidenceFill = document.getElementById('confidenceFill');
        this.confidenceText = document.getElementById('confidenceText');
    }
    
    setupEventListeners() {
        // 파일 업로드 이벤트
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // 드래그 앤 드롭 이벤트
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });
        
        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('dragover');
        });
        
        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFile(files[0]);
            }
        });
        
        // 버튼 이벤트
        this.analyzeBtn.addEventListener('click', () => this.analyzeImage());
        this.clearBtn.addEventListener('click', () => this.clearResult());
        
        // 웹캠 이벤트
        this.startWebcamBtn.addEventListener('click', () => this.startWebcam());
        this.stopWebcamBtn.addEventListener('click', () => this.stopWebcam());
        this.captureBtn.addEventListener('click', () => this.capturePhoto());
    }
    
    updateStatus(text, type = 'loading') {
        this.statusText.textContent = text;
        this.statusIndicator.className = `status-indicator status-${type}`;
    }
    
    async loadModel() {
        try {
            this.updateStatus('AI 모델 로딩 중...', 'loading');
            this.showProgress('🤖 AI 모델 로딩 중...', 20);
            
            // 모델 경로 확인 (상대 경로로 모델 파일 찾기)
            const modelUrl = './model.json';
            
            this.showProgress('🤖 모델 다운로드 중...', 40);
            
            // 모델 로드
            this.model = await tf.loadLayersModel(modelUrl);
            
            this.showProgress('🤖 모델 초기화 중...', 80);
            
            // 모델 워밍업 (첫 번째 예측을 빠르게 하기 위해)
            const dummyInput = tf.zeros([1, 224, 224, 3]);
            this.model.predict(dummyInput).dispose();
            dummyInput.dispose();
            
            this.showProgress('✅ 모델 로딩 완료!', 100);
            this.isModelLoaded = true;
            
            setTimeout(() => {
                this.hideProgress();
                this.updateStatus('AI 모델 준비 완료', 'ready');
            }, 1000);
            
            console.log('모델 로딩 완료');
            console.log('입력 형태:', this.model.inputs[0].shape);
            console.log('출력 형태:', this.model.outputs[0].shape);
            
        } catch (error) {
            console.error('모델 로딩 실패:', error);
            this.hideProgress();
            this.updateStatus('모델 로딩 실패', 'error');
            this.showResult('❌ AI 모델 로딩에 실패했습니다. 페이지를 새로고침해 주세요.', false);
        }
    }
    
    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.handleFile(file);
        }
    }
    
    handleFile(file) {
        if (!file.type.startsWith('image/')) {
            this.showResult('❌ 이미지 파일만 업로드 가능합니다.', false);
            return;
        }
        
        // 파일 크기 체크 (10MB 제한)
        if (file.size > 10 * 1024 * 1024) {
            this.showResult('❌ 파일 크기가 너무 큽니다. 10MB 이하의 파일을 선택해주세요.', false);
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            this.imagePreview.src = e.target.result;
            this.imagePreview.style.display = 'block';
            this.analyzeBtn.style.display = 'inline-block';
            this.clearBtn.style.display = 'inline-block';
            this.hideResult();
            
            // 자동 분석 (모델이 로드된 경우)
            if (this.isModelLoaded) {
                setTimeout(() => this.analyzeImage(), 100);
            }
        };
        reader.readAsDataURL(file);
    }
    
    async startWebcam() {
        try {
            this.webcamStream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 }, 
                    height: { ideal: 480 },
                    facingMode: 'environment' // 후면 카메라 우선
                } 
            });
            this.webcam.srcObject = this.webcamStream;
            this.webcam.style.display = 'block';
            
            this.startWebcamBtn.style.display = 'none';
            this.stopWebcamBtn.style.display = 'inline-block';
            this.captureBtn.style.display = 'inline-block';
        } catch (error) {
            console.error('웹캠 시작 실패:', error);
            this.showResult('❌ 웹캠에 접근할 수 없습니다. 카메라 권한을 확인해주세요.', false);
        }
    }
    
    stopWebcam() {
        if (this.webcamStream) {
            this.webcamStream.getTracks().forEach(track => track.stop());
            this.webcamStream = null;
        }
        
        this.webcam.style.display = 'none';
        this.startWebcamBtn.style.display = 'inline-block';
        this.stopWebcamBtn.style.display = 'none';
        this.captureBtn.style.display = 'none';
    }
    
    capturePhoto() {
        const context = this.canvas.getContext('2d');
        this.canvas.width = this.webcam.videoWidth;
        this.canvas.height = this.webcam.videoHeight;
        
        context.drawImage(this.webcam, 0, 0);
        
        const dataURL = this.canvas.toDataURL('image/jpeg', 0.8);
        this.imagePreview.src = dataURL;
        this.imagePreview.style.display = 'block';
        this.analyzeBtn.style.display = 'inline-block';
        this.clearBtn.style.display = 'inline-block';
        
        this.stopWebcam();
        this.hideResult();
        
        // 자동 분석
        if (this.isModelLoaded) {
            setTimeout(() => this.analyzeImage(), 100);
        }
    }
    
    async analyzeImage() {
        if (!this.isModelLoaded) {
            this.showResult('❌ AI 모델이 아직 로딩 중입니다. 잠시 후 다시 시도해주세요.', false);
            return;
        }
        
        if (!this.imagePreview.src) {
            this.showResult('❌ 분석할 이미지를 선택해주세요.', false);
            return;
        }
        
        try {
            this.showProgress('🔍 이미지 분석 중...', 30);
            this.analyzeBtn.disabled = true;
            
            // 이미지 전처리
            this.showProgress('🔍 이미지 전처리 중...', 50);
            const tensor = await this.preprocessImage(this.imagePreview);
            
            // 예측 수행
            this.showProgress('🤖 AI 분석 중...', 75);
            const prediction = this.model.predict(tensor);
            const score = await prediction.data();
            
            // 메모리 정리
            tensor.dispose();
            prediction.dispose();
            
            this.showProgress('✅ 분석 완료!', 100);
            
            // 결과 해석
            const confidence = score[0];
            const isForeignerCard = confidence > 0.5;
            const percentage = (isForeignerCard ? confidence : (1 - confidence)) * 100;
            
            let resultText, emoji;
            if (isForeignerCard) {
                resultText = `외국인등록증 뒷면으로 판별됩니다`;
                emoji = '✅';
            } else {
                resultText = `외국인등록증 뒷면이 아닙니다`;
                emoji = '❌';
            }
            
            // 신뢰도 표시
            this.showConfidence(percentage, isForeignerCard);
            
            setTimeout(() => {
                this.hideProgress();
                this.showResult(`${emoji} ${resultText}`, isForeignerCard);
                this.analyzeBtn.disabled = false;
            }, 500);
            
        } catch (error) {
            console.error('분석 중 오류:', error);
            this.hideProgress();
            this.analyzeBtn.disabled = false;
            this.showResult('❌ 이미지 분석 중 오류가 발생했습니다. 다시 시도해주세요.', false);
        }
    }
    
    async preprocessImage(imgElement) {
        return tf.tidy(() => {
            // 이미지를 텐서로 변환
            let tensor = tf.browser.fromPixels(imgElement);
            
            // 224x224로 리사이즈
            tensor = tf.image.resizeBilinear(tensor, [224, 224]);
            
            // 배치 차원 추가 (1, 224, 224, 3)
            tensor = tensor.expandDims(0);
            
            // 정규화 (0-1 범위)
            tensor = tensor.div(255.0);
            
            return tensor;
        });
    }
    
    showProgress(text, percentage) {
        this.progress.style.display = 'block';
        this.progressBar.style.width = percentage + '%';
        this.modelStatus.textContent = text;
    }
    
    hideProgress() {
        this.progress.style.display = 'none';
    }
    
    showConfidence(percentage, isPositive) {
        this.confidenceBar.style.display = 'block';
        this.confidenceFill.style.width = percentage + '%';
        this.confidenceFill.style.background = isPositive ? 
            'linear-gradient(135deg, #4caf50 0%, #45a049 100%)' : 
            'linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%)';
        this.confidenceText.textContent = `신뢰도: ${percentage.toFixed(1)}%`;
    }
    
    showResult(text, success) {
        this.resultText.textContent = text;
        this.result.className = success ? 'result success' : 'result error';
        this.result.style.display = 'block';
    }
    
    hideResult() {
        this.result.style.display = 'none';
        this.confidenceBar.style.display = 'none';
    }
    
    clearResult() {
        this.imagePreview.style.display = 'none';
        this.imagePreview.src = '';
        this.analyzeBtn.style.display = 'none';
        this.clearBtn.style.display = 'none';
        this.analyzeBtn.disabled = false;
        this.hideResult();
        this.fileInput.value = '';
        this.stopWebcam();
    }
}

// 페이지 로딩 완료 후 초기화
document.addEventListener('DOMContentLoaded', () => {
    new ForeignerCardClassifier();
});