document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const tabUpload = document.getElementById('tab-upload');
    const tabCamera = document.getElementById('tab-camera');
    const areaUpload = document.getElementById('area-upload');
    const areaCamera = document.getElementById('area-camera');
    
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    
    const cameraFeed = document.getElementById('camera-feed');
    const captureBtn = document.getElementById('capture-btn');
    
    const previewArea = document.getElementById('preview-area');
    const imagePreview = document.getElementById('image-preview');
    const clearBtn = document.getElementById('clear-btn');
    const predictBtn = document.getElementById('predict-btn');
    const loadingArea = document.getElementById('loading-area');
    
    const resultPlaceholder = document.getElementById('result-placeholder');
    const resultContent = document.getElementById('result-content');
    
    const diseaseName = document.getElementById('disease-name');
    const confidenceText = document.getElementById('confidence-text');
    const confidenceFill = document.getElementById('confidence-fill');
    
    const causeText = document.getElementById('cause-text');
    const treatmentText = document.getElementById('treatment-text');
    const preventionText = document.getElementById('prevention-text');
    
    let stream = null;
    let selectedImageFile = null;

    // Tabs
    function setTab(tab) {
        if (tab === 'upload') {
            tabUpload.classList.add('active');
            tabCamera.classList.remove('active');
            areaUpload.classList.add('active');
            areaCamera.classList.remove('active');
            stopCamera();
        } else {
            tabCamera.classList.add('active');
            tabUpload.classList.remove('active');
            areaCamera.classList.add('active');
            areaUpload.classList.remove('active');
            startCamera();
        }
        resetPreview();
    }
    
    tabUpload.addEventListener('click', () => setTab('upload'));
    tabCamera.addEventListener('click', () => setTab('camera'));

    // Upload
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = "var(--accent)"; });
    dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = "var(--border-color)"; });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--border-color)";
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }
        selectedImageFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            showPreview(e.target.result);
        };
        reader.readAsDataURL(file);
    }

    // Camera
    async function startCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            cameraFeed.srcObject = stream;
        } catch (err) {
            console.error(err);
            alert('Could not access camera. Please ensure permissions are granted.');
        }
    }

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            cameraFeed.srcObject = null;
        }
    }

    captureBtn.addEventListener('click', () => {
        if (!cameraFeed.srcObject) return;
        const canvas = document.createElement('canvas');
        canvas.width = cameraFeed.videoWidth;
        canvas.height = cameraFeed.videoHeight;
        canvas.getContext('2d').drawImage(cameraFeed, 0, 0);
        
        canvas.toBlob((blob) => {
            selectedImageFile = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
            showPreview(canvas.toDataURL('image/jpeg'));
            stopCamera();
        }, 'image/jpeg');
    });

    // Preview
    function showPreview(src) {
        imagePreview.src = src;
        areaUpload.style.display = 'none';
        areaCamera.style.display = 'none';
        previewArea.style.display = 'block';
        
        resultPlaceholder.style.display = 'block';
        resultContent.style.display = 'none';
        predictBtn.style.display = 'flex';
        
        // Reset progress bar
        confidenceFill.style.width = '0%';
    }

    function resetPreview() {
        selectedImageFile = null;
        fileInput.value = '';
        previewArea.style.display = 'none';
        
        if (tabUpload.classList.contains('active')) {
            areaUpload.style.display = 'block';
        } else {
            areaCamera.style.display = 'block';
        }
        
        resultPlaceholder.style.display = 'block';
        resultContent.style.display = 'none';
    }

    clearBtn.addEventListener('click', () => {
        resetPreview();
        if (tabCamera.classList.contains('active')) {
            startCamera();
        }
    });

    // Prediction
    predictBtn.addEventListener('click', async () => {
        if (!selectedImageFile) return;

        previewArea.style.display = 'none';
        loadingArea.style.display = 'block';
        resultPlaceholder.style.display = 'block';
        resultContent.style.display = 'none';

        const formData = new FormData();
        formData.append('image', selectedImageFile);

        try {
            // Using standard fetch POST resolving exactly to the pure JSON API endpoint
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                displayResult(data);
            } else {
                throw new Error(data.error || 'Server error occurred during prediction.');
            }
        } catch (error) {
            alert('Prediction Failed: ' + error.message);
            previewArea.style.display = 'block';
        } finally {
            loadingArea.style.display = 'none';
        }
    });

    function displayResult(data) {
        previewArea.style.display = 'block';
        predictBtn.style.display = 'none'; // Hide predict button after success
        
        // Ensure clear button properly resets to let user predict again
        clearBtn.onclick = () => {
            resetPreview();
            predictBtn.style.display = 'flex';
            if (tabCamera.classList.contains('active')) startCamera();
            
            // Reattach standard event
            clearBtn.onclick = () => {
                resetPreview();
                if (tabCamera.classList.contains('active')) startCamera();
            };
        };

        resultPlaceholder.style.display = 'none';
        resultContent.style.display = 'block';

        diseaseName.textContent = data.disease;
        
        const conf = parseFloat(data.confidence);
        confidenceText.textContent = `${conf.toFixed(1)}%`;
        
        // Animate progress bar with a slight delay
        setTimeout(() => {
            confidenceFill.style.width = `${Math.min(conf, 100)}%`;
        }, 300);

        causeText.textContent = data.cause || 'No information available.';
        treatmentText.textContent = data.treatment || 'No information available.';
        preventionText.textContent = data.prevention || 'No information available.';
    }
});
