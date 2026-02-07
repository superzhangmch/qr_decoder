#!/usr/bin/env python3.11
"""
QR Code Decoder Web App
Run: python3.11 qr_web.py
Visit: http://<your-ip>:5000 on your phone
"""

import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from qr_decode import decode_qr, decode_qr_multi
import tempfile

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>QR Decoder</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 20px;
            color: #fff;
        }
        .container {
            max-width: 500px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 20px;
            font-size: 24px;
        }
        .upload-area {
            background: rgba(255,255,255,0.1);
            border: 2px dashed rgba(255,255,255,0.3);
            border-radius: 16px;
            padding: 30px;
            text-align: center;
            margin-bottom: 20px;
        }
        .upload-area.dragover {
            border-color: #4CAF50;
            background: rgba(76, 175, 80, 0.2);
        }
        .btn {
            display: inline-block;
            padding: 14px 28px;
            margin: 8px;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn:active {
            transform: scale(0.95);
        }
        .btn-camera {
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
        }
        .btn-gallery {
            background: linear-gradient(135deg, #2196F3, #1976D2);
            color: white;
        }
        input[type="file"] { display: none; }
        #preview {
            max-width: 100%;
            max-height: 300px;
            border-radius: 12px;
            margin: 15px 0;
            display: none;
        }
        #result {
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
            display: none;
        }
        #result.success {
            background: rgba(76, 175, 80, 0.2);
            border: 1px solid #4CAF50;
        }
        #result.error {
            background: rgba(244, 67, 54, 0.2);
            border: 1px solid #f44336;
        }
        #result-text {
            word-break: break-all;
            font-size: 14px;
            line-height: 1.6;
        }
        #result-text a {
            color: #64B5F6;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid rgba(255,255,255,0.3);
            border-top-color: #fff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .copy-btn {
            background: rgba(255,255,255,0.2);
            border: none;
            padding: 8px 16px;
            border-radius: 8px;
            color: white;
            margin-top: 10px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>QR Code Decoder</h1>

        <div class="upload-area" id="dropZone">
            <p style="margin-bottom: 15px; opacity: 0.8;">Take a photo or select an image</p>
            <label class="btn btn-camera">
                Camera
                <input type="file" id="cameraInput" accept="image/*" capture="environment">
            </label>
            <label class="btn btn-gallery">
                Gallery
                <input type="file" id="galleryInput" accept="image/*">
            </label>
            <img id="preview" alt="Preview">
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Decoding...</p>
        </div>

        <div id="result">
            <div id="result-text"></div>
            <button class="copy-btn" onclick="copyResult()">Copy</button>
        </div>
    </div>

    <script>
        const dropZone = document.getElementById('dropZone');
        const preview = document.getElementById('preview');
        const result = document.getElementById('result');
        const resultText = document.getElementById('result-text');
        const loading = document.getElementById('loading');

        document.getElementById('cameraInput').onchange = handleFile;
        document.getElementById('galleryInput').onchange = handleFile;

        dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add('dragover'); };
        dropZone.ondragleave = () => dropZone.classList.remove('dragover');
        dropZone.ondrop = (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) processFile(e.dataTransfer.files[0]);
        };

        function handleFile(e) {
            if (e.target.files.length) processFile(e.target.files[0]);
        }

        function processFile(file) {
            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                preview.style.display = 'block';
            };
            reader.readAsDataURL(file);

            // Upload and decode
            loading.style.display = 'block';
            result.style.display = 'none';

            const formData = new FormData();
            formData.append('image', file);

            fetch('/decode', { method: 'POST', body: formData })
                .then(r => r.json())
                .then(data => {
                    loading.style.display = 'none';
                    result.style.display = 'block';

                    if (data.success && data.results && data.results.length > 0) {
                        result.className = 'success';
                        let html = '';
                        if (data.results.length > 1) {
                            html += '<div style="margin-bottom:10px;opacity:0.7">Found ' + data.results.length + ' QR codes:</div>';
                        }
                        data.results.forEach((text, i) => {
                            if (data.results.length > 1) {
                                html += '<div style="margin-bottom:8px"><b>' + (i+1) + '.</b> ';
                            }
                            if (text && text.match(/^https?:\/\//i)) {
                                html += '<a href="' + text + '" target="_blank">' + text + '</a>';
                            } else {
                                html += text || '(empty)';
                            }
                            if (data.results.length > 1) html += '</div>';
                        });
                        resultText.innerHTML = html;
                    } else if (data.success) {
                        result.className = 'error';
                        resultText.textContent = 'No QR codes found';
                    } else {
                        result.className = 'error';
                        resultText.textContent = 'Error: ' + data.error;
                    }
                })
                .catch(err => {
                    loading.style.display = 'none';
                    result.style.display = 'block';
                    result.className = 'error';
                    resultText.textContent = 'Network error: ' + err.message;
                });
        }

        function copyResult() {
            const text = resultText.textContent || resultText.innerText;
            navigator.clipboard.writeText(text).then(() => {
                alert('Copied!');
            });
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/decode', methods=['POST'])
def decode():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image uploaded'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})

    try:
        # Save to debug location
        debug_path = '/Users/zhangmiaochang/Desktop/qr_debug/uploaded.jpg'
        file.save(debug_path)
        print(f"[DECODE] Saved to {debug_path}", flush=True)

        # Try to decode up to 3 QR codes
        results = decode_qr_multi(debug_path, max_codes=3)
        print(f"[DECODE] Found {len(results)} QR code(s)", flush=True)
        for i, r in enumerate(results):
            print(f"[DECODE] {i+1}: {r[:60]}...", flush=True)

        return jsonify({'success': True, 'results': results, 'count': len(results)})

    except Exception as e:
        print(f"[DECODE] Error: {e}", flush=True)
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    import socket

    # Get local IP
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except:
        ip = '127.0.0.1'
    finally:
        s.close()

    port = 8080
    print("=" * 50)
    print("QR Code Decoder Web App")
    print("=" * 50)
    print(f"\nVisit on your phone: http://{ip}:{port}")
    print(f"Or on this computer: http://localhost:{port}")
    print("\nPress Ctrl+C to stop\n")

    app.run(host='0.0.0.0', port=port, debug=False)
