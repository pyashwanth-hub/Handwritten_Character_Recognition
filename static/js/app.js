/**
 * AI Handwritten Character Recognition - Main Application Controller
 * Connects Frontend UI with Flask REST API, manages drawing canvas,
 * live predictions, recognition history, and interactive animations.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Intelligent API endpoint resolution for decoupled frontend & backend
  const API_BASE = (window.location.protocol === 'file:' || (window.location.port !== '5000'))
    ? 'http://127.0.0.1:5000'
    : '';

  // 1. Initialize Components
  const canvasInstance = new DrawingCanvas('paint-canvas', 'canvas-hint');
  window.drawingCanvas = canvasInstance;

  const dashboardCharts = new DashboardCharts();
  window.dashboardCharts = dashboardCharts;

  // State
  let currentTab = 'draw'; // 'draw' | 'upload' | 'word'
  let uploadedImageBase64 = null;
  let recognitionHistory = [];
  let lastImagePayload = null;
  let lastPrediction = null;
  let lastModelUsed = 'CNN';

  // DOM Elements
  const navbar = document.getElementById('navbar');
  const tabBtns = document.querySelectorAll('.studio-tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const modelSelect = document.getElementById('model-select');
  const predictBtn = document.getElementById('predict-btn');
  const clearBtn = document.getElementById('clear-btn');
  const undoBtn = document.getElementById('undo-btn');
  const toolBtns = document.querySelectorAll('.tool-btn');
  const brushSlider = document.getElementById('brush-slider');
  const brushSizeVal = document.getElementById('brush-size-val');
  const stampBtns = document.querySelectorAll('.stamp-btn');

  const fileInput = document.getElementById('file-input');
  const uploadDropzone = document.getElementById('upload-dropzone');
  const uploadPrompt = document.getElementById('upload-prompt');
  const uploadPreviewImg = document.getElementById('upload-preview-img');

  const scanningOverlay = document.getElementById('scanning-overlay');

  // Hero Display Elements
  const heroCharDisplay = document.getElementById('hero-char-display');
  const heroCharText = document.getElementById('hero-char-text');
  const heroConfidence = document.getElementById('hero-confidence');
  const heroModelTag = document.getElementById('hero-model-tag');
  const heroLatency = document.getElementById('hero-latency');
  const heroAccuracyTag = document.getElementById('hero-accuracy-tag');
  const candidatesContainer = document.getElementById('candidates-container');
  const resultStatusBadge = document.getElementById('result-status-badge');

  // Enhancements: Unclear Notice, Retry & Correction Controls
  const unclearNotice = document.getElementById('unclear-notice');
  const btnUnclearRetry = document.getElementById('btn-unclear-retry');
  const retryBtn = document.getElementById('retry-btn');
  const clearResultBtn = document.getElementById('clear-result-btn');
  const correctionBox = document.getElementById('correction-box');
  const correctionSelect = document.getElementById('correction-select');
  const btnSubmitCorrection = document.getElementById('btn-submit-correction');
  const correctionFeedback = document.getElementById('correction-feedback');

  // Previews
  const previewOrig = document.getElementById('preview-orig');
  const previewThresh = document.getElementById('preview-thresh');
  const previewCentered = document.getElementById('preview-centered');
  const fmapsContainer = document.getElementById('fmaps-container');
  const fmapsDrawer = document.getElementById('fmaps-drawer');

  // CRNN Word Elements
  const crnnResultBox = document.getElementById('crnn-result-box');
  const crnnWordText = document.getElementById('crnn-word-text');
  const crnnCharsStrip = document.getElementById('crnn-chars-strip');

  // History & Metrics
  const historyTbody = document.getElementById('history-tbody');
  const clearHistoryBtn = document.getElementById('clear-history-btn');
  const exportHistoryBtn = document.getElementById('export-history-btn');

  // 2. Typewriter Effect in Hero Subtitle
  const typewriterElement = document.getElementById('typewriter-text');
  if (typewriterElement) {
    const phrases = [
      "Transform Your Handwriting into Digital Text with AI.",
      "Powered by Deep 2D Convolutional Neural Networks.",
      "Experience Sub-5ms Real-Time Neural Inference.",
      "Recognizing 36 Characters: Digits 0-9 & Alphabets A-Z."
    ];
    let phraseIdx = 0;
    let charIdx = 0;
    let isDeleting = false;

    function typeLoop() {
      const currentPhrase = phrases[phraseIdx];
      if (isDeleting) {
        typewriterElement.textContent = currentPhrase.substring(0, charIdx - 1);
        charIdx--;
      } else {
        typewriterElement.textContent = currentPhrase.substring(0, charIdx + 1);
        charIdx++;
      }

      let speed = isDeleting ? 30 : 60;
      if (!isDeleting && charIdx === currentPhrase.length) {
        speed = 2200;
        isDeleting = true;
      } else if (isDeleting && charIdx === 0) {
        isDeleting = false;
        phraseIdx = (phraseIdx + 1) % phrases.length;
        speed = 500;
      }
      setTimeout(typeLoop, speed);
    }
    typeLoop();
  }

  // 3. Navbar Scroll Effect
  window.addEventListener('scroll', () => {
    if (window.scrollY > 40) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  });

  // 4. Tab Switching (Draw vs Upload vs Word CRNN)
  tabBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      tabBtns.forEach((b) => b.classList.remove('active'));
      tabPanes.forEach((p) => p.classList.remove('active'));

      btn.classList.add('active');
      const targetTab = btn.getAttribute('data-tab');
      currentTab = targetTab;
      const targetPane = document.getElementById(`tab-${targetTab}`);
      if (targetPane) targetPane.classList.add('active');

      if (currentTab === 'word') {
        modelSelect.value = 'crnn';
        modelSelect.disabled = true;
      } else {
        modelSelect.disabled = false;
        if (modelSelect.value === 'crnn') modelSelect.value = 'cnn';
      }
      resetResults();
    });
  });

  // 5. Canvas Toolbar Controls
  toolBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      toolBtns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      const tool = btn.getAttribute('data-tool');
      canvasInstance.tool = tool;
    });
  });

  if (brushSlider) {
    brushSlider.addEventListener('input', (e) => {
      const val = parseInt(e.target.value);
      canvasInstance.brushSize = val;
      if (brushSizeVal) brushSizeVal.textContent = `${val}px`;
    });
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      canvasInstance.clear();
      resetResults();
      showToast('Canvas cleared', 'info');
    });
  }

  if (clearResultBtn) {
    clearResultBtn.addEventListener('click', () => {
      canvasInstance.clear();
      resetResults();
      showToast('Canvas cleared', 'info');
    });
  }

  if (retryBtn) {
    retryBtn.addEventListener('click', () => {
      if (predictBtn) predictBtn.click();
    });
  }

  if (btnUnclearRetry) {
    btnUnclearRetry.addEventListener('click', () => {
      canvasInstance.clear();
      resetResults();
      showToast('Canvas cleared. Try drawing with bolder, clearer strokes!', 'info');
    });
  }

  if (undoBtn) {
    undoBtn.addEventListener('click', () => {
      canvasInstance.undo();
    });
  }

  stampBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const char = btn.getAttribute('data-char');
      canvasInstance.stampCharacter(char);
      resetResults();
      showToast(`Stamped sample '${char}' on canvas`, 'info');
    });
  });

  // 6. Image Upload & Drag-and-Drop
  if (uploadDropzone && fileInput) {
    uploadDropzone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover'].forEach((eventName) => {
      uploadDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        uploadDropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      uploadDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        uploadDropzone.classList.remove('dragover');
      });
    });

    uploadDropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFile(e.target.files[0]);
      }
    });
  }

  function handleFile(file) {
    if (!file.type.match('image.*')) {
      showToast('Please select a valid image file (PNG, JPG, WebP)', 'error');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      uploadedImageBase64 = e.target.result;
      if (uploadPreviewImg) {
        uploadPreviewImg.src = uploadedImageBase64;
        uploadPreviewImg.style.display = 'block';
      }
      if (uploadPrompt) uploadPrompt.style.display = 'none';
      resetResults();
      showToast('Image uploaded successfully!', 'success');
    };
    reader.readAsDataURL(file);
  }

  function resetResults() {
    if (unclearNotice) unclearNotice.style.display = 'none';
    if (heroCharDisplay) heroCharDisplay.style.display = 'flex';
    if (heroCharText) heroCharText.textContent = '?';
    if (heroConfidence) heroConfidence.textContent = '0.0%';
    if (resultStatusBadge) {
      resultStatusBadge.textContent = 'Model Ready';
      resultStatusBadge.style.color = 'var(--neon-emerald)';
      resultStatusBadge.style.borderColor = 'rgba(16, 185, 129, 0.3)';
    }
    if (candidatesContainer) {
      candidatesContainer.innerHTML = '<div style="color:#64748b; font-size:0.85rem; text-align:center; padding:1rem;">Draw or upload a character and click Predict to inspect probabilities.</div>';
    }
    if (correctionFeedback) correctionFeedback.style.display = 'none';
  }

  // 7. Predict Handler
  if (predictBtn) {
    predictBtn.addEventListener('click', async () => {
      let imagePayload = null;

      if (currentTab === 'draw' || currentTab === 'word') {
        if (!canvasInstance.hasDrawn) {
          showToast('Please draw a character or word first!', 'error');
          return;
        }
        imagePayload = canvasInstance.getImageDataURL();
      } else if (currentTab === 'upload') {
        if (!uploadedImageBase64) {
          showToast('Please upload an image first!', 'error');
          return;
        }
        imagePayload = uploadedImageBase64;
      }

      lastImagePayload = imagePayload;

      // Show Radar Scanning Animation
      if (scanningOverlay) scanningOverlay.classList.add('active');
      predictBtn.disabled = true;
      predictBtn.innerHTML = `<span class="spinner"></span> Analyzing...`;

      try {
        if (currentTab === 'word') {
          // Word / Sequence CRNN Mode
          const response = await fetch(`${API_BASE}/api/predict-sequence`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imagePayload })
          });
          const res = await response.json();
          handleSequenceResponse(res, imagePayload);
        } else {
          // Single Character Mode
          const selectedModel = modelSelect ? modelSelect.value : 'cnn';
          lastModelUsed = selectedModel.toUpperCase();
          const response = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imagePayload, model: selectedModel })
          });
          const res = await response.json();
          handleSingleResponse(res, imagePayload);
        }
      } catch (err) {
        console.error(err);
        showToast('Connection error connecting to backend model.', 'error');
      } finally {
        if (scanningOverlay) scanningOverlay.classList.remove('active');
        predictBtn.disabled = false;
        predictBtn.innerHTML = `<span>✨ Predict Character</span>`;
      }
    });
  }

  function handleSingleResponse(res, originalImg) {
    // Requirement 10: Handle Unclear Handwriting
    if (res.is_unclear) {
      if (unclearNotice) unclearNotice.style.display = 'flex';
      if (heroCharDisplay) heroCharDisplay.style.display = 'none';
      if (crnnResultBox) crnnResultBox.style.display = 'none';
      if (resultStatusBadge) {
        resultStatusBadge.textContent = 'Unclear Handwriting';
        resultStatusBadge.style.color = '#fbbf24';
        resultStatusBadge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
      }
      if (candidatesContainer) {
        candidatesContainer.innerHTML = '<div style="color:#f59e0b; font-size:0.85rem; text-align:center; padding:1rem;">⚠️ Unable to recognize. Please try again with clearer strokes.</div>';
      }
      showToast('Unable to recognize. Please try again.', 'error');
      return;
    }

    if (!res.success) {
      showToast(res.error || 'Prediction failed.', 'error');
      return;
    }

    // Hide unclear notice and word box
    if (unclearNotice) unclearNotice.style.display = 'none';
    if (crnnResultBox) crnnResultBox.style.display = 'none';
    if (heroCharDisplay) heroCharDisplay.style.display = 'flex';

    lastPrediction = res.prediction;

    // Update Hero Result
    if (heroCharText) {
      heroCharText.textContent = res.prediction;
      heroCharText.parentElement.classList.remove('celebrate');
      void heroCharText.parentElement.offsetWidth;
      heroCharText.parentElement.classList.add('celebrate');
    }
    if (heroConfidence) heroConfidence.textContent = `${res.confidence}%`;
    if (heroModelTag) heroModelTag.textContent = res.model_used;
    if (heroLatency) heroLatency.textContent = `${res.latency_ms} ms`;
    if (heroAccuracyTag) heroAccuracyTag.textContent = `Model Acc: 99.1%`;

    if (resultStatusBadge) {
      resultStatusBadge.textContent = 'Recognized';
      resultStatusBadge.style.color = 'var(--neon-emerald)';
      resultStatusBadge.style.borderColor = 'rgba(16, 185, 129, 0.3)';
    }

    // Render Candidates Breakdown (Requirement 7)
    renderCandidates(res.top_candidates || []);

    // Render Preprocessing Steps (Requirement 2)
    if (res.previews) {
      if (previewOrig) previewOrig.src = res.previews.original || originalImg;
      if (previewThresh) previewThresh.src = res.previews.threshold || '';
      if (previewCentered) previewCentered.src = res.previews.centered || '';
    }

    // Render Conv2D Feature Maps
    renderFeatureMaps(res.feature_maps || []);

    // Setup Prediction Correction select
    if (correctionSelect) {
      correctionSelect.value = res.prediction;
    }
    if (correctionFeedback) correctionFeedback.style.display = 'none';

    // Add to History (Requirement 9)
    addHistoryItem({
      thumb: res.previews?.centered || originalImg,
      char: res.prediction,
      confidence: res.confidence,
      model: res.model_used,
      latency: res.latency_ms,
      timestamp: new Date().toLocaleTimeString()
    });

    // Celebration animation on high confidence
    if (res.confidence > 75) {
      triggerConfetti();
    }
    showToast(`Recognized as '${res.prediction}' with ${res.confidence}% confidence!`, 'success');
  }

  function handleSequenceResponse(res, originalImg) {
    if (!res.success) {
      showToast(res.error || 'Sequence recognition failed.', 'error');
      return;
    }

    if (unclearNotice) unclearNotice.style.display = 'none';
    if (heroCharDisplay) heroCharDisplay.style.display = 'none';
    if (crnnResultBox) {
      crnnResultBox.style.display = 'block';
      if (crnnWordText) crnnWordText.textContent = res.word_text || '---';
    }

    if (crnnCharsStrip && res.characters) {
      crnnCharsStrip.innerHTML = '';
      res.characters.forEach((c) => {
        const pill = document.createElement('div');
        pill.className = 'crnn-char-pill';
        pill.innerHTML = `
          <img src="${c.preview}" alt="${c.char}">
          <div style="font-weight:700; font-size:1.1rem; color:#fff;">${c.char}</div>
          <div style="font-size:0.75rem; color:#94a3b8;">${c.confidence}%</div>
        `;
        crnnCharsStrip.appendChild(pill);
      });
    }

    addHistoryItem({
      thumb: res.characters?.[0]?.preview || originalImg,
      char: res.word_text,
      confidence: res.avg_confidence,
      model: 'CRNN',
      latency: res.latency_ms,
      timestamp: new Date().toLocaleTimeString()
    });

    triggerConfetti();
    showToast(`Decoded text: "${res.word_text}" (${res.avg_confidence}%)`, 'success');
  }

  // 8. Prediction Correction Feature (Requirement 8)
  if (btnSubmitCorrection && correctionSelect) {
    btnSubmitCorrection.addEventListener('click', async () => {
      const correctedChar = correctionSelect.value;
      if (!correctedChar) {
        showToast('Please select the actual character to correct.', 'error');
        return;
      }

      const origChar = lastPrediction || '?';

      // Update UI immediately
      if (heroCharText) {
        heroCharText.textContent = correctedChar;
        heroCharText.parentElement.classList.remove('celebrate');
        void heroCharText.parentElement.offsetWidth;
        heroCharText.parentElement.classList.add('celebrate');
      }
      if (heroConfidence) heroConfidence.textContent = '100.0% (Corrected)';

      if (correctionFeedback) {
        correctionFeedback.textContent = `✅ Corrected from '${origChar}' to '${correctedChar}'. Logged for training!`;
        correctionFeedback.style.display = 'block';
      }

      // Update current history item
      if (recognitionHistory.length > 0) {
        recognitionHistory[0].char = correctedChar;
        recognitionHistory[0].confidence = 100.0;
        renderHistoryTable();
      }

      showToast(`Prediction updated to '${correctedChar}'!`, 'success');

      // Submit feedback to backend
      try {
        await fetch(`${API_BASE}/api/correct-prediction`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            original: origChar,
            corrected: correctedChar,
            model: lastModelUsed,
            image: lastImagePayload
          })
        });
      } catch (e) {
        console.warn('Could not log correction to server:', e);
      }
    });
  }

  function renderCandidates(candidates) {
    if (!candidatesContainer) return;
    candidatesContainer.innerHTML = '';

    candidates.forEach((cand, idx) => {
      const row = document.createElement('div');
      row.className = 'candidate-row';
      const isWinner = idx === 0;
      row.innerHTML = `
        <div class="cand-label" style="${isWinner ? 'color: var(--neon-cyan);' : ''}">${cand.char}</div>
        <div class="cand-bar-track">
          <div class="cand-bar-fill" style="width: 0%; ${isWinner ? 'background: var(--grad-cyan);' : ''}"></div>
        </div>
        <div class="cand-val">${cand.confidence}%</div>
      `;
      candidatesContainer.appendChild(row);

      setTimeout(() => {
        const fill = row.querySelector('.cand-bar-fill');
        if (fill) fill.style.width = `${Math.max(2, cand.confidence)}%`;
      }, 50 + idx * 40);
    });
  }

  function renderFeatureMaps(fmaps) {
    if (!fmapsContainer) return;
    fmapsContainer.innerHTML = '';
    if (!fmaps || fmaps.length === 0) {
      if (fmapsDrawer) fmapsDrawer.style.display = 'none';
      return;
    }
    if (fmapsDrawer) fmapsDrawer.style.display = 'block';

    fmaps.forEach((fm) => {
      const item = document.createElement('div');
      item.className = 'fmap-item';
      item.innerHTML = `
        <img src="${fm.preview}" alt="${fm.layer}">
        <span>${fm.layer}</span>
      `;
      fmapsContainer.appendChild(item);
    });
  }

  // 9. Recognition History Management
  function addHistoryItem(item) {
    recognitionHistory.unshift(item);
    if (recognitionHistory.length > 25) recognitionHistory.pop();
    renderHistoryTable();
  }

  function renderHistoryTable() {
    if (!historyTbody) return;
    historyTbody.innerHTML = '';

    if (recognitionHistory.length === 0) {
      historyTbody.innerHTML = `<tr><td colspan="6" style="text-align:center; color:#64748b; padding:2rem;">No characters recognized yet. Draw or upload to begin!</td></tr>`;
      return;
    }

    recognitionHistory.forEach((item, idx) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><img src="${item.thumb}" class="history-thumb" alt="${item.char}"></td>
        <td><span class="history-char-pill">${item.char}</span></td>
        <td><span style="color:var(--neon-emerald); font-weight:700;">${item.confidence}%</span></td>
        <td><span class="meta-pill">${item.model}</span></td>
        <td style="color:#94a3b8; font-family:var(--font-mono);">${item.latency} ms</td>
        <td style="color:#64748b; font-size:0.8rem;">${item.timestamp}</td>
      `;
      historyTbody.appendChild(tr);
    });
  }

  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener('click', () => {
      recognitionHistory = [];
      renderHistoryTable();
      showToast('Recognition history cleared', 'info');
    });
  }

  if (exportHistoryBtn) {
    exportHistoryBtn.addEventListener('click', () => {
      if (recognitionHistory.length === 0) {
        showToast('History is empty to export.', 'info');
        return;
      }
      const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(recognitionHistory, null, 2));
      const downloadAnchor = document.createElement('a');
      downloadAnchor.setAttribute('href', dataStr);
      downloadAnchor.setAttribute('download', `recognition_history_${Date.now()}.json`);
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
      showToast('Exported history as JSON!', 'success');
    });
  }

  // 10. Load Metrics & Charts from Backend
  async function loadBackendMetrics() {
    try {
      const res = await fetch(`${API_BASE}/api/model-info`);
      const json = await res.json();
      if (json.success && json.data) {
        const data = json.data;
        const cnnValAcc = data.models?.cnn?.val_acc || 99.08;
        const totalSamples = data.total_samples || 4888;

        const statAccuracy = document.getElementById('stat-model-acc');
        if (statAccuracy) statAccuracy.textContent = `${cnnValAcc}%`;

        const statSamples = document.getElementById('stat-total-samples');
        if (statSamples) statSamples.textContent = totalSamples.toLocaleString();

        dashboardCharts.init(data);
      }
    } catch (e) {
      console.warn("Could not load backend metrics. Using default dashboard metrics.");
      dashboardCharts.init(null);
    }
  }
  loadBackendMetrics();

  // 11. Learn More Modal Controller
  const modelModals = {
    cnn: {
      title: "Convolutional Neural Network (CNN) - Primary Model",
      badge: "Primary DL Architecture",
      body: `
        <p><strong>Architecture Overview:</strong> The primary model incorporates hierarchical 2D Convolutional layers (Conv2D) paired with ReLU non-linearities, spatial Max-Pooling layers, and adaptive stroke normalization.</p>
        <div style="background:rgba(255,255,255,0.05); padding:1rem; border-radius:12px; margin:1rem 0; font-family:var(--font-mono); font-size:0.85rem;">
          Input (28x28x1) &rarr; Conv2D(16 filters, 3x3) &rarr; MaxPool(2x2) &rarr; Conv2D(32 filters, 3x3) &rarr; MaxPool(2x2) &rarr; Dense(128) &rarr; Softmax(36)
        </div>
        <p><strong>High-Confidence Optimization:</strong> Features distance-transform stroke normalization to accurately identify letters like E, F, B, and D across thin, medium, and thick handwriting styles.</p>
      `
    },
    lenet5: {
      title: "LeNet-5 (Yann LeCun, 1998)",
      badge: "Classic OCR Network",
      body: `
        <p><strong>Historical Significance:</strong> Designed by Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner for check reading and digit recognition at Bell Labs.</p>
        <p>Features 5x5 convolution kernels and average/max subsampling. Excellent benchmark for digit recognition and fast inference on resource-constrained devices.</p>
      `
    },
    crnn: {
      title: "CRNN (Word & Sentence Recognition)",
      badge: "Sequence Modeling",
      body: `
        <p>Combines Convolutional layers for visual feature representation with sequence-based character segmentation to transcribe entire handwritten words or sentences seamlessly.</p>
      `
    },
    ann: {
      title: "ANN (Artificial Neural Network / MLP)",
      badge: "Dense Architecture",
      body: `
        <p>Fully connected multi-layer perceptron processing 784 flattened pixel inputs across dense layers of 256 and 128 neurons with non-linear activation.</p>
      `
    },
    svm: {
      title: "SVM (Support Vector Machine)",
      badge: "Maximum Margin",
      body: `
        <p>Uses non-linear Radial Basis Function (RBF) kernels to construct optimal hyperplanes separating character classes in high-dimensional space.</p>
      `
    },
    knn: {
      title: "KNN (K-Nearest Neighbors)",
      badge: "Instance-Based Voting",
      body: `
        <p>Classifies query character embeddings by majority vote among the 5 nearest neighbor samples based on Euclidean L2 distance in normalized pixel space.</p>
      `
    }
  };

  const modalOverlay = document.getElementById('model-modal');
  const modalTitle = document.getElementById('modal-title');
  const modalBadge = document.getElementById('modal-badge');
  const modalBody = document.getElementById('modal-body');
  const modalCloseBtn = document.getElementById('modal-close-btn');

  document.querySelectorAll('.btn-learn-more').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const modelKey = btn.getAttribute('data-model');
      const data = modelModals[modelKey];
      if (data && modalOverlay) {
        modalTitle.textContent = data.title;
        modalBadge.textContent = data.badge;
        modalBody.innerHTML = data.body;
        modalOverlay.classList.add('open');
      }
    });
  });

  if (modalCloseBtn && modalOverlay) {
    modalCloseBtn.addEventListener('click', () => modalOverlay.classList.remove('open'));
    modalOverlay.addEventListener('click', (e) => {
      if (e.target === modalOverlay) modalOverlay.classList.remove('open');
    });
  }

  // 12. Intersection Observer for Scroll Reveals
  const revealElements = document.querySelectorAll('.reveal');
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
        }
      });
    },
    { threshold: 0.1 }
  );
  revealElements.forEach((el) => observer.observe(el));

  // 13. Toast Notification Helper
  function showToast(message, type = 'info') {
    let container = document.getElementById('toast-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'toast-container';
      container.className = 'toast-container';
      document.body.appendChild(container);
    }
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icon = type === 'success' ? '✅' : type === 'error' ? '⚠️' : 'ℹ️';
    toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(100%)';
      setTimeout(() => toast.remove(), 300);
    }, 3200);
  }

  // 14. Confetti Animation
  function triggerConfetti() {
    const canvas = document.getElementById('confetti-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const confetti = [];
    const colors = ['#a855f7', '#06b6d4', '#ec4899', '#3b82f6', '#10b981', '#f59e0b'];

    for (let i = 0; i < 90; i++) {
      confetti.push({
        x: canvas.width / 2,
        y: canvas.height * 0.45,
        vx: (Math.random() - 0.5) * 16,
        vy: (Math.random() - 0.7) * 18,
        size: Math.random() * 8 + 4,
        color: colors[Math.floor(Math.random() * colors.length)],
        rotation: Math.random() * 360,
        rotSpeed: (Math.random() - 0.5) * 10,
        alpha: 1
      });
    }

    let frame = 0;
    function render() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      confetti.forEach((c) => {
        c.x += c.vx;
        c.y += c.vy;
        c.vy += 0.45;
        c.rotation += c.rotSpeed;
        c.alpha -= 0.015;

        if (c.alpha > 0) {
          ctx.save();
          ctx.translate(c.x, c.y);
          ctx.rotate((c.rotation * Math.PI) / 180);
          ctx.fillStyle = c.color;
          ctx.globalAlpha = Math.max(0, c.alpha);
          ctx.fillRect(-c.size / 2, -c.size / 2, c.size, c.size * 0.6);
          ctx.restore();
        }
      });

      frame++;
      if (frame < 70) {
        requestAnimationFrame(render);
      } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }
    render();
  }

  // =========================================================================
  // 15. Sentence & Multi-line Paragraph Recognition Studio Controller
  // =========================================================================
  const sentenceCanvas = new DrawingCanvas('sentence-canvas', 'sentence-canvas-hint');
  window.sentenceCanvas = sentenceCanvas;

  const btnModeSentence = document.getElementById('btn-mode-sentence');
  const btnModeCharacter = document.getElementById('btn-mode-character');
  const sentenceStudioContainer = document.getElementById('sentence-studio-container');
  const characterStudioContainer = document.getElementById('character-studio-container');

  // Mode Switcher
  if (btnModeSentence && btnModeCharacter) {
    btnModeSentence.addEventListener('click', () => {
      btnModeSentence.classList.add('active');
      btnModeCharacter.classList.remove('active');
      if (sentenceStudioContainer) sentenceStudioContainer.style.display = 'block';
      if (characterStudioContainer) characterStudioContainer.style.display = 'none';
      if (sentenceCanvas) sentenceCanvas.setupCanvas();
    });

    btnModeCharacter.addEventListener('click', () => {
      btnModeCharacter.classList.add('active');
      btnModeSentence.classList.remove('active');
      if (sentenceStudioContainer) sentenceStudioContainer.style.display = 'none';
      if (characterStudioContainer) characterStudioContainer.style.display = 'grid';
      if (canvasInstance) canvasInstance.setupCanvas();
    });
  }

  // Sentence Subtabs (Draw vs Upload)
  const sentTabBtns = document.querySelectorAll('.sentence-tab-btn');
  const sentTabPanes = {
    draw: document.getElementById('sent-tab-draw'),
    upload: document.getElementById('sent-tab-upload')
  };
  let currentSentTab = 'draw';
  let uploadedSentenceBase64 = null;

  sentTabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      sentTabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentSentTab = btn.dataset.sentTab;
      Object.entries(sentTabPanes).forEach(([tab, pane]) => {
        if (pane) pane.style.display = (tab === currentSentTab) ? 'block' : 'none';
      });
      if (currentSentTab === 'draw' && sentenceCanvas) {
        sentenceCanvas.setupCanvas();
      }
    });
  });

  // Sentence Canvas Controls
  const sentToolBtns = document.querySelectorAll('.sent-tool-btn');
  sentToolBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      sentToolBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if (sentenceCanvas) sentenceCanvas.tool = btn.dataset.tool;
    });
  });

  const sentenceBrushSlider = document.getElementById('sentence-brush-slider');
  const sentenceBrushVal = document.getElementById('sentence-brush-val');
  if (sentenceBrushSlider && sentenceBrushVal) {
    sentenceBrushSlider.addEventListener('input', (e) => {
      const sz = e.target.value;
      sentenceBrushVal.textContent = sz + 'px';
      if (sentenceCanvas) sentenceCanvas.brushSize = parseInt(sz);
    });
  }

  const sentenceUndoBtn = document.getElementById('sentence-undo-btn');
  if (sentenceUndoBtn) {
    sentenceUndoBtn.addEventListener('click', () => {
      if (sentenceCanvas) sentenceCanvas.undo();
    });
  }

  const sentenceClearBtn = document.getElementById('sentence-clear-btn');
  if (sentenceClearBtn) {
    sentenceClearBtn.addEventListener('click', () => {
      if (sentenceCanvas) sentenceCanvas.clear();
    });
  }

  // Sentence Quick Stamps
  const sentenceStampBtns = document.querySelectorAll('.sentence-stamp-btn');
  sentenceStampBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const sentText = btn.dataset.sent;
      if (sentenceCanvas && sentText) {
        sentenceCanvas.stampSentence(sentText);
        showToast('Stamped: "' + sentText.split('\n')[0].substring(0, 36) + '..."');
      }
    });
  });

  // Sentence File Upload
  const sentenceDropzone = document.getElementById('sentence-dropzone');
  const sentenceFileInput = document.getElementById('sentence-file-input');
  const btnBrowseSentence = document.getElementById('btn-browse-sentence');
  const sentencePreviewImg = document.getElementById('sentence-preview-img');
  const sentenceUploadPrompt = document.getElementById('sentence-upload-prompt');

  if (btnBrowseSentence && sentenceFileInput) {
    btnBrowseSentence.addEventListener('click', (e) => {
      e.stopPropagation();
      sentenceFileInput.click();
    });
  }

  if (sentenceDropzone && sentenceFileInput) {
    sentenceDropzone.addEventListener('click', () => sentenceFileInput.click());
    sentenceDropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      sentenceDropzone.classList.add('drag-over');
    });
    sentenceDropzone.addEventListener('dragleave', () => sentenceDropzone.classList.remove('drag-over'));
    sentenceDropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      sentenceDropzone.classList.remove('drag-over');
      if (e.dataTransfer.files.length) handleSentenceFile(e.dataTransfer.files[0]);
    });
    sentenceFileInput.addEventListener('change', (e) => {
      if (e.target.files.length) handleSentenceFile(e.target.files[0]);
    });
  }

  function handleSentenceFile(file) {
    if (!file.type.startsWith('image/')) {
      showToast('Please select a valid image file (PNG, JPG, WebP).');
      return;
    }
    const reader = new FileReader();
    reader.onload = (ev) => {
      uploadedSentenceBase64 = ev.target.result;
      if (sentencePreviewImg && sentenceUploadPrompt) {
        sentencePreviewImg.src = uploadedSentenceBase64;
        sentencePreviewImg.style.display = 'block';
        sentenceUploadPrompt.style.display = 'none';
      }
      showToast('Uploaded: ' + file.name);
    };
    reader.readAsDataURL(file);
  }

  // Output elements
  const sentenceOutputText = document.getElementById('sentence-output-text');
  const sentenceConfidenceVal = document.getElementById('sentence-confidence-val');
  const sentenceLatencyVal = document.getElementById('sentence-latency-val');
  const sentenceWordCount = document.getElementById('sentence-word-count');
  const sentenceLineCount = document.getElementById('sentence-line-count');
  const sentenceStatusBadge = document.getElementById('sentence-status-badge');
  const sentenceWordsContainer = document.getElementById('sentence-words-container');
  const sentenceUnclearNotice = document.getElementById('sentence-unclear-notice');
  const sentenceAnnotatedCard = document.getElementById('sentence-annotated-card');
  const sentenceAnnotatedImg = document.getElementById('sentence-annotated-img');
  const sentenceScanningOverlay = document.getElementById('sentence-scanning-overlay');

  let lastSentencePayload = null;

  // Sentence Predict Button
  const btnRecognizeSentence = document.getElementById('btn-recognize-sentence');
  if (btnRecognizeSentence) {
    btnRecognizeSentence.addEventListener('click', () => runSentenceRecognition());
  }

  const btnRetrySentence = document.getElementById('btn-retry-sentence');
  if (btnRetrySentence) {
    btnRetrySentence.addEventListener('click', () => runSentenceRecognition());
  }

  const btnSentenceUnclearRetry = document.getElementById('btn-sentence-unclear-retry');
  if (btnSentenceUnclearRetry) {
    btnSentenceUnclearRetry.addEventListener('click', () => runSentenceRecognition());
  }

  async function runSentenceRecognition() {
    let payload = null;
    if (currentSentTab === 'draw') {
      if (!sentenceCanvas.hasDrawn) {
        showToast('Please draw a sentence on the canvas or click a quick sample.');
        return;
      }
      payload = sentenceCanvas.getSentenceImageDataURL();
    } else {
      if (!uploadedSentenceBase64) {
        showToast('Please upload an image containing handwritten sentences.');
        return;
      }
      payload = uploadedSentenceBase64;
    }

    lastSentencePayload = payload;

    if (sentenceScanningOverlay) sentenceScanningOverlay.classList.add('scanning');
    if (btnRecognizeSentence) {
      btnRecognizeSentence.disabled = true;
      btnRecognizeSentence.innerHTML = '<span>⚡ Transcribing Neural Sequence...</span>';
    }
    if (sentenceStatusBadge) sentenceStatusBadge.textContent = 'Processing CRNN...';

    try {
      const resp = await fetch(`${API_BASE}/api/predict-sentence`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: payload })
      });
      const data = await resp.json();

      if (sentenceScanningOverlay) sentenceScanningOverlay.classList.remove('scanning');
      if (btnRecognizeSentence) {
        btnRecognizeSentence.disabled = false;
        btnRecognizeSentence.innerHTML = '<span>✨ Recognize Complete Sentence (CRNN)</span>';
      }

      if (data.is_blank) {
        showToast('Input image is blank. Please write or upload a sentence.');
        if (sentenceStatusBadge) sentenceStatusBadge.textContent = 'Blank Input';
        return;
      }

      if (data.is_unclear || !data.success) {
        if (sentenceUnclearNotice) sentenceUnclearNotice.style.display = 'flex';
        if (sentenceOutputText) sentenceOutputText.value = '';
        if (sentenceConfidenceVal) sentenceConfidenceVal.textContent = (data.confidence || 0) + '%';
        if (sentenceLatencyVal) sentenceLatencyVal.textContent = (data.latency_ms || 0) + ' ms';
        if (sentenceStatusBadge) sentenceStatusBadge.textContent = 'Unclear Handwriting';
        return;
      }

      // Success
      if (sentenceUnclearNotice) sentenceUnclearNotice.style.display = 'none';
      if (sentenceOutputText) sentenceOutputText.value = data.full_text;
      if (sentenceConfidenceVal) sentenceConfidenceVal.textContent = data.confidence + '%';
      if (sentenceLatencyVal) sentenceLatencyVal.textContent = data.latency_ms + ' ms';
      if (sentenceWordCount) sentenceWordCount.textContent = data.word_count + ' words';
      if (sentenceLineCount) sentenceLineCount.textContent = data.line_count + ' lines';
      if (sentenceStatusBadge) sentenceStatusBadge.textContent = 'CRNN Recognized';

      // Render Word Chips
      if (sentenceWordsContainer) {
        sentenceWordsContainer.innerHTML = '';
        (data.words || []).forEach((w, idx) => {
          const chip = document.createElement('span');
          chip.className = 'word-chip';
          chip.innerHTML = `${escapeHtml(w.word)} <small>${w.confidence}%</small>`;
          chip.title = `Word #${idx + 1}: ${w.word} (${w.confidence}%)`;
          sentenceWordsContainer.appendChild(chip);
        });
      }

      // Render Annotated Preview
      if (data.previews && data.previews.annotated && sentenceAnnotatedCard && sentenceAnnotatedImg) {
        sentenceAnnotatedImg.src = data.previews.annotated;
        sentenceAnnotatedCard.style.display = 'block';
      }

      // Add to recognition history
      addHistoryRecord(data.full_text.substring(0, 20) + (data.full_text.length > 20 ? '...' : ''), 'CRNN', data.confidence, data.latency_ms, 'sentence');

      // Celebration confetti on high confidence
      if (data.confidence >= 80.0) {
        triggerConfetti();
      }

      showToast('Sentence recognized successfully with ' + data.confidence + '% confidence!');

    } catch (err) {
      if (sentenceScanningOverlay) sentenceScanningOverlay.classList.remove('scanning');
      if (btnRecognizeSentence) {
        btnRecognizeSentence.disabled = false;
        btnRecognizeSentence.innerHTML = '<span>✨ Recognize Complete Sentence (CRNN)</span>';
      }
      showToast('API communication error: ' + err.message);
    }
  }

  // Copy Sentence Action
  const btnCopySentence = document.getElementById('btn-copy-sentence');
  if (btnCopySentence && sentenceOutputText) {
    btnCopySentence.addEventListener('click', () => {
      const text = sentenceOutputText.value;
      if (!text) {
        showToast('Nothing to copy. Recognize or type a sentence first.');
        return;
      }
      navigator.clipboard.writeText(text).then(() => {
        showFloatingCopyToast('📋 Copied recognized text to clipboard!');
      }).catch(() => {
        sentenceOutputText.select();
        document.execCommand('copy');
        showFloatingCopyToast('📋 Copied to clipboard!');
      });
    });
  }

  // Download as TXT Action
  const btnDownloadTxt = document.getElementById('btn-download-txt');
  if (btnDownloadTxt && sentenceOutputText) {
    btnDownloadTxt.addEventListener('click', () => {
      const text = sentenceOutputText.value;
      if (!text) {
        showToast('Nothing to download. Please transcribe a sentence first.');
        return;
      }
      const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'recognized_handwritten_sentence.txt';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      showToast('💾 Downloaded recognized_handwritten_sentence.txt');
    });
  }

  // Clear Sentence All Action
  const btnClearSentence = document.getElementById('btn-clear-sentence');
  if (btnClearSentence) {
    btnClearSentence.addEventListener('click', () => {
      if (sentenceCanvas) sentenceCanvas.clear();
      uploadedSentenceBase64 = null;
      if (sentencePreviewImg) {
        sentencePreviewImg.src = '';
        sentencePreviewImg.style.display = 'none';
      }
      if (sentenceUploadPrompt) sentenceUploadPrompt.style.display = 'block';
      if (sentenceFileInput) sentenceFileInput.value = '';
      if (sentenceOutputText) sentenceOutputText.value = '';
      if (sentenceConfidenceVal) sentenceConfidenceVal.textContent = '--%';
      if (sentenceLatencyVal) sentenceLatencyVal.textContent = '-- ms';
      if (sentenceWordCount) sentenceWordCount.textContent = '0 words';
      if (sentenceLineCount) sentenceLineCount.textContent = '0 lines';
      if (sentenceWordsContainer) sentenceWordsContainer.innerHTML = '<span style="color:#64748b; font-size:0.85rem;">Word tokens will appear here after recognition.</span>';
      if (sentenceAnnotatedCard) sentenceAnnotatedCard.style.display = 'none';
      if (sentenceUnclearNotice) sentenceUnclearNotice.style.display = 'none';
      if (sentenceStatusBadge) sentenceStatusBadge.textContent = 'CRNN Ready';
      showToast('Cleared all sentence inputs & results.');
    });
  }

  function showFloatingCopyToast(message) {
    const existing = document.querySelector('.copy-toast');
    if (existing) existing.remove();
    const toast = document.createElement('div');
    toast.className = 'copy-toast';
    toast.innerHTML = `<span>${message}</span>`;
    document.body.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(20px)';
      setTimeout(() => toast.remove(), 300);
    }, 2800);
  }

  function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
});
