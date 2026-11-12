/**
 * High-Precision Digital Drawing Canvas
 * Smooth Bezier handwriting strokes, brush sizing, eraser, undo, and sample stamps.
 */

class DrawingCanvas {
  constructor(canvasId, hintId) {
    this.canvas = document.getElementById(canvasId);
    this.hint = document.getElementById(hintId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');

    this.isDrawing = false;
    this.tool = 'brush'; // 'brush' or 'eraser'
    this.brushSize = 22;
    this.strokeColor = '#ffffff';
    this.hasDrawn = false;

    this.undoStack = [];
    this.points = [];

    this.setupCanvas();
    this.bindEvents();
  }

  setupCanvas() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.width = rect.width || 380;
    this.height = rect.height || 380;

    this.canvas.width = this.width * dpr;
    this.canvas.height = this.height * dpr;
    this.ctx.scale(dpr, dpr);

    this.clear(false);
  }

  clear(saveState = true) {
    if (saveState) this.saveState();
    this.ctx.fillStyle = '#000000';
    this.ctx.fillRect(0, 0, this.width, this.height);
    this.hasDrawn = false;
    if (this.hint) this.hint.classList.remove('hidden');
  }

  saveState() {
    this.undoStack.push(this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height));
    if (this.undoStack.length > 20) this.undoStack.shift();
  }

  undo() {
    if (this.undoStack.length > 0) {
      const state = this.undoStack.pop();
      this.ctx.putImageData(state, 0, 0);
      this.hasDrawn = true;
    } else {
      this.clear(false);
    }
  }

  getPointerPos(e) {
    const rect = this.canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: clientX - rect.left,
      y: clientY - rect.top
    };
  }

  startDrawing(e) {
    e.preventDefault();
    this.saveState();
    this.isDrawing = true;
    this.hasDrawn = true;
    if (this.hint) this.hint.classList.add('hidden');

    const pos = this.getPointerPos(e);
    this.points = [pos];

    this.ctx.beginPath();
    this.ctx.arc(pos.x, pos.y, this.brushSize / 2, 0, Math.PI * 2);
    this.ctx.fillStyle = this.tool === 'eraser' ? '#000000' : this.strokeColor;
    this.ctx.fill();
  }

  draw(e) {
    if (!this.isDrawing) return;
    e.preventDefault();

    const pos = this.getPointerPos(e);
    this.points.push(pos);

    if (this.points.length < 3) {
      const b = this.points[0];
      this.ctx.beginPath();
      this.ctx.arc(b.x, b.y, this.brushSize / 2, 0, Math.PI * 2);
      this.ctx.fillStyle = this.tool === 'eraser' ? '#000000' : this.strokeColor;
      this.ctx.fill();
      return;
    }

    this.ctx.lineWidth = this.brushSize;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
    this.ctx.strokeStyle = this.tool === 'eraser' ? '#000000' : this.strokeColor;

    this.ctx.beginPath();
    this.ctx.moveTo(this.points[0].x, this.points[0].y);

    for (let i = 1; i < this.points.length - 2; i++) {
      const xc = (this.points[i].x + this.points[i + 1].x) / 2;
      const yc = (this.points[i].y + this.points[i + 1].y) / 2;
      this.ctx.quadraticCurveTo(this.points[i].x, this.points[i].y, xc, yc);
    }

    const last = this.points.length - 1;
    this.ctx.quadraticCurveTo(
      this.points[last - 1].x,
      this.points[last - 1].y,
      this.points[last].x,
      this.points[last].y
    );
    this.ctx.stroke();
  }

  stopDrawing() {
    this.isDrawing = false;
    this.points = [];
  }

  bindEvents() {
    // Mouse
    this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
    this.canvas.addEventListener('mousemove', (e) => this.draw(e));
    window.addEventListener('mouseup', () => this.stopDrawing());

    // Touch
    this.canvas.addEventListener('touchstart', (e) => this.startDrawing(e), { passive: false });
    this.canvas.addEventListener('touchmove', (e) => this.draw(e), { passive: false });
    this.canvas.addEventListener('touchend', () => this.stopDrawing());

    window.addEventListener('resize', () => {
      // Re-setup while keeping ratio
      const tempImg = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
      const rect = this.canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      this.width = rect.width || 380;
      this.height = rect.height || 380;
      this.canvas.width = this.width * dpr;
      this.canvas.height = this.height * dpr;
      this.ctx.scale(dpr, dpr);
      this.ctx.putImageData(tempImg, 0, 0);
    });
  }

  getImageDataURL() {
    // Create an unscaled square image copy with black background
    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = 280;
    exportCanvas.height = 280;
    const expCtx = exportCanvas.getContext('2d');

    expCtx.fillStyle = '#000000';
    expCtx.fillRect(0, 0, 280, 280);
    expCtx.drawImage(this.canvas, 0, 0, 280, 280);

    return exportCanvas.toDataURL('image/png');
  }

  /**
   * Stamp a handwriting sample onto canvas for instant testing.
   */
  stampCharacter(char) {
    this.clear();
    this.saveState();
    this.hasDrawn = true;
    if (this.hint) this.hint.classList.add('hidden');

    this.ctx.fillStyle = '#ffffff';
    this.ctx.font = `bold ${Math.floor(this.width * 0.55)}px 'Segoe UI', 'Comic Sans MS', sans-serif`;
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText(char, this.width / 2, this.height / 2 + 5);
  }

  /**
   * Stamp a handwriting sentence or paragraph sample onto canvas.
   */
  stampSentence(text) {
    this.clear();
    this.saveState();
    this.hasDrawn = true;
    if (this.hint) this.hint.classList.add('hidden');

    this.ctx.fillStyle = '#ffffff';
    this.ctx.font = '24px "Ink Free", "Segoe Script", "Comic Sans MS", cursive, sans-serif';
    this.ctx.textAlign = 'left';
    this.ctx.textBaseline = 'middle';

    const lines = text.split('\n');
    const lineHeight = 44;
    let startY = lines.length === 1 ? (this.height / 2) : 52;

    lines.forEach((line, idx) => {
      this.ctx.fillText(line, 20, startY + idx * lineHeight);
    });
  }

  getSentenceImageDataURL() {
    return this.canvas.toDataURL('image/png');
  }
}

window.DrawingCanvas = DrawingCanvas;
