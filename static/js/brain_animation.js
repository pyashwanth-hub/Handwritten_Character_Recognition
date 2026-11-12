/**
 * Animated Futuristic Neural Network Illustration for Hero Section
 * Draws dynamic multi-layer neurons, synaptic connections, and firing action potentials.
 */

(function () {
  const canvas = document.getElementById('brain-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  let width = (canvas.width = canvas.parentElement.clientWidth || 450);
  let height = (canvas.height = canvas.parentElement.clientHeight || 450);

  // Define 4 neural layers
  const layerCounts = [5, 8, 8, 5];
  const layers = [];
  const synapses = [];
  const signals = [];

  const neonColors = ['#06b6d4', '#ec4899', '#a855f7', '#3b82f6'];

  class Neuron {
    constructor(x, y, layerIndex, indexInLayer) {
      this.x = x;
      this.y = y;
      this.layerIndex = layerIndex;
      this.index = indexInLayer;
      this.radius = 6.5;
      this.baseColor = neonColors[layerIndex % neonColors.length];
      this.pulse = Math.random() * Math.PI * 2;
      this.pulseSpeed = 0.04 + Math.random() * 0.03;
      this.energy = 0.5;
    }

    update() {
      this.pulse += this.pulseSpeed;
      this.currentRadius = this.radius + Math.sin(this.pulse) * 1.5;
    }

    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.currentRadius, 0, Math.PI * 2);
      ctx.fillStyle = this.baseColor;
      ctx.shadowBlur = 15;
      ctx.shadowColor = this.baseColor;
      ctx.globalAlpha = 0.9;
      ctx.fill();

      // Outer glowing ring
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.currentRadius + 3, 0, Math.PI * 2);
      ctx.strokeStyle = this.baseColor;
      ctx.globalAlpha = 0.35 + Math.sin(this.pulse) * 0.2;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.shadowBlur = 0;
    }
  }

  class Synapse {
    constructor(fromNode, toNode) {
      this.from = fromNode;
      this.to = toNode;
      this.alpha = 0.12 + Math.random() * 0.15;
    }

    draw() {
      ctx.beginPath();
      ctx.moveTo(this.from.x, this.from.y);
      ctx.lineTo(this.to.x, this.to.y);
      ctx.strokeStyle = this.from.baseColor;
      ctx.globalAlpha = this.alpha;
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }
  }

  class ActionSignal {
    constructor(synapse) {
      this.synapse = synapse;
      this.progress = 0;
      this.speed = 0.015 + Math.random() * 0.02;
      this.color = synapse.from.baseColor;
      this.size = 3.5;
    }

    update() {
      this.progress += this.speed;
      return this.progress < 1;
    }

    draw() {
      const x = this.synapse.from.x + (this.synapse.to.x - this.synapse.from.x) * this.progress;
      const y = this.synapse.from.y + (this.synapse.to.y - this.synapse.from.y) * this.progress;

      ctx.beginPath();
      ctx.arc(x, y, this.size, 0, Math.PI * 2);
      ctx.fillStyle = '#fff';
      ctx.shadowBlur = 12;
      ctx.shadowColor = this.color;
      ctx.globalAlpha = 1;
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  }

  function setup() {
    layers.length = 0;
    synapses.length = 0;
    signals.length = 0;

    const layerSpacing = (width - 100) / (layerCounts.length - 1);

    layerCounts.forEach((count, lIdx) => {
      const layer = [];
      const x = 50 + lIdx * layerSpacing;
      const verticalSpacing = (height - 90) / (count + 1);

      for (let i = 0; i < count; i++) {
        const y = 45 + (i + 1) * verticalSpacing;
        layer.push(new Neuron(x, y, lIdx, i));
      }
      layers.push(layer);
    });

    // Connect adjacent layers
    for (let l = 0; l < layers.length - 1; l++) {
      for (const fromNode of layers[l]) {
        for (const toNode of layers[l + 1]) {
          synapses.push(new Synapse(fromNode, toNode));
        }
      }
    }
  }

  function loop() {
    ctx.clearRect(0, 0, width, height);

    // Draw synapses
    synapses.forEach((s) => s.draw());

    // Randomly fire action signals
    if (Math.random() < 0.28 && synapses.length > 0) {
      const randSyn = synapses[Math.floor(Math.random() * synapses.length)];
      signals.push(new ActionSignal(randSyn));
    }

    // Update and draw action signals
    for (let i = signals.length - 1; i >= 0; i--) {
      const active = signals[i].update();
      if (active) {
        signals[i].draw();
      } else {
        signals.splice(i, 1);
      }
    }

    // Update and draw neurons
    layers.forEach((layer) => {
      layer.forEach((node) => {
        node.update();
        node.draw();
      });
    });

    requestAnimationFrame(loop);
  }

  window.addEventListener('resize', () => {
    if (canvas.parentElement) {
      width = canvas.width = canvas.parentElement.clientWidth || 450;
      height = canvas.height = canvas.parentElement.clientHeight || 450;
      setup();
    }
  });

  setup();
  loop();
})();
