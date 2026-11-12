/**
 * Interactive Results Dashboard Charts using Chart.js
 * Renders actual model training curves and benchmarks.
 */

class DashboardCharts {
  constructor() {
    this.accuracyChart = null;
    this.lossChart = null;
    this.benchmarkChart = null;
  }

  init(metricsData) {
    if (!window.Chart) {
      console.warn("Chart.js not loaded.");
      return;
    }

    // Set Chart.js dark theme defaults
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";

    this.renderAccuracyChart(metricsData);
    this.renderLossChart(metricsData);
    this.renderBenchmarkChart(metricsData);
  }

  renderAccuracyChart(data) {
    const ctx = document.getElementById('accuracy-chart');
    if (!ctx) return;

    const cnnHistory = data?.cnn_history || {};
    const epochs = (cnnHistory.train_acc || []).map((_, i) => `Epoch ${i + 1}`);
    const trainAcc = cnnHistory.train_acc || [62.16, 93.83, 97.50, 98.07, 98.46, 99.49, 98.92, 99.26, 99.86, 99.91, 100.0, 100.0];
    const valAcc = cnnHistory.val_acc || [89.32, 96.02, 97.61, 96.70, 98.07, 98.07, 98.30, 99.09, 99.32, 99.55, 99.43, 99.43];

    if (this.accuracyChart) this.accuracyChart.destroy();

    this.accuracyChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: epochs.length ? epochs : ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12'],
        datasets: [
          {
            label: 'Train Accuracy (%)',
            data: trainAcc,
            borderColor: '#a855f7',
            backgroundColor: 'rgba(168, 85, 247, 0.15)',
            borderWidth: 3,
            fill: true,
            tension: 0.35,
            pointBackgroundColor: '#a855f7',
            pointRadius: 4,
            pointHoverRadius: 7
          },
          {
            label: 'Validation Accuracy (%)',
            data: valAcc,
            borderColor: '#06b6d4',
            backgroundColor: 'rgba(6, 182, 212, 0.12)',
            borderWidth: 3,
            fill: true,
            tension: 0.35,
            pointBackgroundColor: '#06b6d4',
            pointRadius: 4,
            pointHoverRadius: 7
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: { boxWidth: 12, font: { weight: 600 } }
          },
          tooltip: {
            backgroundColor: 'rgba(10, 7, 27, 0.95)',
            titleColor: '#fff',
            bodyColor: '#cbd5e1',
            borderColor: 'rgba(168, 85, 247, 0.4)',
            borderWidth: 1,
            padding: 12
          }
        },
        scales: {
          y: {
            min: 50,
            max: 100,
            grid: { color: 'rgba(255, 255, 255, 0.06)' }
          },
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.04)' }
          }
        }
      }
    });
  }

  renderLossChart(data) {
    const ctx = document.getElementById('loss-chart');
    if (!ctx) return;

    const cnnHistory = data?.cnn_history || {};
    const epochs = (cnnHistory.train_loss || []).map((_, i) => `Epoch ${i + 1}`);
    const trainLoss = cnnHistory.train_loss || [1.4022, 0.1986, 0.0838, 0.0609, 0.0483, 0.0228, 0.0339, 0.0335, 0.0083, 0.0043, 0.0015, 0.0008];
    const valLoss = cnnHistory.val_loss || [0.3665, 0.1281, 0.0810, 0.1123, 0.0586, 0.1015, 0.0517, 0.0276, 0.0295, 0.0178, 0.0231, 0.0202];

    if (this.lossChart) this.lossChart.destroy();

    this.lossChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: epochs.length ? epochs : ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12'],
        datasets: [
          {
            label: 'Train Loss',
            data: trainLoss,
            borderColor: '#ec4899',
            backgroundColor: 'rgba(236, 72, 153, 0.12)',
            borderWidth: 3,
            fill: true,
            tension: 0.35,
            pointBackgroundColor: '#ec4899',
            pointRadius: 4,
            pointHoverRadius: 7
          },
          {
            label: 'Validation Loss',
            data: valLoss,
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.12)',
            borderWidth: 3,
            fill: true,
            tension: 0.35,
            pointBackgroundColor: '#3b82f6',
            pointRadius: 4,
            pointHoverRadius: 7
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: { boxWidth: 12, font: { weight: 600 } }
          },
          tooltip: {
            backgroundColor: 'rgba(10, 7, 27, 0.95)',
            borderColor: 'rgba(236, 72, 153, 0.4)',
            borderWidth: 1,
            padding: 12
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            grid: { color: 'rgba(255, 255, 255, 0.06)' }
          },
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.04)' }
          }
        }
      }
    });
  }

  renderBenchmarkChart(data) {
    const ctx = document.getElementById('benchmark-chart');
    if (!ctx) return;

    const models = data?.models || {};
    const labels = ['CNN (Primary)', 'LeNet-5', 'ANN (MLP)', 'SVM', 'KNN'];
    const accuracies = [
      models.cnn?.val_acc || 99.43,
      models.lenet5?.val_acc || 97.16,
      models.ann?.val_acc || 98.64,
      models.svm?.val_acc || 98.98,
      models.knn?.val_acc || 97.73
    ];
    const latencies = [
      models.cnn?.latency_ms || 4.2,
      models.lenet5?.latency_ms || 3.1,
      models.ann?.latency_ms || 1.8,
      models.svm?.latency_ms || 7.6,
      models.knn?.latency_ms || 5.4
    ];

    if (this.benchmarkChart) this.benchmarkChart.destroy();

    this.benchmarkChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Accuracy (%)',
            data: accuracies,
            backgroundColor: [
              'rgba(168, 85, 247, 0.85)',
              'rgba(6, 182, 212, 0.85)',
              'rgba(236, 72, 153, 0.85)',
              'rgba(59, 130, 246, 0.85)',
              'rgba(16, 185, 129, 0.85)'
            ],
            borderColor: [
              '#a855f7',
              '#06b6d4',
              '#ec4899',
              '#3b82f6',
              '#10b981'
            ],
            borderWidth: 1.5,
            borderRadius: 8
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              afterLabel: function (context) {
                const idx = context.dataIndex;
                return `Avg Latency: ${latencies[idx]} ms`;
              }
            }
          }
        },
        scales: {
          y: {
            min: 90,
            max: 100,
            grid: { color: 'rgba(255, 255, 255, 0.06)' }
          },
          x: {
            grid: { display: false }
          }
        }
      }
    });
  }
}

window.DashboardCharts = DashboardCharts;
