const API_BASE = 'http://127.0.0.1:8000';
let currentMode = null;

document.addEventListener('DOMContentLoaded', () => {
    const featureCards = document.querySelectorAll('.feature-card');
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '0';
                entry.target.style.transform = 'translateY(20px)';

                setTimeout(() => {
                    entry.target.style.transition = 'all 0.6s ease';
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }, 100);

                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    featureCards.forEach(card => observer.observe(card));

    const demoPlaceholder = document.querySelector('.demo-placeholder');
    if (demoPlaceholder) {
        demoPlaceholder.addEventListener('click', () => {
            alert('Demo video would play here in production. Connect your recorded detection output for live showcase.');
        });
        demoPlaceholder.style.cursor = 'pointer';
    }

    initializeControls();
    checkAPIStatus();
    setInterval(checkAPIStatus, 2000);
});

function initializeControls() {
    const modeButtons = document.querySelectorAll('.mode-btn');

    modeButtons.forEach(btn => {
        btn.addEventListener('click', async () => {
            const mode = btn.dataset.mode || 'stop';
            await switchMode(mode);
        });
    });
}

async function switchMode(mode) {
    try {
        let endpoint = '/mode/stop';
        let modeLabel = 'Stopped';

        if (mode === '1') {
            endpoint = '/mode/1';
            modeLabel = 'Suspicious Detection + SOS';
        } else if (mode === '2') {
            endpoint = '/mode/2';
            modeLabel = 'Overcrowd Detection';
        } else if (mode === '3') {
            endpoint = '/mode/3';
            modeLabel = 'Target Lock System';
        }

        const response = await fetch(`${API_BASE}${endpoint}`, { method: 'POST' });

        if (!response.ok) throw new Error('API request failed');

        const data = await response.json();
        currentMode = data.mode;
        updateModeDisplay(modeLabel);
        updateActiveButton(data.mode);
    } catch (error) {
        console.error('Failed to switch mode:', error);
        updateApiStatus('Error', 'error');
    }
}

function updateModeDisplay(label) {
    const modeDisplay = document.getElementById('current-mode');
    if (modeDisplay) {
        modeDisplay.textContent = label;
    }
    updateApiStatus('Connected', 'connected');
}

function updateActiveButton(mode) {
    const buttons = document.querySelectorAll('.mode-btn');
    buttons.forEach(btn => {
        btn.classList.remove('active');
        if ((mode === 'sos' && btn.id === 'btn-sos') ||
            (mode === 'crowd' && btn.id === 'btn-crowd') ||
            (mode === 'target' && btn.id === 'btn-target') ||
            (mode === null && btn.id === 'btn-stop')) {
            btn.classList.add('active');
        }
    });
}

async function checkAPIStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        if (response.ok) {
            const data = await response.json();
            currentMode = data.current_mode;
            updateActiveButton(data.current_mode);
            updateApiStatus('Connected', 'connected');
        }
    } catch (error) {
        updateApiStatus('Offline', 'offline');
    }
}

function updateApiStatus(text, status) {
    const statusElement = document.getElementById('api-status');
    if (statusElement) {
        statusElement.textContent = text;
        statusElement.style.color = status === 'connected' ? '#00ff00' :
                                     status === 'error' ? '#ff4444' : '#ffaa00';
    }
}
