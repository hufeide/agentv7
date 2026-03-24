class AgentWebInterface {
    constructor() {
        this.currentTaskId = null;
        this.eventSource = null;
        this.isExecuting = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000;
        this.currentSessionId = null;
        this.sessions = new Map();
        
        this.initializeElements();
        this.attachEventListeners();
        this.initializeSession('initial', 'Initial Session');
    }

    initializeElements() {
        this.taskInput = document.getElementById('taskInput');
        this.executeBtn = document.getElementById('executeBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.statusIcon = document.getElementById('statusIcon');
        this.statusText = document.getElementById('statusText');
        this.progressBar = document.getElementById('progressBar');
        this.taskId = document.getElementById('taskId');
        this.taskTime = document.getElementById('taskTime');
        this.logsContainer = document.getElementById('logsContainer');
        this.resultContent = document.getElementById('resultContent');
        
        // 新增的控制按钮
        this.collapseAllBtn = document.getElementById('collapseAllBtn');
        this.expandAllBtn = document.getElementById('expandAllBtn');
        this.clearLogsBtn = document.getElementById('clearLogsBtn');
    }

    attachEventListeners() {
        this.executeBtn.addEventListener('click', () => this.executeTask());
        this.clearBtn.addEventListener('click', () => this.clearAll());
        
        // 新增的日志控制按钮事件
        this.collapseAllBtn.addEventListener('click', () => this.collapseAllSessions());
        this.expandAllBtn.addEventListener('click', () => this.expandAllSessions());
        this.clearLogsBtn.addEventListener('click', () => this.clearAllLogs());
    }

    // 会话管理方法
    initializeSession(sessionId, title) {
        this.currentSessionId = sessionId;
        this.sessions.set(sessionId, {
            id: sessionId,
            title: title,
            logs: [],
            collapsed: false
        });
    }

    createNewSession(taskId) {
        const sessionId = `session_${taskId}`;
        const title = `Task: ${taskId.substring(0, 8)}...`;
        this.initializeSession(sessionId, title);
        this.renderSession(sessionId);
        return sessionId;
    }

    renderSession(sessionId) {
        const session = this.sessions.get(sessionId);
        if (!session) return;

        const sessionGroup = document.createElement('div');
        sessionGroup.className = `session-group ${session.collapsed ? 'collapsed' : ''}`;
        sessionGroup.setAttribute('data-session-id', sessionId);

        sessionGroup.innerHTML = `
            <div class="session-header">
                <span class="session-title">${session.title}</span>
                <span class="session-toggle">▼</span>
            </div>
            <div class="session-logs">
                ${session.logs.map(log => this.createLogEntry(log)).join('')}
            </div>
        `;

        // 移除现有的会话（如果存在）
        const existingSession = this.logsContainer.querySelector(`[data-session-id="${sessionId}"]`);
        if (existingSession) {
            existingSession.remove();
        }

        // 添加到日志容器顶部
        this.logsContainer.insertBefore(sessionGroup, this.logsContainer.firstChild);

        // 添加点击事件
        const header = sessionGroup.querySelector('.session-header');
        header.addEventListener('click', () => this.toggleSession(sessionId));
    }

    toggleSession(sessionId) {
        const session = this.sessions.get(sessionId);
        if (!session) return;

        session.collapsed = !session.collapsed;
        this.renderSession(sessionId);
    }

    collapseAllSessions() {
        this.sessions.forEach((session, sessionId) => {
            session.collapsed = true;
            this.renderSession(sessionId);
        });
    }

    expandAllSessions() {
        this.sessions.forEach((session, sessionId) => {
            session.collapsed = false;
            this.renderSession(sessionId);
        });
    }

    clearAllLogs() {
        this.sessions.clear();
        this.logsContainer.innerHTML = '';
        this.initializeSession('initial', 'Initial Session');
        this.renderSession('initial');
        this.addLog('INFO', 'Logs cleared');
    }

    // 日志添加方法
    addLog(level, message, timestamp = null) {
        const logEntry = {
            timestamp: timestamp || this.getCurrentTime(),
            level: level,
            message: message
        };

        const session = this.sessions.get(this.currentSessionId);
        if (session) {
            session.logs.push(logEntry);
            this.renderSession(this.currentSessionId);
        }

        // 自动滚动到最新日志
        this.scrollToLatestLog();
    }

    createLogEntry(log) {
        return `
            <div class="log-entry log-${log.level.toLowerCase()}">
                <span class="log-time">${log.timestamp}</span>
                <span class="log-level">${log.level}</span>
                <span class="log-message">${log.message}</span>
            </div>
        `;
    }

    scrollToLatestLog() {
        const sessionGroup = this.logsContainer.querySelector(`[data-session-id="${this.currentSessionId}"]`);
        if (sessionGroup && !sessionGroup.classList.contains('collapsed')) {
            const sessionLogs = sessionGroup.querySelector('.session-logs');
            if (sessionLogs) {
                sessionLogs.scrollTop = sessionLogs.scrollHeight;
            }
        }
    }

    async executeTask() {
        const taskDescription = this.taskInput.value.trim();
        
        if (!taskDescription) {
            this.showError('Please enter a task description');
            return;
        }

        if (this.isExecuting) {
            this.showError('A task is already executing');
            return;
        }

        this.setExecutingState(true);
        this.clearResult();
        
        // 创建新会话
        const taskId = this.generateTaskId();
        const sessionId = this.createNewSession(taskId);
        this.currentSessionId = sessionId;
        
        this.addLog('INFO', 'Creating task...');

        try {
            const response = await fetch('/api/execute', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ task: taskDescription })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.currentTaskId = data.task_id;
            this.taskId.textContent = `Task ID: ${this.currentTaskId.substring(0, 8)}...`;
            this.taskTime.textContent = 'Time: Running...';
            
            this.addLog('INFO', data.message);
            this.connectToLogs();
        } catch (error) {
            this.showError(`Failed to execute task: ${error.message}`);
            this.setExecutingState(false);
        }
    }

    connectToLogs() {
        if (!this.currentTaskId) {
            return;
        }

        this.closeEventSource();

        this.eventSource = new EventSource(`/api/logs/${this.currentTaskId}`);

        this.eventSource.onmessage = (event) => {
            try {
                const logData = JSON.parse(event.data);
                
                // 检查是否是心跳消息
                if (logData.heartbeat) {
                    // 心跳消息，不做任何处理，只是保持连接活跃
                    return;
                }
                
                // 检查是否是错误消息
                if (logData.error) {
                    this.addLog('ERROR', logData.error);
                    
                    // 如果任务不存在，停止重连
                    if (logData.error.includes('not found') || logData.error.includes('expired')) {
                        this.addLog('ERROR', 'Task not found or expired, stopping reconnection');
                        this.reconnectAttempts = this.maxReconnectAttempts; // 停止重连
                        this.setExecutingState(false);
                        this.closeEventSource();
                        return;
                    }
                } else {
                    this.addLog(logData.level, logData.message, logData.timestamp);
                }
            } catch (error) {
                console.error('Failed to parse log data:', error);
            }
        };

        this.eventSource.addEventListener('complete', (event) => {
            try {
                const statusData = JSON.parse(event.data);
                this.handleTaskCompletion(statusData);
            } catch (error) {
                console.error('Failed to parse completion data:', error);
            }
        });

        this.eventSource.onerror = (error) => {
            console.error('SSE error:', error);
            
            if (this.eventSource.readyState === EventSource.CLOSED) {
                this.addLog('ERROR', 'Connection closed');
                
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    this.addLog('INFO', `Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
                    
                    setTimeout(() => {
                        this.connectToLogs();
                    }, this.reconnectDelay);
                } else {
                    this.addLog('ERROR', 'Max reconnection attempts reached');
                    this.setExecutingState(false);
                }
            } else {
                this.addLog('ERROR', 'Connection error, waiting for recovery...');
            }
        };
    }

    handleTaskCompletion(statusData) {
        this.setExecutingState(false);
        this.taskTime.textContent = `Time: ${this.getCurrentTime()}`;
        
        // 显示完整的结果
        this.resultContent.textContent = JSON.stringify(statusData, null, 2);
        
        if (statusData.status === 'completed') {
            this.addLog('INFO', 'Task completed successfully');
            this.progressBar.style.width = '100%';
        } else if (statusData.status === 'failed') {
            this.addLog('ERROR', 'Task failed');
        } else {
            this.addLog('WARNING', 'Task completed with partial success');
        }
        
        this.closeEventSource();
    }

    closeEventSource() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
    }

    setExecutingState(executing) {
        this.isExecuting = executing;
        this.executeBtn.disabled = executing;
        
        if (executing) {
            this.statusIcon.className = 'status-icon status-running';
            this.statusText.textContent = 'Running';
            this.progressBar.style.width = '0%';
        } else {
            this.statusIcon.className = 'status-icon status-idle';
            this.statusText.textContent = 'Idle';
        }
    }

    clearAll() {
        this.taskInput.value = '';
        this.clearResult();
    }

    clearResult() {
        this.resultContent.textContent = 'No result yet';
    }

    showError(message) {
        this.addLog('ERROR', message);
        alert(message);
    }

    generateTaskId() {
        return 'task_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    getCurrentTime() {
        const now = new Date();
        return now.toTimeString().split(' ')[0];
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new AgentWebInterface();
});