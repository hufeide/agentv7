# Agent Web Interface Design

**Date:** 2026-03-24
**Status:** Approved

## Overview
Create a simple web interface to interact with agent_v7.py, allowing users to submit tasks and view real-time execution logs and results.

## Architecture

### Technology Stack
- **Backend:** Flask (Python web framework)
- **Real-time Communication:** Server-Sent Events (SSE)
- **Frontend:** HTML5 + CSS3 + Vanilla JavaScript
- **Task Execution:** asyncio-based agent_v7.py

### System Architecture
```
HTML Frontend <--HTTP--> Flask API <--asyncio--> agent_v7.py
     ↑                                              ↓
     └──────── SSE Real-time Logs ─────────────────┘
```

## Components

### 1. Backend API (Flask)

#### Endpoints

**POST /api/execute**
- Submit a new task for execution
- Request: `{"task": "task description"}`
- Response: `{"task_id": "uuid", "status": "pending"}`
- Starts async execution of agent_v7.py

**GET /api/status/<task_id>**
- Get current task status
- Response: `{"status": "pending|running|completed|failed", "result": {...}}`

**GET /api/logs/<task_id>**
- SSE endpoint for real-time log streaming
- Streams log lines as they are generated
- Content-Type: text/event-stream

#### Task Management
- Use in-memory dictionary to store task states
- Each task has: id, status, logs, result, error
- Background thread runs asyncio event loop for agent_v7.py

### 2. Frontend Interface

#### Layout
```
+-----------------------------------+
|         Agent Web Interface       |
+-----------------------------------+
| Task Input:                       |
| [_________________________]       |
| [Execute] [Stop] [Clear]          |
+-----------------------------------+
| Status: Running                   |
| Progress: [████████░░] 80%        |
+-----------------------------------+
| Execution Logs:                   |
| 19:17:40 - AgentOS initialized     |
| 19:17:41 - Plan generated         |
| ...                               |
+-----------------------------------+
| Result:                           |
| Status: completed                 |
| Total Steps: 1                    |
| Completed: 1                      |
+-----------------------------------+
```

#### Features
- **Task Input:** Textarea for task description
- **Control Buttons:** Execute, Stop, Clear
- **Status Display:** Current task status with visual indicator
- **Progress Bar:** Visual progress indicator
- **Log Display:** Auto-scrolling log window with color-coded messages
- **Result Display:** Formatted JSON result display

### 3. Data Flow

1. **Task Submission**
   - User enters task → Click Execute
   - Frontend POST to /api/execute
   - Backend creates task, starts execution
   - Backend returns task_id

2. **Log Streaming**
   - Frontend connects to /api/logs/<task_id> via EventSource
   - Backend captures agent_v7.py logs
   - Logs streamed in real-time to frontend
   - Frontend appends logs to display

3. **Task Completion**
   - agent_v7.py completes execution
   - Backend stores result in task state
   - Frontend polls /api/status or receives completion event
   - Frontend displays final result

## Error Handling

### Backend Errors
- **Task Timeout:** Return timeout error after 300s
- **Execution Failure:** Capture exception, return error message
- **Invalid Input:** Validate task description, return 400 error

### Frontend Errors
- **Network Error:** Display connection error message
- **SSE Disconnection:** Auto-reconnect with exponential backoff
- **Task Failure:** Display error message from backend

## Implementation Details

### Backend File Structure
```
web_server.py          # Flask application
agent_wrapper.py       # Wrapper for agent_v7.py execution
```

### Frontend File Structure
```
templates/
  index.html           # Main HTML interface
static/
  css/
    style.css          # Styling
  js/
    app.js             # Frontend logic
```

### Key Implementation Points

1. **Async Execution in Flask**
   - Run asyncio event loop in background thread
   - Use `asyncio.run_coroutine_threadsafe()` for task execution

2. **Log Capture**
   - Redirect agent_v7.py logger to custom handler
   - Collect logs in memory for SSE streaming

3. **SSE Implementation**
   - Use Flask's `Response` with generator
   - Format messages as SSE events: `data: {log_line}\n\n`

4. **Frontend State Management**
   - Store current task_id in variable
   - Manage EventSource connection lifecycle
   - Handle UI updates based on task status

## Testing Plan

1. **Unit Tests**
   - Test API endpoints with various inputs
   - Test log capture and streaming
   - Test error handling

2. **Integration Tests**
   - Test complete task execution flow
   - Test SSE connection and reconnection
   - Test concurrent task handling

3. **Browser Tests**
   - Test on Chrome, Firefox, Safari
   - Test responsive design on mobile
   - Test console for JavaScript errors

## Future Enhancements

- Add task history and persistence
- Support file upload for tasks
- Add task cancellation functionality
- Implement user authentication
- Add task scheduling
