from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import threading
import time
import json
from agent_wrapper import AgentWrapper

app = Flask(__name__)
agent_wrapper = AgentWrapper()

# 定期清理旧任务
def cleanup_scheduler():
    while True:
        time.sleep(300)  # 每 5 分钟清理一次
        removed = agent_wrapper.cleanup_old_tasks(max_age_hours=1)
        if removed > 0:
            print(f"Cleaned up {removed} old tasks")

cleanup_thread = threading.Thread(target=cleanup_scheduler, daemon=True)
cleanup_thread.start()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/execute', methods=['POST'])
def execute_task():
    data = request.get_json()
    task_description = data.get('task', '').strip()
    
    if not task_description:
        return jsonify({'error': 'Task description is required'}), 400
    
    task_id = agent_wrapper.create_task(task_description)
    
    def run_in_thread():
        agent_wrapper.execute_task(task_id)
    
    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    
    return jsonify({
        'task_id': task_id,
        'status': 'pending',
        'message': 'Task created and started'
    })


@app.route('/api/status/<task_id>')
def get_task_status(task_id):
    status = agent_wrapper.get_task_status(task_id)
    if not status:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(status)


@app.route('/api/logs/<task_id>')
def stream_logs(task_id):
    def generate():
        last_log_count = 0
        empty_count = 0
        max_empty_count = 20  # 最多等待 10 秒（20 * 0.5秒）
        
        while True:
            logs = agent_wrapper.get_task_logs(task_id)
            
            # 如果任务不存在，等待一段时间再检查
            if not logs:
                empty_count += 1
                if empty_count > max_empty_count:
                    yield f"data: {json.dumps({'error': 'Task not found or expired'})}\n\n"
                    break
                time.sleep(0.5)
                continue
            
            # 任务存在，重置空计数器
            empty_count = 0
            
            new_logs = logs[last_log_count:]
            for log in new_logs:
                log_data = {
                    'timestamp': log['timestamp'],
                    'level': log['level'],
                    'message': log['message']
                }
                yield f"data: {json.dumps(log_data)}\n\n"
            
            last_log_count = len(logs)
            
            status = agent_wrapper.get_task_status(task_id)
            if status and status['status'] in ['completed', 'failed']:
                yield f"event: complete\n"
                yield f"data: {json.dumps(status)}\n\n"
                break
            
            # 如果没有新日志，发送心跳消息保持连接
            if len(new_logs) == 0:
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"
            
            time.sleep(0.5)
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/tasks', methods=['GET'])
def list_tasks():
    task_ids = list(agent_wrapper.tasks.keys())
    tasks_info = []
    for task_id in task_ids:
        status = agent_wrapper.get_task_status(task_id)
        if status:
            tasks_info.append({
                'task_id': task_id,
                'status': status['status'],
                'created_at': status['created_at']
            })
    return jsonify({'tasks': tasks_info})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
