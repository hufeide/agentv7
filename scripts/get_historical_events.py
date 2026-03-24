#!/usr/bin/env python3
"""
获取历史上的今天的重要事件数据
使用维基百科API获取历史上的今天事件
"""

import requests
import json
from datetime import datetime

def get_historical_events_today():
    """获取历史上的今天的重要事件"""
    # 获取当前日期
    today = datetime.now()
    month = today.month
    day = today.day
    
    # 维基百科API端点
    url = f"https://en.wikipedia.org/api/rest_v1/page/featured/{month:02d}/{day:02d}"
    
    try:
        # 发送请求
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # 提取事件信息
        events = []
        
        if 'selected' in data:
            # 处理选中的事件
            selected = data['selected']
            if 'text' in selected:
                # 解析文本中的事件
                text = selected['text']
                # 简单分割事件（按换行符）
                for line in text.split('\n'):
                    if line.strip() and len(line.strip()) > 10:
                        events.append(line.strip())
        
        # 如果没有获取到事件，尝试另一种方法
        if not events:
            # 使用维基百科的"历史上的今天"页面
            wiki_url = f"https://en.wikipedia.org/wiki/Wikipedia:Selected_anniversaries/{month:02d}_{day:02d}"
            
            # 尝试获取页面内容
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(wiki_url, headers=headers)
            if response.status_code == 200:
                # 简单解析HTML获取事件
                content = response.text
                
                # 提取事件文本（简化处理）
                import re
                # 查找事件列表
                event_pattern = r'<li>([^<]+)</li>'
                matches = re.findall(event_pattern, content)
                
                for match in matches[:20]:  # 获取前20个事件
                    if 'edit' not in match.lower() and len(match.strip()) > 15:
                        events.append(match.strip())
        
        return {
            "date": f"{month}/{day}",
            "events": events[:15] if events else ["无法获取详细事件列表"],
            "source": "Wikipedia",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "date": f"{month}/{day}",
            "events": [f"获取数据时出错: {str(e)}"],
            "source": "Error",
            "timestamp": datetime.now().isoformat()
        }

def main():
    """主函数"""
    print("正在获取历史上的今天的重要事件...")
    
    result = get_historical_events_today()
    
    # 格式化输出
    print(f"\n📅 日期: {result['date']}")
    print(f"📚 来源: {result['source']}")
    print("\n📋 历史上的今天重要事件:")
    print("-" * 50)
    
    for i, event in enumerate(result['events'], 1):
        print(f"{i}. {event}")
    
    print("-" * 50)
    print(f"⏱️ 时间戳: {result['timestamp']}")
    
    # 保存为JSON文件
    with open('historical_events_today.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print("\n✅ 数据已保存到 historical_events_today.json")

if __name__ == "__main__":
    main()
