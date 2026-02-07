# datetime json.py

::: info 文件信息
- 📄 原文件：`02_datetime_json.py`
- 🔤 语言：python
:::

Python 标准库：datetime 和 json 模块
本文件介绍日期时间处理和 JSON 数据处理。

## 完整代码

```python
import json
from datetime import datetime, date, time, timedelta, timezone
from typing import Any


def main01_datetime_basics():
    """
    ============================================================
                    1. datetime 基础
    ============================================================
    """
    print("=" * 60)
    print("1. datetime 基础")
    print("=" * 60)

    # 【获取当前时间】
    print("--- 获取当前时间 ---")
    now = datetime.now()
    today = date.today()
    utc_now = datetime.now(timezone.utc)

    print(f"datetime.now(): {now}")
    print(f"date.today(): {today}")
    print(f"UTC 时间: {utc_now}")

    # 【创建日期时间】
    print(f"\n--- 创建日期时间 ---")
    dt = datetime(2024, 6, 15, 14, 30, 45)
    d = date(2024, 6, 15)
    t = time(14, 30, 45)

    print(f"datetime: {dt}")
    print(f"date: {d}")
    print(f"time: {t}")

    # 【访问属性】
    print(f"\n--- 访问属性 ---")
    print(f"年: {now.year}")
    print(f"月: {now.month}")
    print(f"日: {now.day}")
    print(f"时: {now.hour}")
    print(f"分: {now.minute}")
    print(f"秒: {now.second}")
    print(f"微秒: {now.microsecond}")
    print(f"星期: {now.weekday()} (0=周一)")
    print(f"星期: {now.isoweekday()} (1=周一)")


def main02_datetime_formatting():
    """
    ============================================================
                2. 日期时间格式化
    ============================================================
    """
    print("\n" + "=" * 60)
    print("2. 日期时间格式化")
    print("=" * 60)

    now = datetime.now()

    # 【格式化输出 strftime】
    print("--- strftime 格式化 ---")
    print(f"默认: {now}")
    print(f"ISO格式: {now.isoformat()}")
    print(f"%Y-%m-%d: {now.strftime('%Y-%m-%d')}")
    print(f"%Y/%m/%d %H:%M:%S: {now.strftime('%Y/%m/%d %H:%M:%S')}")
    print(f"%Y年%m月%d日: {now.strftime('%Y年%m月%d日')}")
    print(f"%A, %B %d, %Y: {now.strftime('%A, %B %d, %Y')}")

    print(f"\n常用格式化代码:")
    print("  %Y: 四位年份")
    print("  %m: 月份 (01-12)")
    print("  %d: 日期 (01-31)")
    print("  %H: 小时 (00-23)")
    print("  %M: 分钟 (00-59)")
    print("  %S: 秒 (00-59)")
    print("  %A: 星期全称")
    print("  %a: 星期缩写")
    print("  %B: 月份全称")
    print("  %b: 月份缩写")

    # 【解析字符串 strptime】
    print(f"\n--- strptime 解析 ---")
    date_str = "2024-06-15 14:30:00"
    parsed = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    print(f"解析 '{date_str}': {parsed}")

    # 【ISO 格式】
    print(f"\n--- ISO 格式 ---")
    iso_str = now.isoformat()
    print(f"isoformat(): {iso_str}")
    parsed = datetime.fromisoformat(iso_str)
    print(f"fromisoformat(): {parsed}")


def main03_timedelta():
    """
    ============================================================
                    3. 时间差 timedelta
    ============================================================
    """
    print("\n" + "=" * 60)
    print("3. 时间差 timedelta")
    print("=" * 60)

    # 【创建 timedelta】
    print("--- 创建 timedelta ---")
    delta1 = timedelta(days=7)
    delta2 = timedelta(hours=3, minutes=30)
    delta3 = timedelta(weeks=2, days=3, hours=5)

    print(f"7天: {delta1}")
    print(f"3小时30分: {delta2}")
    print(f"2周3天5小时: {delta3}")

    # 【日期运算】
    print(f"\n--- 日期运算 ---")
    now = datetime.now()
    tomorrow = now + timedelta(days=1)
    yesterday = now - timedelta(days=1)
    next_week = now + timedelta(weeks=1)

    print(f"现在: {now}")
    print(f"明天: {tomorrow}")
    print(f"昨天: {yesterday}")
    print(f"下周: {next_week}")

    # 【计算时间差】
    print(f"\n--- 计算时间差 ---")
    date1 = datetime(2024, 1, 1)
    date2 = datetime(2024, 12, 31)
    diff = date2 - date1

    print(f"从 {date1.date()} 到 {date2.date()}")
    print(f"相差: {diff.days} 天")
    print(f"总秒数: {diff.total_seconds()}")

    # 【timedelta 属性】
    print(f"\n--- timedelta 属性 ---")
    delta = timedelta(days=5, hours=3, minutes=30, seconds=45)
    print(f"timedelta: {delta}")
    print(f"days: {delta.days}")
    print(f"seconds: {delta.seconds}")  # 不包括天数
    print(f"total_seconds: {delta.total_seconds()}")


def main04_timezone():
    """
    ============================================================
                    4. 时区处理
    ============================================================
    """
    print("\n" + "=" * 60)
    print("4. 时区处理")
    print("=" * 60)

    # 【UTC 时间】
    print("--- UTC 时间 ---")
    utc_now = datetime.now(timezone.utc)
    print(f"UTC: {utc_now}")

    # 【自定义时区】
    print(f"\n--- 自定义时区 ---")
    china_tz = timezone(timedelta(hours=8))
    china_now = datetime.now(china_tz)
    print(f"北京时间 (UTC+8): {china_now}")

    # 【时区转换】
    print(f"\n--- 时区转换 ---")
    utc_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    china_time = utc_time.astimezone(china_tz)

    print(f"UTC: {utc_time}")
    print(f"北京: {china_time}")

    # 【推荐使用 zoneinfo（Python 3.9+）】
    print(f"\n--- zoneinfo (Python 3.9+) ---")
    try:
        from zoneinfo import ZoneInfo

        utc = ZoneInfo("UTC")
        shanghai = ZoneInfo("Asia/Shanghai")
        new_york = ZoneInfo("America/New_York")

        now_utc = datetime.now(utc)
        now_shanghai = now_utc.astimezone(shanghai)
        now_ny = now_utc.astimezone(new_york)

        print(f"UTC: {now_utc}")
        print(f"上海: {now_shanghai}")
        print(f"纽约: {now_ny}")
    except ImportError:
        print("需要 Python 3.9+ 或安装 tzdata")


def main05_json_basics():
    """
    ============================================================
                    5. JSON 基础
    ============================================================
    """
    print("\n" + "=" * 60)
    print("5. JSON 基础")
    print("=" * 60)

    # 【Python 对象转 JSON 字符串】
    print("--- json.dumps ---")
    data = {
        "name": "Alice",
        "age": 25,
        "is_student": False,
        "courses": ["Python", "JavaScript"],
        "address": {
            "city": "Beijing",
            "country": "China"
        },
        "score": None
    }

    json_str = json.dumps(data)
    print(f"dumps: {json_str}")

    # 格式化输出
    json_pretty = json.dumps(data, indent=2)
    print(f"\n格式化:\n{json_pretty}")

    # 中文处理
    json_chinese = json.dumps(data, ensure_ascii=False, indent=2)
    print(f"\n中文:\n{json_chinese}")

    # 【JSON 字符串转 Python 对象】
    print("\n--- json.loads ---")
    json_str = '{"name": "Bob", "age": 30, "active": true}'
    obj = json.loads(json_str)
    print(f"loads: {obj}")
    print(f"类型: {type(obj)}")
    print(f"name: {obj['name']}")


def main06_json_file():
    """
    ============================================================
                    6. JSON 文件操作
    ============================================================
    """
    print("\n" + "=" * 60)
    print("6. JSON 文件操作")
    print("=" * 60)

    import tempfile
    from pathlib import Path

    data = {
        "users": [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ],
        "total": 2
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "data.json"

        # 【写入 JSON 文件】
        print("--- 写入 JSON 文件 ---")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"写入: {json_path}")

        # 【读取 JSON 文件】
        print(f"\n--- 读取 JSON 文件 ---")
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        print(f"读取: {loaded}")


def main07_json_custom():
    """
    ============================================================
                7. 自定义 JSON 序列化
    ============================================================
    """
    print("\n" + "=" * 60)
    print("7. 自定义 JSON 序列化")
    print("=" * 60)

    # 【自定义编码器】
    print("--- 自定义编码器 ---")

    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, date):
                return obj.isoformat()
            if isinstance(obj, set):
                return list(obj)
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return super().default(obj)

    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age

    data = {
        "person": Person("Alice", 25),
        "created": datetime.now(),
        "tags": {"python", "coding"},
    }

    json_str = json.dumps(data, cls=CustomEncoder, indent=2)
    print(json_str)

    # 【自定义解码器】
    print(f"\n--- 自定义解码 ---")

    def custom_decoder(dct):
        if 'created' in dct:
            dct['created'] = datetime.fromisoformat(dct['created'])
        return dct

    json_str = '{"name": "Alice", "created": "2024-06-15T14:30:00"}'
    obj = json.loads(json_str, object_hook=custom_decoder)
    print(f"解码后: {obj}")
    print(f"created 类型: {type(obj['created'])}")

    # 【使用 default 参数】
    print(f"\n--- 使用 default 参数 ---")

    def json_serializer(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return {"__datetime__": obj.isoformat()}
        if isinstance(obj, date):
            return {"__date__": obj.isoformat()}
        raise TypeError(f"不支持的类型: {type(obj)}")

    data = {"event": "meeting", "date": datetime.now()}
    json_str = json.dumps(data, default=json_serializer)
    print(f"序列化: {json_str}")


def main08_json_tips():
    """
    ============================================================
                    8. JSON 实用技巧
    ============================================================
    """
    print("\n" + "=" * 60)
    print("8. JSON 实用技巧")
    print("=" * 60)

    # 【排序键】
    print("--- 排序键 ---")
    data = {"c": 3, "a": 1, "b": 2}
    print(f"原始: {json.dumps(data)}")
    print(f"排序: {json.dumps(data, sort_keys=True)}")

    # 【紧凑输出】
    print(f"\n--- 紧凑输出 ---")
    data = {"name": "Alice", "age": 25}
    compact = json.dumps(data, separators=(',', ':'))
    print(f"紧凑: {compact}")

    # 【验证 JSON】
    print(f"\n--- 验证 JSON ---")

    def is_valid_json(s: str) -> bool:
        try:
            json.loads(s)
            return True
        except json.JSONDecodeError:
            return False

    print(f'是否有效 JSON: {is_valid_json("{\\"name\\": \\"Alice\\"}")}')
    print(f'是否有效 JSON: {is_valid_json("not json")}')

    # 【安全解析】
    print(f"\n--- 安全解析 ---")
    json_str = '{"name": "Alice", "extra": "unknown"}'

    # 只提取需要的字段
    data = json.loads(json_str)
    safe_data = {
        "name": data.get("name", ""),
        "age": data.get("age", 0),
    }
    print(f"安全提取: {safe_data}")


if __name__ == "__main__":
    main01_datetime_basics()
    main02_datetime_formatting()
    main03_timedelta()
    main04_timezone()
    main05_json_basics()
    main06_json_file()
    main07_json_custom()
    main08_json_tips()
```
