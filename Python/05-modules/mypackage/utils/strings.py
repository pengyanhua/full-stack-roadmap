"""
字符串处理工具
"""
from typing import List


def capitalize_words(text: str) -> str:
    """将每个单词的首字母大写"""
    return ' '.join(word.capitalize() for word in text.split())


def reverse_string(text: str) -> str:
    """反转字符串"""
    return text[::-1]


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """截断字符串"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def slugify(text: str) -> str:
    """将字符串转换为 URL 友好的格式"""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    return text


def word_count(text: str) -> dict:
    """统计单词频率"""
    words = text.lower().split()
    count = {}
    for word in words:
        word = ''.join(c for c in word if c.isalnum())
        if word:
            count[word] = count.get(word, 0) + 1
    return count


if __name__ == "__main__":
    print(capitalize_words("hello world python"))
    print(reverse_string("hello"))
    print(truncate("This is a long text", 10))
    print(slugify("Hello World! This is Python"))
    print(word_count("hello world hello python world"))
