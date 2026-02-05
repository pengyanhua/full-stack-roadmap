#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
            项目2：文本分析器（词频统计）
============================================================
一个文本分析工具，可以统计词频、字符数等信息。

功能：
- 统计单词频率
- 计算字符数、单词数、行数
- 找出最常见的单词
- 支持文件输入和字符串输入

知识点：
- 文件操作
- 字符串处理
- 正则表达式
- collections.Counter
- 数据分析
============================================================
"""
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class TextStats:
    """文本统计结果"""
    char_count: int
    char_count_no_spaces: int
    word_count: int
    line_count: int
    sentence_count: int
    avg_word_length: float
    avg_words_per_sentence: float


class TextAnalyzer:
    """文本分析器"""

    def __init__(self, text: str = ""):
        self.text = text
        self._words: List[str] = []
        self._word_freq: Counter = Counter()

        if text:
            self._analyze()

    def load_file(self, filepath: str) -> None:
        """从文件加载文本"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        self.text = path.read_text(encoding='utf-8')
        self._analyze()

    def load_text(self, text: str) -> None:
        """加载文本字符串"""
        self.text = text
        self._analyze()

    def _analyze(self) -> None:
        """分析文本"""
        # 提取单词（只包含字母和数字）
        self._words = re.findall(r'\b[a-zA-Z]+\b', self.text.lower())
        self._word_freq = Counter(self._words)

    def get_stats(self) -> TextStats:
        """获取文本统计信息"""
        lines = self.text.split('\n')
        sentences = re.split(r'[.!?]+', self.text)
        sentences = [s.strip() for s in sentences if s.strip()]

        total_word_length = sum(len(word) for word in self._words)
        avg_word_length = total_word_length / len(self._words) if self._words else 0

        return TextStats(
            char_count=len(self.text),
            char_count_no_spaces=len(self.text.replace(' ', '').replace('\n', '')),
            word_count=len(self._words),
            line_count=len(lines),
            sentence_count=len(sentences),
            avg_word_length=round(avg_word_length, 2),
            avg_words_per_sentence=round(len(self._words) / len(sentences), 2) if sentences else 0
        )

    def word_frequency(self) -> Counter:
        """获取词频统计"""
        return self._word_freq

    def most_common(self, n: int = 10) -> List[Tuple[str, int]]:
        """获取最常见的 n 个单词"""
        return self._word_freq.most_common(n)

    def search_word(self, word: str) -> Dict:
        """搜索单词出现信息"""
        word = word.lower()
        count = self._word_freq.get(word, 0)

        # 找出所有出现位置
        positions = []
        for match in re.finditer(rf'\b{re.escape(word)}\b', self.text.lower()):
            positions.append(match.start())

        return {
            'word': word,
            'count': count,
            'positions': positions,
            'percentage': round(count / len(self._words) * 100, 2) if self._words else 0
        }

    def get_unique_words(self) -> List[str]:
        """获取所有不重复的单词"""
        return list(self._word_freq.keys())

    def get_word_length_distribution(self) -> Dict[int, int]:
        """获取单词长度分布"""
        distribution = Counter(len(word) for word in self._words)
        return dict(sorted(distribution.items()))

    def find_words_by_length(self, length: int) -> List[str]:
        """找出指定长度的单词"""
        return list(set(word for word in self._words if len(word) == length))

    def print_report(self) -> None:
        """打印分析报告"""
        stats = self.get_stats()
        most_common = self.most_common(10)
        length_dist = self.get_word_length_distribution()

        print("\n" + "=" * 50)
        print("           文本分析报告")
        print("=" * 50)

        print("\n【基本统计】")
        print(f"  字符数: {stats.char_count}")
        print(f"  字符数（不含空格）: {stats.char_count_no_spaces}")
        print(f"  单词数: {stats.word_count}")
        print(f"  行数: {stats.line_count}")
        print(f"  句子数: {stats.sentence_count}")
        print(f"  平均单词长度: {stats.avg_word_length}")
        print(f"  平均每句单词数: {stats.avg_words_per_sentence}")

        print("\n【最常见的 10 个单词】")
        for i, (word, count) in enumerate(most_common, 1):
            percentage = count / stats.word_count * 100 if stats.word_count else 0
            bar = "█" * int(percentage)
            print(f"  {i:2}. {word:15} {count:5} ({percentage:5.2f}%) {bar}")

        print("\n【单词长度分布】")
        max_count = max(length_dist.values()) if length_dist else 0
        for length, count in length_dist.items():
            bar_width = int(count / max_count * 20) if max_count else 0
            bar = "█" * bar_width
            print(f"  {length:2} 字母: {count:5} {bar}")

        print("\n【其他统计】")
        print(f"  不重复单词数: {len(self.get_unique_words())}")
        print(f"  词汇丰富度: {len(self.get_unique_words()) / stats.word_count * 100:.2f}%"
              if stats.word_count else "  词汇丰富度: 0%")

        print("=" * 50)


def demo():
    """演示"""
    sample_text = """
    Python is a programming language that lets you work quickly
    and integrate systems more effectively. Python is powerful and fast.
    Python plays well with others. Python runs everywhere.
    Python is friendly and easy to learn. Python is Open.

    The Python Software Foundation is an organization devoted to
    advancing open source technology related to the Python programming language.
    """

    print("【文本分析器演示】\n")

    analyzer = TextAnalyzer(sample_text)
    analyzer.print_report()

    # 搜索特定单词
    print("\n【搜索单词 'python'】")
    result = analyzer.search_word('python')
    print(f"  出现次数: {result['count']}")
    print(f"  占比: {result['percentage']}%")
    print(f"  位置: {result['positions'][:5]}...")

    # 找出特定长度的单词
    print("\n【6 个字母的单词】")
    words = analyzer.find_words_by_length(6)
    print(f"  {', '.join(words[:10])}")


def main():
    """主函数 - 交互模式"""
    print("=" * 50)
    print("      欢迎使用 Python 文本分析器")
    print("=" * 50)
    print("""
命令:
  load <file>    从文件加载文本
  text           输入文本（多行，空行结束）
  stats          显示统计信息
  top <n>        显示最常见的 n 个单词
  search <word>  搜索单词
  report         显示完整报告
  demo           运行演示
  quit           退出
""")

    analyzer = TextAnalyzer()

    while True:
        try:
            user_input = input(">>> ").strip()
            if not user_input:
                continue

            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if command == "quit" or command == "exit":
                print("再见！")
                break

            elif command == "demo":
                demo()

            elif command == "load":
                if not args:
                    print("请提供文件路径")
                else:
                    try:
                        analyzer.load_file(args)
                        print(f"已加载文件: {args}")
                    except FileNotFoundError as e:
                        print(e)

            elif command == "text":
                print("输入文本（空行结束）:")
                lines = []
                while True:
                    line = input()
                    if not line:
                        break
                    lines.append(line)
                analyzer.load_text('\n'.join(lines))
                print("文本已加载")

            elif command == "stats":
                if not analyzer.text:
                    print("请先加载文本")
                else:
                    stats = analyzer.get_stats()
                    print(f"\n单词数: {stats.word_count}")
                    print(f"字符数: {stats.char_count}")
                    print(f"行数: {stats.line_count}")

            elif command == "top":
                n = int(args) if args else 10
                top_words = analyzer.most_common(n)
                print(f"\n最常见的 {n} 个单词:")
                for word, count in top_words:
                    print(f"  {word}: {count}")

            elif command == "search":
                if not args:
                    print("请提供要搜索的单词")
                else:
                    result = analyzer.search_word(args)
                    print(f"\n'{result['word']}' 出现 {result['count']} 次 ({result['percentage']}%)")

            elif command == "report":
                if not analyzer.text:
                    print("请先加载文本")
                else:
                    analyzer.print_report()

            else:
                print(f"未知命令: {command}")

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    main()
