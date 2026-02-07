# 合规与审计

## 目录
- [概述](#概述)
- [GDPR合规](#gdpr合规)
- [SOC2合规](#soc2合规)
- [等保三级](#等保三级)
- [审计日志](#审计日志)

## 概述

### 主要合规框架对比

```
┌────────────────┬──────────────┬──────────────┬──────────────┐
│    框架        │  GDPR        │  SOC 2       │  等保三级    │
├────────────────┼──────────────┼──────────────┼──────────────┤
│ 适用范围       │ 欧盟个人数据 │ 美国服务组织 │  中国信息系统│
│ 强制性         │ 法律要求     │ 客户要求     │  法律要求    │
│ 罚款           │ 最高4%营收   │ N/A          │  行政处罚    │
│ 审计周期       │ 持续         │ 年度         │  年度        │
│ 核心关注       │ 隐私保护     │ 安全控制     │  系统安全    │
└────────────────┴──────────────┴──────────────┴──────────────┘
```

## GDPR合规

### GDPR核心原则

```
1. 合法性、公平性、透明性
   - 明确告知用户数据收集目的
   - 获取明确同意
   - 透明的隐私政策

2. 目的限制
   - 只收集必要数据
   - 不得用于其他目的
   - 明确数据用途

3. 数据最小化
   - 仅收集最少必要数据
   - 定期删除不需要的数据
   - 匿名化/假名化

4. 准确性
   - 确保数据准确
   - 及时更新
   - 允许用户纠正

5. 存储限制
   - 限制保留期限
   - 自动删除过期数据
   - 明确保留政策

6. 完整性和保密性
   - 加密存储
   - 访问控制
   - 安全传输

7. 问责制
   - 数据保护官(DPO)
   - 隐私影响评估(PIA)
   - 数据泄露通知
```

### GDPR技术实现

```python
# 数据主体权利实现
from datetime import datetime, timedelta
from typing import List, Dict
import hashlib

class GDPRCompliance:
    """GDPR合规实现"""

    def __init__(self, db_connection):
        self.db = db_connection

    # 1. 访问权 (Right to Access)
    def export_user_data(self, user_id: str) -> Dict:
        """导出用户所有数据"""
        data = {
            'personal_info': self._get_personal_info(user_id),
            'orders': self._get_orders(user_id),
            'activities': self._get_activities(user_id),
            'consents': self._get_consents(user_id),
            'export_date': datetime.now().isoformat()
        }
        return data

    # 2. 更正权 (Right to Rectification)
    def update_user_data(self, user_id: str, updates: Dict):
        """允许用户更正数据"""
        self.db.execute("""
            UPDATE users 
            SET {updates}, updated_at = NOW()
            WHERE user_id = %s
        """, (user_id,))

        # 记录修改日志
        self._log_modification(user_id, updates)

    # 3. 删除权 (Right to Erasure / Right to be Forgotten)
    def delete_user_data(self, user_id: str, reason: str):
        """删除或匿名化用户数据"""

        # 匿名化个人信息
        self.db.execute("""
            UPDATE users 
            SET 
                email = %s,
                name = 'Deleted User',
                phone = NULL,
                address = NULL,
                deleted_at = NOW(),
                deletion_reason = %s
            WHERE user_id = %s
        """, (
            self._anonymize_email(user_id),
            reason,
            user_id
        ))

        # 删除非必要数据
        self.db.execute("DELETE FROM user_activities WHERE user_id = %s", (user_id,))
        self.db.execute("DELETE FROM user_preferences WHERE user_id = %s", (user_id,))

        # 保留交易记录(法律要求)但匿名化
        self.db.execute("""
            UPDATE orders 
            SET user_name = 'Deleted User',
                shipping_address = 'REDACTED'
            WHERE user_id = %s
        """, (user_id,))

        # 记录删除日志
        self._log_deletion(user_id, reason)

    # 4. 限制处理权 (Right to Restriction)
    def restrict_processing(self, user_id: str, restriction_type: str):
        """限制数据处理"""
        self.db.execute("""
            UPDATE users 
            SET processing_restricted = true,
                restriction_type = %s,
                restriction_date = NOW()
            WHERE user_id = %s
        """, (restriction_type, user_id))

    # 5. 数据可移植权 (Right to Data Portability)
    def export_portable_format(self, user_id: str, format='json') -> bytes:
        """导出机器可读格式"""
        data = self.export_user_data(user_id)

        if format == 'json':
            import json
            return json.dumps(data, indent=2).encode()
        elif format == 'xml':
            # 实现XML导出
            pass
        elif format == 'csv':
            # 实现CSV导出
            pass

    # 6. 反对权 (Right to Object)
    def object_to_processing(self, user_id: str, processing_type: str):
        """反对特定处理"""
        self.db.execute("""
            INSERT INTO processing_objections (user_id, processing_type, created_at)
            VALUES (%s, %s, NOW())
        """, (user_id, processing_type))

        # 停止相关处理
        if processing_type == 'marketing':
            self.db.execute("""
                UPDATE users SET marketing_consent = false WHERE user_id = %s
            """, (user_id,))

    # 同意管理
    def record_consent(self, user_id: str, consent_type: str, granted: bool):
        """记录用户同意"""
        self.db.execute("""
            INSERT INTO user_consents (
                user_id, consent_type, granted, 
                ip_address, user_agent, created_at
            ) VALUES (%s, %s, %s, %s, %s, NOW())
        """, (user_id, consent_type, granted, request.remote_addr, request.user_agent))

    # 数据泄露通知
    def report_data_breach(self, breach_details: Dict):
        """72小时内通知监管机构"""
        breach = {
            'breach_id': self._generate_breach_id(),
            'discovered_at': datetime.now(),
            'affected_users': breach_details['affected_users'],
            'data_types': breach_details['data_types'],
            'severity': breach_details['severity'],
            'mitigation': breach_details['mitigation']
        }

        # 记录泄露
        self._log_breach(breach)

        # 通知监管机构 (72小时内)
        if breach['severity'] in ['high', 'critical']:
            self._notify_dpa(breach)

        # 通知受影响用户
        self._notify_affected_users(breach)

    def _anonymize_email(self, user_id: str) -> str:
        """匿名化邮箱"""
        hash_value = hashlib.sha256(user_id.encode()).hexdigest()[:10]
        return f"deleted_{hash_value}@deleted.local"

    def _log_modification(self, user_id, updates):
        """记录数据修改"""
        pass

    def _log_deletion(self, user_id, reason):
        """记录数据删除"""
        pass

    def _log_breach(self, breach):
        """记录数据泄露"""
        pass

    def _notify_dpa(self, breach):
        """通知数据保护机构"""
        pass

    def _notify_affected_users(self, breach):
        """通知受影响用户"""
        pass
