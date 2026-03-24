# Skills 文档目录

本目录包含所有可用 Skills 的说明文档。

---

## Skill 标准规范

### Skill 格式

```markdown
# Skill 名称

**功能描述**

## 支持的操作

1. 操作1 - 描述
2. 操作2 - 描述

## 输入参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| param1 | string | 是 | 参数说明 |
| param2 | int | 否 | 参数说明 |

## 输出格式

```json
{
    "success": true,
    "output": "结果内容",
    "metadata": {}
}
```

## 使用示例

```python
from skills import SkillExecutor

result = await executor.execute("skill_name", {
    "param1": "value1",
    "param2": 123
})
```
```

### Skills 调用规范

```markdown
# Skill 执行流程

1. 接收输入参数
2. 执行对应操作
3. 返回标准格式结果
```

---

## 已注册 Skills

### 1. code_writer - 代码编写

根据需求编写完整代码。

**支持的语言**: Python, JavaScript, TypeScript, Java, Go, Rust

**支持的代码类型**: function, class, script

**输入参数**:
- `language` - 编程语言
- `code_name` - 文件名
- `requirements` - 需求描述
- `code_type` - 代码类型
- `existing_code` - 现有代码（可选）

**输出**: 生成的代码字符串

---

### 2. code_debugger - 代码调试

分析和修复代码错误。

**输入参数**:
- `code` - 待调试代码
- `error` - 错误类型（可选）
- `error_message` - 错误信息（可选）
- `test_cases` - 测试用例（可选）

**输出**: 修复后的代码

---

### 3. code_updater - 代码更新

补充和更新代码功能。

**输入参数**:
- `code` - 原始代码
- `update_type` - 更新类型
- `update_content` - 更新内容
- `insert_position` - 插入位置

**输出**: 更新后的代码

---

### 4. code_refactor - 代码重构

重构代码以提高可读性和性能。

**输入参数**:
- `code` - 待重构代码
- `refactoring_type` - 重构类型

**输出**: 重构后的代码

---

### 5. code_review - 代码审查

审查代码质量和风格。

**输入参数**:
- `code` - 待审查代码
- `style_guide` - 代码规范

**输出**: 审查报告
