---
name: code
description: Use when writing, debugging, updating, refactoring, or reviewing code - supports Python, JavaScript, TypeScript, Java, Go, and Rust with specific actions for each development task
---

# Code Development Skill

## Overview

Comprehensive code development skill supporting the full development lifecycle: writing new code, debugging errors, updating functionality, refactoring for improvement, and reviewing for quality.

## When to Use

Use this skill when you need to:

- **Write**: Create new functions, classes, or scripts from requirements
- **Debug**: Analyze and fix errors (SyntaxError, TypeError, ZeroDivisionError, etc.)
- **Update**: Add new functions, methods, or imports to existing code
- **Refactor**: Extract functions, simplify logic, or optimize performance
- **Review**: Check code quality against style guides (pep8, google, standard)

## Quick Reference

| Action | Parameters | Use Case |
|--------|------------|----------|
| `write` | language, code_name, requirements | Create new code from scratch |
| `debug` | code, error, error_message | Fix runtime or syntax errors |
| `update` | code, update_type, update_content | Add functions/methods/imports |
| `refactor` | code, refactoring_type | Improve code structure |
| `review` | code, style_guide | Check code quality |

## Supported Languages

| Language | Extension | Characteristics |
|----------|-----------|-----------------|
| Python | .py | Dynamic typing, rapid development |
| JavaScript | .js | Browser scripting |
| TypeScript | .ts | JavaScript superset with types |
| Java | .java | Object-oriented programming |
| Go | .go | High-performance concurrency |
| Rust | .rs | Memory safety without GC |

## Action Details

### write - Create New Code

```yaml
action: write
language: python
code_name: analysis_utils.py
requirements: Calculate AH stock premium ratio
code_type: function
```

### debug - Fix Errors

```yaml
action: debug
language: python
code: def divide(a, b): return a / b
error: ZeroDivisionError
error_message: division by zero
```

### update - Add Functionality

```yaml
action: update
language: python
code: def foo(): pass
update_type: add_function
update_content: Add calculation function
insert_position: end
```

### refactor - Improve Code

```yaml
action: refactor
language: python
code: def process(): pass
refactoring_type: extract_function
```

### review - Check Quality

```yaml
action: review
language: python
code: def FooBar(): pass
style_guide: pep8
```

## Output Format

All actions return structured JSON:

```json
{
  "success": true,
  "output": "code content or report",
  "metadata": {
    "skill": "code",
    "action": "write",
    "language": "python",
    "code_name": "example.py",
    "lines": 45,
    "size": 1234,
    "issues_found": 0,
    "suggestions": []
  }
}
```

## Examples

### 1. Write Python Function

```yaml
action: write
language: python
code_name: premium_calculator.py
requirements: Calculate AH stock premium ratio and percentage
code_type: function
```

### 2. Debug Division by Zero

```yaml
action: debug
language: python
code: |
  def divide(a, b):
      return a / b
  print(divide(10, 0))
error: ZeroDivisionError
error_message: division by zero
```

### 3. Add Function to Existing Code

```yaml
action: update
language: python
code: def foo(): pass
update_type: add_function
update_content: Add add(a, b) function for addition
insert_position: end
```

### 4. Extract Function

```yaml
action: refactor
language: python
code: |
  def process():
      # validation logic here
      # processing logic here
      pass
refactoring_type: extract_function
```

### 5. Review Code Style

```yaml
action: review
language: python
code: def FooBar(): pass
style_guide: pep8
```

## Style Guides

| Guide | Description |
|------- |-------------|
| `pep8` | Python style guide (default) |
| `google` | Google language conventions |
| `standard` | JavaScript Standard Style |
