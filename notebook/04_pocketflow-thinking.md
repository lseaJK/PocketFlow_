这是一个**结构化思维链（Chain of Thought）推理系统**，通过递归调用LLM来实现复杂问题的分步求解和计划执行。

## 核心设计理念

### 1. 结构化计划执行
- **计划驱动**：不是简单的问答，而是创建和执行一个动态调整的计划
- **状态管理**：每个计划步骤有明确状态（Pending/Done/Verification Needed）
- **嵌套分解**：复杂步骤可以分解为子步骤，形成树状结构

### 2. 自循环机制
```python
# 关键的自循环逻辑
cot[ChainOfThoughtNode] -->|"continue"| cot
```
节点根据计划状态决定是否继续思考，直到问题解决。

## 关键组件详解

### ChainOfThoughtNode-prep()
```python
def prep(self, shared):
    # 1. 提取基础信息
    problem = shared.get("problem", "")
    thoughts = shared.get("thoughts", [])
    current_thought_number = shared.get("current_thought_number", 0)

    shared["current_thought_number"] = current_thought_number + 1

    # 2. 格式化历史思考（如果有） 
    thoughts_text = ""
    last_plan_structure = None # Will store the list of dicts
    if thoughts:
        thoughts_text_list = []
        for i, t in enumerate(thoughts):
                thought_block = f"Thought {t.get('thought_number', i+1)}:\n"
                thinking = textwrap.dedent(t.get('current_thinking', 'N/A')).strip()
                thought_block += f"  Thinking:\n{textwrap.indent(thinking, '    ')}\n"

                plan_list = t.get('planning', [])
                # Use the recursive helper for display formatting
                plan_str_formatted = format_plan(plan_list, indent_level=2)
                thought_block += f"  Plan Status After Thought {t.get('thought_number', i+1)}:\n{plan_str_formatted}"

                if i == len(thoughts) - 1:
                    last_plan_structure = plan_list # Keep the actual structure

                thoughts_text_list.append(thought_block)

        thoughts_text = "\n--------------------\n".join(thoughts_text_list)
    else: # 处理首次思考的情况
        thoughts_text = "No previous thoughts yet."
        # Suggest an initial plan structure using dictionaries
        last_plan_structure = [
            {'description': "Understand the problem", 'status': "Pending"},
            {'description': "Develop a high-level plan", 'status': "Pending"},
            {'description': "Conclusion", 'status': "Pending"}
        ]

    # 3. 格式化计划用于提示
    last_plan_text_for_prompt = format_plan_for_prompt(last_plan_structure) if last_plan_structure else "# No previous plan available."

    return {
        "problem": problem,
        "thoughts_text": thoughts_text,
        "last_plan_text": last_plan_text_for_prompt,
        "last_plan_structure": last_plan_structure, # Pass the raw structure too if needed for complex updates
        "current_thought_number": current_thought_number + 1,
        "is_first_thought": not thoughts
    }
```

格式化历史思考的输出格式
```
Thought 1:
Thinking:
    [思考内容...]
Plan Status After Thought 1:
    - [Done] Understand the problem: 理解了问题要求
    - [Pending] Develop mathematical model
--------------------
Thought 2:
Thinking:
    [思考内容...]
Plan Status After Thought 2:
    - [Done] Understand the problem: 理解了问题要求
    - [Done] Develop mathematical model: 建立了Markov链模型
    - [Pending] Solve Markov chain equations
```

### ChainOfThoughtNode-exec()
负责构造提示、调用LLM并解析结果。

```python
def exec(self, prep_res):
    # 接受输入参数
    problem = prep_res["problem"]
    thoughts_text = prep_res["thoughts_text"]
    last_plan_text = prep_res["last_plan_text"]
    # last_plan_structure = prep_res["last_plan_structure"] # Can use if needed
    current_thought_number = prep_res["current_thought_number"]
    is_first_thought = prep_res["is_first_thought"]

    # 构造基础指令
    instruction_base = textwrap.dedent(f"""
    你的任务是生成下一个思考（思考 {current_thought_number}）。

    指令：
    1.  **评估前一个思考：** 如果不是第一次思考，在 `current_thinking` 中首先评估思考 {current_thought_number - 1}。说明："对思考 {current_thought_number - 1} 的评估：[正确/轻微问题/重大错误 - 解释]"。优先处理错误。
    2.  **执行步骤：** 执行计划中第一个状态为 `Pending` 的步骤。
    3.  **维护计划（结构）：** 生成更新的 `planning` 列表。每个项目应为包含以下键的字典：`description`（字符串）、`status`（字符串："Pending"、"Done"、"Verification Needed"），以及可选的 `result`（字符串，步骤完成时的简洁摘要）或 `mark`（字符串，需要验证的原因）。子步骤通过包含这些字典列表的 `sub_steps` 键表示。
    4.  **更新当前步骤状态：** 在更新后的计划中，将已执行步骤的 `status` 改为 "Done" 并添加包含简洁摘要的 `result` 键。如果基于评估需要验证，将状态改为 "Verification Needed" 并添加 `mark`。
    5.  **细化计划（子步骤）：** 如果一个 "Pending" 步骤很复杂，在其字典中添加 `sub_steps` 键，包含分解后的新步骤字典列表（状态："Pending"）。保持父步骤状态为 "Pending" 直到所有子步骤都标记为 "Done"。
    6.  **细化计划（错误处理）：** 根据评估发现逻辑修改计划（例如，更改状态、添加纠正步骤）。
    7.  **最终步骤：** 确保计划朝着最终的步骤字典如 `{{'description': "Conclusion", 'status': "Pending"}}` 推进。
    8.  **终止条件：** 仅当执行描述为 "Conclusion" 的步骤时，将 `next_thought_needed` 设置为 `false`。
""")
    
    if is_first_thought: # 首次思考的情况
        instruction_context = textwrap.dedent("""
            **这是第一次思考：** 创建初始计划作为字典列表（键：description, status）。如果需要，通过 `sub_steps` 键包含子步骤。然后，在 `current_thinking` 中执行第一步，并提供更新后的计划（将步骤1标记为 `status: Done` 并附带 `result`）。
        """)
    else: # 后续思考的情况
        instruction_context = textwrap.dedent(f"""
            **之前的计划（简化视图）：**
            {last_plan_text}

            在 `current_thinking` 中首先评估思考 {current_thought_number - 1}。然后，继续执行第一个 `status: Pending` 的步骤。更新计划结构（字典列表）以反映评估、执行和细化。
        """)

    # 定义输出格式
    instruction_format = textwrap.dedent("""
        你的响应必须且仅以用 ```yaml ... ``` 包裹的YAML结构格式：
        ```yaml
        current_thinking: |
        # 对思考N的评估：[评估]...（如果适用）
        # 当前步骤的思考...
        planning:
        # 字典列表（键：description, status, 可选的[result, mark, sub_steps]）
        - description: "步骤1"
            status: "Done"
            result: "简洁的结果摘要"
        - description: "步骤2 复杂任务" # 现在被分解
            status: "Pending" # 父步骤保持Pending
            sub_steps:
            - description: "子任务2a"
                status: "Pending"
            - description: "子任务2b"
                status: "Verification Needed"
                mark: "思考X的结果似乎有问题"
        - description: "步骤3"
            status: "Pending"
        - description: "Conclusion"  # 注意：这个键名保持英文不变
            status: "Pending"
        next_thought_needed: true # 仅当执行Conclusion步骤时设置为false。
        ```
    """)

    # 组合完整提示
    prompt = textwrap.dedent(f"""
        你是一个细致严谨的AI助手，通过结构化计划逐步解决复杂问题。你批判性地评估先前步骤，如果需要则细化计划加入子步骤，并逻辑地处理错误。请使用指定的字典结构表示计划。

        问题：{problem}

        先前思考：
        {thoughts_text}
        --------------------
        {instruction_base}
        {instruction_context}
        {instruction_format}
    """)

    # 完整的执行指令结构

    response = call_llm(prompt)

    # Simple YAML extraction
    yaml_str = response.split("```yaml")[1].split("```")[0].strip()
    thought_data = yaml.safe_load(yaml_str) # Can raise YAMLError

    # --- Validation (using assert) ---
    assert thought_data is not None, "YAML parsing failed, result is None"
    assert "current_thinking" in thought_data, "LLM response missing 'current_thinking'"
    assert "next_thought_needed" in thought_data, "LLM response missing 'next_thought_needed'"
    assert "planning" in thought_data, "LLM response missing 'planning'"
    assert isinstance(thought_data.get("planning"), list), "LLM response 'planning' is not a list"
    # Optional: Add deeper validation of list items being dicts if needed
    # --- End Validation ---

    # Add thought number
    thought_data["thought_number"] = current_thought_number
    return thought_data
```

### ChainOfThoughtNode-post()
负责处理执行结果、更新状态、格式化输出并决定是否继续循环。
```python
def post(self, shared, prep_res, exec_res):
    # 存储当前思考
    if "thoughts" not in shared:
        shared["thoughts"] = []
    shared["thoughts"].append(exec_res)

    # 提取并格式化数据
    plan_list = exec_res.get("planning", ["Error: Planning data missing."])
    plan_str_formatted = format_plan(plan_list, indent_level=1)

    thought_num = exec_res.get('thought_number', 'N/A')
    current_thinking = exec_res.get('current_thinking', 'Error: Missing thinking content.')
    dedented_thinking = textwrap.dedent(current_thinking).strip()

    # 判断是否为结论步骤（可选逻辑）
    is_conclusion = False
    if isinstance(plan_list, list):
        # 从后向前检查，因为最新步骤通常在末尾
        for item in reversed(plan_list):
            if isinstance(item, dict) and item.get('description') == "Conclusion":
                # 如果Conclusion已标记为Done，或者为Pending但next_thought_needed为False
                if item.get('status') == "Done" or (item.get('status') == "Pending" and not exec_res.get("next_thought_needed", True)):
                    is_conclusion = True
                    break
                # Simple check, might need nested search if Conclusion could be a sub-step

    # 终止条件判断和处理
    if not exec_res.get("next_thought_needed", True): # Primary termination signal
        shared["solution"] = dedented_thinking # Solution is the thinking content of the final step
        print(f"\nThought {thought_num} (Conclusion):")
        print(f"{textwrap.indent(dedented_thinking, '  ')}")
        print("\nFinal Plan Status:")
        print(textwrap.indent(plan_str_formatted, '  '))
        print("\n=== FINAL SOLUTION ===")
        print(dedented_thinking)
        print("======================\n")
        return "end"

    # 继续循环的处理
    print(f"\nThought {thought_num}:")
    print(f"{textwrap.indent(dedented_thinking, '  ')}")
    print("\nCurrent Plan Status:")
    print(textwrap.indent(plan_str_formatted, '  '))
    print("-" * 50)

    return "continue"
```

**继续循环的输出：**
```
Thought 3:
  对思考2的评估：正确 - Markov链模型建立正确
  思考：现在求解Markov链的稳态方程...

Current Plan Status:
  - [Done] 理解问题: 明确了连续掷出3,4,5的条件
  - [Done] 建立模型: 建立了六状态Markov链模型
  - [Pending] 求解方程
    - [Pending] 建立状态转移矩阵
    - [Pending] 求解线性方程组
  - [Pending] Conclusion
--------------------------------------------------
```

**终止循环的输出：**
```
Thought 5 (Conclusion):
  对思考4的评估：正确 - 计算结果验证通过
  思考：最终概率为 1/8...

Final Plan Status:
  - [Done] 理解问题: 明确了连续掷出3,4,5的条件
  - [Done] 建立模型: 建立了六状态Markov链模型
  - [Done] 求解方程: 得到稳态概率分布
  - [Done] Conclusion: 计算最终概率为1/8

=== FINAL SOLUTION ===
对思考4的评估：正确 - 计算结果验证通过
思考：最终概率为 1/8...
======================
```

**后处理阶段的决策流程**
```
开始
  ↓
存储当前思考到共享状态
  ↓
提取并格式化计划和思考内容
  ↓
检查next_thought_needed标志
  ├── 如果为False → 终止分支
  │     ↓
  │   保存解决方案
  │     ↓
  │   显示结论内容
  │     ↓
  │   返回"end" → 结束循环
  │
  └── 如果为True → 继续分支
        ↓
      显示当前进度
        ↓
      返回"continue" → 继续下一次思考
```
