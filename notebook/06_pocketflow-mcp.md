这个MCP项目实现了一个基于工作流（Flow）的工具调用系统。


## 项目架构

### 1. **MCP服务器部分** (`simple_server.py`)
```python
# 创建了一个数学运算服务器，提供四个基本数学工具
mcp = FastMCP("Math Operations Server")

# 工具定义
@mcp.tool()
def add(a: int, b: int) -> int:  # 加法工具
@mcp.tool()
def subtract(a: int, b: int) -> int:  # 减法工具
@mcp.tool()
def multiply(a: int, b: int) -> int:  # 乘法工具
@mcp.tool()
def divide(a: int, b: int) -> float:  # 除法工具
```

### 2. **工作流系统** (Node/Flow)
```python
question = default_question
for arg in sys.argv[1:]:
    if arg.startswith("--"):
        question = arg[2:]
        break

print(f"🤔 Processing question: {question}")

# Create nodes
get_tools_node = GetToolsNode()
decide_node = DecideToolNode()
execute_node = ExecuteToolNode()

# Connect nodes
get_tools_node - "decide" >> decide_node
decide_node - "execute" >> execute_node

# Create and run flow
flow = Flow(start=get_tools_node)
shared = {"question": question}
flow.run(shared)
```

### 工具调用具体流程

#### 获取可用工具 (`GetToolsNode`)

`@mcp.tool()`是FastMCP开发框架中的一个核心装饰器，它的核心作用是：将一个普通的Python函数注册为可供AI模型（或MCP客户端）发现和调用的标准化工具。

```python
def get_tools(server_script_path=None):
    """Get available tools, either from MCP server or locally based on MCP global setting."""
    if MCP:
        return mcp_get_tools(server_script_path)
    else:
        return local_get_tools(server_script_path)

def mcp_get_tools(server_script_path):
    """Get available tools from an MCP server.
    """
    async def _get_tools():
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path]
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_response = await session.list_tools()
                return tools_response.tools
    
    return asyncio.run(_get_tools())

class GetToolsNode(Node):
    def prep(self, shared):
        """Initialize and get tools"""
        # The question is now passed from main via shared
        print("🔍 Getting available tools...")
        return "simple_server.py"

    def exec(self, server_path):
        """Retrieve tools from the MCP server"""
        tools = get_tools(server_path)
        return tools

    def post(self, shared, prep_res, exec_res):
        """Store tools and process to decision node"""
        tools = exec_res # exec 返回的原始工具列表
        shared["tools"] = tools # 存储原始数据
        
        # 2. 格式化数据，生成给LLM看的描述
        tool_info = []
        for i, tool in enumerate(tools, 1):
            properties = tool.inputSchema.get('properties', {})
            required = tool.inputSchema.get('required', [])
            
            params = []
            for param_name, param_info in properties.items():
                param_type = param_info.get('type', 'unknown')
                req_status = "(Required)" if param_name in required else "(Optional)"
                params.append(f"    - {param_name} ({param_type}): {req_status}")
            
            tool_info.append(f"[{i}] {tool.name}\n  Description: {tool.description}\n  Parameters:\n" + "\n".join(params))
        
        shared["tool_info"] = "\n".join(tool_info) # 存入共享状态
        return "decide"
```

#### 智能决策 (`DecideToolNode`)
```python
class DecideToolNode(Node):
    def prep(self, shared):
        """准备提示词，供LLM处理问题"""
        tool_info = shared["tool_info"]
        question = shared["question"]
        
        prompt = f"""
### 上下文
你是一个可以通过模型上下文协议使用工具的助手。

### 可用工具
{tool_info}

### 任务
回答这个问题："{question}"

## 下一步行动
分析问题，提取任何数字或参数，并决定使用哪个工具。
请严格按以下格式返回你的响应：

\`\`\`yaml
thinking: |
    <你对此问题的逐步推理：它在问什么，需要提取什么数字/参数>
tool: <要使用的工具名称>
reason: <选择此工具的原因>
parameters:
    <参数名>: <参数值>
    <参数名>: <参数值>
\`\`\`
重要提示：
1. 请从问题中正确提取数字
2. 为多行字段使用适当的缩进（4个空格）
3. 对多行文本字段使用 | 字符
"""
        return prompt

    def exec(self, prompt):
        """调用LLM处理问题并决定使用哪个工具"""
        print("🤔 正在分析问题并决定使用哪个工具...")
        response = call_llm(prompt)
        return response

    def post(self, shared, prep_res, exec_res):
        """从YAML中提取决策并保存到共享上下文中"""
        try:
            # 提取YAML部分
            yaml_str = exec_res.split("```yaml")[1].split("```")[0].strip()
            # 解析YAML
            decision = yaml.safe_load(yaml_str)
            
            # 将决策保存到共享上下文中
            shared["tool_name"] = decision["tool"]
            shared["parameters"] = decision["parameters"]
            shared["thinking"] = decision.get("thinking", "")
            
            print(f"💡 已选择工具：{decision['tool']}")
            print(f"🔢 已提取参数：{decision['parameters']}")
            
            # 指定下一个要执行的节点
            return "execute"
        except Exception as e:
            print(f"❌ 解析LLM响应时出错：{e}")
            print("原始响应：", exec_res)
            return None
```
这个节点的设计体现了典型的**规划-执行**模式：先由LLM进行理解、规划和决策，然后将结构化的决策结果交给后续节点去具体执行。

### 执行工具 (`ExecuteToolNode`)
`ExecuteToolNode`负责将前序节点的“决策”转化为“实际行动”并交付最终结果。

```python
class ExecuteToolNode(Node):
    def prep(self, shared):
        # 读取 `DecideToolNode` 存入的 `tool_name`（工具名）和 `parameters`（参数字典）
        return shared["tool_name"], shared["parameters"]

    def exec(self, inputs):
        """Execute the chosen tool"""
        tool_name, parameters = inputs
        print(f"🔧 Executing tool '{tool_name}' with parameters: {parameters}")
        # 调用关键的 `call_tool` 函数，该函数会与 `simple_server.py` 这个 MCP 服务器通信，指示其运行指定的工具（如 `add`），并传入对应参数。
        result = call_tool("simple_server.py", tool_name, parameters)
        return result

    def post(self, shared, prep_res, exec_res):
        print(f"\n✅ Final Answer: {exec_res}")
        return "done"
```

`ExecuteToolNode` 虽然代码简短，但在实际应用中却是**可靠性、安全性和性能的关键所在**，需要特别注意以下几点：

| 注意事项 | 说明与潜在风险 | 改进建议 |
| :--- | :--- | :--- |
| **1. 异常处理** | 当前 `exec` 方法未捕获异常。如果网络不通、服务器出错、参数类型不匹配或工具执行内部报错（如除零），整个流程会**崩溃**。 | 在 `exec` 或 `call_tool` 外部添加 `try…except`，优雅地处理异常，并将错误信息存入 `shared` 或返回给用户，而非直接中断。 |
| **2. 输入验证与净化** | 它完全信任来自LLM的 `parameters`。如果LLM被诱导或被攻击而传入了恶意参数（如注入命令、极大数字导致资源耗尽），可能引发安全问题。 | 在 `prep` 或 `exec` 阶段增加验证逻辑，例如检查参数类型、数值范围、字符串长度等，或使用安全的数据转换。 |
| **3. 执行上下文隔离** | 工具执行可能**有状态或产生副作用**（如写入文件、修改数据库）。在高并发或多次执行同一流程时，如果不隔离，可能导致数据污染。 | 确保工具函数设计为无状态的，或在工作流层面为每次执行创建独立的临时上下文/会话。 |
| **4. 超时与资源限制** | 如果某个工具执行时间过长（如复杂计算、等待外部API），会**阻塞整个工作流线程**，影响系统响应。 | 为 `call_tool` 设置**执行超时**机制，防止长时间挂起。对于耗时任务，可考虑异步执行模式。 |
| **5. 结果格式化与后处理** | 当前 `post` 只是简单打印。实际应用中，工具返回的可能是复杂对象（如JSON、列表），需要进一步**提取、转换或渲染**才能成为友好的“答案”。 | 将 `post` 扩展为一个小型的**结果处理器**，根据 `tool_name` 或结果结构进行定制化处理，再输出。 |
| **6. 可观测性** | 仅打印“执行中”和“最终答案”对于调试和监控是不够的，缺乏**执行耗时、内部状态**等信息。 | 添加更详细的日志（如开始/结束时间戳），或集成监控指标（Metrics）来跟踪工具执行的成功率、延迟等。 |

## MCP通信流程

```
用户问题
    ↓
GetToolsNode
    ↓ (获取工具列表)
DecideToolNode
    ↓ (LLM分析决策)
ExecuteToolNode
    ↓ (调用实际工具)
结果输出
```

关键组件交互

1. **MCP客户端** (`utils.py`中的函数)：
   - `get_tools()`: 获取服务器工具列表
   - `call_tool()`: 调用具体工具

2. **工作流引擎** (`pocketflow`):
   - `Node`: 基础节点类
   - `Flow`: 工作流管理器

3. **LLM集成**:
   - `call_llm()`: 与大语言模型通信，用于分析决策
