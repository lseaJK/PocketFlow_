
# 一文理清 Langfuse：定义、作用 + 极简可运行例子
Langfuse核心定位——**它是给 LLM 应用（比如 AI 助手、AI 工作流）做“运行监控+问题排查+性能优化”的工具**，让运行过程变得清晰可查。


## 一、Langfuse 是什么？有什么用？
1. **什么是 Langfuse？**

Langfuse 是一款 **开源的 LLM 应用可观测性平台**（支持本地部署/云服务），核心是“捕获 LLM 应用运行时的全量数据，并用可视化界面展示”。

核心概念：
- **Trace（追踪）**：一次完整的 AI 任务执行（比如“用户问问题→AI 调用模型→返回答案”），是所有数据的“顶层容器”；
- **Span（跨度）**：Trace 下的细分环节（比如“预处理用户输入”“调用 GPT-4”“格式化输出结果”），每个环节的耗时、输入输出都存在这里；
- **Dashboard（仪表盘）**：Langfuse 的 Web 界面，能直观看到所有 Trace、Span 的数据（比如哪个环节耗时最长、哪个请求失败了、总共用了多少 Token）。

2. **Langfuse 核心作用（解决什么痛点？）**

开发 LLM 应用时，你一定会遇到这些问题，而 Langfuse 就是专门解决它们的：
| 开发痛点 | Langfuse 怎么解决？ |
|----------|--------------------|
| 调用 LLM 失败了，但不知道是“提示词写错”“API 密钥过期”还是“网络问题” | 自动捕获异常信息（错误类型、栈轨迹、当时的输入参数），点一下就能看到 |
| AI 回答太慢，但不知道是“模型调用耗时久”还是“中间数据处理卡壳” | 记录每个环节的耗时（比如“预处理 0.2s → 调用 GPT-4 3.5s → 格式化 0.1s”），快速定位瓶颈 |
| 想统计“今天用了多少 Token”“哪个模型最常用”“成功率多少” | 自动生成指标报表，不用自己写统计代码 |
| 上线后用户反馈“AI 答非所问”，但没法复现当时的场景 | 留存每次请求的“用户输入→AI 输出→中间所有数据”，支持回溯复现 |
| 多智能体/复杂工作流（比如“AI 1 分析需求→AI 2 生成内容→AI 3 校验”），某个环节出错没法定位 | 生成“链路图”，清晰展示每个智能体的调用顺序和数据流转，哪个环节红了就是哪出错 |


## 二、极简可运行例子：5 分钟上手 Langfuse
下面用“调用 OpenAI API 生成文案”的场景，写一个最简化的例子，让你跑通“代码→Langfuse 可视化”全流程。

**前置准备**
1. **获取 Langfuse 凭证**（免费）：
   - 访问 Langfuse 官网（https://langfuse.com/）注册账号（支持 GitHub 登录）；
   - 登录后，进入「Settings → API Keys」，复制 `PUBLIC KEY` 和 `SECRET KEY`（后续配置用）；
   - 记住 Langfuse 服务地址（默认是 `https://cloud.langfuse.com`，本地部署的话填自己的地址）。

2. **安装依赖**：
   打开终端，执行命令（只需要两个核心依赖）：
   ```bash
   pip install langfuse openai  # langfuse是核心，openai用于调用LLM
   ```

**完整代码（可直接复制运行）**
```python
# 1. 导入依赖
from langfuse import Langfuse
from openai import OpenAI
import os

# 2. 配置 Langfuse（替换成你自己的凭证）
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "你的Langfuse公钥"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", "你的Langfuse私钥"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")  # 默认云服务地址
)

# 3. 初始化 OpenAI 客户端（替换成你的 OpenAI API 密钥）
openai_client = OpenAI(
    api_key="你的OpenAI API密钥"  # 从OpenAI官网获取：https://platform.openai.com/
)

# 4. 定义一个“生成产品文案”的函数（包含多个环节，方便展示 Span）
def generate_product_copy(product_name: str, product_feature: str) -> str:
    # 4.1 创建顶层 Trace（对应“生成文案”这个完整任务）
    trace = langfuse.trace(
        name="product-copy-generation",  # Trace 名称（自定义，方便识别）
        user_id="test-user-001"  # 可选：关联用户ID，方便统计单个用户的使用情况
    )

    try:
        # 4.2 环节1：预处理输入（创建第一个 Span）
        with trace.span(name="preprocess-input") as span:
            # 记录这个环节的输入
            span.update(inputs={"product_name": product_name, "product_feature": product_feature})
            # 预处理逻辑（比如拼接成提示词模板）
            prompt = f"为产品「{product_name}」写一段文案，突出卖点：{product_feature}，风格活泼亲切，不超过50字"
            # 记录这个环节的输出
            span.update(outputs={"processed_prompt": prompt})

        # 4.3 环节2：调用 OpenAI 模型（创建第二个 Span，LLM 调用是核心环节）
        with trace.span(name="call-openai") as span:
            span.update(inputs={"prompt": prompt, "model": "gpt-3.5-turbo"})
            # 调用 OpenAI API
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            copy = response.choices[0].message.content.strip()
            # 记录输出、Token 消耗（Langfuse 会自动解析 OpenAI 响应的 Token 数据）
            span.update(
                outputs={"generated_copy": copy},
                # 手动记录 Token（也可以让 Langfuse 自动捕获，这里演示手动配置）
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            )

        # 4.4 环节3：格式化输出（创建第三个 Span）
        with trace.span(name="format-output") as span:
            span.update(inputs={"raw_copy": copy})
            # 简单格式化（比如加个表情符号）
            formatted_copy = f"✨ {copy} ✨"
            span.update(outputs={"formatted_copy": formatted_copy})

        # 4.5 标记 Trace 成功
        trace.update(status="success", outputs={"final_copy": formatted_copy})
        return formatted_copy

    except Exception as e:
        # 4.6 捕获异常，标记 Trace 失败
        trace.update(
            status="error",
            error={"message": str(e), "type": type(e).__name__}  # 记录异常信息
        )
        raise e  # 抛出异常，不影响原有业务逻辑

# 5. 运行函数，触发 Langfuse 追踪
if __name__ == "__main__":
    result = generate_product_copy(
        product_name="无线蓝牙耳机",
        product_feature="超长续航20小时、降噪效果好"
    )
    print("最终文案：", result)
```

**运行后查看结果（关键步骤）**
1. 执行代码：终端会输出最终文案（比如 `✨ 无线蓝牙耳机来啦！20小时超长续航+超强降噪，通勤听歌超自在～ ✨`）；
2. 打开 Langfuse 仪表盘：访问 `https://cloud.langfuse.com`（或你的本地部署地址）；
3. 查看 Trace 数据：
   - 在「Traces」页面，能看到刚刚运行的 `product-copy-generation` 任务；
   - 点击这个 Trace，会看到三个 Span 按顺序排列：`preprocess-input`→`call-openai`→`format-output`；
   - 点击每个 Span，能看到它的「输入/输出/耗时」（比如 `call-openai` 耗时 1.8s，用了 35 个 Prompt Token）；
   - 如果代码出错（比如 OpenAI 密钥错了），Trace 会显示红色“error”，点击能看到完整的异常信息。


## 三、例子核心逻辑拆解（帮你理解 Langfuse 工作原理）
这个例子虽然简单，但覆盖了 Langfuse 的核心用法，对应实际开发场景：
1. **Trace 是“任务容器”**：一个完整的业务逻辑（比如“生成文案”）对应一个 Trace，所有相关环节都归属于它；
2. **Span 是“环节监控”**：每个细分步骤（预处理、调用 LLM、格式化）对应一个 Span，精准记录每个环节的细节；
3. **自动/手动结合采集数据**：Langfuse 能自动捕获 LLM 调用的 Token 消耗、耗时，也支持手动记录自定义数据（比如用户 ID、产品名称）；
4. **异常自动上报**：只要用 `try-except` 包裹，异常信息会自动关联到 Trace，不用额外写日志代码。


回到之前的 PocketFlow 项目
之前提到的「PocketFlow Tracing」，本质就是把上面的“手动创建 Trace/Span”逻辑，通过「装饰器+框架钩子」自动化了：
- 你不用手动写 `langfuse.trace()` 和 `with trace.span()`；
- 只需要给 PocketFlow 的 Flow 加 `@trace_flow` 装饰器，框架会自动给“整个工作流”创建 Trace，给“每个 Node 的 prep/exec/post 阶段”创建 Span；
- 核心原理和上面的例子完全一致，只是把“手动埋点”变成了“无侵入式自动埋点”。

# pocket-tracing 目录下 setup.py 与 test_tracing.py 详解

这两个文件是项目「**规范化分发**」和「**核心功能验证**」的关键——setup.py 负责让项目能被快速安装复用，test_tracing.py 负责保障追踪功能的正确性，二者分别对应 Python 项目开发的「打包部署」和「测试验证」两大核心环节。

## 一、setup.py：项目打包与分发配置文件
1. 核心定位

setup.py 是 Python 项目的「**安装配置文件**」，通过项目依赖和打包规则，通过`pip install .`本地安装。有了它，用户只需一行命令就能安装并使用 `@trace_flow` 装饰器等核心功能。

2. 核心使用场景
```bash
# 1. 本地安装项目（开发模式，修改代码后立即生效）
pip install -e .  # 安装后可在任何地方导入 from tracing import trace_flow

# 2. 安装开发依赖（用于运行测试、调试）
pip install -e .[dev]

# 3. 打包项目（生成 .whl 或 .tar.gz 文件，可分发他人）
python setup.py bdist_wheel  # 生成 .whl 文件（推荐，安装更快）
python setup.py sdist        # 生成 .tar.gz 文件

# 4. 上传到 PyPI（开源后供他人通过 pip install 下载）
twine upload dist/*  # 需要先安装 twine：pip install twine
```

## 二、test_tracing.py：核心追踪功能的单元测试文件
1. 核心定位

test_tracing.py 是项目的「**核心功能验证文件**」，专门测试 PocketFlow Tracing 的核心能力（如配置加载、Tracer 初始化、装饰器追踪、异常捕获），确保代码修改后不破坏原有功能，是项目可维护性的关键。

测试的核心目标：
- 配置类 `TracingConfig` 能正确读取环境变量、校验参数；
- `LangfuseTracer` 能正确创建 Trace/Span、采集数据、上报异常；
- `@trace_flow` 装饰器能无侵入式给 Flow 加追踪（同步/异步都支持）；
- 节点的 `prep/exec/post` 阶段能被正确追踪，Span 层级正确。

2. 典型内容结构

```python
# 1. 导入依赖（测试框架 + 项目核心模块 + 模拟工具）
import pytest
import asyncio
from unittest.mock import Mock, patch
from pocketflow import Node, Flow, AsyncFlow, AsyncNode
from tracing.config import TracingConfig
from tracing.core.tracer import LangfuseTracer
from tracing.decorators.trace_decorators import trace_flow

# 2. 测试配置类 TracingConfig（对应 config/ 目录）
def test_tracing_config_load_from_env(monkeypatch):
    """测试：TracingConfig 能正确从环境变量加载配置"""
    # 模拟环境变量（用 monkeypatch 临时修改环境变量，不影响系统环境）
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("LANGFUSE_HOST", "https://test-langfuse.com")
    monkeypatch.setenv("POCKETFLOW_TRACING_DEBUG", "true")

    # 初始化配置类
    config = TracingConfig()

    # 断言：配置值与模拟环境变量一致
    assert config.public_key == "test-public-key"
    assert config.secret_key == "test-secret-key"
    assert config.host == "https://test-langfuse.com"
    assert config.debug is True

def test_tracing_config_missing_required_env(monkeypatch):
    """测试：缺少必填环境变量（如 secret_key）时，会抛出异常"""
    # 只设置 public_key，不设置 secret_key（必填项）
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public-key")
    
    # 断言：初始化时会抛出 ValueError
    with pytest.raises(ValueError, match="Langfuse secret key is required"):
        TracingConfig()

# 3. 测试 LangfuseTracer 核心逻辑（对应 core/ 目录）
@patch("tracing.core.tracer.Langfuse")  # Mock Langfuse 客户端，避免真实网络请求
def test_langfuse_tracer_create_trace(mock_langfuse):
    """测试：LangfuseTracer 能正确创建 Trace 和 Span"""
    # 1. 准备配置和 Mock 客户端
    config = TracingConfig(
        public_key="test-pub",
        secret_key="test-sec",
        host="https://test-host.com"
    )
    mock_client = Mock()
    mock_langfuse.return_value = mock_client  # 让 Langfuse() 返回 Mock 客户端

    # 2. 初始化 Tracer
    tracer = LangfuseTracer(config)

    # 3. 创建顶层 Trace
    trace = tracer.start_trace(flow_name="TestFlow", session_id="session-001")

    # 断言：Langfuse 客户端的 trace() 方法被调用，参数正确
    mock_client.trace.assert_called_once_with(
        name="TestFlow",
        session_id="session-001",
        user_id=None  # 默认 None
    )

    # 4. 创建子 Span（模拟 Node 的 prep 阶段）
    with tracer.create_span(trace, name="node-prep", inputs={"data": "test-input"}):
        pass  # 模拟 Span 执行过程

    # 断言：Span 被创建，且更新了输入数据
    mock_trace = mock_client.trace.return_value  # Trace 的 Mock 实例
    mock_trace.span.assert_called_once_with(name="node-prep")
    mock_span = mock_trace.span.return_value
    mock_span.update.assert_any_call(inputs={"data": "test-input"})

# 4. 测试装饰器 @trace_flow（对应 decorators/ 目录）
def test_trace_decorator_sync_flow(monkeypatch, mock_langfuse):
    """测试：同步 Flow 加 @trace_flow 装饰器后，能自动追踪"""
    # 1. Mock 环境变量和 Langfuse 客户端
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-pub")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-sec")
    mock_client = Mock()
    mock_langfuse.return_value = mock_client

    # 2. 定义测试 Node 和 Flow
    class TestNode(Node):
        def prep(self, shared):
            return shared["input"]
        
        def exec(self, data):
            return f"processed: {data}"
        
        def post(self, shared, prep_res, exec_res):
            shared["output"] = exec_res
            return "default"

    # 3. 给 Flow 加装饰器
    @trace_flow(flow_name="TestSyncFlow", session_id="sync-session-001")
    class TestSyncFlow(Flow):
        def __init__(self):
            super().__init__(start=TestNode())

    # 4. 运行 Flow
    flow = TestSyncFlow()
    shared = {"input": "test-data"}
    flow.run(shared)

    # 断言：Trace 被创建，且包含 Node 各阶段的 Span
    mock_client.trace.assert_called_once_with(
        name="TestSyncFlow",
        session_id="sync-session-001",
        user_id=None
    )
    mock_trace = mock_client.trace.return_value
    # 断言：prep/exec/post 三个 Span 都被创建
    assert mock_trace.span.call_count >= 3
    span_names = [call[1]["name"] for call in mock_trace.span.call_args_list]
    assert "node-prep" in span_names
    assert "node-exec" in span_names
    assert "node-post" in span_names

# 5. 测试异步 Flow 追踪（适配 AsyncFlow）
@pytest.mark.asyncio  # 标记为异步测试用例
async def test_trace_decorator_async_flow(monkeypatch, mock_langfuse):
    """测试：异步 AsyncFlow 加装饰器后，能正确追踪"""
    # 1. Mock 环境变量和 Langfuse 客户端
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-pub")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-sec")
    mock_client = Mock()
    mock_langfuse.return_value = mock_client

    # 2. 定义异步 Node 和 AsyncFlow
    class TestAsyncNode(AsyncNode):
        async def prep(self, shared):
            await asyncio.sleep(0.1)  # 模拟异步操作
            return shared["input"]
        
        async def exec(self, data):
            await asyncio.sleep(0.1)
            return f"async-processed: {data}"
        
        async def post(self, shared, prep_res, exec_res):
            shared["output"] = exec_res
            return "default"

    # 3. 给 AsyncFlow 加装饰器
    @trace_flow(flow_name="TestAsyncFlow", session_id="async-session-001")
    class TestAsyncFlow(AsyncFlow):
        def __init__(self):
            super().__init__(start=TestAsyncNode())

    # 4. 运行异步 Flow
    flow = TestAsyncFlow()
    shared = {"input": "async-test-data"}
    await flow.arun(shared)

    # 断言：异步 Flow 的 Trace 和 Span 被正确创建
    mock_client.trace.assert_called_once_with(
        name="TestAsyncFlow",
        session_id="async-session-001",
        user_id=None
    )
    mock_trace = mock_client.trace.return_value
    assert mock_trace.span.call_count >= 3

# 6. 测试异常场景追踪
def test_trace_decorator_error_capture(monkeypatch, mock_langfuse):
    """测试：Node 执行异常时，Trace 能捕获异常信息"""
    # 1. Mock 环境变量和 Langfuse 客户端
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-pub")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-sec")
    mock_client = Mock()
    mock_langfuse.return_value = mock_client

    # 2. 定义会抛出异常的 Node
    class ErrorNode(Node):
        def exec(self, data):
            raise ValueError("模拟执行失败！")  # 故意抛出异常

    # 3. 装饰 Flow 并运行
    @trace_flow(flow_name="TestErrorFlow")
    class TestErrorFlow(Flow):
        def __init__(self):
            super().__init__(start=ErrorNode())

    flow = TestErrorFlow()
    shared = {"input": "error-data"}

    # 4. 运行时断言会抛出异常，且 Trace 记录了错误
    with pytest.raises(ValueError, match="模拟执行失败！"):
        flow.run(shared)

    # 断言：Trace 状态被标记为 error，且包含异常信息
    mock_trace = mock_client.trace.return_value
    mock_trace.update.assert_any_call(
        status="error",
        error={"message": "模拟执行失败！", "type": "ValueError"}
    )
```

**关键代码解析**
（1）Mock 技术的应用
- `@patch("tracing.core.tracer.Langfuse")`：模拟 Langfuse 客户端，避免测试时依赖真实的 Langfuse 服务（无需网络、无需真实凭证），专注测试项目自身的逻辑；
- `monkeypatch`：pytest 的工具，用于临时修改环境变量、函数返回值等，测试不同场景（如“缺少环境变量”“正常加载配置”）。

（2）测试用例设计逻辑
每个测试函数对应一个“核心场景”，遵循「**准备→执行→断言**」三步法：
- 准备：模拟环境（如环境变量、Mock 客户端）、定义测试用的 Node/Flow；
- 执行：初始化配置/Tracer、运行 Flow；
- 断言：验证结果是否符合预期（如配置值正确、Trace/Span 被创建、异常被捕获）。

（3）同步/异步场景全覆盖
- 同步测试：测试普通 Flow 和 Node；
- 异步测试：用 `@pytest.mark.asyncio` 标记，测试 AsyncFlow 和 AsyncNode，确保项目支持异步工作流（之前提到的核心特性）。

（4）异常场景测试
专门设计 `test_trace_decorator_error_capture`，验证“Node 执行失败时，Langfuse 能正确记录异常信息”——这是可观测性工具的核心价值（排障），必须覆盖。

**核心使用场景**（怎么运行测试？）
```bash
# 1. 确保安装了开发依赖（含 pytest）
pip install -e .[dev]

# 2. 运行所有测试用例
pytest test_tracing.py -v  # -v 显示详细日志

# 3. 运行指定测试用例（如只测试装饰器功能）
pytest test_tracing.py::test_trace_decorator_sync_flow -v

# 4. 运行异步测试用例（自动识别 @pytest.mark.asyncio）
pytest test_tracing.py::test_trace_decorator_async_flow -v
```


## 三、两个文件的关联与项目角色总结
| 文件名称       | 核心角色                | 依赖关系                                  | 学习重点                                  |
|----------------|-------------------------|-------------------------------------------|-------------------------------------------|
| setup.py       | 项目打包与分发          | 依赖 requirements.txt，定义项目依赖和打包规则 | Python 项目规范化打包、依赖分层、兼容性控制 |
| test_tracing.py| 核心功能单元测试        | 依赖 pytest、mock，测试 config/core/decorators 目录 | 单元测试设计、Mock 技术、同步/异步测试    |

**关联逻辑：**
1. setup.py 中定义的 `extras_require["dev"]` 包含 `pytest` 和 `mock`，是运行 test_tracing.py 的前提；
2. test_tracing.py 测试的核心功能（Tracer、装饰器），是 setup.py 打包后用户使用的核心能力——确保打包的项目“能用且好用”。


# tracing 目录深度解析
**tracing 目录是整个追踪功能的“核心代码仓库”**，所有与 Langfuse 整合、PocketFlow 生命周期对接、追踪逻辑封装的核心代码都集中于此。它是项目的“业务逻辑层”，区别于示例（examples）、测试（tests）、打包配置（setup.py）等辅助目录，直接决定了“如何实现无侵入式全链路追踪”。

tracing 目录典型结构：
```
tracing/
├─ __init__.py          # 包初始化文件，暴露核心 API 供外部调用
├─ config.py            # 配置管理模块（定义 TracingConfig 类）
├─ tracer.py            # 核心追踪模块（定义 LangfuseTracer 类）
├─ decorators.py        # 装饰器模块（定义 @trace_flow 装饰器）
├─ hooks.py             # 框架钩子模块（对接 PocketFlow 节点生命周期）
└─ utils/               # 工具子目录（序列化、日志、异常处理等辅助工具）
   ├─ __init__.py
   ├─ serialization.py  # 数据序列化工具
   ├─ logging.py        # 调试日志工具
   └─ error_handling.py # 异常格式化工具
```

## 核心文件/子目录逐一详解
### 1. `tracer.py`：核心追踪模块
**核心定位**：封装 Langfuse 的 Trace/Span 操作，**所有追踪数据的采集、组织、上报都由它完成**。

```python
from langfuse import Langfuse
from langfuse.model import Trace, Span
from typing import Dict, Any, Optional
from .config import TracingConfig
from .utils.serialization import serialize_data
from .utils.error_handling import format_exception

class LangfuseTracer:
    """Langfuse 追踪器，负责 Trace/Span 的创建、管理和数据上报"""
    def __init__(self, config: TracingConfig):
        self.config = config
        # 初始化 Langfuse 客户端
        self.langfuse_client = Langfuse(
            public_key=config.public_key,
            secret_key=config.secret_key,
            host=config.host,
            debug=config.debug
        )
        self.current_trace: Optional[Trace] = None  # 存储当前顶层 Trace

    def start_trace(
        self,
        flow_name: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Trace:
        """创建顶层 Trace（对应一个完整的 PocketFlow 工作流）"""
        self.current_trace = self.langfuse_client.trace(
            name=flow_name,
            session_id=session_id or self.config.session_id,
            user_id=user_id or self.config.user_id
        )
        return self.current_trace

    def end_trace(self, status: str = "success", outputs: Optional[Dict[str, Any]] = None):
        """结束顶层 Trace，上报最终状态和输出"""
        if not self.current_trace:
            return
        # 序列化输出数据（处理复杂对象）
        serialized_outputs = serialize_data(outputs) if self.config.trace_outputs else None
        self.current_trace.update(
            status=status,
            outputs=serialized_outputs
        )
        # 手动刷新数据（确保上报到 Langfuse 服务）
        self.langfuse_client.flush()

    def start_span(
        self,
        trace: Trace,
        span_name: str,
        inputs: Optional[Dict[str, Any]] = None
    ) -> Span:
        """创建子 Span（对应 Node 的某个阶段：prep/exec/post）"""
        # 序列化输入数据
        serialized_inputs = serialize_data(inputs) if self.config.trace_inputs else None
        span = trace.span(
            name=span_name,
            inputs=serialized_inputs
        )
        return span

    def end_span(
        self,
        span: Span,
        outputs: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ):
        """结束 Span，上报输出或异常"""
        # 处理输出
        if outputs and self.config.trace_outputs:
            serialized_outputs = serialize_data(outputs)
            span.update(outputs=serialized_outputs)
        # 处理异常
        if exception and self.config.trace_errors:
            error_info = format_exception(exception)
            span.update(
                status="error",
                error=error_info
            )
        span.end()  # 标记 Span 结束，记录耗时

    def capture_exception(self, trace: Trace, exception: Exception):
        """给顶层 Trace 上报全局异常"""
        if not trace or not self.config.trace_errors:
            return
        error_info = format_exception(exception)
        trace.update(
            status="error",
            error=error_info
        )
```

**关键知识点**：
- **Trace/Span 层级关联**：顶层 Trace 对应 PocketFlow 工作流，子 Span 对应节点的每个阶段，形成“Trace→Node Span→阶段子Span”的链路；
- **数据序列化**：通过 `utils.serialization` 处理复杂对象（如 PocketFlow 的 `shared` 状态），确保能上报到 Langfuse；
- **异常格式化**：通过 `utils.error_handling` 统一异常信息格式，让 Langfuse 展示的错误更易读。

### 2. `decorators.py`：装饰器模块
**核心定位**：提供**无侵入式接入的入口**，通过 `@trace_flow` 装饰器，让用户无需修改 PocketFlow 工作流的核心逻辑，即可开启追踪。

```python
import asyncio
from functools import wraps
from typing import Type, Callable
from pocketflow import Flow, AsyncFlow
from .config import TracingConfig
from .tracer import LangfuseTracer
from .hooks import register_node_hooks

def trace_flow(
    config: Optional[TracingConfig] = None,
    flow_name: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
):
    """装饰器：为 PocketFlow Flow/AsyncFlow 添加 Langfuse 追踪"""
    # 优先从环境变量加载配置
    config = config or TracingConfig.from_env()
    tracer = LangfuseTracer(config)

    def decorator(flow_cls: Type[Flow]):
        # 确定 Flow 名称（装饰器参数 > 类名）
        actual_flow_name = flow_name or flow_cls.__name__
        # 注册节点生命周期钩子（实现阶段级 Span 采集）
        register_node_hooks(tracer, actual_flow_name)

        # 处理同步 Flow 的 run 方法
        if issubclass(flow_cls, Flow):
            original_run = flow_cls.run
            @wraps(original_run)
            def wrapped_run(self, shared: dict):
                try:
                    # 启动顶层 Trace
                    trace = tracer.start_trace(actual_flow_name, session_id, user_id)
                    # 执行原 run 方法
                    result = original_run(self, shared)
                    # 结束 Trace，标记成功
                    tracer.end_trace(status="success", outputs=shared)
                    return result
                except Exception as e:
                    # 捕获全局异常，上报到 Trace
                    tracer.capture_exception(trace, e)
                    tracer.end_trace(status="error")
                    raise e
            # 替换原 run 方法
            flow_cls.run = wrapped_run

        # 处理异步 AsyncFlow 的 arun 方法
        elif issubclass(flow_cls, AsyncFlow):
            original_arun = flow_cls.arun
            @wraps(original_arun)
            async def wrapped_arun(self, shared: dict):
                try:
                    trace = tracer.start_trace(actual_flow_name, session_id, user_id)
                    result = await original_arun(self, shared)
                    tracer.end_trace(status="success", outputs=shared)
                    return result
                except Exception as e:
                    tracer.capture_exception(trace, e)
                    tracer.end_trace(status="error")
                    raise e
            flow_cls.arun = wrapped_arun

        return flow_cls
    return decorator
```

**关键知识点**：
- **装饰器+函数重写**：通过 `wraps` 保留原函数的元信息，重写 `run`/`arun` 方法，注入 Trace 启动/结束逻辑；
- **同步/异步兼容**：区分 `Flow` 和 `AsyncFlow`，分别处理同步 `run` 和异步 `arun` 方法；
- **钩子关联**：调用 `register_node_hooks` 对接 PocketFlow 节点生命周期，实现阶段级 Span 采集。


### 3. `hooks.py`：框架钩子模块
**核心定位**：对接 PocketFlow 的**节点生命周期**，在 `prep/exec/post` 阶段自动创建 Span，是实现“阶段级追踪”的关键。

```python
from typing import Any, Dict
from pocketflow import Node
from .tracer import LangfuseTracer
from .utils.logging import debug_log

def register_node_hooks(tracer: LangfuseTracer, flow_name: str):
    """注册 PocketFlow 节点的生命周期钩子，实现阶段级 Span 采集"""
    # 钩子函数：阶段开始前创建 Span
    def before_phase(node: Node, phase: str, **kwargs):
        if not tracer.current_trace:
            return
        # 构造 Span 名称（格式：flow名-节点名-阶段）
        span_name = f"{flow_name}-{node.__class__.__name__}-{phase}"
        # 提取阶段输入数据
        inputs = kwargs.get("shared") if phase == "prep" else kwargs.get("data")
        # 启动 Span 并存储在节点的临时属性中
        span = tracer.start_span(tracer.current_trace, span_name, inputs=inputs)
        setattr(node, f"_current_{phase}_span", span)
        debug_log(f"Started span for {span_name}")

    # 钩子函数：阶段结束后上报数据/异常
    def after_phase(node: Node, phase: str, result: Any, exception: Exception = None):
        span = getattr(node, f"_current_{phase}_span", None)
        if not span:
            return
        # 结束 Span，上报输出或异常
        tracer.end_span(span, outputs={"result": result}, exception=exception)
        debug_log(f"Ended span for {span.name}")
        # 清理节点临时属性
        delattr(node, f"_current_{phase}_span")

    # 注册 PocketFlow 节点的生命周期钩子
    Node.register_hook("before_prep", lambda node, shared: before_phase(node, "prep", shared=shared))
    Node.register_hook("after_prep", lambda node, result, exc: after_phase(node, "prep", result, exc))
    Node.register_hook("before_exec", lambda node, data: before_phase(node, "exec", data=data))
    Node.register_hook("after_exec", lambda node, result, exc: after_phase(node, "exec", result, exc))
    Node.register_hook("before_post", lambda node, shared, prep, exec: before_phase(node, "post", shared=shared, prep=prep, exec=exec))
    Node.register_hook("after_post", lambda node, result, exc: after_phase(node, "post", result, exc))
```

**关键知识点**：
- **框架钩子机制**：利用 PocketFlow 提供的 `register_hook` 接口，在节点的每个阶段前后插入自定义逻辑；
- **临时属性存储**：将 Span 存储在节点的临时属性中（如 `_current_prep_span`），实现阶段前后的 Span 关联；
- **钩子参数适配**：不同阶段（prep/exec/post）的输入参数不同，通过 lambda 函数统一适配到 `before_phase`。

### 4. `utils/`：工具子目录
**核心定位**：封装通用辅助工具，为核心模块提供支撑，**让核心代码更简洁、更健壮**，避免重复逻辑。

| 工具文件                | 核心功能                                                                 |
|-------------------------|--------------------------------------------------------------------------|
| `serialization.py`      | 处理复杂数据序列化（如 datetime、自定义对象），转化为 Langfuse 可存储的 JSON 格式 |
| `logging.py`            | 封装调试日志（仅开启 `debug` 时输出），区分开发/生产环境日志级别             |
| `error_handling.py`     | 统一异常格式化（提取异常类型、栈轨迹、错误信息），确保上报数据规范           |

## 三、tracing 目录的整体逻辑串联
tracing 目录的代码遵循“**分层设计+职责单一**”原则，各模块协同工作的流程如下：
1. **配置加载**：用户通过 `TracingConfig` 加载 Langfuse 凭证和追踪开关，传入 `LangfuseTracer`；
2. **装饰器注入**：`@trace_flow` 装饰器重写 Flow 的 `run`/`arun` 方法，启动顶层 Trace；
3. **钩子采集**：`register_node_hooks` 注册的生命周期钩子，在节点每个阶段前后创建/结束 Span；
4. **数据上报**：`LangfuseTracer` 负责将 Trace/Span 数据（输入/输出/耗时/异常）上报到 Langfuse；
5. **工具支撑**：`utils` 目录的工具函数处理序列化、日志、异常等通用逻辑，保障核心流程稳定。

# examples 目录详解（实战落地参考库）
在 PocketFlow Tracing 项目中，**examples 目录是“核心功能的实战演示集合”**，目标是让用户通过“复制→运行→修改”快速上手追踪功能，覆盖从“基础使用”到“复杂场景”的全流程。

**examples 目录典型结构**（按学习优先级排序）
```
examples/
├─ 01_basic_sync_flow.py        # 基础同步流：单个节点+默认配置（入门首选）
├─ 02_basic_async_flow.py       # 基础异步流：异步节点+AsyncFlow（适配异步场景）
├─ 03_custom_config.py          # 自定义配置：手动指定Langfuse参数、关闭输入追踪等
├─ 04_error_tracing.py          # 异常场景：节点执行失败时的追踪效果
├─ 05_complex_flow.py           # 复杂工作流：多节点+嵌套流+用户/Session关联
└─ 06_tool_call_tracing.py      # 工具调用场景：AI调用数据库/PDF解析的追踪（实战高频）
```

## 三、逐示例详细解析（代码+场景+学习重点）
所有示例均遵循「**环境准备→核心代码→运行步骤→结果验证**」四步结构，确保可复现。

### 1. 01_basic_sync_flow.py：基础同步流（入门首选）
核心场景：最简化的同步工作流（1个Node+1个Flow），演示「默认配置下的无侵入式追踪」——仅需添加 `@trace_flow` 装饰器，无需额外配置。

```python
# 导入核心依赖（PocketFlow基础+追踪装饰器）
from pocketflow import Node, Flow
from tracing import trace_flow  # 从tracing目录导入核心装饰器

# 1. 定义普通同步Node（PocketFlow标准节点）
class SimpleNode(Node):
    def prep(self, shared):
        """预处理：提取shared中的输入数据"""
        return shared["user_input"]  # 输入：用户提问
    
    def exec(self, data):
        """核心执行：模拟AI生成回答"""
        return f"AI回复：你问的是「{data}」，这是默认回答～"  # 简单模拟LLM调用
    
    def post(self, shared, prep_res, exec_res):
        """后处理：更新shared状态"""
        shared["ai_output"] = exec_res
        return "default"  # PocketFlow要求post返回动作标识

# 2. 给Flow添加追踪装饰器（核心：仅需这一行）
@trace_flow(flow_name="BasicSyncFlow")  # 指定Flow名称（方便Langfuse中识别）
class MySyncFlow(Flow):
    def __init__(self):
        super().__init__(start=SimpleNode())  # 定义工作流：仅一个节点

# 3. 运行工作流
if __name__ == "__main__":
    # 初始化共享状态（工作流输入）
    shared = {"user_input": "如何使用PocketFlow Tracing？"}
    # 执行Flow
    flow = MySyncFlow()
    flow.run(shared)
    # 打印结果（验证工作流正常执行）
    print("工作流输出：", shared["ai_output"])
```

执行命令：`python examples/01_basic_sync_flow.py`；
查看结果：
   - 终端输出工作流结果；
   - 打开 Langfuse 仪表盘，在「Traces」中找到 `BasicSyncFlow`，可看到：
     - 顶层 Trace（工作流）；
     - 3个 Span（对应 Node 的 `prep`/`exec`/`post` 阶段）；
     - 每个 Span 的输入/输出/耗时。


### 2. 02_basic_async_flow.py：基础异步流
核心场景：演示异步工作流（`AsyncNode`+`AsyncFlow`）的追踪适配——PocketFlow 支持异步节点，examples 展示追踪功能如何兼容异步场景（无需修改装饰器，自动适配）。

```python
from pocketflow import AsyncNode, AsyncFlow  # 导入异步组件
from tracing import trace_flow
import asyncio

# 1. 定义异步Node（区别于同步Node：方法前加async）
class SimpleAsyncNode(AsyncNode):
    async def prep(self, shared):
        await asyncio.sleep(0.1)  # 模拟异步操作（如网络请求）
        return shared["user_input"]
    
    async def exec(self, data):
        await asyncio.sleep(0.2)  # 模拟异步LLM调用
        return f"异步AI回复：你问的是「{data}」～"
    
    async def post(self, shared, prep_res, exec_res):
        shared["ai_output"] = exec_res
        return "default"

# 2. 装饰AsyncFlow（与同步流完全相同，无需改装饰器）
@trace_flow(flow_name="BasicAsyncFlow")
class MyAsyncFlow(AsyncFlow):
    def __init__(self):
        super().__init__(start=SimpleAsyncNode())

# 3. 异步运行工作流
if __name__ == "__main__":
    async def main():
        shared = {"user_input": "异步工作流如何追踪？"}
        flow = MyAsyncFlow()
        await flow.arun(shared)  # 异步执行：arun()
        print("异步工作流输出：", shared["ai_output"])
    
    asyncio.run(main())
```

- 核心：`@trace_flow` 装饰器自动兼容同步/异步 Flow（底层通过重写 `run`/`arun` 方法实现）；
- 关联：tracing 目录 `decorators.py` 中对 `AsyncFlow` 的处理逻辑；
- 目标：掌握异步工作流的追踪使用，覆盖实际开发中“异步优先”的场景。

### 3. 03_custom_config.py：自定义配置
**核心场景**：演示「不依赖 `.env` 文件，通过 `TracingConfig` 手动配置」——适用于需要动态调整追踪参数的场景（如关闭输入追踪、指定 User ID、切换 Langfuse 环境）。

```python
from pocketflow import Node, Flow
from tracing import trace_flow, TracingConfig  # 导入配置类

# 1. 定义自定义配置（替代.env文件）
custom_config = TracingConfig(
    langfuse_public_key="你的Langfuse公钥",  # 手动指定凭证
    langfuse_secret_key="你的Langfuse私钥",
    langfuse_host="https://cloud.langfuse.com",
    debug=True,  # 开启调试日志
    trace_inputs=False,  # 关闭输入数据追踪（隐私敏感场景可用）
    trace_outputs=True,  # 保留输出数据追踪
    user_id="user-123",  # 关联具体用户（方便Langfuse统计用户行为）
    session_id="session-456"  # 关联会话（方便追溯同一用户的连续请求）
)

# 2. 装饰器传入自定义配置（核心）
@trace_flow(
    config=custom_config,
    flow_name="CustomConfigFlow"
)
class MyConfigFlow(Flow):
    def __init__(self):
        super().__init__(start=SimpleNode())  # 复用之前的SimpleNode

# 运行逻辑同基础示例...
```

### 4. 04_error_tracing.py：异常场景追踪
**核心场景**：演示「节点执行失败时，Langfuse 如何捕获异常信息」——这是可观测性工具的核心价值（排障），示例中故意让 Node 抛出异常，验证追踪效果。

```python
from pocketflow import Node, Flow
from tracing import trace_flow

# 1. 定义会抛出异常的Node
class ErrorNode(Node):
    def prep(self, shared):
        return shared["user_input"]
    
    def exec(self, data):
        # 故意抛出异常（模拟LLM调用失败、参数错误等场景）
        raise ValueError(f"执行失败：输入数据「{data}」不符合要求！")
    
    def post(self, shared, prep_res, exec_res):
        shared["ai_output"] = exec_res
        return "default"

# 2. 装饰Flow（正常使用装饰器）
@trace_flow(flow_name="ErrorTracingFlow")
class MyErrorFlow(Flow):
    def __init__(self):
        super().__init__(start=ErrorNode())

# 3. 运行并捕获异常
if __name__ == "__main__":
    shared = {"user_input": "错误的输入"}
    flow = MyErrorFlow()
    try:
        flow.run(shared)
    except Exception as e:
        print("捕获异常：", str(e))  # 终端打印异常
```

**结果验证**
- 终端会打印异常信息；
- Langfuse 仪表盘上，`ErrorTracingFlow` 的 Trace 状态为「error」，点击可查看：
  - 异常类型（`ValueError`）；
  - 异常信息（`执行失败：输入数据「错误的输入」不符合要求！`）；
  - 异常发生的 Span（`exec` 阶段）；
  - 完整的栈轨迹（方便定位代码问题）。

### 5. 05_complex_flow.py：复杂工作流
**核心场景**：演示「多节点+嵌套流+用户/Session 关联」——模拟真实 AI 工作流（如“需求分析→内容生成→结果校验”），展示追踪功能在复杂拓扑中的可用性。

```python
from pocketflow import Node, Flow
from tracing import trace_flow

# 1. 定义子节点（多节点协作）
class AnalyzeNode(Node):
    """节点1：需求分析"""
    def exec(self, data):
        return f"分析结果：用户需要关于「{data}」的详细说明"

class GenerateNode(Node):
    """节点2：内容生成"""
    def exec(self, data):
        return f"生成内容：{data}...（详细说明文本）"

class ValidateNode(Node):
    """节点3：结果校验"""
    def exec(self, data):
        return f"校验结果：内容合规，长度达标（{len(data)}字）"

# 2. 定义嵌套流（子Flow作为父Flow的节点）
@trace_flow(flow_name="SubGenerateFlow")  # 子流也可单独追踪
class SubGenerateFlow(Flow):
    def __init__(self):
        super().__init__(start=GenerateNode())

# 3. 定义父Flow（多节点+嵌套流）
@trace_flow(
    flow_name="ComplexWorkflow",
    user_id="user-789",  # 关联用户
    session_id="session-007"  # 关联会话
)
class ParentFlow(Flow):
    def __init__(self):
        # 定义执行顺序：AnalyzeNode → 子流 SubGenerateFlow → ValidateNode
        super().__init__(
            start=AnalyzeNode(),
            nodes=[
                (AnalyzeNode, SubGenerateFlow),  # 分析后执行子流
                (SubGenerateFlow, ValidateNode)  # 子流执行后校验
            ]
        )

# 运行逻辑：父流运行时，子流的追踪自动关联到父Trace
```

Langfuse 仪表盘上会显示「层级化 Trace 链路」：
- 顶层 Trace：`ComplexWorkflow`（父流）；
- 子 Trace：`SubGenerateFlow`（子流，作为父 Trace 的子 Span）；
- 每个节点的 `prep`/`exec`/`post` 阶段 Span 按执行顺序排列，清晰展示数据流转。

### 6. 06_tool_call_tracing.py：工具调用场景
**核心场景**：演示「AI 工作流调用外部工具（如数据库查询、PDF 解析）」的追踪——这是 AI Agent 开发的高频场景，需追踪“工具调用的输入/输出/耗时/异常”。

```python
from pocketflow import Node, Flow
from tracing import trace_flow
import sqlite3  # 模拟数据库工具

# 1. 定义工具调用Node（数据库查询）
class DBQueryNode(Node):
    def prep(self, shared):
        """预处理：提取查询条件"""
        return shared["query_sql"]
    
    def exec(self, sql):
        """核心执行：调用数据库工具"""
        # 模拟数据库查询
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE users (id INT, name TEXT)")
        cursor.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')")
        conn.commit()
        
        cursor.execute(sql)
        result = cursor.fetchall()  # 工具返回结果
        conn.close()
        return result
    
    def post(self, shared, prep_res, exec_res):
        shared["db_result"] = exec_res
        return "default"

# 2. 装饰Flow，追踪工具调用
@trace_flow(flow_name="ToolCallFlow")
class ToolFlow(Flow):
    def __init__(self):
        super().__init__(start=DBQueryNode())

# 运行：查询数据库
if __name__ == "__main__":
    shared = {"query_sql": "SELECT * FROM users WHERE id = 1"}
    flow = ToolFlow()
    flow.run(shared)
    print("数据库查询结果：", shared["db_result"])  # 输出：[(1, 'Alice')]
```

Langfuse 仪表盘上会捕获：
- 工具调用的输入（`query_sql: SELECT * FROM users WHERE id = 1`）；
- 工具调用的输出（`db_result: [(1, 'Alice')]`）；
- 工具调用的耗时（如 0.05s）；
- 若数据库查询失败（如 SQL 语法错误），会自动捕获异常信息。
