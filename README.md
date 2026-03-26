%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#e1f5fe', 'primaryTextColor': '#01579b', 'primaryBorderColor': '#0288d1', 'lineColor': '#0288d1', 'secondaryColor': '#fff3e0', 'tertiaryColor': '#e8f5e9'}}}%%

flowchart TB
    subgraph "入口层 Entry Layer"
        Main["main()"]
        ProductionAgentOS["ProductionAgentOS<br/>统一入口/生命周期管理"]
    end

    subgraph "配置与基础设施 Infrastructure"
        ConfigManager["ConfigManager<br/>配置管理（单例）"]
        EventBus["EventBus<br/>事件总线（伪异步）"]
        EventSubscriptionManager["EventSubscriptionManager<br/>订阅管理"]
        ToolExecutorPool["ToolExecutorPool<br/>进程池（未使用）"]
    end

    subgraph "Context 管理系统 Context System"
        ContextManager["ContextManager<br/>Context 生命周期管理"]
        Context["Context<br/>共享可变对象 ⚠️"]
        ContextFormatter["ContextFormatter<br/>Prompt 格式化"]
        ContextCompressor["ContextCompressor<br/>三层压缩"]
        GlobalContext["GlobalContext"]
        TaskContext["TaskContext"]
    end

    subgraph "智能层 Intelligence Layer - God Object ⚠️"
        LLMRuntime["LLMRuntime<br/>万能对象（问题核心）"]
        
        subgraph "LLMRuntime 内部职责过载"
            Reasoning["reason() - 纯推理"]
            ToolCall["tool_call() - ReAct循环"]
            ToolDispatch["_execute_tool() - 工具调度"]
            Deduplication["去重逻辑"]
            ContextWrite["直接写 Context ⚠️"]
            Degradation["降级策略"]
            HistoryCompression["历史压缩"]
        end
    end

    subgraph "规划层 Planning"
        Planner["Planner<br/>计划生成器"]
        Replanner["Replanner<br/>动态重规划"]
        Critic["Critic<br/>执行评估器"]
        StepPlan["StepPlan"]
        Plan["Plan"]
        DynamicPlan["DynamicPlan<br/>执行时计划"]
    end

    subgraph "执行层 Execution"
        ExecutionEngine["ExecutionEngine<br/>执行引擎"]
        Worker["Worker<br/>结构化执行器"]
        ErrorRecovery["ErrorRecovery<br/>错误恢复"]
    end

    subgraph "能力层 Capability Layer - 双注册中心 ⚠️"
        subgraph "注册中心 1"
            CapabilityRegistry["CapabilityRegistry<br/>执行入口"]
            ExecutableCapability["ExecutableCapability<br/>工具"]
            InstructableCapability["InstructableCapability<br/>技能"]
            AgenticCapability["AgenticCapability"]
        end
        
        subgraph "注册中心 2（冗余）"
            ToolRegistry["ToolRegistry<br/>实例+Schema"]
        end
        
        ToolCapability["ToolCapability"]
        SkillCapability["SkillCapability<br/>渐进披露三层"]
        SkillPolicy["SkillPolicy<br/>偷偷做 Agent ⚠️"]
    end

    subgraph "数据层 Data"
        State["State<br/>认知状态（artifacts+memory+trace）"]
        Artifact["Artifact"]
        StepTrace["StepTrace"]
        Step["Step"]
        StepState["StepState"]
    end

    subgraph "事件类型 Event Types"
        EventType["EventType<br/>枚举"]
        Event["Event"]
        STEP_READY["STEP_READY"]
        STEP_COMPLETED["STEP_COMPLETED"]
        STEP_FAILED["STEP_FAILED"]
        TASK_COMPLETED["TASK_COMPLETED"]
    end

    %% 入口调用关系
    Main --> ProductionAgentOS
    ProductionAgentOS --> initialize
    ProductionAgentOS --> run

    %% 初始化流程
    initialize --> LLMRuntime
    initialize --> ContextManager
    initialize --> Planner
    initialize --> Critic
    initialize --> Replanner
    initialize --> ErrorRecovery
    initialize --> ExecutionEngine
    initialize --> EventBus
    
    %% 核心问题：LLMRuntime 是 God Object
    LLMRuntime -.->|包含| Reasoning
    LLMRuntime -.->|包含| ToolCall
    LLMRuntime -.->|包含| ToolDispatch
    LLMRuntime -.->|包含| Deduplication
    LLMRuntime -.->|直接修改| ContextWrite
    LLMRuntime -.->|包含| Degradation
    LLMRuntime -.->|包含| HistoryCompression
    
    %% Context 系统关系（问题：共享可变）
    ContextManager -->|创建/管理| Context
    Context -->|被修改| LLMRuntime
    Context -->|被修改| SkillPolicy
    Context -->|被读取| ContextFormatter
    Context -->|被压缩| ContextCompressor
    ContextManager -->|包含| GlobalContext
    ContextManager -->|包含| TaskContext
    
    %% 规划流程
    ProductionAgentOS -->|调用| Planner
    Planner -->|生成| Plan
    Plan -->|包含| StepPlan
    StepPlan -->|转换为| Step
    Step -->|放入| DynamicPlan
    ExecutionEngine -->|管理| DynamicPlan
    
    %% 执行流程（问题：缺少 Scheduler）
    ExecutionEngine -->|发布| STEP_READY
    EventBus -->|分发| Worker
    Worker -->|消费| STEP_READY
    Worker -->|执行| SkillPolicy
    Worker -->|或执行| ToolCapability
    
    %% SkillPolicy 问题：偷偷做 Agent
    SkillPolicy -->|调用| LLMRuntime
    SkillPolicy -->|构建工具| ToolCapability
    SkillPolicy -->|直接修改| Context
    
    %% 双注册中心问题
    ToolRegistry -->|加载| ToolCapability
    CapabilityRegistry -->|注册| ToolCapability
    CapabilityRegistry -->|注册| SkillCapability
    SkillCapability -->|被使用| SkillPolicy
    
    %% 工具执行（问题：ToolExecutorPool 未使用）
    ToolCapability -.->|应该使用| ToolExecutorPool
    ToolCapability -.->|实际直接执行| State
    
    %% EventBus 问题：伪异步
    EventBus -->|串行调用| EventSubscriptionManager
    EventSubscriptionManager -->|管理订阅| Worker
    EventSubscriptionManager -->|管理订阅| ExecutionEngine
    
    %% 状态管理
    ExecutionEngine -->|更新| State
    State -->|包含| Artifact
    State -->|包含| StepTrace
    Worker -->|创建| Artifact
    
    %% 错误恢复
    Worker -->|失败触发| ErrorRecovery
    ErrorRecovery -->|决策| Replanner
    Replanner -->|生成新| Step
    ErrorRecovery -->|或重试| STEP_READY
    
    %% 评估流程
    Worker -->|可选| Critic
    Critic -->|评估| Artifact
    Critic -->|调用| LLMRuntime
    
    %% 完成流程
    Worker -->|发布| STEP_COMPLETED
    STEP_COMPLETED -->|触发| ExecutionEngine
    ExecutionEngine -->|检查完成| TASK_COMPLETED

    %% 样式标记问题区域
    style LLMRuntime fill:#ffebee,stroke:#c62828,stroke-width:3px
    style Context fill:#ffebee,stroke:#c62828,stroke-width:2px
    style EventBus fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style SkillPolicy fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style ToolExecutorPool fill:#eceff1,stroke:#546e7a,stroke-dasharray: 5 5
    style ToolRegistry fill:#eceff1,stroke:#546e7a,stroke-dasharray: 5 5
    
    %% 标记 God Object 内部
    style Reasoning fill:#ffebee,stroke:#c62828,stroke-dasharray: 3 3
    style ToolCall fill:#ffebee,stroke:#c62828,stroke-dasharray: 3 3
    style Deduplication fill:#ffebee,stroke:#c62828,stroke-dasharray: 3 3
    style ContextWrite fill:#ffebee,stroke:#c62828,stroke-dasharray: 3 3


    <img width="18778" height="6072" alt="deepseek_mermaid_20260325_006bd9" src="https://github.com/user-attachments/assets/78dce30c-40b2-42a1-80d1-a855f7938521" />





sequenceDiagram
    participant User
    participant AgentOS as ProductionAgentOS
    participant Planner
    participant Engine as ExecutionEngine
    participant Worker
    participant LLM as LLMRuntime
    participant State as State/Context

    User->>AgentOS: run(task="分析市场")
    
    Note over AgentOS: 【阶段1: 初始化】
    AgentOS->>AgentOS: initialize()
    AgentOS->>LLM: 创建 AsyncOpenAI 客户端
    AgentOS->>AgentOS: 创建 ContextManager(长生命周期)
    AgentOS->>AgentOS: 加载 Tools + Skills 到 CapabilityRegistry
    
    Note over AgentOS,Planner: 【阶段2: 规划】
    AgentOS->>Planner: plan(task, tools, skills)
    Planner->>LLM: reason(prompt="分解任务...")
    LLM-->>Planner: 返回 JSON 计划
    Planner->>Planner: 解析为 Plan(steps, dag)
    Planner-->>AgentOS: Plan(plan_id, 3 steps)
    
    Note over AgentOS,Engine: 【阶段3: 执行准备】
    AgentOS->>Engine: set_plan(plan)
    AgentOS->>Engine: 创建 DynamicPlan(Step 对象)
    AgentOS->>Engine: start()
    Engine->>Engine: _publish_ready_steps()
    Engine->>Engine: 检查 DAG 依赖 → step_1 就绪
    Engine->>Worker: 发布 STEP_READY(step_1)
    
    Note over Worker,LLM: 【阶段4: 步骤执行循环】
    loop 每个步骤执行
        Worker->>Worker: claim_step(step_1)
        Worker->>ContextMgr: get_or_create(step_1)
        Worker->>LLM: tool_call(system_prompt, user_prompt, tools)
        
        loop ReAct 迭代 (max 10 次)
            LLM->>LLM: 调用 vLLM API
            alt 需要工具调用
                LLM->>CapabilityReg: execute(tool_name, args)
                CapabilityReg-->>LLM: 返回工具结果
                LLM->>LLM: 记录 tool_trace + history
            else 直接返回答案
                LLM-->>Worker: (success=True, output)
            end
        end
        
        Worker->>State: update_artifact(step_1, output)
        Worker->>Engine: 发布 STEP_COMPLETED(step_1)
    end
    
    Note over Engine: 【阶段5: 依赖推进】
    Engine->>Engine: process_completed(event)
    Engine->>Engine: _publish_ready_steps()
    Engine->>Engine: 检查 step_2 依赖 step_1 ✓ → 发布 STEP_READY
    Engine->>Worker: 发布 STEP_READY(step_2)
    
    Note over AgentOS: 【阶段6: 任务完成】
    Engine->>Engine: _check_and_publish_completion()
    Engine->>AgentOS: 发布 TASK_COMPLETED
    AgentOS->>AgentOS: 收集 artifacts → 构建结果
    AgentOS-->>User: 返回 JSON 结果





<img width="6705" height="8523" alt="mermaid-1774501451203" src="https://github.com/user-attachments/assets/b5c56826-e169-4a1d-b219-bf762678c053" />

    
