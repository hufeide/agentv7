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
