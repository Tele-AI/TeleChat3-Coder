### 针对 VLLM 框架的 TeleCoder3 模型补丁

#### 目前支持以下功能
- 工具调用解析
- 支持 Interleaved Thinking 模式：chat 接口的 message 支持 reasoning_content 字段，并拼接到 模型上下文中

#### 使用教程
1. 安装插件（只需要执行一次）
```shell
pip install -e .
```
2. 如果需要工具解析，启动vllm服务的时候加上参数`--enable-auto-tool-choice --tool-call-parser telechat3`
