@docs/task_specs/html_rewrite_cot/image_to_html_cot_pipeline_design.md 先不写代码，先理解和分析这个需求设计文档里面的方案，理解目的，看看有什么问题，然后给出一个整体设计方案 @html_rewrite_cot 。参考数据在 /home/yanghaitao/Projects/Data/FineWebEdu/rewrite_samples

这是我对你的问题的回答：
问题1：分开输出
问题2：用选项B
问题3：按 sample 粒度做 resume
问题4：配置中加 max_input_tokens 限制，超出则跳过并记录 warning ；但这个max_input_tokens可以设得很大 （比如64K），我们HTML是改写得到的，本身长度就不长。不是互联网脏HTML。
问题5：我不懂，但确保尽可能把真实渲染的信息给获取到，如果过长超时应该也要记录log下来并进行统计，方便我们后续分析结果
问题6：分两个阶段
问题7：用base64
问题8：user 侧 text prompt不用改
帮我实现完整代码，并给出详细的设计报告和使用说明文档


stop用 "<html" 是不是很危险。比如后面如果要造慢思考数据，要对html代码进行分析，是不是就会被中途截断？
另外，看看API调用部分的代码能不能和 @html_rewrite 里面的复用，尽可能统一接口和config配置方法（只是特指API调用这一个部分），避免重复工作。
另外README应该还是要放在 @html_rewrite_cot 这个目录里面

