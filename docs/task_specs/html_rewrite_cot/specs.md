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



@outputs/html_rewrite_cot/ 这里面是使用命令： python -m html_rewrite_cot.main --config html_rewrite_cot/configs/default_local.yaml 得到的最终6条数据的结果。你自己做一下质检：（1）首先确定质检方案：应该从哪些维度判断CoT质量，比如说CoT里面有没有幻觉，CoT 和图片/代码是否对应，中间结果有什么问题，格式有什么问题，是否容易给模型进行学习，等等其它方面，（2）然后你作为多模态大模型根据质检的维度自己进行看图片、CoT、代码、输出结果分析，给出分析报告

（1）首先将你的质检方案写成文档，作为后续版本你自己质检的依据（2）然后根据你这次质检结果和分析修改对应代码和跑相应的实验（6条数据）（3）用修改后的代码跑完之后用之前已经存的质检依据重新质检，看看问题有没有修改，将整个过程记录成实验报告存下来，方便后续迭代


针对M/G 维度仍有 placeholder 外观语言泄漏的问题，帮我继续修改，实验，并质检评测，写出第二版实验报告

现在 retry_on_timeout 为什么能够保证可以重新跑playwright会成功？是不是这种数据可能会出错最好还是去掉？因为CoT的数据质量要求很高。现在这个代码里面，如果stage1最后还是有一些数据 没有成功运行，即结果不ok，那么在stage2会怎么样？stage2是不是确保都是用stage1结果比较好？即stage2是不是应该依赖stage1已经成功的数据，然后最后会输出统计，说明用的数据里面没成功 的原因（包括stage1没成功）。你觉得呢？另外确保各个阶段resume的逻辑是正确的。



