@.claude/skills/panguml-format/
@docs/task_specs/convert_sagoge.md
请根据统一数据格式 skill 和本次任务说明，生成一个完整可运行的 Python 数据转换脚本。

@.claude/skills/panguml-format/ @docs/task_specs/visualization.md
请根据统一数据格式 skill 和本次可视化任务说明，生成一个完整可运行的可视化代码，代码存在./visualization/中，并给使用说明。


请根据统一数据格式 skill 和本需求文档，生成一个完整可运行的 Python 种子 query 筛选模块。代码应支持百万级数据规模、分阶段运行和 resume。先不写代码，帮忙审视这套方案，看看是否有不合理之处必须改进，还是可以作为一个v1版本


我们筛选完种子query后（即指令问题），现在想基于这些种子进行模型的蒸馏。我需要可以自己指定api_key，base_url和需要蒸馏的model。我们通过requests.post （verify=False）的方式进行调用。代码读取前面筛选出来的种子的文件（先从100k开始），然后把instruction提取出来，拼上方便自定义的user prompt（用于设定SVG生成的人设以及最必要的根据下面文本指令生成SVG代码的英文prompt）。代码需要先验证API能正常工作，然后并行调用API，并能够自动retry， 完整记录调用结果。同时完整记录调用的tokens量，包括输入、输出和总tokens量。注意，失败调用的也要尽可能记录。记录的log存在一个指定的目录下(比如用不同日 期、不同模型进行区分)，每次实验我们都存在那里，后续还可以很方便地进行统计所有实验总的tokens用量。注意代码应该尽可能解耦，比如我可以直接运行一个单独 的python文件直接debug API是否正常。最后结果存成jsonl，然后可以通过一个沙箱（ @utils/sandbox ）用cairosvg渲染SVG图片（可以设置大小，默认用SVG代码里面的viewBox的大小），然后数据存成 @.claude/skills/panguml-format/SKILL.md 这种统一格式，其中image_root自己指定，然后height/width严格和渲染的图片大小一致。 基于本需求，生成对应的需求文档 @docs/task_specs/distillation.md ，生成完整可运行的 Python 代码并给出对应README。先不写代码，帮忙审视这套方案，看看是否有不合理之处必须改进，以及对于蒸馏代码我还有什么是遗忘的。

1. Prompt结果是不是用 user message即可，这里的问题是如果我们想尽可能保证格式正确，是不是应该加格式要求，比如markdown ```svg\n...\n```？Prompt应该可以去选择，然后可以有脚本可以跑很少几个case去调用不同prompt，用来看看不同prompt下效果如何 2. ：finish_reason == "length" 记录下来即可，默认直接丢弃。我们如果要蒸馏比如Gemini3，设置多大的max_token比较合理呢？3. 最终jsonl要保留原始_meta信息，最好还能留有原始对应 的SVG代码，这样我们可以在后续可视化中看在同一个指令下，蒸馏的SVG和原始数据集里的SVG差别有多大，也许还能给后面做质检使用 4. utils/sandbox 应该是一个简单但鲁棒的模块，支持大规模并行沙箱调用进行渲染。在渲染前肯定需要做SVG校验，但是否放沙箱你可以根据整个系统逻辑、同时考虑未来加入更多语言 等可拓展性、易用性进行设计