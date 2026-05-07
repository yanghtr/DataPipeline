@utils/renderers/html 文件夹里面有一个代码脚本，可以将输入文件里面的html代码渲染成图片。这个代码考虑了如何确保代码能够正常渲染，解决了不少corner case，已有的逻辑都需要进行保留。现在我需要对这个文件夹下面的代码进行修改整合。其中 @utils/renderers/html/README.md 给了各个代码文件的目标（比如某个文件用法），但 parse_render_log.py/AesCode_358K_filter_by_render_issues.py 具体怎么用没有说（而且文件的命名也应该修改，避免出现 AesCode_358K 这种特殊的字符）。我现在不知道整体代码的逻辑和使用说明。帮我理解并整理 @utils/renderers/html 里面的代码，保持原有逻辑，让它们更方便地被使用。比如 --mode render 进行渲染，--mode filter进行筛选，或者更好的设计



现在的jsonl文件是我用大模型改写真实互联网HTML代码得到的，用于训练模型的前端代码生成能力。我们现在需要把WARN+ERROR的都去掉。在这个背景下，我还希望增加几个功能：（1）有时候CDN不能访问，比如一些Google font: "msg": "Refused to apply style from 'https://fonts.googleapis.com/xxxx ，参考 @utils/renderers/html/logs/all.txt 。这会导致我们丢失不少数据。既然代码是大模型改写得到的，是不是对jsonl里面的HTML去做一个预处理（比如用一个预处理的模块代码）：对于不能访问的Google 字体，我们去把链接改成能访问的。比如 https://fonts.googleapis.com/css2?family=PT+Sans:300,400,600,700&family=Raleway:300,400,600,700&display=swap 不能访问，但是 https://fonts.googleapis.com/css?family=Lato:100,100i,300,300i,400,400i,700,700i,900,900i&display=swap  好像可以访问，我们能有一些简单轻便的规则化的方式去预处理链接方便后面留下更多的数据吗？你觉得呢？（2）filter阶段我还希望能够增加几个功能：比如对于渲染得到的图片，如果图片过高（比如网站很长，height:width > 3:1），我们就把这部分数据也filter掉（相当于在WARN/ERROR基础上再加一个level tag，后面filter根据tag去除某部分数据）。相关信息也要记录下来 （3）filter阶段还应该默认加一个flag，默认会存jsonl文件到一个文件路径（比如run.sh里面设成和images同样目录的jsonl），这里的jsonl是panguml格式 @.claude/skills/panguml-format/SKILL.md ,我们可以指定图片根目录（比如就是 images这个文件夹），然后panguml jsonl里面的图片相对路径可以是相对于这个目录。这样整个流程下来就可以直接得到用于模型训练的数据。


（1）功能 1 — Google Fonts 预处理（preprocess_html.py 新模块）关键规律：css2? API 网络受限：不要直接替换掉然后用系统默认字体。我意思是指虽然css2不能访问，但是链接里面的css2改成css是不是就可以访问了，我是说这种直接人工修改链接的规则。你可以帮忙检查一下是不是就能访问了。我们不需要保证所有都能捞回来，只要能尽可能捞回一部分数据即可（2）panguml 里 user turn 里面需要放文字，你需要自己显示设计50条 instruction template，然后每条数据随机从中采样。 这些 instruction 都有类似下面的意思（image to HTML）： Provide the HTML code for the input image. 你可以参考之前SVG的模板 @/home/yanghaitao/Projects/DataPipeline/converters/templates/image_to_svg_templates.txt  （我们需要显式存这个文件）


（1）我发现CDN cache很不稳定，比如我第一次跑遇到下面的错误日志：
[CDN] Cache ready: 29 succeeded, 2 failed
[CDN] FAILED: HTTP 404 | https://fonts.googleapis.com
[CDN] FAILED: HTTP 404 | https://fonts.gstatic.com

第二次跑之后变成：
[CDN] Cache ready: 16 succeeded, 15 failed
[CDN] FAILED: HTTP 404 | https://fonts.gstatic.com
[CDN] FAILED: HTTP 404 | https://fonts.googleapis.com
[CDN] FAILED: URLError: <urlopen error timed out> | https://fonts.googleapis.com/css?family=Open+Sans:400,400i,700,700i|Montserrat:300,400,600,700&amp;subset=latin,latin-ext
[CDN] FAILED: URLError: <urlopen error timed out> | https://fonts.googleapis.com/css?family=Open+Sans:300,300i,400,400i,600,600i,700,700i,800,800i&display=swap
[CDN] FAILED: URLError: <urlopen error timed out> | https://fonts.googleapis.com/css?family=Open+Sans:300,400,600|Amatic+SC:400,700&display=swap
[CDN] FAILED: URLError: <urlopen error timed out> | https://fonts.googleapis.com/css?family=Open+Sans:400,600,700
[CDN] FAILED: URLError: <urlopen error timed out> | https://fonts.googleapis.com/css?family=Open+Sans:300,400,700,800
[CDN] FAILED: URLError: <urlopen error timed out> | https://fonts.googleapis.com/css?family=Open+Sans:300,400,600,700
[CDN] FAILED: URLError: <urlopen error timed out> | https://fonts.googleapis.com/css?family=Inter:300,400,500,600,700,800
[CDN] FAILED: URLError: <urlopen error timed out> | https://fonts.googleapis.com/css?family=Roboto+Condensed:300,400,700,300italic,400italic,700italic|Roboto:100,300,400,500,700,900,100italic,300italic,400italic,500italic,700italic,900italic|PT+Serif:400,700,400italic,700italic
[CDN]   ... and 5 more failures

这部分代码会对后面的WARN/ERROR造成影响吗？可以让它更鲁棒些吗？

（2）tall_ratio_threshold应该暴露在最外一层的run.sh里面，能够在这里面进行指定（默认改成4.0）

（3）filter模式之后需要加入一些统计信息，比如图片的宽高分布，一个统计图画出所有的每一个文件去除的比例等，最好能画出图片存下来

（4）现在的打印信息非常误导人，比如下面是一次实验结果：
┌─ STEP 3/3  过滤原数据  (level=all)
│
│  ── part2026-03-23-00000
│    [INFO] 已加载 50 条 instruction 模板: /home/ma-user/work/yanghaitao/Code/data_creation/FineWebEdu/DataPipeline/converters/templates/image_to_html_templates.txt
│    [LOAD] issues: /home/ma-user/work/yanghaitao/Code/data_creation/FineWebEdu/run/20260324_0_20//images/issues.json  level=all
│    [INFO] 共 2 个源文件，1000 个待剔除 id
│    [INFO] 输入文件数: 2
│      api_calls.jsonl                          无需过滤  (stem=stage2_part2026-03-23-00000_api_calls)
│      output.jsonl                             剔除 495/3638  (stem=stage2_part2026-03-23-00000_output)
│    
│    [SUMMARY]
│      处理文件数    : 2
│      原始总条目数  : 7287
│      共剔除条目数  : 495
│      保留条目数    : 6792
│      无对应 issue 的文件数: 1
│      输出目录      : /home/ma-user/work/yanghaitao/Code/data_creation/FineWebEdu/run/20260324_0_20/filtered/part2026-03-23-00000
│      panguml 输出  : /home/ma-user/work/yanghaitao/Code/data_creation/FineWebEdu/run/20260324_0_20/jsonl/data_000000.jsonl
│      panguml 写入  : 3143 条  跳过: 3649 条（PNG 不存在或 HTML 为空）
│
│  ── part2026-03-23-00001
│    [INFO] 已加载 50 条 instruction 模板: /home/ma-user/work/yanghaitao/Code/data_creation/FineWebEdu/DataPipeline/converters/templates/image_to_html_templates.txt
│    [LOAD] issues: /home/ma-user/work/yanghaitao/Code/data_creation/FineWebEdu/run/20260324_0_20//images/issues.json  level=all
│    [INFO] 共 2 个源文件，1000 个待剔除 id
│    [INFO] 输入文件数: 2
│      api_calls.jsonl                          无需过滤  (stem=stage2_part2026-03-23-00001_api_calls)
│      output.jsonl                             剔除 505/3582  (stem=stage2_part2026-03-23-00001_output)
│    
│    [SUMMARY]
│      处理文件数    : 2
│      原始总条目数  : 7178
│      共剔除条目数  : 505
│      保留条目数    : 6673
│      无对应 issue 的文件数: 1
│      输出目录      : /home/ma-user/work/yanghaitao/Code/data_creation/FineWebEdu/run/20260324_0_20/filtered/part2026-03-23-00001
│      panguml 输出  : /home/ma-user/work/yanghaitao/Code/data_creation/FineWebEdu/run/20260324_0_20/jsonl/data_000000.jsonl
│      panguml 写入  : 3077 条  跳过: 3596 条（PNG 不存在或 HTML 为空）
│
└─ STEP 3/3  完成  →  /home/ma-user/work/yanghaitao/Code/data_creation/FineWebEdu/run/20260324_0_20//filtered
这里把整体和这单个文件的数目搞错了，比如SUMMARY都是记录的完整的统计数，导致有大量的重复、误读，难以理解这些数据。稍微简化一下，让部分统计更加清晰，打印完每一个文件信息最后再打印全部合并的统计信息

(5) 现在的resume也有很大的bug，应该显示默认设置--resume这个flag。现在又3个stage，应该确保不管每一个stage中途失败，重新跑了之后数据能够正确统计，也能够正确print，现在重新跑之后会很多统计上的错误。

（6）更新相关README.md和设计文档DESIGN.md