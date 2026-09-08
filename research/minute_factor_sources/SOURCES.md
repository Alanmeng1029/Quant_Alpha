# 分钟因子外部资料：来源登记

首轮检索 2026-09-04；扩充检索 2026-09-06（GitHub 仓库、HuggingFace 研报数据集、萝卜投研/东财/quant-wiki/BigQuant 等渠道，走本机代理 127.0.0.1:7897）。范围是“由分钟 OHLCV 聚合为日频横截面因子”的公开研报、开源复现与论文；不包含只能以逐笔、委托簿、订单流或另类数据（点击率等）获得的因子作为可生产来源。

目录约定：

- `reports/`：独立下载的研报 PDF；`reports/hf_stock_factors/` 子目录表示来自 HuggingFace `MMInstruction/stock_factors` 数据集（保留原始文件名）。
- `papers/`：国内外论文 PDF。
- `metadata/`：无法获得 PDF 时的落地页/全文 HTML 存档。
- 代码仓库统一浅克隆在 `external/minute-factor-references/`（gitignore，不入库；本文件登记来源与内容）。`repos/` 目录保留但未使用。

## 一、GitHub 仓库（优先级 1）

2026-09-06 经代理全部浅克隆成功，位于 `external/minute-factor-references/`：

| 本地目录 | 来源 | 内容 | 备注 |
| --- | --- | --- | --- |
| `QuantsPlaybook/` | [hugo2046/QuantsPlaybook](https://github.com/hugo2046/QuantsPlaybook)（6k★） | 100+ 券商金工研报复现。`B-因子构建类/` 含高频价量相关性（东吴 2020-02-23）、聪明钱 2.0、APM、振幅因子的隐藏结构、开源微观结构系列（1）、量价买卖压力（东方六十）、球队硬币、筹码因子等，多数目录同时内嵌研报 PDF 原件 | 内嵌 PDF 约 30 份，含 4 份开源《市场微观结构研究系列》 |
| `Quantitative-analysis/` | [ZFgan/Quantitative-analysis](https://github.com/ZFgan/Quantitative-analysis) | 券商研报复现 notebook。内嵌 41 份研报 PDF，其中开源《市场微观结构研究系列》1（A股反转之力的微观来源 2019-12-23）、3（聪明钱因子模型的 2.0 版本 2020-02-09）、5（APM 因子模型的进阶版 2020-03-07）、7（振幅因子的隐藏结构 2020-05-16）四篇为分钟因子核心文献 | 前次登记为“无法下载”，本次解决 |
| `Quant-Report/` | [QuantNi/Quant-Report](https://github.com/QuantNi/Quant-Report) | 182 份研报 PDF 合集。关键子目录：`量价_分钟级别研报/方正证券/`（8 篇：成交量激增时刻、成交量潮汐、勇攀高峰波动率、球队硬币动量、波动率的波动率、股价跳跃改进振幅、草木皆兵、水中行舟）；`量价_level2级别研报/海通证券/`（20 篇选股因子系列，多为逐笔/L2，见“数据边界”）；另含《高频特质偏度因子全解析》、光大七（K 线最短路径非流动性）、海通四十六/五十九/六十四/六十九/七十/七十六等 | 一站式研报库，含少量重复文件 |
| `Kysec_Quant_Project/` | [John15380/Kysec_Quant_Project](https://github.com/John15380/Kysec_Quant_Project) | 研报复现：开源《独家量价因子的高频测试》。`factors/` 有 smart_money、apm、amplitude、money_flow 的工程化实现 | 前次登记为“无法下载”，本次解决 |
| `Replication-of-Minute-Frequency-Factor-refer-CICC/` | [C-X-Lu/...-refer-CICC](https://github.com/C-X-Lu/Replication-of-Minute-Frequency-Factor-refer-CICC) | 中金《量化多因子系列（12）：高频因子手册》分钟频因子复现（py/polars + notebook） | 中金手册本体仅微信全文（见 metadata） |
| `Replication-of-Minute-Frequency-Factor-refer-FZ/` | [C-X-Lu/...-refer-FZ](https://github.com/C-X-Lu/Replication-of-Minute-Frequency-Factor-refer-FZ) | 方正金工多因子选股系列（分钟频）复现，polars 实现 | 与 `Quant-Report` 方正 PDF 配套 |
| `minute_factors/` | [shandonguzi/minute_factors](https://github.com/shandonguzi/minute_factors) | 分钟级数据因子分析框架（engine/bt/factors） | 框架参考 |
| `BARRA-Volatility/` | [1030692824/BARRA-Volatility](https://github.com/1030692824/BARRA-Volatility) | 分钟波动率因子挖掘 + Barra 风格归因工程 | 波动率类因子工程参考 |

原有（2026-08-31 克隆）：`ashare-5min-sequence-alpha/`、`quant_strategy/`（B1ueDrops，中证 500 分钟多因子）、`systematic-alpha-ml-pipeline/`。

## 二、独立下载的研报 PDF（优先级 2）

`reports/` 与 `reports/hf_stock_factors/`，全部通过 `file` 校验为有效 PDF：

| 本地文件 | 券商/日期 | 内容要点 |
| --- | --- | --- |
| `reports/cpv_difference_lead_lag.pdf` | 东吴 2021 | CPV 因子抢跑版：差分价量、象限拆分（首轮已有） |
| `reports/xinda_intraday_factor_review.pdf` | 信达 2023 | 放量分钟收益波动、尾盘成交额/流通市值（首轮已有） |
| `reports/fangzheng_intraday_momentum_overnight_reversal_industry.pdf` | 方正 2017-11-21 | 《行业轮动的黄金律：日内动量与隔夜反转》，市场行为的宝藏系列（一），魏建榕。个股日内/隔夜收益分解的行业层面应用；仅用 OHLC |
| `reports/hf_stock_factors/15、...已实现高阶矩因子及改进-20页.pdf` | 东北证券（因子选股系列之四） | 5/10min 已实现方差、偏度、峰度构造周/月因子及改进，中证 1000 回测 |
| `reports/hf_stock_factors/18、...基于高频数据的风险不确定性因子-33页.pdf` | 东北证券 2023-06-01 | UOIDR：基于高频数据的风险不确定性因子 |
| `reports/hf_stock_factors/20、...日内成交量分布因子及LogsigAlpha因子生成-33页.pdf` | 东北证券 2023-11-29 | 日内成交量分布因子 + Logsig 路径签名生成因子 |
| `reports/hf_stock_factors/14、...波动率因子的逻辑与非对称使用-26页.pdf` | 东方证券（因子选股系列） | 波动率因子逻辑与非对称使用（日频为主，作体系参考） |
| `reports/hf_stock_factors/东方证券_20160811_日内残差高阶矩与股票收益....pdf` | 东方证券 2016-08-11 | 因子选股系列之九：日内残差高阶矩（对市场回归后的残差矩） |
| `reports/hf_stock_factors/方正证券_21060708_跟踪聪明钱：从分钟行情数据到选股因子.pdf` | 方正 2016-07-08 | “聆听高频世界的声音”系列（三），魏建榕：聪明钱 S 因子原始出处（分钟量价，Q 值排序 pooled 语义） |
| `reports/hf_stock_factors/方正证券_20161025_凤鸣朝阳：股价日内模式中蕴藏的选股因子.pdf` | 方正 2016-10-25 | 日内价格模式（形态聚类）选股因子 |
| `reports/hf_stock_factors/爱建证券_20161010_高频选股因子梳理以及新因子探索.pdf` | 爱建 2016-10-10 | 5 类热门高频因子梳理（走势、量价、高频收益、与大盘关系等）+ 新因子 |
| `reports/hf_stock_factors/“量价淘金”...(七)...羊群效应...-240806-国盛证券-15页.pdf` | 国盛 2024-08-06 | 量价淘金系列（七）：趋势资金的极端交易行为、羊群效应识别 |
| `reports/hf_stock_factors/20170302-海通证券-...价量新因子测试.pdf` | 海通 2017-03-02 | 价量新因子批量测试（冯佳睿） |
| `reports/hf_stock_factors/84-...国泰君安-...神秘的尾盘30分钟.pdf` | 国泰君安 2016-12-23 | 数量化专题之八十四：尾盘 30 分钟的短期预测信息 |
| `reports/hf_stock_factors/20200218-华泰人工智能系列之二十八：基于量价的人工智能选股体系概览.pdf` | 华泰 2020-02-18 | 量价 AI 选股体系概览（方法论地图，非具体因子） |
| `reports/hf_stock_factors/银河证券_20160425_...成交量波动选股研究.pdf` | 银河 2016-04-22 | ⚠️ 基于**行情软件点击率大数据**而非分钟 OHLCV，不符合数据边界，仅留作目录参考 |

仓库内嵌（未复制，路径相对 `external/minute-factor-references/`）：

| 位置 | 内容 |
| --- | --- |
| `Quant-Report/量价_分钟级别研报/方正证券/` | （1）成交量激增时刻 alpha；（2）成交量潮汐；（3）勇攀高峰（波动率变动）；（4）球队硬币（动量效应识别）；（5）波动率的波动率（模糊性厌恶）；（6）股价跳跃改进振幅；（8）草木皆兵（极端收益决策权重）；（9）水中行舟（成交额市场跟随性） |
| `Quant-Report/`（根目录） | 《博彩偏好还是风险补偿？高频特质偏度因子全解析》；20171122 光大七《基于 K 线最短路径构造的非流动性因子》；20190416 海通四十六《日内分时成交中的玄机》；20200424 海通六十四《基于直观逻辑和机器学习的高频数据低频化应用》；20200829 海通七十《日内市场微观结构与高频因子选股能力》；选股因子七十六《基于深度学习的高频因子挖掘》；20200730 六十九《高频因子的现实与幻想》 |
| `Quant-Report/量价_level2级别研报/海通证券/` | 选股因子系列 11/46/47/49/56/57/58/59/64/66/69/70/71/72/75/79/85 等 20 篇 —— **多为逐笔/L2 数据，不符合当前数据边界**，仅四十六/六十四/七十等少数以分钟或综述视角可参考 |
| `Quantitative-analysis/开源证券-市场微观结构研究系列（1）/` | 20191223 系列（1）《A 股反转之力的微观来源》（W 反转分解）PDF+notebook |
| `Quantitative-analysis/聪明钱因子模型的2.0版本/` | 20200209 系列（3）PDF + notebook（与 QuantsPlaybook 同源） |
| `Quantitative-analysis/APM因子模型/` | 20200307 系列（5）《APM 因子模型的进阶版》PDF + notebook |
| `Quantitative-analysis/振幅因子的隐藏结构/` | 20200516 系列（7）PDF + notebook |
| `Quantitative-analysis/基于量价关系度量股票的买卖压力/` | 20191029 东方六十 PDF（量价关系度量买卖压力，分钟非流动性类） |
| `QuantsPlaybook/B-因子构建类/高频价量相关性，意想不到的选股因子/` | 20200223 东吴“技术分析拥抱选股因子”系列（一）PDF + notebook（CPV 姊妹篇） |

## 三、论文（优先级 3）

`papers/`：

| 本地文件 | 出处 | 内容 |
| --- | --- | --- |
| `amaya_realized_skewness_jfe2015.pdf` | Amaya, Christoffersen, Jacobs, Vasquez（JFE 2015 工作稿, CREATES rp13_41） | 5 分钟数据构造已实现偏度/峰度，周频横截面预测的奠基论文 |
| `rjef_higher_realized_moments_2021.pdf` | Romanian J. Economic Forecasting 2021 | 已实现高阶矩可预测性的复制与扩展 |
| `lou_polk_skouras_tug_of_war_nber_w24465.pdf` | Lou, Polk, Skouras（NBER w24465） | A Tug of War：隔夜与日内预期收益的对抗（日内/隔夜分解的经典框架） |
| `arxiv_strata_intraday_mamba_2608.28060.pdf` | arXiv 2608.28060 | STRATA：5 分钟 bar 直接端到端预测次日横截面排名（Mamba/SSM 结构） |
| `arxiv_intraday_momentum_microstructure_2607.01550.pdf` | arXiv 2607.01550 | 日内动量的微观结构解释（横截面日内模式） |
| `arxiv_skewness_dispersion_2604.07870.pdf` | arXiv 2604.07870 | 个股已实现偏度的截面离散度预测市场收益 |
| `arxiv_echo_state_intraday_2504.19623.pdf` | arXiv 2504.19623 | Echo State Network 多 horizon 日内收益横截面预测 |
| `reports/hf_stock_factors/基于高频数据的市场情绪择时研究_胡海涛_20100526.pdf` | 广发量化择时系列收录 | 高频数据市场情绪择时（择时向，论文体） |

## 四、已检索、可核验定义但未取得本地 PDF 的来源

| 来源 | 可提取内容 | 状态与原因 |
| --- | --- | --- |
| [长江证券 高频因子系列专题（fxbaogao 780，2019-2026 共 19 篇）](https://www.fxbaogao.com/zhuanti/detail/780) | 高频因子（二）结构化反转、（三）研究框架、（六）特异视角波动率、（八）高位成交、（九）高频波动中的时序信息、（十）量价反转微观结构、（十一）微观划分、（十二）日内与日间、（十三）广义拥挤度等 | 平台需登录；BigQuant 镜像（[二](https://bigquant.com/wiki/doc/k57UdBQ6GS)、[框架](https://bigquant.com/wiki/doc/7D2q95vg7V)）与 [microbell 十三](http://wt.microbell.com/data/488ebf5aebd247f9bb91bd06dc4305a0.html) 页面可检索定义 |
| [中金 量化多因子系列（12）：高频因子手册](https://mp.weixin.qq.com/s/N48b27FAPwcbQZkbnlBhwQ) | 高频价量因子字典（分钟 + L2 混合，含中金自研因子） | 微信全文已存 `metadata/cicc_high_freq_factor_handbook_wechat.html`；[Scribd PDF](https://www.scribd.com/document/1038445439/) 需登录；复现代码在 GitHub（见上表） |
| [光大 多因子五：见微知著，成交量占比高频因子解析（2017-09-01）](https://bigquant.com/wiki/doc/61etdmb4NI) | 集合竞价成交量占比因子；注意 1 分钟首根 bar 是否含竞价量的数据口径 | 各下载渠道需登录 |
| [光大 多因子八：高频因子——日内分时成交量蕴藏玄机（2017-11-23）](https://bigquant.com/wiki/doc/qkCecoflIQ) | 49 个 5 分钟时段成交量占比因子、日内 U 型季节性 | 同上 |
| [海通 高频量价因子在股票与期货中的表现（2018-11-01）](https://bigquant.com/wiki/doc/FMNopvZkg1) | 收益率分布/成交量分布/量价相关性三类高频因子体系 | 同上（早期登记误标为华泰，该报告为海通） |
| [广发 高频价量数据的因子化方法（多因子 Alpha 四十一，2021）](https://bigquant.com/wiki/doc/xCReVHKVy1) | 日内价格相关、日内价量相关、盘前信息、特定时段采样共 46 因子 | 登录平台，仅记录元数据 |
| [东吴 CPV 分时版（2024）](https://www.nxny.com/report/view_5844264.html) | 全天 8 个 30 分钟区间价量相关性及分散度 | 下载需登录（首轮已登记，仍未取得） |
| [国盛 量价淘金（四）高、低位放量](https://www.scribd.com/document/892581540/) | 高/低价格状态下的分钟波动率占比 | 登录平台 |
| [开源证券官网研报页（聪明钱 2.0, id=1595）](https://www.kysec.cn/index.php?m=content&c=index&a=show&catid=108&id=1595) | 官方入口（PDF 已通过 GitHub 仓库取得，见上表） | 已解决，仅存链接 |

下载渠道说明：`bigdata-s3.wmcloud.com/researchreport/{YYYY-MM}/{hash}.pdf?download=true` 为萝卜投研（robo.datayes.com）后端直链，需其站内检索获取 hash，搜索引擎不索引；东财 `pdf.dfcfw.com` 直链需 reportapi 的 infoCode，其搜索接口（search-api-web）对研报类型返回空。quant-wiki 的 `asset.quant-wiki.com/pdf/...` 直链测试返回 1 页错误 PDF，已弃用。

## 五、数据边界

当前分钟湖可直接支持：`open/high/low/close/volume_share/amount_cny` 与交易日、分钟序号、股票代码。需要额外日频数据才可支持：流通市值、行业/市值中性化、指数分钟收益（APM、残差矩、水中行舟等需要）。需要 L2/逐笔才可支持：盘口中间价、买卖价差、委托簿深度、订单失衡、撤单、主动买卖方向和真实大单 —— 对应 `Quant-Report/量价_level2级别研报/海通证券/` 大部分、中金手册的 L2 部分、天风买卖压力失衡、国金量价背离（快照）、东兴行为追踪（快照）等，不进入当前候选。另类数据（银河点击率）同样排除。

具体公式映射、同现有 core24 的重合判断，以及候选优先级见 [ANALYSIS.md](ANALYSIS.md)。
