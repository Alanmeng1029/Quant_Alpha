# 分钟因子外部资料：来源登记

检索日期：2026-09-04。范围是“由分钟 OHLCV 聚合为日频横截面因子”的公开研报和开源复现；不包含只能以逐笔、委托簿或付费终端获得的数据作为立即可生产来源。

## 已下载并解析的公开 PDF

| 本地文件 | 来源 | 内容 | 获取与校验 |
| --- | --- | --- | --- |
| `reports/cpv_difference_lead_lag.pdf` | [东吴证券：CPV 因子抢跑版，差分视角下的价量互动关系（2021）](https://bigdata-s3.wmcloud.com/researchreport/2021-05/5a3a96171b8fd33cb8c0733dccbe162e.pdf) | 分钟价量相关性、差分、价先量行、量先价行、符号分段合成 | HTTP 直链下载成功；PDF 22 页；已提取并人工视觉复核公式页 8。 |
| `reports/xinda_intraday_factor_review.pdf` | [信达证券：基于基金持仓、特色基本面因子以及高频价量因子的 1000 指数增强（2023）](https://bigdata-s3.wmcloud.com/researchreport/2023-02/e23c267e3113ff6f4f35038da586fea4.pdf) | 放量分钟收益波动率、尾盘成交额/流通市值、日频到周频平滑 | HTTP 直链下载成功；PDF 26 页；已提取并人工视觉复核公式页 6。 |

文件仅用于本地研究和公式复现；保留各报告的原始版权声明，不重新分发。

## 已检索、可用于定义核验但未能作为本地下载件的来源

| 来源 | 可提取内容 | 状态与原因 |
| --- | --- | --- |
| [方正：跟踪聪明钱](https://bigquant.com/wiki/doc/4OvIDMRuTH) | 聪明钱的 `|r| / volume^0.25` 排序与 10 日窗口定义 | 页面可检索；页面引用的旧 PDF 静态链接返回 404。 |
| [John15380/Kysec_Quant_Project](https://github.com/John15380/Kysec_Quant_Project) | 聪明钱、APM、振幅、资金流复现代码结构与 README 公式 | GitHub 页面可检索；本机 `codeload.github.com` HTTP 下载连接超时，未把不完整 clone 作为资料。 |
| [hugo2046/QuantsPlaybook](https://github.com/hugo2046/QuantsPlaybook) | 券商研报复现导航，含聪明钱/APM 类专题 | GitHub 页面可检索；同一网络限制，未下载。 |
| [东吴：CPV 分时版（2024）](https://www.nxny.com/report/view_5844264.html) | 将全天切为 8 个 30 分钟区间，研究区间价量相关性及其分散度 | 页面摘要可检索；下载需要登录，未绕过。 |
| [广发：高频价量数据的因子化方法（2021）](https://www.scribd.com/document/783273984/20210712-%E5%B9%BF%E5%8F%91%E8%AF%81%E5%88%B8-%E5%A4%9A%E5%9B%A0%E5%AD%90Alpha%E7%B3%BB%E5%88%97%E6%8A%A5%E5%91%8A%E4%B9%8B-%E5%9B%9B%E5%8D%81%E4%B8%80-%E9%AB%98%E9%A2%91%E4%BB%B7%E9%87%8F%E6%95%B0%E6%8D%AE%E7%9A%84%E5%9B%A0%E5%AD%90%E5%8C%96%E6%96%B9%E6%B3%95) | 46 个日频价量因子及日频后平滑的两阶段框架 | 登录/订阅平台，仅记录元数据，不下载。 |
| [国盛：高、低位放量](https://www.scribd.com/document/892581540/%E5%9B%BD%E7%9B%9B%E8%AF%81%E5%88%B8-%E9%87%8F%E4%BB%B7%E6%B7%98%E9%87%91-%E9%80%89%E8%82%A1%E5%9B%A0%E5%AD%90%E7%B3%BB%E5%88%97%E7%A0%94%E7%A9%B6-%E5%9B%9B-%E9%AB%98-%E4%BD%8E%E4%BD%8D%E6%94%BE%E9%87%8F-%E4%BB%8E%E4%BA%8B%E4%BB%B6%E9%A9%B1%E5%8A%A8%E5%88%B0%E9%80%89%E8%82%A1%E5%9B%A0%E5%AD%90-%E7%83%BD%E7%81%AB%E7%A0%94%E6%8A%A5www-fhyanbao-com) | 高/低价格状态下的分钟波动率占比 | 仅页面解析可见，下载受平台限制。 |
| [华泰：高频量价因子在股票与期货中的表现（2018）](https://www.htsec.com/jfimg/colimg/upload/20181106/32441541468174586.pdf) | 已实现偏度、峰度、上下行波动率、成交量分布等分类 | 原直链目前返回券商站点 HTML 而非 PDF；响应留存于 `metadata/huatai_high_frequency_price_volume_20181101_landing_page.html`，不误标为报告。 |

## 数据边界

当前分钟湖可直接支持：`open/high/low/close/volume_share/amount_cny` 与交易日、分钟序号、股票代码。需要额外日频数据才可支持：流通市值、行业/市值中性化、指数分钟收益。需要 L2/逐笔才可支持：盘口中间价、买卖价差、委托簿深度、订单失衡、撤单、主动买卖方向和真实大单。

具体公式映射、同现有 core24 的重合判断，以及候选优先级见 [ANALYSIS.md](ANALYSIS.md)。
