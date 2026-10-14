# 增量训练D相对C：新增匹配10题、退步7题逐题分析

范围：相同100题开发集里的17道EM状态变化题。当前助手逐篇阅读两版查询、返回文档、最终引用及金标支持文档，非独立人工审定。官方EM/F1与别名不修改；没有重算全100题语义准确率，也未全面审阅12道D弃答题。

结论：**条件核验、目标实体排除与缺失末跳搜索有可见改善；作品版本、人物关系和答案类型绑定仍不稳定。**不能把10道新增EM都算作完整可靠回答，也不能把7道EM损失全部视为事实遗忘。

## 新增匹配的10题

| 题目简述 | C → D | 归因与证据限制 |
|---|---|---|
| Boren提案的另一参与者 | Boren → McCurdy | 两版已取得相同共同提案与Boren身份文档；D正确理解beside/who else，排除已给人物。第二查未补必要证据。 |
| Casino Royale作者 | 未提交 → Ian Fleming | C正确答案因错误长ID被拒；D引用实际见过的替代文档完成提交。D仍重复同一查询；并没有实际演示“先出错再恢复”。 |
| 两项游乐设施的乐园 | Hollywood → Florida | 全文同时支持Florida与Hollywood的历史替换关系，题目未区分。唯一金标有歧义，不算干净的事实纠正。 |
| 1954出生的持有人 | Jerome → Jay | D继续核验年份，得到Jay出生1954并更新候选；虽二查仍以Jerome为假设，最终选择和引用正确。 |
| 地名意为pleasant place的村庄 | Baldwin → Mineola | D保持词义约束，虽然先猜Floral Park，仍检索到Mineola并改选；C丢掉词义约束且引用同名演员。 |
| Peter Daou网站口号 | Faith Goldy → media for the 65.8 million | 首查加slogan使Verrit进入结果，答案类型正确。但D额外引用无关Sandusky Register，作者身份链未补完整。 |
| 教会所在城市 | Topeka, Kansas → Topeka | 两者都受证据支持，纯EM表述收益。C一查已足够，D二查多余，效率反而退步。 |
| Elgar配音者对应漫画 | 另一漫画 → Love Hina | D首查保留Keitaro Arima关键词，找到目标角色。但第二查追日本配音Ueda，未核验英语Derek Stephen Prince与Elgar关系；答案匹配不等于完整链可靠。 |
| Roger Groot大学所在县 | Lexington → Rockbridge County | D补查大学county关系，修复把城市当县的提前停止错误；完整人物—大学—县引用。 |
| 与Rumer合演者出生年 | 1988 → 1986 | D改选另一位演员Brittany Snow并查出生年，完整演员关系与年份证据。 |

主分类：8题答案或提交改善、1题表述收益、1题题目歧义。8题中至少网站口号与配音漫画仍有证据链或无关引用问题，不能宣称8题全部满足“完整证据可靠”标准。

## 退步的7题

| 题目简述 | C → D | 归因与证据限制 |
|---|---|---|
| Michael Owen退役职业 | breeder and owner → breeder | 职业与父子关系均检索/引用正确；breeder受支持，但省略owner，属于信息不完整，不是编造职业。 |
| Paul DeBoy所参演游戏开发商 | Rockstar San Diego → Rockstar Games | D查到原作及II，却选并引用II的开发商，回答原作问题。主要是作品版本绑定失败；原作还区分开发和发行。C也引用了另一前作Revolver，旧版EM正确不代表旧证据链可靠。 |
| 2008电影导演 | Douglas Jackson → Dead at 17 | D已取得明确导演的正文，却用电影名回答directed by who；不是缺少检索，属于答案类型/关系抽取错误。 |
| Gamer中的前球员 | Terry Crews → Gino Cappelletti | D二查得到一位前球员，但其名字不在Gamer演员表。只满足职业条件，没有满足参演条件，错误把两个实体的属性拼接。 |
| Devonshire Club命名人物 | 带完整爵位姓名 → William Cavendish | 同一第七代公爵文档支持简称，但省略爵位可能增加同名歧义。D又未检索/引用俱乐部命名关系，存在额外证据链退步，不能只归为别名。 |
| 巴西最多参赛车手获胜数 | 11 → Cannot determine | 首篇已明确Rubens，D却查询Emerson并重复该查询，最终弃答。可见主因是实体选择和查询推进失败，不应直接归因“更保守”。 |
| Chef Mung Daal配音 | William Dwight Schultz → Dwight Schultz | 同一人物的常用名与文档标题，角色/剧集证据链仍在；主要是EM字符串差异。 |

主分类：4题真实答案或查询失败、1题受支持但省略信息、2题简称差异。其中Cavendish简称案例还发生证据链退步。

## 对后续工作的含义

1. **约束核验不是完全没学到。**1954出生、另一演员、词义、城市/县、排除既有人物这几类出现可核验改善；但没有控制实验，不能断言某一训练样本导致某题修复。
2. **优先增加实体与关系绑定的训练诊断。**同名作品/不同年份/续作、开发商与发行商、演员表与职业交集、问导演却回答电影；这比单纯训练更多搜索次数具体。
3. **重复查询和无关引用仍需单独评价。**答对、正常结束、完整金标ID覆盖分别只是不同指标。Casino Royale、Topeka和Verrit展示答案匹配但策略/引用仍有缺陷。
4. **暂不据此改算语义总分或宣称遗忘。**只有17题审阅，未覆盖全部100题或所有弃答，EM净+3并不等于真实能力净+3。历史单种子开发结果也不能证明因果机制。
5. **保留评分，另外维护语义审阅层。**记录支持简称、部分回答、金标歧义和引用缺陷；以后若调整别名/评分，需预先冻结规则并同时重新评测两版，不能只修补有利于D的题。

来源：C运行`outputs/baselines_20261013T234622Z_601d10b9`，D运行`outputs/baselines_20261014T045233Z_5c5a6ab6`，各自agent原始轨迹及题目/语料快照。逐题正文、动作和文件SHA256审计：D运行`evaluation/incremental_change_attribution_17.json`；原始逐题EM变化为同目录`paired_answer_changes.json`。
