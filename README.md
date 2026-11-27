# 查询算子的磁盘溢写排序

`with sorted_rows(input_rows, run_rows=..., fan_in=..., scratch=...) as rows`
接收一次性行迭代器，按带符号 64 位整数键升序输出；相同键保留输入次序。
输入分批排序为临时运行集，超出归并路数则多轮归并，最后向下游逐行输出。
输出只在上下文内有效，正常退出、提前停止或异常都会关闭并清理临时文件。

Python 3.9+，标准库。演示使用本仓自制重复键并强制多轮归并：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --no-index -r requirements.txt
.venv/bin/python -m unittest -v
.venv/bin/python demo.py
```

预期键为 `0,1,2,2,4,4,4`，键 2 的负载依次为 B/E、键 4 为 A/C/F，
且输出 `scratch files remaining: 0`。

单行负载是有 UTF-8 字节上限的字符串；内部 JSONL 保留全局序号和换行负载。
内存批次至多 `run_rows` 行，单次归并最多同时读 `fan_in` 个输入文件并另写一个；
这不是精确的 RSS 或磁盘配额。需完整消费输入再产出第一行；
不支持多进程共用运行集、断电续跑或磁盘配额预留。

## 带额度的全外连接

```py
with budgeted_full_join(left, right, pair_budget=..., run_rows=...,
                        fan_in=..., max_payload=..., scratch=...) as (
        joined, deferred_left, deferred_right):
    ...
```

`left`、`right` 各为一次性 `Row` 流（每侧至多 200,000 行），进入上下文时各遍历
一次并按 `(key, 到达序号)` 溢写为临时文件；随后按键升序决定归属。两侧都有同键行
需要 `左行数 × 右行数` 个名额，只有一侧需要该侧行数；剩余额度够则整键接收并扣除，
不够则整键延期、不预扣，继续看后面的键。归属只看键的升序位置和剩余额度，与三个游标
被读取的顺序/行数无关。

`joined` 按 `(left_row, right_row)` 产出，单侧缺失一侧为 `None`；同键按左侧到达
顺序逐行配对、每行内部再按右侧到达顺序。两个延期流按本侧原始顺序交付延期键的原始
`Row`，可直接作为下一次调用的输入。三个游标各自持有独立句柄，可交错消费、可以只用
其中任意一个；离开上下文后都不再产出。连接对按需生成，同键超热不会预生成完整结果。
非法参数、非法行或超出 200,000 行上限抛 `ValueError`；临时数据随上下文退出清理。

## 只压缩部分运行集的版本压缩

```py
with compact_versions(selected, untouched, snapshots=(...),
                      run_rows=128, fan_in=8, scratch=...) as batch:
    for version in batch.replacement():     # 按 (key, revision) 升序的替换流
        ...
    for version in batch.visible(at=None):  # 最新；at=登记版本号读历史时点
        ...
```

`selected` 是要被替换的版本流，`untouched` 是保持原样、继续参与读查询的版本流；两路
都是无序、一次性迭代器（各至多 200,000 条），进入上下文时各验证并外部排序一次。
`Version(key, revision, payload)` 中负载为字符串、空字符串也是值，`None` 表示删除。
每个键在某时点读到版本号不超过该时点的最后一条记录；没有记录或最后一条是删除时该键
不可见，不回退更早的值。`snapshots` 是至多 32 个互不重复的登记版本号，放行时点为
全部登记时点加上无版本上界的“最新”；未登记时点调用 `visible` 立即抛 `ValueError`，
不推进任何游标。

替换流只交付 `selected` 中仍被放行时点需要的原始记录，从不合成、改写或交付
`untouched` 的行。决策在一次按 `(key, revision)` 的流式归并里完成，多个时点共享同
一次单键扫描：一个单调的时点指针标出每条记录胜出的登记时点，而“交付视图当前赢家”
（最新的 untouched 行或已保留 selected 行，删除也计入）在时点之间持续传递，后面的
时点直接复用前面的结论，不为每个时点重扫历史，也不预先落盘每个时点的可见副本。
selected 值只要在任意放行时点胜出就保留；selected 删除只在“原始视图不可见、而交付
视图仍会读出值”的第一个时点保留，删除一旦成为交付赢家就持续遮住更旧的值，更新的
删除若只是再次遮住已不可见的时点则不重复交付。

批次游标只读替换溢写文件和 untouched 溢写文件，原始 selected 全量既不留内存副本也不
留隐藏文件；替换文件与 untouched 文件按 `(key, revision)` 二路归并即可回答任意查询，
归并结果与游标读取顺序、提前停读无关。上下文退出后旧游标不再产出，新调用抛
`RuntimeError`；本次临时文件和句柄一律清理，不触碰调用方文件。同键版本链可以远超
内存：内存峰值受 `run_rows` 批次约束（`tracemalloc` 在上下文内、生成器已全部耗尽时
读取构建期峰值，而不是等文件删除后再测），临时逻辑字节数由后台线程在整个上下文期间
持续采样目录得到，覆盖构建、归并和读游标各阶段。

## 增量有序交换

```py
with ordered_exchange([reserve_0, reserve_1], shared_bytes=...,
                      scratch=None) as exchange:
    exchange.offer(shard, Row(key, payload))
    exchange.advance(shard, lower_bound)
    row = exchange.take()       # (shard, seq, Row) or None
    exchange.finish(shard)
    exchange.done()
```

`ordered_exchange` 是单进程、由调用方交错驱动的算子，不启动线程或后台任务。
分片数为 `len(reserved_bytes)`（1 到 8），各分片保留额度之和加 `shared_bytes`
至多 16 MiB；参数在进入上下文时读取。每行额度为 `32 + payload 的 UTF-8 字节数`：
各分片先使用自己的保留额度，只有超出保留的部分合计占用共享额度，空闲保留额度不借给
其他分片。额度不足时 `offer` 返回 `False`，且不推进序号或改变暂存状态；额度允许时
返回 `True`。成功接收总数硬上限为 200,000，消费行不能恢复该上限。

`advance(shard, lower_bound)` 承诺之后不会收到严格小于该下界的新键；下界只能持平或
增大。`take()` 不等待分片：只有 `key` 严格小于每个未结束分片下界的行才可交付，全部分片
结束后取消该限制。全局顺序为 `(key, shard, seq)`。交付立即释放该行额度，调用方继续持有
返回的 `Row` 不占暂存额度。`finish` 可重复调用；结束分片不能再 `offer`/`advance`，但其
限制随即失效。`done()` 仅在全部分片结束且没有待交付行时为真。

实现把宽负载和 `(key, shard, seq)` 主键放入 scratch 下的私有 SQLite 文件（BLOB 原样保存
UTF-8 字节），由主键完成排序，删除提交后以 `auto_vacuum=FULL` 回收文件空间；调用方逐行
生成、不累积已消费负载时，Python 可跟踪分配峰值远低于 8 MiB，存活暂存字节也低于
64 MiB。退出上下文会关闭连接并删除本次私有目录，不删除 scratch 中的调用方原有文件。

### 向下游装载器发布有序页

```py
with ordered_exchange([reserve_0, reserve_1], shared_bytes=...,
                      scratch=None, publisher=load, page_rows=128,
                      delivery_bytes=0) as exchange:
    exchange.offer(shard, Row(key, payload))
    exchange.advance(shard, lower_bound)
    confirmed = exchange.flush()   # loader-confirmed row count, 0 if none
    exchange.finish(shard)
    exchange.done()
```

传入 `publisher` 时改用按页发布，而 `take()` 会抛 `RuntimeError`（不改状态）；
未传发布器时保持逐行取数，此时调用 `flush()` 同样抛 `RuntimeError`。进入上下文时校验：
`publisher` 为 `None` 或可调用对象，`page_rows` 为 1–128、`delivery_bytes` 为 0–1 MiB，
二者均为整数且不接受布尔值；非发布模式也校验这两个新参数，但不使用发布额度。

`publisher(page_id, rows)` 使用仅属于本次交换的命名空间，页号从 0 连续分配、`rows`
为非空元组，元素与 `take()` 返回同形 `(shard, seq, Row)`。装载器保证整页写入或不写入，
同一页号、完全相同的有序序列重复调用不增加副作用；正常返回表示已提交，抛出的异常不表明
是否已提交，因此交换会保留该页并在下次 `flush()` 用相同页号、行身份、顺序与负载重试，
即使其后又有更多行变得安全。每次 `flush()` 至多调用装载器一次；确认后返回本页行数。

页的取法与 `take()` 一致：只取严格小于每个未结束分片下界的安全行（全部结束后取消限制），
按 `(key, shard, seq)` 顺序，不足 `page_rows` 也照发；无安全行返回 0 且不占页号。分片保留
与共享额度仍只约束待交付行；`delivery_bytes` 是独立的待确认页留存额度。发布前若整页代价
可完整计入该留存额度且仍有余量，则把整页一次性转由它承担重试责任、立即归还原生产额度；
否则整页继续占用原生产额度，绝不拆页分账，也绝不因额度不足而拒绝一个原本可交付的页。
确认时只清本页实际仍占用的额度，不重复返还；有未确认页期间 `offer()` 仍按真实占用的
生产额度准入。只有全部分片结束、无待发布行且无待确认页时 `done()` 才为真。

## 分片向前看滑动计数

```py
with partitioned_window(rows, cuts, preceding=..., carry_keys=...,
                        run_rows=128, fan_in=8, scratch=...) as batch:
    row, count = batch.take(shard)      # (Row, count) 或 None
    path = batch.owner_path(shard)      # 该分片独占的暂存目录
    batch.retire(shard)                 # 归还分片暂存，返回 True/False
```

`rows` 是一次性无序 `Row` 流（至多 200,000 行，负载至多 4096 UTF-8 字节），进入上下文时
只遍历一次，按 `(key, 全局到达序号)` 外部排序后写入每个分片独占的 `shard-N/` 暂存目录；
切点 `cuts` 为零到七个严格递增的合法键，等于切点的行进右侧，分片自 0 编号、允许空分片。
`take(shard)` 在分片内按键升序、同键按输入到达次序交付 `(Row, count)`，读尽返回 `None`；
各分片游标互相独立，可任意交错读取，不阻塞也不推进其他分片。对键 `t` 的每一行，`count`
是本批原始输入中键落在数学整数闭区间 `[t-preceding, t]` 的行数（含尚未取走的同键行），
左端按任意精度整数计算，最小键附近不回绕；计数只取决于原始批次，与读取/归还次序无关，
也不会预先物化整批带计数结果。

计数由只存整数的 SQLite 账本给出：全局不同键计数表、分片归属表、已归还范围留存表均不含
负载，无法替代任何分片。每个分片游标在不同键计数表上维护一对单调的分页位置（窗口上下沿），
按页推进、每游标至多缓存一页，因此同键组或跨范围窗口远超内存时也只做线性扫描，绝不为一条
结果重扫整个来源分片。

`retire(shard)` 尝试立即归还：分片仍有未取行时返回 `False` 且保留真实可用的源暂存、不改动
任何消费位置、不预扣留存额度；已读尽且没有任何未取行的窗口再覆盖其键时必须成功。成功时
真正删除该分片目录及其暂存（而非仅标记不可见），此后该分片不能再 `take`（抛
`RuntimeError`），重复归还返回 `True`。仍被右侧未取行需要的源键按“不同键”迁移为纯整数
计数留存：一键一名额、同一份留存可被多个下游共享，所有已归还范围合计不超过
`carry_keys`；不再被任何未取行窗口触及的留存会在后续归还判定时释放。归还成功后不再读取
原分片目录，剩余窗口计数依旧精确。分片号非法抛 `ValueError`，退出批次后各方法抛
`RuntimeError`；正常退出、提前退出、上游或下游异常都只清理本次暂存与句柄、不吞原异常，
也不触碰调用方在 `scratch` 下的既有文件。生成器供给、调用方不囤积负载时，进入到退出的
Python 可跟踪分配峰值低于 12 MiB；同时存活的暂存逻辑字节不超过
`16 MiB + 16 ×（负载 UTF-8 总字节 + 128 × 行数）`。




