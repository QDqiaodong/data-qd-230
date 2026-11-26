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

