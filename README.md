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
