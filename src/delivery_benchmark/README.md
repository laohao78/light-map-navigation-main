# Benchmark使用说明

## 安装依赖

```sh
apt install psmisc
```

## 生成配送任务指令

```sh
python3 delivery_request_generator.py --num 3 --per-block 3 --output-file data/medium0323_reqs.txt --coords-file data/medium0323_coords.txt
```

## 自动化执行配送任务指令

```sh
./start_delivery.sh data/xxx.txt
```