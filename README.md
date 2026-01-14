# puzzlesBatchEval

面向“已有 reference + predict（可选 processed/）”的批量评估入口。

## 目录约定（每个 target）

```
<target>/
  references/
    *.pdb
    processed/*.pdb        # 可选（若存在则优先使用）
  models/
    *.pdb
    processed/*.pdb        # 可选（若存在则优先使用）
  chain_mapping.json       # 可选：全局链映射（model->reference）
  chain_mappings/          # 可选：按 pair 覆盖
    <ref_basename>__<model_basename>.json
    <ref_stem>__<model_stem>.json
```

## 运行（示例）

```
python puzzlesBatchEval/rpz_assess_batch.py \
  --targets-root /path/to/targets \
  --config puzzlesBatchEval/config.json
```

冒烟测试（仅取前 1 个 reference 与前 3 个 model）：

```
python puzzlesBatchEval/rpz_assess_batch.py \
  --targets-root /path/to/targets \
  --config puzzlesBatchEval/config.json \
  --limit-references 1 \
  --limit-models 3
```

输出目录结构：

```
results/<run_id>/<target>/<ref_stem>/<model_stem>/{rmsd,inf,tm,mcq,gdt,lddt,clash}/...
results/<run_id>/<target>/tables/summary_all_pairs.csv
results/<run_id>/<target>/tables/errors.csv
```

`summary_all_pairs.csv` 的行键为 `"<ref_basename>__<model_basename>"`（同时提供 `reference`/`model` 两列），
每个指标都有对应的 `<metric>_rank` 列；失败/缺失的指标不会参与 rank，并会出现在 `errors.csv`。
