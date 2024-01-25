# mosGraphGen

## 1. UCSC Dataset Processing
Check the jupyter nodebook 'UCSC_union_raw_data_process.ipynb' for details.
![](./Paper-figures/UCSC-flowchart.png)

## 2. ROSMAP Dataset Processing
Check the jupyter nodebook 'ROSMAP_union_raw_data_process.ipynb' for details.
![](./Paper-figures/ROSMAP-flowchart.png)

## 3. Run the Graph Neural Network Model
![](./Paper-figures/Model.png)

```bash
python geo_tmain_gcn-UCSC.py
```

```bash
python geo_tmain_gat-UCSC.py
```

```bash
python geo_tmain_gcn-ROSMAP.py
```

```bash
python geo_tmain_gat-ROSMAP.py
```
