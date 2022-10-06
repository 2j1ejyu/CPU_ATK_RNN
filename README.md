# CPU_ATK_RNN

## Requirements
```
pip install -r requirements.txt
```
## Training Code

A script for running the training code

```bash
python train.py --exp_name [name] --data_root [data_root] --attack_type ["FR" or "PA"] --gpu [gpu_num]
```

## Datas
the datas should be in the form of csv, and each of the file should consist of feature columns 
```
IPC || Instructions || Cycles || Started RTM Exec || Aborted RTM Exec || TX Write Abort
```
