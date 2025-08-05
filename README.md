# Fine_Tuning


## Instructions to run the code:

Can run these commands through a notebook if using AWS notebook job. Else in the terminal of the machine being used. 

```bash
pip install -r requirements.txt
```

```bash 
pip uninstall torchvision -y
```

Change --nproc-per-node to number of GPUS as required
```bash
torchrun --nproc-per-node=1 train.py
```

For notebook job following set up also needed. Skip if train and requirments files are available on machine. Run this before the above commands

```python
!pip install boto3
import boto3
s3 = boto3.client('s3',
                  aws_access_key_id=aws_key,
                  aws_secret_access_key=aws_secret)

bucket_name = 'train-job-input-folder'
files_to_download = {
    'requirements.txt': 'requirements_s3.txt',
    'single_gpu_faq_train.py': 'train.py'
}

for s3_key, local_name in files_to_download.items():
    s3.download_file(bucket_name, s3_key, local_name)
    print(f"Downloaded {s3_key} to {local_name}")
```
