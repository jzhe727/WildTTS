from huggingface_hub import HfApi
api = HfApi()

api.upload_large_folder(folder_path="../../training/data/titw_hf", repo_id="jzhe727/wildtts-titw", repo_type="dataset")