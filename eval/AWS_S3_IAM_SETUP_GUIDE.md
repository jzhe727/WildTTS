## AWS S3 IAM setup guide for MLflow

### Configure credentials on your system

Option A — AWS credentials file (recommended for local development)

Create or edit `~/.aws/credentials`:

```ini
[default]
aws_access_key_id = YOUR_ACCESS_KEY_ID
aws_secret_access_key = YOUR_SECRET_ACCESS_KEY
region = us-east-1
```

Option B — Environment variables (recommended for production / CI)

```bash
export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="us-east-1"
```

### Testing your access

Save and run the following Python script to verify S3 access. Replace `your-mlflow-bucket` with the name of the bucket you intend to use.

```python
#!/usr/bin/env python3
import boto3
from botocore.exceptions import ClientError

def test_s3_access(bucket_name):
    try:
        s3_client = boto3.client('s3')

        # Test list objects (read/list permission)
        response = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
        print(f"✓ Successfully accessed bucket: {bucket_name}")

        # Test write access
        test_key = "mlflow-test/test.txt"
        s3_client.put_object(Bucket=bucket_name, Key=test_key, Body=b"test")
        print(f"✓ Successfully wrote test object: {test_key}")

        # Test read access
        obj = s3_client.get_object(Bucket=bucket_name, Key=test_key)
        print("✓ Successfully read test object")

        # Cleanup
        s3_client.delete_object(Bucket=bucket_name, Key=test_key)
        print("✓ Successfully deleted test object")

        print("\nAll S3 permissions verified successfully!")
        return True

    except ClientError as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    bucket_name = "your-mlflow-bucket"  # Replace with your bucket name
    test_s3_access(bucket_name)
```

### Notes
- If you use IAM roles (EC2, ECS, EKS), you generally don't need to set credentials locally — ensure the instance/task role has the required S3 permissions.
- Replace `region` / `AWS_DEFAULT_REGION` with your AWS region if different from `us-east-1`.
- The script requires `boto3` and `botocore`; install with `pip install boto3`.

### How to run the test

```bash
python test_s3_access.py
```
