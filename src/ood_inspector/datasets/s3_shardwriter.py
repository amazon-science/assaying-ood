import os
from tempfile import TemporaryDirectory

import boto3
import webdataset


class S3ShardWriter(webdataset.ShardWriter):
    def __init__(self, s3_path: str, pattern: str, **kwargs) -> None:
        self.base_pattern = pattern
        self.temporary_directory = None
        self.s3 = boto3.client("s3")
        s3_path = s3_path.split("s3://")[1].split("/")
        self.bucket_name = s3_path[0]
        self.s3_path = "/".join(s3_path[1:])
        self.shards_copied_to_s3 = []
        super().__init__(pattern=pattern, **kwargs)

    def finish(self):
        super().finish()
        if self.temporary_directory is not None:
            self.move_shard_to_s3()
            self.temporary_directory.__exit__(None, None, None)
        self.temporary_directory = TemporaryDirectory()
        self.pattern = os.path.join(self.temporary_directory.__enter__(), self.base_pattern)

    def move_shard_to_s3(self):
        shard_basename = os.path.basename(self.fname)
        shard_path = os.path.join(self.s3_path, shard_basename)
        with open(self.fname, "rb") as f:
            self.s3.upload_fileobj(f, self.bucket_name, shard_path)
        self.shards_copied_to_s3.append(shard_path)

    def __exit__(self, exception_type, exception_value, exception_traceback):
        if exception_type is not None:
            for shard_path in self.shards_copied_to_s3:
                self.s3.delete_object(Bucket=self.bucket_name, Key=shard_path)
            raise RuntimeError(
                "Shard writer exited without being done! (ctrl+c)'ed out? Cleaned up."
            )
        super().__exit__(exception_type, exception_value, exception_traceback)
