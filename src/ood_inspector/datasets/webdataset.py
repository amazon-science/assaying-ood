import collections
import contextlib
import logging
import mmap
import os
import subprocess
import sys
import threading
import urllib.parse
from typing import Dict, List, Optional, Union

import boto3
import tqdm
import webdataset
import webdataset.gopen
import webdataset.tariterators
from torch.utils.data import Dataset

from ood_inspector.datasets.s3_shardwriter import S3ShardWriter
from ood_inspector.utils import s3_folder_exists_and_not_empty, s3_list

LOGGER = logging.getLogger(__name__)

# We need to patch a part of webdatsets.  Instead of copying all their code and applying the patch
# there, this seems like a more "minimally invasive" approach.


def patched_url_opener(data, handler=webdataset.tariterators.reraise_exception, **kw):
    """Given a stream of url names (packaged in `dict(url=url)`), yield opened streams."""
    for sample in data:
        assert isinstance(sample, dict), sample
        assert "url" in sample
        url = sample["url"]
        try:
            # Use a context manager, to ensure the stream is closed when the Pipe object goes out of
            # scope.
            with webdataset.gopen.gopen(url, **kw) as stream:
                sample.update(stream=stream)
                yield sample
        except GeneratorExit:
            # This exception is triggered when the generator is closed outside of this code.
            # We do not want it to be handled similarly to other exceptions as this exception does
            # not indicate a failure case.
            break
        except Exception as exn:
            exn.args = exn.args + (url,)
            if handler(exn):
                continue
            else:
                break


webdataset.tariterators.url_opener = patched_url_opener


# Another patch to log errors from the subprocess via pythons logging module.


class PipeLogWriter(threading.Thread):
    """Class to allow redirecting subprocess stderr output to log.

    Subprocess uses low level os functionality to redirect streams and thus we need a correct
    `fileno` property for things to work. This is done by creating a pipe and using a separate
    thread to read new data from the pipe and write it to log.
    """

    def __init__(self, command):
        super().__init__()
        self.daemon = True  # End thread as soon as the parent thread ends.
        self.command = command
        self.logger = logging.getLogger("Pipe stderr")
        self.fdRead, self.fdWrite = os.pipe()
        self.start()

    def fileno(self):
        """Return the file descriptor which allows writing to the log.

        This is used by Popen in order to redirect process output.
        """
        return self.fdWrite

    def close(self):
        """Close the pipe."""
        try:
            os.close(self.fdRead)
        except IOError:
            # Sometimes, the reading file descriptor is already closed. This can occur when the run
            # process exits and thus the context manager `fdopen` exists.
            pass
        os.close(self.fdWrite)

    def run(self):
        with os.fdopen(self.fdRead) as f:
            for line in iter(f.readline, ""):
                self.logger.error(line.strip("\n"))


class Pipe:
    """Wrapper class for subprocess.Pipe.
    This class looks like a stream from the outside, but it checks
    subprocess status and handles timeouts with exceptions.
    This way, clients of the class do not need to know that they are
    dealing with subprocesses.

    This class is a slight adaptation of the original which can be found at
    https://github.com/webdataset/webdataset/blob/d616fe4eac6715c939d1511e6c9348319dc5e057/webdataset/gopen.py#L20

    It includes additional logging functionality for the stderr of subprocesses.

    :param *args: passed to `subprocess.Pipe`
    :param **kw: passed to `subprocess.Pipe`
    :param timeout: timeout for closing/waiting
    :param ignore_errors: don't raise exceptions on subprocess errors
    :param ignore_status: list of status codes to ignore
    """

    def __init__(
        self,
        *args,
        mode=None,
        timeout=7200.0,
        ignore_errors=False,
        ignore_status=[],
        **kw,
    ):
        """Create an IO Pipe."""
        self.ignore_errors = ignore_errors
        self.ignore_status = [0] + ignore_status
        self.timeout = timeout
        self.args = (args, kw)
        self.logger = PipeLogWriter(" ".join(args))
        if mode[0] == "r":
            self.proc = subprocess.Popen(*args, stdout=subprocess.PIPE, stderr=self.logger, **kw)
            self.stream = self.proc.stdout
            if self.stream is None:
                raise ValueError(f"{args}: couldn't open")
        elif mode[0] == "w":
            self.proc = subprocess.Popen(*args, stdin=subprocess.PIPE, stderr=self.logger, **kw)
            self.stream = self.proc.stdin
            if self.stream is None:
                raise ValueError(f"{args}: couldn't open")
        self.status = None

    def __str__(self):
        return f"<Pipe {self.args}>"

    def check_status(self):
        """Poll the process and handle any errors."""
        status = self.proc.poll()
        if status is not None:
            self.wait_for_child()

    def wait_for_child(self):
        """Check the status variable and raise an exception if necessary."""
        verbose = int(os.environ.get("GOPEN_VERBOSE", 0))
        if self.status is not None and verbose:
            # print(f"(waiting again [{self.status} {os.getpid()}:{self.proc.pid}])",
            # file=sys.stderr)
            return
        self.status = self.proc.wait()
        if verbose:
            print(
                f"pipe exit [{self.status} {os.getpid()}:{self.proc.pid}] "
                f"{self.args} {webdataset.gopen.info}",
                file=sys.stderr,
            )
        if self.status not in self.ignore_status and not self.ignore_errors:
            raise Exception(f"{self.args}: exit {self.status} (read) {webdataset.gopen.info}")

    def read(self, *args, **kw):
        """Wrap stream.read and checks status."""
        result = self.stream.read(*args, **kw)
        self.check_status()
        return result

    def write(self, *args, **kw):
        """Wrap stream.write and checks status."""
        result = self.stream.write(*args, **kw)
        self.check_status()
        return result

    def readLine(self, *args, **kw):  # noqa: N802
        """Wrap stream.readLine and checks status."""
        result = self.stream.readLine(*args, **kw)
        self.status = self.proc.poll()
        self.check_status()
        return result

    def close(self):
        """Wrap stream.close, wait for the subprocess, and handle errors."""
        self.stream.close()
        self.status = self.proc.wait(self.timeout)
        self.logger.close()
        self.wait_for_child()

    def __enter__(self):
        """Context handler."""
        return self

    def __exit__(self, etype, value, traceback):
        """Context handler."""
        self.close()


webdataset.gopen.Pipe = Pipe


def get_s3_mmap(uri: str):
    """Get bytes representation of s3 object using boto3.

    In this implementation, we don't maintain a long-running connection to S3 to  incrementally
    stream the content, but write everything directly into mmap memory. This is due to issues when
    interacting with s3. See the following issue:
    https://forums.aws.amazon.com/thread.jspa?threadID=111195

    Args:
        uri (str): S3 uri for example "s3://my_bucket/my_file.txt"

    Returns:
        mmap: The file content in memory.
    """
    # TODO(hornmax): This is probably not the most efficient implementation due to recreation of the
    # s3 client, but seems to work for now. Generally, the overhead of initializing the client
    # should be negligible compared to the download of the file in our case though, because shards
    # are relatively large.
    parsed = urllib.parse.urlparse(uri, allow_fragments=False)
    bucket_name = parsed.netloc
    key = parsed.path.strip("/")
    s3 = boto3.client("s3")
    header = s3.head_object(Bucket=bucket_name, Key=key)
    size = header["ContentLength"]

    # Get some anonymous memory to write to.
    memory = mmap.mmap(-1, size)
    s3.download_fileobj(bucket_name, key, memory)
    # Reset the position to read from beginning of data stream.
    memory.flush()
    memory.seek(0)
    return contextlib.closing(memory)


def s3_mmap_handler(uri, *args, **kwargs):
    """Handler for webdataset.gopen to support s3 streams."""
    return get_s3_mmap(uri)


webdataset.gopen.gopen_schemes["s3"] = s3_mmap_handler


def log_and_reraise(exn):
    """Handler to log exceptions occurring during webdataset processing.

    If we don't explicitly log these, we will not see them in the error logs as they are ignored by
    python.
    """
    LOGGER.warning("Caught exception in webdataset pipeline:")
    LOGGER.exception(exn)
    raise exn


def _s3_pipe(s3key: str) -> str:
    return f"pipe:aws s3 cp --quiet {s3key} -"


def get_number_of_datapoints(dataset):
    LOGGER.info("Counting size of webdataset...")
    n = sum([1 for _ in dataset])
    LOGGER.info(f"Webdataset contains {n} examples.")
    return n


def get_webdataset(
    uri_expression: Union[str, List[str]],
    number_of_classes_per_attribute: Optional[Dict[str, int]] = None,
    number_of_datapoints: Optional[int] = None,
) -> Dataset:
    """Returns a webdataset specified by the shards in uri_expression.

    Args:
        uri_expression: Either a string or a list of strings specifying the set of shards.
        number_of_classes_per_attribute [optional]: The numbers of attribute-classes for each
                                                attribute. If not provided, these are computed.
        number_of_datapoints [optional]: The number of datapoints in all shards. If not provided, it
                                        is computed.

    Returns:
        A webdataset.
    """
    if isinstance(uri_expression, list):
        uri_expression = [u for u in uri_expression]
    elif isinstance(uri_expression, str):
        uri_expression = uri_expression
    else:
        raise ValueError()
    shard_list = webdataset.PytorchShardList(
        uri_expression,
        shuffle=False,
    )
    dataset = webdataset.WebDataset(shard_list)
    dataset = dataset.decode("pil")
    dataset = dataset.rename(image="jpg;png")
    number_of_datapoints = number_of_datapoints or get_number_of_datapoints(dataset)

    if number_of_classes_per_attribute is None:
        number_of_classes_per_attribute = get_number_of_classes_per_attribute(dataset)

    attributes = list(number_of_classes_per_attribute.keys())

    dataset = dataset.rename(
        **{
            attribute: f"{attribute}.cls" if not attribute == "label_" else "cls"
            for attribute in attributes
        }
    )

    dataset = dataset.with_length(number_of_datapoints)

    return dataset


def get_number_of_classes_per_attribute(dataset):
    logging.info("Counting how many different values each attribute attains...")

    classes_per_attribute = collections.defaultdict(lambda: set())

    for sample in dataset:
        for attribute in sample.keys():
            if "." not in attribute or attribute.startswith("__") or attribute.endswith("__"):
                continue
            cleansed_attribute = ".".join(attribute.split(".")[:-1])
            classes_per_attribute[cleansed_attribute].add(sample[attribute])

    number_of_classes_per_attribute = {
        attribute: len(classes_per_attribute[attribute])
        for attribute in classes_per_attribute.keys()
    }

    logging.info("Result: %s", number_of_classes_per_attribute)

    return number_of_classes_per_attribute


def get_fewshot_subsampled_dataset(
    dataset: Dataset,
    s3_cache_folder: str,
    number_datapoints_per_class: int,
    maxsize: int,
    maxcount: int,
    target_attribute: str,
    number_of_classes_per_attribute: Union[int, Dict[str, int]],
    force_create: bool = False,
) -> Dataset:
    total_datapoints = None
    if not s3_folder_exists_and_not_empty(s3_cache_folder) or force_create:
        LOGGER.info(
            "Subsampling webdataset. You might see an 'Exception ignored' message (expected)."
        )
        if isinstance(number_of_classes_per_attribute, int):
            number_of_classes_per_attribute = {target_attribute: number_of_classes_per_attribute}
        number_of_classes = number_of_classes_per_attribute[target_attribute]
        with S3ShardWriter(
            s3_cache_folder,
            pattern="shard-%06d.tar",
            maxsize=maxsize,
            maxcount=maxcount,
            keep_meta=True,
        ) as dataset_writer:
            datapoints_per_class = collections.defaultdict(lambda: 0)
            progress_iterator = tqdm.tqdm(iter(dataset))
            for data in progress_iterator:
                image = data["image"]
                label = data[target_attribute]
                if datapoints_per_class[label] == number_datapoints_per_class:
                    continue
                datapoints_per_class[label] += 1
                per_class_numbers = [n for n in datapoints_per_class.values()]
                dataset_writer.write(
                    {
                        "__key__": f"{(sum(per_class_numbers)-1):07d}",
                        "cls": label,
                        "jpg": image,
                    }
                )

                # Compute number of elements in least-seen class.
                # (the product ensures that it is 0 if we have not seen one class)
                min_elements_per_class = min(per_class_numbers) * (
                    len(per_class_numbers) == number_of_classes
                )
                description = (
                    "Generating fewshot dataset - "
                    + f"Min. class: {min_elements_per_class}/{number_datapoints_per_class}, "
                    + f"Max. class: {max(per_class_numbers)}/{number_datapoints_per_class}, "
                    + f"Seen classes: {len(per_class_numbers)}/{number_of_classes}"
                )
                progress_iterator.set_description(description)

                if sum(per_class_numbers) == number_of_classes * number_datapoints_per_class:
                    break

            total_datapoints = sum(per_class_numbers)

    subsampled_dataset = get_webdataset(
        s3_list(s3_cache_folder),
        number_of_classes_per_attribute=number_of_classes_per_attribute,
        number_of_datapoints=total_datapoints,
    )
    return subsampled_dataset
