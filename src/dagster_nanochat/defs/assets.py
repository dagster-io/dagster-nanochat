import dagster as dg
import os
from dagster_nanochat.utils.file_downloader import download_file


BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle"
MAX_SHARDS = 1823
FILE_DIRECTORY = "data/raw"

shards = [f"{BASE_URL}/resolve/main/shard_{i:05d}.parquet" for i in range(MAX_SHARDS)]

training_set = shards[:-1]
validation_set = shards[-1:]


base_parquet_files = dg.StaticPartitionsDefinition(shards)


raw_data = dg.AssetSpec(
    "huggface_karpathy_datasets",
    description="Raw data from Hugging Face Karpathy datasets",
    metadata={
        "url": dg.MetadataValue.url(BASE_URL),
    },
)


@dg.asset(
    deps=[raw_data],
    partitions_def=training_set,
)
def training_files(context: dg.AssetExecutionContext):
    url_path = context.partition_key
    file_path = os.path.join(FILE_DIRECTORY, os.path.basename(url_path))

    download_file(url_path, file_path)


@dg.asset(
    deps=[raw_data],
)
def validation_files(context: dg.AssetExecutionContext):
    url_path = validation_set[0]
    file_path = os.path.join(FILE_DIRECTORY, os.path.basename(url_path))

    download_file(url_path, file_path)
