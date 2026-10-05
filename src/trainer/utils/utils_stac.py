import os
from typing import Any
from urllib.parse import urlparse

import boto3
from pystac import Catalog, Link
from pystac.stac_io import DefaultStacIO, StacIO

s3 = boto3.client("s3")


class CustomStacIO(DefaultStacIO):
    """From: https://pystac.readthedocs.io/en/stable/concepts.html#i-o-in-pystac"""

    def __init__(self):
        self.s3 = boto3.resource("s3")
        super().__init__()

    def read_text(self, source: str | Link, *args: Any, **kwargs: Any) -> str:
        """Reads texts"""
        parsed = urlparse(source)
        if parsed.scheme == "s3":
            bucket = parsed.netloc
            key = parsed.path[1:]

            obj = self.s3.Object(bucket, key)
            return obj.get()["Body"].read().decode("utf-8")
        else:
            return super().read_text(source, *args, **kwargs)

    def write_text(self, dest: str | Link, txt: str, *args: Any, **kwargs: Any) -> None:
        """Writes text"""
        parsed = urlparse(dest)
        if parsed.scheme == "s3":
            bucket = parsed.netloc
            key = parsed.path[1:]
            self.s3.Object(bucket, key).put(Body=txt, ContentEncoding="utf-8")
        else:
            super().write_text(dest, txt, *args, **kwargs)


StacIO.set_default(CustomStacIO)


class STACReader:
    """reads from STAC catalog"""

    def __init__(
        self, bucket: str = "fimc-data", catalog: str = "s3://fimc-data/benchmark/stac-bench-cat/catalog.json"
    ) -> None:
        self.bucket = bucket
        self.root_catalog = Catalog.from_file(catalog)

    def read_stac(
        self,
        collection: str,
        item_id: str,
        asset_id: str,
        output_dir: str,
        output_name: str = None,
    ) -> str:
        """Reads an asset from a known collection and item.

        Args:
            collection (str): valid collection name in STAC
            item_id (str): valid item id in STAC
            asset_id (str): valid asset id in STAC
            output_dir (str): directory to save to
            output_name (str, optional): output file name. Defaults to name of asset file.

        Returns
        -------
            str: output path of saved file
        """
        col = self.root_catalog.get_child(collection)

        # get the asset path and split it from the bucket
        href = col.get_item(item_id).get_assets()[asset_id].href
        asset_path = href.split(f"{self.bucket}/")[-1]

        # if no output name, take file name from href
        if not output_name:
            output_name = href.split("/")[-1]
        output_file = os.path.join(output_dir, output_name)

        try:
            s3.download_file(self.bucket, asset_path, output_file)
            print(f"Downloaded {self.bucket}/{asset_path} to {output_file}")
            return output_file
        except Exception:
            # force raise because jupyter wasn't raising
            raise
