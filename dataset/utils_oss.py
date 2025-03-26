import webdataset
# from aoss_client.client import Client


# from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Set, Tuple
# def url_opener_oss(oss_config):
#     oss_client = Client(conf_path=oss_config)
#     def helper(
#         data: Iterable[Dict[str, Any]],
#         handler: Callable[[Exception], bool] = webdataset.handlers.reraise_exception,
#         **kw: Dict[str, Any],
#     ):
#         for sample in data:
#             assert isinstance(sample, dict), sample
#             assert "url" in sample
#             url = sample["url"]
#             try:
#                 if url.startswith('cluster1') or url.startswith('s3://'):
#                     stream = oss_client.get(url, enable_stream=True)
#                 else:
#                     stream = webdataset.tariterators.gopen.gopen(url, **kw)
#                 sample.update(stream=stream)
#                 yield sample
#             except Exception as exn:
#                 exn.args = exn.args + (url,)
#                 if handler(exn):
#                     continue
#                 else:
#                     break
#     return helper