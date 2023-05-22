import utils
import concurrent.futures
import argparse

args = argparse.ArgumentParser()

args.add_argument("output_dir", default="./data/videos")
args.add_argument("max_files", type=int, default=10)
args.add_argument("--tmp_path", type=str, default="tmp")
args.add_argument("--offset", type=int, default=0)
args_parsed = args.parse_args()

# for youtube_id, _ in zip(utils.youtube_ids(), range(10)):
#     print(youtube_id)

with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    futures = []
    for youtube_url, _ in zip(utils.youtube_ids(offset=args_parsed.offset), range(args_parsed.max_files)):
        print(youtube_url)
        try:
            futures.append(executor.submit(utils.download_video, youtube_url=youtube_url, output_dir=args_parsed.output_dir, tmp_path=args_parsed.tmp_path))
        except:
            pass
        # break

    for future in concurrent.futures.as_completed(futures):
        print(future.result())

# utils.download_videos([url for url, _ in zip(utils.youtube_urls(), range(10))])

