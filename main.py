import utils
import concurrent.futures
import argparse

args = argparse.ArgumentParser()

args.add_argument("output_dir", default="./data/videos")
args.add_argument("max_files", type=int, default=10)
args_parsed = args.parse_args()

# for youtube_id, _ in zip(utils.youtube_ids(), range(10)):
#     print(youtube_id)

with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    futures = []
    for youtube_url, _ in zip(utils.youtube_ids(), range(args_parsed.max_files)):
        print(youtube_url)
        futures.append(executor.submit(utils.download_video, youtube_url=youtube_url, output_dir=args_parsed.output_dir))
        # break

    for future in concurrent.futures.as_completed(futures):
        print(future.result())

# utils.download_videos([url for url, _ in zip(utils.youtube_urls(), range(10))])

